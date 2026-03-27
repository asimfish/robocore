"""Flow Policy 变种集合。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import StateEncoder
from robocore.policy.networks.dit import DiTActionModel
from robocore.policy.networks.unet_1d import ConditionalUNet1d
from robocore.policy.registry import PolicyRegistry


# ============================================================
# 共享的 Flow 基类
# ============================================================

class _FlowBase(BasePolicy):
    """Flow Policy 变种的共享基类。"""

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        num_inference_steps: int = 10,
        backbone: str = "unet",  # "unet" or "dit"
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_inference_steps = num_inference_steps
        self.backbone = backbone
        self._extra_kwargs = kwargs
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_encoder(self) -> int:
        cond_dim = 0
        if self.obs_dim is not None and self.obs_dim > 0:
            self.state_encoder = StateEncoder(
                input_dim=self.obs_dim * self.obs_horizon,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
            cond_dim += self.hidden_dim
        else:
            self.state_encoder = None
        return cond_dim

    def _build_velocity_net(self, cond_dim: int) -> nn.Module:
        if self.backbone == "dit":
            return DiTActionModel(
                action_dim=self.action_dim,
                cond_dim=cond_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self._extra_kwargs.get("num_layers", 4),
                num_heads=self._extra_kwargs.get("num_heads", 4),
                max_horizon=self.pred_horizon,
            )
        else:
            return ConditionalUNet1d(
                action_dim=self.action_dim,
                cond_dim=cond_dim,
                down_dims=self._extra_kwargs.get("down_dims", (256, 512, 1024)),
            )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []
        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state.flatten(1)
            features.append(self.state_encoder(state))
        if not features:
            raise ValueError("No observation provided")
        return torch.cat(features, dim=-1)


# ============================================================
# MeanFlow: 一步生成的 Flow Policy
# ============================================================

@PolicyRegistry.register("mean_flow")
class MeanFlowPolicy(_FlowBase):
    """MeanFlow: 均值流策略。

    核心思想（Geng et al. 2025）：
    - 训练一个网络直接预测 "平均速度场"
    - 推理时只需 1 步 Euler 即可生成动作（极快）
    - 训练目标：v_theta(x_t, t) 应该等于从 x_t 到 x_1 的平均速度

    关键公式：
    - 平均速度 = (x_1 - x_t) / (1 - t)
    - 损失 = MSE(v_theta(x_t, t), (x_1 - x_t) / (1 - t))
    - 推理：x_1 = x_0 + v_theta(x_0, 0)  （一步！）
    """

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        self.velocity_net = self._build_velocity_net(cond_dim)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        # 从噪声开始
        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)

        if self.num_inference_steps == 1:
            # 一步生成（MeanFlow 的核心优势）
            t = torch.zeros(B, device=device)
            velocity = self.velocity_net(x, t, cond)
            action = x + velocity
        else:
            # 多步 Euler（更精确但更慢）
            dt = 1.0 / self.num_inference_steps
            for step in range(self.num_inference_steps):
                t = torch.full((B,), step * dt, device=device)
                velocity = self.velocity_net(x, t, cond)
                x = x + velocity * dt
            action = x

        return PolicyOutput(action=action)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        # 采样 t ~ U(0, 1)，避免 t=1
        t = torch.rand(B, device=device) * 0.999

        noise = torch.randn_like(action)
        t_expand = t.unsqueeze(-1).unsqueeze(-1)

        # x_t = (1-t) * noise + t * action
        x_t = (1 - t_expand) * noise + t_expand * action

        # 平均速度目标 = (action - x_t) / (1 - t)
        mean_velocity = (action - x_t) / (1 - t_expand + 1e-8)

        # 预测
        pred_velocity = self.velocity_net(x_t, t, cond)

        loss = nn.functional.mse_loss(pred_velocity, mean_velocity)
        return {"loss": loss}


# ============================================================
# ConsistencyFlow: 一致性蒸馏 + Flow Matching
# ============================================================

@PolicyRegistry.register("consistency_flow")
class ConsistencyFlowPolicy(_FlowBase):
    """Consistency Flow Policy。

    结合 Consistency Models 和 Flow Matching：
    - 训练一个自一致的模型：f(x_t, t) 在同一条 ODE 轨迹上输出一致
    - 推理时可以 1-2 步生成
    - 比纯 flow matching 更快，比纯 consistency model 更稳定

    训练方式：
    1. 先训练标准 flow matching teacher
    2. 蒸馏到 consistency model（或直接 consistency training）
    """

    def __init__(self, ema_decay: float = 0.999, **kwargs: Any):
        self.ema_decay = ema_decay
        super().__init__(**kwargs)

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        # 在线网络
        self.velocity_net = self._build_velocity_net(cond_dim)
        # EMA 目标网络
        self.target_net = self._build_velocity_net(cond_dim)
        # 初始化 target = online
        self.target_net.load_state_dict(self.velocity_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False

    def _update_target(self) -> None:
        """EMA 更新目标网络。"""
        for p_online, p_target in zip(
            self.velocity_net.parameters(), self.target_net.parameters()
        ):
            p_target.data.mul_(self.ema_decay).add_(
                p_online.data, alpha=1 - self.ema_decay
            )

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)

        # 一步或少步生成
        dt = 1.0 / max(self.num_inference_steps, 1)
        for step in range(self.num_inference_steps):
            t = torch.full((B,), step * dt, device=device)
            velocity = self.velocity_net(x, t, cond)
            x = x + velocity * dt

        return PolicyOutput(action=x)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        # 采样两个相邻时间步
        t = torch.rand(B, device=device) * 0.9
        dt = 1.0 / self.num_inference_steps
        t_next = (t + dt).clamp(max=1.0)

        noise = torch.randn_like(action)
        t_exp = t.unsqueeze(-1).unsqueeze(-1)
        t_next_exp = t_next.unsqueeze(-1).unsqueeze(-1)

        x_t = (1 - t_exp) * noise + t_exp * action
        x_t_next = (1 - t_next_exp) * noise + t_next_exp * action

        # 在线网络在 t 处的预测
        pred_t = self.velocity_net(x_t, t, cond)

        # 目标网络在 t+dt 处的预测（不计算梯度）
        with torch.no_grad():
            pred_next = self.target_net(x_t_next, t_next, cond)

        # 一致性损失：两个预测应该一致
        consistency_loss = nn.functional.mse_loss(pred_t, pred_next)

        # 标准 flow matching 损失
        target_velocity = action - noise
        flow_loss = nn.functional.mse_loss(pred_t, target_velocity)

        loss = flow_loss + 0.5 * consistency_loss

        # 更新 EMA
        self._update_target()

        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "consistency_loss": consistency_loss,
        }


# ============================================================
# RectifiedFlow: 直线化 Flow
# ============================================================

@PolicyRegistry.register("rectified_flow")
class RectifiedFlowPolicy(_FlowBase):
    """Rectified Flow Policy。

    核心思想（Liu et al. 2023）：
    - 标准 flow matching 的 ODE 轨迹可能弯曲
    - Rectified Flow 通过 "reflow" 操作让轨迹变直
    - 直线轨迹 → 更少的步数就能精确采样

    训练流程：
    1. 先训练标准 flow matching
    2. 用训练好的模型生成 (noise, data) 配对
    3. 在配对上重新训练（reflow），轨迹变直
    4. 可以多次 reflow

    这里实现的是 1-rectified flow（一次 reflow）。
    """

    def __init__(self, reflow_iter: int = 0, **kwargs: Any):
        self.reflow_iter = reflow_iter
        super().__init__(**kwargs)

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        self.velocity_net = self._build_velocity_net(cond_dim)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t = torch.full((B,), step * dt, device=device)
            velocity = self.velocity_net(x, t, cond)
            x = x + velocity * dt

        return PolicyOutput(action=x)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        t = torch.rand(B, device=device)
        noise = torch.randn_like(action)
        t_exp = t.unsqueeze(-1).unsqueeze(-1)

        # 线性插值
        x_t = (1 - t_exp) * noise + t_exp * action

        # 目标：直线速度 = action - noise（与 t 无关）
        target = action - noise

        pred = self.velocity_net(x_t, t, cond)
        loss = nn.functional.mse_loss(pred, target)

        # 可选：直线度正则化
        # 鼓励速度场在不同 t 处保持一致
        if self.training and self.reflow_iter > 0:
            t2 = torch.rand(B, device=device)
            t2_exp = t2.unsqueeze(-1).unsqueeze(-1)
            x_t2 = (1 - t2_exp) * noise + t2_exp * action
            pred2 = self.velocity_net(x_t2, t2, cond)
            straightness_loss = nn.functional.mse_loss(pred, pred2.detach())
            loss = loss + 0.1 * straightness_loss

        return {"loss": loss}

    def generate_reflow_pairs(
        self, obs: dict[str, torch.Tensor], num_samples: int = 100
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成 reflow 配对数据。

        用当前模型从噪声生成数据，形成 (noise, generated_data) 配对。
        """
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        noise_list, data_list = [], []
        for _ in range(num_samples // B + 1):
            noise = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
            x = noise.clone()
            dt = 1.0 / self.num_inference_steps
            for step in range(self.num_inference_steps):
                t = torch.full((B,), step * dt, device=device)
                velocity = self.velocity_net(x, t, cond)
                x = x + velocity * dt
            noise_list.append(noise)
            data_list.append(x)

        return torch.cat(noise_list)[:num_samples], torch.cat(data_list)[:num_samples]


# ============================================================
# Flow-DiT: Flow Matching with DiT backbone
# ============================================================

@PolicyRegistry.register("flow_dit")
class FlowDiTPolicy(_FlowBase):
    """Flow Matching + DiT backbone。

    标准 flow matching 但使用 DiT 替代 U-Net。
    """

    def __init__(self, **kwargs: Any):
        kwargs["backbone"] = "dit"
        super().__init__(**kwargs)

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        self.velocity_net = self._build_velocity_net(cond_dim)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t = torch.full((B,), step * dt, device=device)
            velocity = self.velocity_net(x, t, cond)
            x = x + velocity * dt

        return PolicyOutput(action=x)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        t = torch.rand(B, device=device)
        noise = torch.randn_like(action)
        t_exp = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t_exp) * noise + t_exp * action
        target = action - noise

        pred = self.velocity_net(x_t, t, cond)
        loss = nn.functional.mse_loss(pred, target)
        return {"loss": loss}
