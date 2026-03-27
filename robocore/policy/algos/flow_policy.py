"""Flow Policy 实现：基于 Flow Matching 的策略。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import StateEncoder
from robocore.policy.networks.unet_1d import ConditionalUNet1d
from robocore.policy.registry import PolicyRegistry
from robocore.policy.schedulers import FlowMatchingScheduler


@PolicyRegistry.register("flow_policy")
class FlowPolicy(BasePolicy):
    """Flow Policy：基于 Flow Matching 的动作生成。

    相比 Diffusion Policy：
    - 训练目标是 velocity field 而非噪声
    - 推理用 ODE 求解（Euler 步进），通常只需 10 步
    - 速度快 7-10x
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str | torch.device = "cuda",
        # 网络参数
        hidden_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        # Flow matching 参数
        num_inference_steps: int = 10,
        sigma_min: float = 1e-4,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.down_dims = down_dims
        self.num_inference_steps = num_inference_steps
        self.sigma_min = sigma_min

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_model(self) -> None:
        """构建网络。"""
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
            cond_dim = self.hidden_dim

        # 速度场预测网络（复用 U-Net 架构）
        self.velocity_net = ConditionalUNet1d(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            down_dims=self.down_dims,
        )

        # Flow matching 调度器
        self.scheduler = FlowMatchingScheduler(
            num_steps=self.num_inference_steps,
            sigma_min=self.sigma_min,
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """编码观测。"""
        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state.flatten(1)
            return self.state_encoder(state)
        return torch.zeros(obs["state"].shape[0], self.hidden_dim, device=self._device)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """推理：ODE 求解生成动作。"""
        cond = self._encode_obs(obs)
        batch_size = cond.shape[0]
        device = cond.device

        # 从噪声开始 (t=0)
        x = torch.randn(batch_size, self.pred_horizon, self.action_dim, device=device)

        # Euler 步进 (t: 0 → 1)
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            velocity = self.velocity_net(x, t, cond)
            x = x + velocity * dt

        return PolicyOutput(action=x)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """计算 flow matching 损失。"""
        cond = self._encode_obs(obs)
        batch_size = action.shape[0]
        device = action.device

        # 采样噪声和时间步
        noise = torch.randn_like(action)
        timesteps = self.scheduler.get_timesteps(batch_size, device)

        # 构造 x_t = (1-t) * noise + t * action
        x_t = self.scheduler.add_noise(action, noise, timesteps)

        # 目标速度 = action - noise
        target_velocity = self.scheduler.get_velocity_target(action, noise)

        # 预测速度
        pred_velocity = self.velocity_net(x_t, timesteps, cond)

        # MSE 损失
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)

        return {"loss": loss}
