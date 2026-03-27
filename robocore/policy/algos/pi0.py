"""Pi0 策略实现：基于 Flow Matching 的 VLA。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.registry import PolicyRegistry
from robocore.policy.schedulers import FlowMatchingScheduler


@PolicyRegistry.register("pi0")
class Pi0Policy(BasePolicy):
    """Pi0: Physical Intelligence 的 VLA 模型。

    核心特点：
    1. 基于预训练 VLM 的 action expert
    2. 使用 Flow Matching 生成连续动作（而非离散 token）
    3. 支持多模态输入（图像 + 语言 + 本体感知）
    4. Action chunking 输出

    架构：
    - VLM backbone（PaliGemma 等）处理图像+语言
    - Action expert（独立 Transformer）处理动作生成
    - Cross-attention 连接 VLM 和 action expert
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 16,
        action_horizon: int = 16,
        device: str | torch.device = "cuda",
        # 网络参数
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        # Flow matching 参数
        num_inference_steps: int = 10,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_inference_steps = num_inference_steps

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_model(self) -> None:
        """构建 Pi0 模型。"""
        # 观测编码
        obs_input_dim = self.obs_dim if self.obs_dim else 256
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 时间步编码
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 噪声动作投影
        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)

        # 位置编码
        self.pos_embed = nn.Embedding(self.pred_horizon, self.hidden_dim)

        # Action Expert: Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.action_expert = nn.TransformerDecoder(
            decoder_layer, num_layers=self.num_layers
        )

        # 输出头：预测速度场
        self.velocity_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

        # Flow matching 调度器
        self.scheduler = FlowMatchingScheduler(
            num_steps=self.num_inference_steps
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """编码观测为条件 token 序列。"""
        if "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state[:, -1]
            return self.obs_encoder(state).unsqueeze(1)  # (B, 1, hidden)
        batch_size = 1
        return torch.zeros(batch_size, 1, self.hidden_dim, device=self._device)

    def _predict_velocity(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """预测速度场。"""
        batch_size = noisy_action.shape[0]

        # 时间步编码
        t_emb = self.time_embed(timestep.float().unsqueeze(-1))  # (B, hidden)
        t_token = t_emb.unsqueeze(1)  # (B, 1, hidden)

        # 条件 = obs + time
        memory = torch.cat([cond, t_token], dim=1)  # (B, 2, hidden)

        # 动作 token
        action_feat = self.action_proj(noisy_action)  # (B, T, hidden)
        positions = torch.arange(self.pred_horizon, device=noisy_action.device)
        action_feat = action_feat + self.pos_embed(positions).unsqueeze(0)

        # Transformer 解码
        decoded = self.action_expert(action_feat, memory)

        # 输出速度
        velocity = self.velocity_head(decoded)
        return velocity

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """推理：Flow matching ODE 求解。"""
        cond = self._encode_obs(obs)
        batch_size = cond.shape[0]
        device = cond.device

        # 从噪声开始
        x = torch.randn(batch_size, self.pred_horizon, self.action_dim, device=device)

        # Euler 步进
        dt = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = torch.full((batch_size,), step / self.num_inference_steps, device=device)
            velocity = self._predict_velocity(x, t, cond)
            x = x + velocity * dt

        return PolicyOutput(action=x)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """计算 flow matching 损失。"""
        cond = self._encode_obs(obs)
        batch_size = action.shape[0]
        device = action.device

        # 采样时间步 t ~ U(0, 1)
        t = torch.rand(batch_size, device=device)

        # 采样噪声
        noise = torch.randn_like(action)

        # 构造 x_t = (1-t) * noise + t * action
        t_expand = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t_expand) * noise + t_expand * action

        # 目标速度 = action - noise
        target_velocity = action - noise

        # 预测速度
        pred_velocity = self._predict_velocity(x_t, t, cond)

        # MSE 损失
        loss = nn.functional.mse_loss(pred_velocity, target_velocity)

        return {"loss": loss}
