"""DiT Policy: 基于 Diffusion Transformer 的策略。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import ImageEncoder, StateEncoder
from robocore.policy.networks.dit import DiTActionModel
from robocore.policy.registry import PolicyRegistry
from robocore.policy.schedulers import DDIMScheduler, DDPMScheduler


@PolicyRegistry.register("dit")
class DiTPolicy(BasePolicy):
    """DiT Policy: Diffusion Transformer 策略。

    用 DiT (Peebles & Xie, 2023) 架构做动作生成：
    - AdaLN-Zero 条件注入
    - Cross-attention 融合观测
    - 比 U-Net 更好的 scaling 特性

    支持：
    - 状态观测 / 图像观测
    - DDPM / DDIM 采样
    - 可配置深度和宽度
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str | torch.device = "cuda",
        # DiT 参数
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        # 扩散参数
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        # 图像
        image_encoder: str = "resnet18",
        image_keys: list[str] | None = None,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps
        self.image_encoder_name = image_encoder
        self.image_keys = image_keys or []

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_model(self) -> None:
        cond_dim = 0

        # 状态编码器
        if self.obs_dim is not None and self.obs_dim > 0:
            self.state_encoder = StateEncoder(
                input_dim=self.obs_dim * self.obs_horizon,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
            cond_dim += self.hidden_dim
        else:
            self.state_encoder = None

        # 图像编码器
        if self.image_keys:
            self.image_encoder = ImageEncoder(
                backbone=self.image_encoder_name,
                output_dim=self.hidden_dim,
            )
            cond_dim += self.hidden_dim * len(self.image_keys)
        else:
            self.image_encoder = None

        # DiT 主网络
        self.dit = DiTActionModel(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_horizon=self.pred_horizon,
            mlp_ratio=self.mlp_ratio,
            dropout=self.dropout,
        )

        # 调度器
        self.train_scheduler = DDPMScheduler(num_steps=self.num_diffusion_steps)
        self.inference_scheduler = DDIMScheduler(
            num_train_steps=self.num_diffusion_steps,
            num_inference_steps=self.num_inference_steps,
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state.flatten(1)
            features.append(self.state_encoder(state))

        if self.image_encoder is not None:
            for key in self.image_keys:
                if key in obs:
                    img = obs[key]
                    if img.ndim == 5:  # (B, T, C, H, W)
                        img = img[:, -1]
                    features.append(self.image_encoder(img))

        if not features:
            raise ValueError("No observation provided")
        return torch.cat(features, dim=-1)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)

        for t in self.inference_scheduler.inference_timesteps:
            ts = torch.full((B,), t.item(), device=device, dtype=torch.long)
            noise_pred = self.dit(x, ts, cond)
            x = self.inference_scheduler.step(noise_pred, t.item(), x)

        return PolicyOutput(action=x)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        timesteps = self.train_scheduler.get_timesteps(B, device)
        noise = torch.randn_like(action)
        noisy = self.train_scheduler.add_noise(action, noise, timesteps)

        pred = self.dit(noisy, timesteps, cond)
        loss = nn.functional.mse_loss(pred, noise)

        return {"loss": loss}
