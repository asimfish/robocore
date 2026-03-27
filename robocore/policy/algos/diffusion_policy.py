"""Diffusion Policy 实现。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import ImageEncoder, MultiViewEncoder, StateEncoder
from robocore.policy.networks.unet_1d import ConditionalUNet1d
from robocore.policy.registry import PolicyRegistry
from robocore.policy.schedulers import DDIMScheduler, DDPMScheduler


@PolicyRegistry.register("dp")
class DiffusionPolicy(BasePolicy):
    """Diffusion Policy：基于 DDPM 的动作生成。

    核心思路：
    1. 编码观测 → 条件向量
    2. 训练：给 GT 动作加噪 → U-Net 预测噪声 → MSE 损失
    3. 推理：从纯噪声开始 → 迭代去噪 → 得到动作序列

    支持：
    - 状态观测 / 图像观测 / 多视角图像
    - DDPM / DDIM 采样
    - Action chunking + temporal ensemble
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
        diffusion_step_embed_dim: int = 128,
        # 扩散参数
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        noise_schedule: str = "squaredcos_cap_v2",
        # 图像编码器
        image_encoder: str = "resnet18",
        image_keys: list[str] | None = None,
        freeze_image_encoder: bool = False,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.down_dims = down_dims
        self.diffusion_step_embed_dim = diffusion_step_embed_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps
        self.noise_schedule = noise_schedule
        self.image_encoder_name = image_encoder
        self.image_keys = image_keys or []
        self.freeze_image_encoder = freeze_image_encoder

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
            self.image_encoder = MultiViewEncoder(
                camera_names=self.image_keys,
                backbone=self.image_encoder_name,
                per_view_dim=self.hidden_dim,
                freeze=self.freeze_image_encoder,
            )
            cond_dim += self.image_encoder.output_dim
        else:
            self.image_encoder = None

        # 如果没有任何编码器，用一个默认维度
        if cond_dim == 0:
            cond_dim = self.hidden_dim

        # 噪声预测网络
        self.noise_pred_net = ConditionalUNet1d(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            diffusion_step_embed_dim=self.diffusion_step_embed_dim,
            down_dims=self.down_dims,
        )

        # 噪声调度器
        self.train_scheduler = DDPMScheduler(
            num_steps=self.num_diffusion_steps,
            beta_schedule=self.noise_schedule,
        )
        self.inference_scheduler = DDIMScheduler(
            num_train_steps=self.num_diffusion_steps,
            num_inference_steps=self.num_inference_steps,
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """编码观测为条件向量。"""
        features = []

        # 状态编码
        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]  # (batch, obs_horizon, state_dim)
            if state.ndim == 3:
                state = state.flatten(1)  # (batch, obs_horizon * state_dim)
            features.append(self.state_encoder(state))

        # 图像编码
        if self.image_encoder is not None:
            image_dict = {
                k.replace("image_", ""): v
                for k, v in obs.items()
                if k.startswith("image_")
            }
            if image_dict:
                # 取最后一帧
                for k, v in image_dict.items():
                    if v.ndim == 5:  # (batch, T, C, H, W)
                        image_dict[k] = v[:, -1]
                features.append(self.image_encoder(image_dict))

        if not features:
            raise ValueError("No valid observation found for encoding")

        return torch.cat(features, dim=-1)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """推理：从噪声生成动作序列。"""
        cond = self._encode_obs(obs)
        batch_size = cond.shape[0]
        device = cond.device

        # 从纯噪声开始
        noisy_action = torch.randn(
            batch_size, self.pred_horizon, self.action_dim, device=device
        )

        # DDIM 去噪
        for t in self.inference_scheduler.inference_timesteps:
            timestep = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(noisy_action, timestep, cond)
            noisy_action = self.inference_scheduler.step(noise_pred, t.item(), noisy_action)

        return PolicyOutput(action=noisy_action)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """计算扩散损失。"""
        cond = self._encode_obs(obs)
        batch_size = action.shape[0]
        device = action.device

        # 采样时间步
        timesteps = self.train_scheduler.get_timesteps(batch_size, device)

        # 采样噪声
        noise = torch.randn_like(action)

        # 加噪
        noisy_action = self.train_scheduler.add_noise(action, noise, timesteps)

        # 预测噪声
        noise_pred = self.noise_pred_net(noisy_action, timesteps, cond)

        # MSE 损失
        loss = nn.functional.mse_loss(noise_pred, noise)

        return {"loss": loss}
