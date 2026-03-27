"""视频预测模型：预测未来图像帧。"""
from __future__ import annotations

import torch
import torch.nn as nn


class VideoPredictor(nn.Module):
    """视频预测世界模型。

    给定历史图像帧 + 动作序列，预测未来帧。
    用于：
    - 视觉 MPC
    - 数据增强
    - 奖励学习

    架构：轻量 ConvLSTM / Transformer-based。
    这里实现 latent-space 版本（先编码再预测）。
    """

    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 64,
        action_dim: int = 7,
        latent_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # 图像编码器 (简化版 CNN)
        self.img_encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
        )

        # 图像解码器
        self.img_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 4 * 4),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(image_size),
            nn.Conv2d(16, image_channels, 3, 1, 1),
        )

        # 动力学 (latent space)
        self.dynamics = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.hidden_dim = hidden_dim

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, latent_dim)"""
        return self.img_encoder(images)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """(B, latent_dim) → (B, C, H, W)"""
        return self.img_decoder(z)

    def predict_next(
        self, z: torch.Tensor, action: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单步预测。返回 (z_next, h_next)。"""
        inp = torch.cat([z, action], dim=-1)
        h_next = self.dynamics(inp, h)
        z_next = self.latent_proj(h_next)
        return z_next, h_next

    def rollout_latent(
        self, init_image: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """多步 latent rollout。

        Args:
            init_image: (B, C, H, W)
            actions: (B, T, action_dim)
        Returns:
            z_seq: (B, T, latent_dim)
        """
        B, T, _ = actions.shape
        z = self.encode(init_image)
        h = torch.zeros(B, self.hidden_dim, device=z.device)

        z_list = []
        for t in range(T):
            z, h = self.predict_next(z, actions[:, t], h)
            z_list.append(z)
        return torch.stack(z_list, dim=1)

    def predict_video(
        self, init_image: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """预测未来视频帧。

        Returns:
            images: (B, T, C, H, W)
        """
        z_seq = self.rollout_latent(init_image, actions)
        B, T, D = z_seq.shape
        images = self.decode(z_seq.reshape(B * T, D))
        C, H, W = images.shape[1:]
        return images.reshape(B, T, C, H, W)

    def compute_loss(
        self,
        image_seq: torch.Tensor,
        action_seq: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            image_seq: (B, T+1, C, H, W)
            action_seq: (B, T, action_dim)
        """
        B, T_plus_1, C, H, W = image_seq.shape
        T = T_plus_1 - 1

        pred = self.predict_video(image_seq[:, 0], action_seq[:, :T])
        target = image_seq[:, 1:]

        recon_loss = nn.functional.mse_loss(pred, target)
        return {"loss": recon_loss, "recon_loss": recon_loss}
