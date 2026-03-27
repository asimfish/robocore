"""DP3 (3D Diffusion Policy) 实现。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import StateEncoder
from robocore.policy.networks.unet_1d import ConditionalUNet1d
from robocore.policy.registry import PolicyRegistry
from robocore.policy.schedulers import DDIMScheduler, DDPMScheduler


class SimplePointNetEncoder(nn.Module):
    """简化版 PointNet 编码器。"""

    def __init__(self, input_dim: int = 3, output_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.projection = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, N, input_dim) 点云
        Returns:
            (batch, output_dim) 全局特征
        """
        feat = self.mlp(x)  # (batch, N, 256)
        global_feat = feat.max(dim=1)[0]  # (batch, 256)
        return self.projection(global_feat)


@PolicyRegistry.register("dp3")
class DP3Policy(BasePolicy):
    """DP3: 3D Diffusion Policy。

    相比 DP 的改进：
    - 使用 3D 点云作为观测（而非 2D 图像）
    - PointNet 编码器提取 3D 几何特征
    - 对 SE(3) 空间的操作更自然
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
        # 点云参数
        point_dim: int = 3,  # xyz or xyz+rgb
        num_points: int = 1024,
        # 扩散参数
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.down_dims = down_dims
        self.point_dim = point_dim
        self.num_points = num_points
        self.num_diffusion_steps = num_diffusion_steps
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
        cond_dim = 0

        # 点云编码器
        self.pointcloud_encoder = SimplePointNetEncoder(
            input_dim=self.point_dim,
            output_dim=self.hidden_dim,
        )
        cond_dim += self.hidden_dim

        # 状态编码器（可选）
        if self.obs_dim is not None and self.obs_dim > 0:
            self.state_encoder = StateEncoder(
                input_dim=self.obs_dim * self.obs_horizon,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
            cond_dim += self.hidden_dim
        else:
            self.state_encoder = None

        # 噪声预测网络
        self.noise_pred_net = ConditionalUNet1d(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            down_dims=self.down_dims,
        )

        # 调度器
        self.train_scheduler = DDPMScheduler(num_steps=self.num_diffusion_steps)
        self.inference_scheduler = DDIMScheduler(
            num_train_steps=self.num_diffusion_steps,
            num_inference_steps=self.num_inference_steps,
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # 点云编码
        if "pointcloud" in obs:
            pc = obs["pointcloud"]
            if pc.ndim == 4:  # (batch, obs_horizon, N, D)
                pc = pc[:, -1]  # 取最后一帧
            features.append(self.pointcloud_encoder(pc))

        # 状态编码
        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state.flatten(1)
            features.append(self.state_encoder(state))

        if not features:
            raise ValueError("DP3 requires pointcloud observation")

        return torch.cat(features, dim=-1)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        batch_size = cond.shape[0]
        device = cond.device

        noisy_action = torch.randn(
            batch_size, self.pred_horizon, self.action_dim, device=device
        )

        for t in self.inference_scheduler.inference_timesteps:
            timestep = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            noise_pred = self.noise_pred_net(noisy_action, timestep, cond)
            noisy_action = self.inference_scheduler.step(noise_pred, t.item(), noisy_action)

        return PolicyOutput(action=noisy_action)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        batch_size = action.shape[0]
        device = action.device

        timesteps = self.train_scheduler.get_timesteps(batch_size, device)
        noise = torch.randn_like(action)
        noisy_action = self.train_scheduler.add_noise(action, noise, timesteps)
        noise_pred = self.noise_pred_net(noisy_action, timesteps, cond)

        loss = nn.functional.mse_loss(noise_pred, noise)
        return {"loss": loss}
