"""IMLE Policy：基于 Implicit Maximum Likelihood Estimation 的策略。

参考：Luo et al. 2025 "IMLE Policy: Fast and Sample Efficient Visuomotor
Policy Learning via Implicit Maximum Likelihood Estimation"
(https://arxiv.org/abs/2502.12371)

核心思想：
- 生成器 G(obs, z) 从随机噪声 z 直接映射到动作序列（单步前向）
- 训练用 IMLE：对每个 GT 动作，从 K 个噪声样本中找最近邻，只回传最近邻的梯度
- 推理只需 1 次前向，比 Diffusion (10-100步) 和 Flow (10步) 快得多
- 天然支持多模态动作分布
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import StateEncoder
from robocore.policy.networks.unet_1d import ConditionalUNet1d
from robocore.policy.registry import PolicyRegistry


@PolicyRegistry.register("imle")
class IMLEPolicy(BasePolicy):
    """IMLE Policy：单步生成 + IMLE 训练。"""

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        down_dims: tuple[int, ...] = (256, 512, 1024),
        latent_dim: int = 32,
        num_samples: int = 64,
        noise_scale: float = 1.0,
        num_inference_samples: int = 1,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.down_dims = down_dims
        self.latent_dim = latent_dim
        self.num_samples = num_samples
        self.noise_scale = noise_scale
        self.num_inference_samples = num_inference_samples
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim, obs_horizon=obs_horizon,
            pred_horizon=pred_horizon, action_horizon=action_horizon, device=device,
        )

    def _build_model(self) -> None:
        cond_dim = 0
        if self.obs_dim is not None and self.obs_dim > 0:
            self.state_encoder = StateEncoder(
                input_dim=self.obs_dim * self.obs_horizon,
                hidden_dim=self.hidden_dim, output_dim=self.hidden_dim,
            )
            cond_dim += self.hidden_dim
        else:
            self.state_encoder = None
            cond_dim = self.hidden_dim

        self.noise_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim * self.pred_horizon),
        )
        self.generator = ConditionalUNet1d(
            action_dim=self.action_dim, cond_dim=cond_dim, down_dims=self.down_dims,
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state.flatten(1)
            return self.state_encoder(state)
        return torch.zeros(obs["state"].shape[0], self.hidden_dim, device=self._device)

    def _generate(self, cond: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        init = self.noise_proj(z).view(-1, self.pred_horizon, self.action_dim)
        t = torch.zeros(cond.shape[0], device=cond.device, dtype=torch.long)
        return self.generator(init, t, cond)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        if self.num_inference_samples <= 1:
            z = torch.randn(B, self.latent_dim, device=cond.device) * self.noise_scale
            action = self._generate(cond, z)
        else:
            actions = []
            for _ in range(self.num_inference_samples):
                z = torch.randn(B, self.latent_dim, device=cond.device) * self.noise_scale
                actions.append(self._generate(cond, z))
            action = torch.stack(actions, dim=0).mean(dim=0)
        return PolicyOutput(action=action)

    def compute_loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device
        K = self.num_samples
        z = torch.randn(B, K, self.latent_dim, device=device) * self.noise_scale
        z_flat = z.view(B * K, self.latent_dim)
        cond_expanded = cond.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        candidates = self._generate(cond_expanded, z_flat)
        candidates = candidates.view(B, K, self.pred_horizon, self.action_dim)
        gt_expanded = action.unsqueeze(1).expand_as(candidates)
        distances = ((candidates - gt_expanded) ** 2).sum(dim=(-1, -2))
        nn_idx = distances.argmin(dim=1)
        nn_actions = candidates[torch.arange(B, device=device), nn_idx]
        loss = nn.functional.mse_loss(nn_actions, action)
        nn_dist = distances[torch.arange(B, device=device), nn_idx].mean()
        return {"loss": loss, "nn_dist": nn_dist.detach()}
