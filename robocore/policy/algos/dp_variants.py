"""Diffusion Policy 变种集合。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import StateEncoder
from robocore.policy.networks.dit import DiTActionModel
from robocore.policy.networks.unet_1d import ConditionalUNet1d
from robocore.policy.registry import PolicyRegistry
from robocore.policy.schedulers import DDIMScheduler, DDPMScheduler


# ============================================================
# 共享的编码 + 扩散逻辑
# ============================================================

class _DPBase(BasePolicy):
    """DP 变种的共享基类。子类只需提供 noise_pred_net。"""

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps
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
        """构建观测编码器，返回 cond_dim。"""
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

    def _build_schedulers(self) -> None:
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
            noise_pred = self.noise_pred_net(x, ts, cond)
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
        pred = self.noise_pred_net(noisy, timesteps, cond)
        loss = nn.functional.mse_loss(pred, noise)
        return {"loss": loss}


# ============================================================
# DP-T: Diffusion Policy with DiT (Transformer) backbone
# ============================================================

@PolicyRegistry.register("dp_t")
class DPTransformer(_DPBase):
    """Diffusion Policy — DiT (Transformer) backbone。

    用 DiT 替代 U-Net，在长 horizon 和高维动作上更有优势。
    参考：Chi et al. "Diffusion Policy" 中的 Transformer 变体。
    """

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        self.noise_pred_net = DiTActionModel(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self._extra_kwargs.get("num_layers", 4),
            num_heads=self._extra_kwargs.get("num_heads", 4),
            max_horizon=self.pred_horizon,
        )
        self._build_schedulers()


# ============================================================
# DP-CNN: 经典 CNN (U-Net) backbone（即原版 DP，别名）
# ============================================================

@PolicyRegistry.register("dp_cnn")
class DPCNN(_DPBase):
    """Diffusion Policy — CNN (1D U-Net) backbone。

    这就是原版 Diffusion Policy 的 CNN 变体，提供显式别名。
    """

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        self.noise_pred_net = ConditionalUNet1d(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            down_dims=self._extra_kwargs.get("down_dims", (256, 512, 1024)),
        )
        self._build_schedulers()


# ============================================================
# DP-BE: Diffusion Policy with Behavior Ensemble
# ============================================================

@PolicyRegistry.register("dp_be")
class DPBehaviorEnsemble(_DPBase):
    """Diffusion Policy — Behavior Ensemble。

    多个去噪头并行预测，推理时取加权平均。
    适用于多模态动作分布的场景。
    """

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        n_heads = self._extra_kwargs.get("num_ensemble_heads", 3)
        self.ensemble_heads = nn.ModuleList([
            ConditionalUNet1d(
                action_dim=self.action_dim,
                cond_dim=cond_dim,
                down_dims=self._extra_kwargs.get("down_dims", (128, 256)),
            )
            for _ in range(n_heads)
        ])
        # 用第一个头作为默认
        self.noise_pred_net = self.ensemble_heads[0]
        self._build_schedulers()

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device

        # 每个头独立采样
        all_actions = []
        for head in self.ensemble_heads:
            x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
            for t in self.inference_scheduler.inference_timesteps:
                ts = torch.full((B,), t.item(), device=device, dtype=torch.long)
                pred = head(x, ts, cond)
                x = self.inference_scheduler.step(pred, t.item(), x)
            all_actions.append(x)

        # 取均值
        action = torch.stack(all_actions).mean(dim=0)
        return PolicyOutput(
            action=action,
            info={"ensemble_actions": all_actions},
        )

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        timesteps = self.train_scheduler.get_timesteps(B, device)
        noise = torch.randn_like(action)
        noisy = self.train_scheduler.add_noise(action, noise, timesteps)

        total_loss = torch.tensor(0.0, device=device)
        for head in self.ensemble_heads:
            pred = head(noisy, timesteps, cond)
            total_loss = total_loss + nn.functional.mse_loss(pred, noise)
        total_loss = total_loss / len(self.ensemble_heads)

        return {"loss": total_loss}


# ============================================================
# DP-G: Diffusion Policy with Guidance (Classifier-Free)
# ============================================================

@PolicyRegistry.register("dp_cfg")
class DPClassifierFreeGuidance(_DPBase):
    """Diffusion Policy — Classifier-Free Guidance。

    训练时随机 drop 条件（p_uncond），推理时用 guidance scale 增强条件信号。
    适用于语言条件或目标条件的场景。
    """

    def __init__(self, p_uncond: float = 0.1, guidance_scale: float = 1.5, **kwargs: Any):
        self.p_uncond = p_uncond
        self.guidance_scale = guidance_scale
        super().__init__(**kwargs)

    def _build_model(self) -> None:
        cond_dim = self._build_encoder()
        self.noise_pred_net = ConditionalUNet1d(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            down_dims=self._extra_kwargs.get("down_dims", (256, 512, 1024)),
        )
        self._build_schedulers()

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        cond = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        # 随机 drop 条件
        mask = torch.rand(B, 1, device=device) > self.p_uncond
        cond = cond * mask.float()

        timesteps = self.train_scheduler.get_timesteps(B, device)
        noise = torch.randn_like(action)
        noisy = self.train_scheduler.add_noise(action, noise, timesteps)
        pred = self.noise_pred_net(noisy, timesteps, cond)
        loss = nn.functional.mse_loss(pred, noise)
        return {"loss": loss}

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        cond = self._encode_obs(obs)
        B = cond.shape[0]
        device = cond.device
        uncond = torch.zeros_like(cond)

        x = torch.randn(B, self.pred_horizon, self.action_dim, device=device)
        for t in self.inference_scheduler.inference_timesteps:
            ts = torch.full((B,), t.item(), device=device, dtype=torch.long)
            pred_cond = self.noise_pred_net(x, ts, cond)
            pred_uncond = self.noise_pred_net(x, ts, uncond)
            # Classifier-free guidance
            pred = pred_uncond + self.guidance_scale * (pred_cond - pred_uncond)
            x = self.inference_scheduler.step(pred, t.item(), x)
        return PolicyOutput(action=x)
