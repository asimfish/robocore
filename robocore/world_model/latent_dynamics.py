"""潜在动力学模型：在 latent space 预测状态转移。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class LatentDynamicsModel(nn.Module):
    """Latent Dynamics Model (LDM)。

    在学到的潜在空间中预测下一状态，用于：
    - Model-based planning (MPC / CEM)
    - 数据增强 (imagination rollout)
    - 表征学习的辅助任务

    架构：encoder → dynamics → decoder
    - encoder: obs → z
    - dynamics: (z, a) → z'
    - decoder: z → obs_pred (可选)

    支持确定性和随机性动力学 (RSSM style)。
    """

    def __init__(
        self,
        obs_dim: int = 64,
        action_dim: int = 7,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        stochastic: bool = False,
        stoch_dim: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.stochastic = stochastic
        self.stoch_dim = stoch_dim

        # Encoder: obs → latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Deterministic dynamics: (z, a) → h
        rnn_input = latent_dim + action_dim
        if stochastic:
            rnn_input += stoch_dim
        self.rnn = nn.GRUCell(rnn_input, latent_dim)

        # Stochastic component (RSSM-style)
        if stochastic:
            # Prior: h → (mu, sigma)
            self.prior_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, stoch_dim * 2),
            )
            # Posterior: (h, obs_enc) → (mu, sigma)
            self.posterior_net = nn.Sequential(
                nn.Linear(latent_dim + latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, stoch_dim * 2),
            )

        # Decoder: z → obs_pred
        dec_input = latent_dim + stoch_dim if stochastic else latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

        # Reward predictor head
        self.reward_head = nn.Sequential(
            nn.Linear(dec_input, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def _get_stoch(self, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_std = params.chunk(2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        dist = torch.distributions.Normal(mu, std)
        sample = dist.rsample()
        return sample, mu, std

    def step(
        self,
        h: torch.Tensor,
        action: torch.Tensor,
        stoch: torch.Tensor | None = None,
        obs_enc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """单步动力学预测。

        Args:
            h: (B, latent_dim) 确定性状态
            action: (B, action_dim)
            stoch: (B, stoch_dim) 上一步随机状态 (stochastic only)
            obs_enc: (B, latent_dim) 当前观测编码 (posterior, training only)

        Returns:
            dict with h_next, obs_pred, reward_pred, prior/posterior stats
        """
        if self.stochastic:
            rnn_input = torch.cat([h, action, stoch if stoch is not None else torch.zeros(h.shape[0], self.stoch_dim, device=h.device)], dim=-1)
        else:
            rnn_input = torch.cat([h, action], dim=-1)

        h_next = self.rnn(rnn_input, h)

        result: dict[str, torch.Tensor] = {"h": h_next}

        if self.stochastic:
            # Prior
            prior_params = self.prior_net(h_next)
            prior_sample, prior_mu, prior_std = self._get_stoch(prior_params)
            result["prior_mu"] = prior_mu
            result["prior_std"] = prior_std

            if obs_enc is not None:
                # Posterior (training)
                post_params = self.posterior_net(torch.cat([h_next, obs_enc], dim=-1))
                post_sample, post_mu, post_std = self._get_stoch(post_params)
                result["posterior_mu"] = post_mu
                result["posterior_std"] = post_std
                stoch_out = post_sample
            else:
                stoch_out = prior_sample

            result["stoch"] = stoch_out
            feat = torch.cat([h_next, stoch_out], dim=-1)
        else:
            feat = h_next

        result["obs_pred"] = self.decoder(feat)
        result["reward_pred"] = self.reward_head(feat).squeeze(-1)

        return result

    def rollout(
        self,
        init_obs: torch.Tensor,
        actions: torch.Tensor,
        obs_seq: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """多步 rollout。

        Args:
            init_obs: (B, obs_dim)
            actions: (B, T, action_dim)
            obs_seq: (B, T, obs_dim) GT 观测序列 (training, for posterior)

        Returns:
            dict with stacked predictions
        """
        B, T, _ = actions.shape
        h = self.encode(init_obs)
        stoch = torch.zeros(B, self.stoch_dim, device=h.device) if self.stochastic else None

        all_obs, all_reward = [], []
        all_prior_mu, all_prior_std = [], []
        all_post_mu, all_post_std = [], []

        for t in range(T):
            obs_enc = self.encode(obs_seq[:, t]) if obs_seq is not None else None
            out = self.step(h, actions[:, t], stoch, obs_enc)
            h = out["h"]
            if self.stochastic:
                stoch = out["stoch"]
                all_prior_mu.append(out["prior_mu"])
                all_prior_std.append(out["prior_std"])
                if "posterior_mu" in out:
                    all_post_mu.append(out["posterior_mu"])
                    all_post_std.append(out["posterior_std"])

            all_obs.append(out["obs_pred"])
            all_reward.append(out["reward_pred"])

        result = {
            "obs_pred": torch.stack(all_obs, dim=1),
            "reward_pred": torch.stack(all_reward, dim=1),
        }
        if self.stochastic and all_prior_mu:
            result["prior_mu"] = torch.stack(all_prior_mu, dim=1)
            result["prior_std"] = torch.stack(all_prior_std, dim=1)
            if all_post_mu:
                result["posterior_mu"] = torch.stack(all_post_mu, dim=1)
                result["posterior_std"] = torch.stack(all_post_std, dim=1)

        return result

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        reward_seq: torch.Tensor | None = None,
        kl_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """计算世界模型损失。"""
        B, T_plus_1, D = obs_seq.shape
        T = T_plus_1 - 1

        out = self.rollout(obs_seq[:, 0], action_seq[:, :T], obs_seq[:, 1:])

        # 重建损失
        recon_loss = nn.functional.mse_loss(out["obs_pred"], obs_seq[:, 1:])

        loss = recon_loss
        result = {"loss": loss, "recon_loss": recon_loss}

        # 奖励损失
        if reward_seq is not None:
            reward_loss = nn.functional.mse_loss(out["reward_pred"], reward_seq[:, :T])
            loss = loss + reward_loss
            result["reward_loss"] = reward_loss

        # KL 散度 (stochastic)
        if self.stochastic and "posterior_mu" in out:
            prior = torch.distributions.Normal(out["prior_mu"], out["prior_std"])
            posterior = torch.distributions.Normal(out["posterior_mu"], out["posterior_std"])
            kl = torch.distributions.kl_divergence(posterior, prior).mean()
            loss = loss + kl_weight * kl
            result["kl_loss"] = kl

        result["loss"] = loss
        return result
