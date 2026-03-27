"""SAC (Soft Actor-Critic) 实现。"""
from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, layers: int = 2):
        super().__init__()
        mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            mods.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        mods.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SACAgent(nn.Module):
    """SAC: Soft Actor-Critic。

    Haarnoja et al. 2018: 最大熵 RL。
    - Actor: 高斯策略 π(a|s)
    - Critic: 双 Q 网络 Q(s, a)
    - 自动温度调节 α

    用于：
    - 在线 RL 微调
    - RLPD (RL with Prior Data): 用离线数据初始化 buffer
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        gamma: float = 0.99,
        tau: float = 0.005,
        init_alpha: float = 0.1,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self._device = device

        # Actor
        self.actor = _MLP(obs_dim, action_dim * 2, hidden_dim)  # mu + log_std

        # Twin Q
        self.q1 = _MLP(obs_dim + action_dim, 1, hidden_dim)
        self.q2 = _MLP(obs_dim + action_dim, 1, hidden_dim)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # Temperature
        self.log_alpha = nn.Parameter(torch.tensor(init_alpha).log())
        self.target_entropy = -action_dim

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        out = self.actor(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        std = log_std.exp()

        if deterministic:
            return torch.tanh(mu)

        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        return torch.tanh(x)

    def _sample_action_and_logprob(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.actor(obs)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-20, 2)
        std = log_std.exp()

        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)

        # Log prob with tanh correction
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def update(self, batch: dict[str, Any]) -> dict[str, float]:
        obs = batch["obs"]["state"] if isinstance(batch["obs"], dict) else batch["obs"]
        action = batch["action"]
        reward = batch["reward"].unsqueeze(-1)
        next_obs = batch["next_obs"]["state"] if isinstance(batch["next_obs"], dict) else batch["next_obs"]
        done = batch["done"].unsqueeze(-1)

        # === Critic update ===
        with torch.no_grad():
            next_action, next_log_prob = self._sample_action_and_logprob(next_obs)
            q1_next = self.q1_target(torch.cat([next_obs, next_action], dim=-1))
            q2_next = self.q2_target(torch.cat([next_obs, next_action], dim=-1))
            q_next = torch.min(q1_next, q2_next) - self.alpha.detach() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * q_next

        q1_pred = self.q1(torch.cat([obs, action], dim=-1))
        q2_pred = self.q2(torch.cat([obs, action], dim=-1))
        critic_loss = F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # === Actor update ===
        new_action, log_prob = self._sample_action_and_logprob(obs)
        q1_new = self.q1(torch.cat([obs, new_action], dim=-1))
        q2_new = self.q2(torch.cat([obs, new_action], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # === Alpha update ===
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # === Target update ===
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.lerp_(p.data, self.tau)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.lerp_(p.data, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha.item(),
        }
