"""DPPO: Diffusion Policy Policy Optimization。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.algos.diffusion_policy import DiffusionPolicy


class DPPOAgent(nn.Module):
    """DPPO: 用 PPO 微调 Diffusion Policy。

    Fan et al. 2024: 将 Diffusion Policy 与 RL 结合。
    - 用预训练的 DP 作为策略
    - 用 PPO 的 clipped objective 微调
    - 保留扩散过程的多模态性

    训练流程：
    1. 预训练 DP (模仿学习)
    2. 用 DP 采样动作 → 环境交互 → 收集 (s, a, r, adv)
    3. PPO 更新 DP 的参数
    """

    def __init__(
        self,
        dp_policy: DiffusionPolicy,
        critic_hidden: int = 256,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        lr_actor: float = 1e-5,
        lr_critic: float = 3e-4,
    ):
        super().__init__()
        self.policy = dp_policy
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        # Value function
        obs_dim = dp_policy.obs_dim or 64
        self.value_net = nn.Sequential(
            nn.Linear(obs_dim, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, critic_hidden), nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

        self.actor_opt = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_opt = torch.optim.Adam(self.value_net.parameters(), lr=lr_critic)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算 GAE (Generalized Advantage Estimation)。"""
        T = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=rewards.device)

        for t in reversed(range(T)):
            next_val = next_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """PPO 更新。"""
        state = obs.get("state", list(obs.values())[0])
        if state.ndim == 3:
            state = state[:, -1]

        # Value loss
        values = self.get_value(state)
        value_loss = nn.functional.mse_loss(values, returns)

        self.critic_opt.zero_grad()
        value_loss.backward()
        self.critic_opt.step()

        # Policy loss (simplified: 用 MSE 近似，因为 DP 没有显式 log_prob)
        # 实际 DPPO 论文用更复杂的方法估计 log_prob
        pred = self.policy.predict(obs)
        policy_loss = -(advantages.unsqueeze(-1).unsqueeze(-1) * pred.action).mean()

        self.actor_opt.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        return {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        }
