"""经典模仿学习算法：BC, BC-RNN, IBC, DAgger。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import StateEncoder
from robocore.policy.registry import PolicyRegistry


# ============================================================
# BC: Behavior Cloning (MLP)
# ============================================================

@PolicyRegistry.register("bc")
class BCPolicy(BasePolicy):
    """Behavior Cloning：最基础的模仿学习。

    直接用 MLP 回归 obs → action。
    简单但有效的 baseline。
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        action_horizon: int = 1,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.0,
        loss_type: str = "mse",  # "mse" or "l1"
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.loss_type = loss_type
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            obs_horizon=obs_horizon, pred_horizon=pred_horizon,
            action_horizon=action_horizon, device=device,
        )

    def _build_model(self) -> None:
        input_dim = (self.obs_dim or 0) * self.obs_horizon
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(self.num_layers - 1):
            layers.extend([nn.Linear(in_d, self.hidden_dim), nn.ReLU()])
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            in_d = self.hidden_dim
        layers.append(nn.Linear(in_d, self.action_dim * self.pred_horizon))
        self.mlp = nn.Sequential(*layers)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        state = obs["state"]
        if state.ndim == 3:
            state = state.flatten(1)
        out = self.mlp(state)
        return PolicyOutput(action=out.view(-1, self.pred_horizon, self.action_dim))

    def compute_loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.predict(obs).action
        if action.ndim == 2:
            action = action.unsqueeze(1)
        fn = nn.functional.mse_loss if self.loss_type == "mse" else nn.functional.l1_loss
        return {"loss": fn(pred, action)}


# ============================================================
# BC-RNN: Behavior Cloning with RNN
# ============================================================

@PolicyRegistry.register("bc_rnn")
class BCRNNPolicy(BasePolicy):
    """BC-RNN：用 LSTM 建模时序依赖。

    比 MLP-BC 更好地处理部分可观测和时序相关的任务。
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 10,
        pred_horizon: int = 1,
        action_horizon: int = 1,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        rnn_layers: int = 2,
        rnn_type: str = "lstm",  # "lstm" or "gru"
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            obs_horizon=obs_horizon, pred_horizon=pred_horizon,
            action_horizon=action_horizon, device=device,
        )

    def _build_model(self) -> None:
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=self.obs_dim or 0,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_layers,
            batch_first=True,
        )
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        state = obs["state"]  # (B, T, obs_dim)
        if state.ndim == 2:
            state = state.unsqueeze(1)
        rnn_out, _ = self.rnn(state)
        last = rnn_out[:, -1]  # (B, hidden)
        action = self.action_head(last)
        return PolicyOutput(action=action.unsqueeze(1))

    def compute_loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.predict(obs).action
        if action.ndim == 3:
            action = action[:, -1:]
        elif action.ndim == 2:
            action = action.unsqueeze(1)
        return {"loss": nn.functional.mse_loss(pred, action)}


# ============================================================
# IBC: Implicit Behavior Cloning
# ============================================================

@PolicyRegistry.register("ibc")
class IBCPolicy(BasePolicy):
    """Implicit Behavior Cloning (IBC)。

    Florence et al. 2022: 用 EBM (Energy-Based Model) 建模策略。
    - 训练：学习能量函数 E(obs, action)，GT action 能量低
    - 推理：用 Langevin dynamics 或 derivative-free optimization 采样低能量动作

    优势：天然支持多模态动作分布。
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        action_horizon: int = 1,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        num_neg_samples: int = 256,
        langevin_steps: int = 50,
        langevin_lr: float = 1e-2,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_neg_samples = num_neg_samples
        self.langevin_steps = langevin_steps
        self.langevin_lr = langevin_lr
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            obs_horizon=obs_horizon, pred_horizon=pred_horizon,
            action_horizon=action_horizon, device=device,
        )

    def _build_model(self) -> None:
        input_dim = (self.obs_dim or 0) + self.action_dim
        self.energy_net = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def _energy(self, obs_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """计算能量 E(obs, action)。"""
        x = torch.cat([obs_flat, action], dim=-1)
        return self.energy_net(x).squeeze(-1)

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """Langevin dynamics 采样。"""
        state = obs["state"]
        if state.ndim == 3:
            state = state.flatten(1)
        B = state.shape[0]

        # 初始化随机动作
        action = torch.randn(B, self.action_dim, device=state.device)
        action.requires_grad_(True)

        for _ in range(self.langevin_steps):
            energy = self._energy(state, action)
            grad = torch.autograd.grad(energy.sum(), action)[0]
            action = action - self.langevin_lr * grad
            action = action + 0.01 * torch.randn_like(action)  # noise
            action = action.detach().requires_grad_(True)

        return PolicyOutput(action=action.detach().unsqueeze(1))

    def compute_loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        state = obs["state"]
        if state.ndim == 3:
            state = state.flatten(1)
        if action.ndim == 3:
            action = action[:, 0]
        B = state.shape[0]

        # 正样本能量
        pos_energy = self._energy(state, action)

        # 负样本
        neg_actions = torch.randn(B, self.num_neg_samples, self.action_dim, device=state.device)
        state_exp = state.unsqueeze(1).expand(-1, self.num_neg_samples, -1)
        neg_energy = self._energy(
            state_exp.reshape(-1, state.shape[-1]),
            neg_actions.reshape(-1, self.action_dim),
        ).reshape(B, self.num_neg_samples)

        # InfoNCE-style loss
        logits = torch.cat([pos_energy.unsqueeze(1), neg_energy], dim=1)  # (B, 1+N)
        logits = -logits  # 低能量 = 高概率
        labels = torch.zeros(B, dtype=torch.long, device=state.device)
        loss = nn.functional.cross_entropy(logits, labels)

        return {"loss": loss}


# ============================================================
# DAgger: Dataset Aggregation
# ============================================================

@PolicyRegistry.register("dagger")
class DAggerPolicy(BasePolicy):
    """DAgger: Dataset Aggregation。

    Ross et al. 2011: 迭代式模仿学习。
    1. 用当前策略收集数据
    2. 用专家标注收集到的观测
    3. 聚合数据集，重新训练

    这里实现 DAgger 的策略网络部分（与 BC 相同），
    DAgger 的核心逻辑在训练循环中。
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        action_horizon: int = 1,
        device: str | torch.device = "cuda",
        hidden_dim: int = 256,
        beta_schedule: str = "linear",  # "linear", "constant"
        initial_beta: float = 1.0,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.beta_schedule = beta_schedule
        self.initial_beta = initial_beta
        self._current_beta = initial_beta
        self._iteration = 0
        super().__init__(
            obs_dim=obs_dim, action_dim=action_dim,
            obs_horizon=obs_horizon, pred_horizon=pred_horizon,
            action_horizon=action_horizon, device=device,
        )

    def _build_model(self) -> None:
        input_dim = (self.obs_dim or 0) * self.obs_horizon
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    @property
    def beta(self) -> float:
        """当前混合系数：beta * expert + (1-beta) * policy。"""
        if self.beta_schedule == "linear":
            return max(0.0, self.initial_beta * (1 - self._iteration / 100))
        return self.initial_beta

    def step_iteration(self) -> None:
        self._iteration += 1

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        state = obs["state"]
        if state.ndim == 3:
            state = state.flatten(1)
        action = self.mlp(state)
        return PolicyOutput(action=action.unsqueeze(1))

    def compute_loss(self, obs: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.predict(obs).action
        if action.ndim == 3:
            action = action[:, -1:]
        elif action.ndim == 2:
            action = action.unsqueeze(1)
        return {"loss": nn.functional.mse_loss(pred, action)}
