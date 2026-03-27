"""统一策略接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class PolicyOutput:
    """策略输出。"""

    # 预测的动作序列 (batch, horizon, action_dim)
    action: torch.Tensor

    # 可选：动作分布参数（用于采样或分析）
    action_dist: Any = None

    # 可选：额外信息（attention weights, latent codes, etc.）
    info: dict[str, Any] = field(default_factory=dict)


class BasePolicy(nn.Module, ABC):
    """所有策略的基类。

    统一接口：
    - predict: 推理时预测动作
    - compute_loss: 训练时计算损失
    - get_action: 从 PolicyOutput 中提取可执行动作

    子类需要实现：
    - _build_model: 构建网络
    - predict: 前向推理
    - compute_loss: 计算训练损失
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 2,
        pred_horizon: int = 16,
        action_horizon: int = 8,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self._device = torch.device(device)

        # 子类在 __init__ 中调用 _build_model
        self._build_model()

    @abstractmethod
    def _build_model(self) -> None:
        """构建网络结构。子类实现。"""

    @abstractmethod
    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """推理：给定观测，预测动作序列。

        Args:
            obs: 观测字典，可能包含：
                - state: (batch, obs_horizon, state_dim)
                - image_*: (batch, obs_horizon, C, H, W)
                - language: list[str] (batch,)

        Returns:
            PolicyOutput with action shape (batch, pred_horizon, action_dim)
        """

    @abstractmethod
    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """计算训练损失。

        Args:
            obs: 观测字典
            action: ground truth 动作 (batch, pred_horizon, action_dim)

        Returns:
            dict with at least "loss" key, plus optional auxiliary losses
        """

    def get_action(self, output: PolicyOutput) -> torch.Tensor:
        """从 PolicyOutput 中提取可执行动作。

        默认取 action_horizon 步。子类可覆盖实现 temporal ensemble。

        Returns:
            (batch, action_horizon, action_dim)
        """
        return output.action[:, : self.action_horizon]

    def save(self, path: str | Path) -> None:
        """保存模型权重和配置。"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "obs_horizon": self.obs_horizon,
                    "pred_horizon": self.pred_horizon,
                    "action_horizon": self.action_horizon,
                },
            },
            path / "policy.pt",
        )

    def load(self, path: str | Path, strict: bool = True) -> None:
        """加载模型权重。"""
        path = Path(path)
        ckpt = torch.load(path / "policy.pt", map_location="cpu", weights_only=False)
        self.load_state_dict(ckpt["state_dict"], strict=strict)

    @property
    def num_params(self) -> int:
        """可训练参数数量。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
