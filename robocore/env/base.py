"""统一环境接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class EnvSpec:
    """环境规格描述。"""

    # 观测空间
    obs_keys: list[str] = field(default_factory=list)  # e.g. ["state", "image_front"]
    obs_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    obs_dtypes: dict[str, str] = field(default_factory=dict)

    # 动作空间
    action_dim: int = 7
    action_type: str = "joint_position"  # joint_position, ee_pose, etc.
    action_range: tuple[float, float] = (-1.0, 1.0)

    # 任务信息
    task_name: str = ""
    max_episode_steps: int = 400
    num_envs: int = 1  # 并行环境数


@dataclass
class StepResult:
    """环境 step 返回值。"""

    obs: dict[str, np.ndarray | torch.Tensor]
    reward: float | np.ndarray = 0.0
    done: bool | np.ndarray = False
    truncated: bool | np.ndarray = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnv(ABC):
    """所有环境的基类。

    遵循 Gymnasium 风格接口，但扩展了对多模态观测的支持。
    """

    def __init__(self, task_name: str = "", **kwargs: Any):
        self.task_name = task_name
        self._spec: EnvSpec | None = None

    @abstractmethod
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """重置环境。

        Returns:
            obs: 初始观测字典
        """

    @abstractmethod
    def step(self, action: np.ndarray | torch.Tensor) -> StepResult:
        """执行一步。

        Args:
            action: (action_dim,) 动作向量

        Returns:
            StepResult
        """

    @abstractmethod
    def get_obs(self) -> dict[str, Any]:
        """获取当前观测。"""

    @abstractmethod
    def close(self) -> None:
        """关闭环境。"""

    @property
    def spec(self) -> EnvSpec:
        """环境规格。"""
        if self._spec is None:
            raise RuntimeError("EnvSpec not initialized. Call reset() first.")
        return self._spec

    def render(self) -> np.ndarray | None:
        """渲染当前帧（可选）。"""
        return None

    def seed(self, seed: int) -> None:
        """设置随机种子。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
