"""核心数据结构：Episode, Observation, Action."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Observation:
    """统一观测数据结构。

    支持多模态观测：图像（多视角）、状态向量、点云、语言指令。
    所有字段可选，按需填充。
    """

    # 状态向量 (joint positions, velocities, etc.)
    state: torch.Tensor | np.ndarray | None = None

    # 图像观测 {camera_name: image_tensor}
    # image shape: (C, H, W) or (T, C, H, W) for temporal
    images: dict[str, torch.Tensor | np.ndarray] = field(default_factory=dict)

    # 点云 (N, 3) or (N, 6) with colors
    pointcloud: torch.Tensor | np.ndarray | None = None

    # 语言指令
    language: str | None = None

    # 额外信息
    extra: dict[str, Any] = field(default_factory=dict)

    def to_tensor(self, device: str | torch.device = "cpu") -> Observation:
        """将所有数据转为 torch.Tensor 并移到指定设备。"""
        def _convert(x: Any) -> torch.Tensor | None:
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        return Observation(
            state=_convert(self.state),
            images={k: _convert(v) for k, v in self.images.items()},
            pointcloud=_convert(self.pointcloud),
            language=self.language,
            extra=self.extra,
        )


@dataclass
class Action:
    """统一动作数据结构。

    支持多种动作表示：关节位置、关节速度、末端执行器位姿、夹爪。
    """

    # 主动作向量 (action_dim,) or (horizon, action_dim) for chunked
    data: torch.Tensor | np.ndarray | None = None

    # 动作类型标记
    action_type: str = "joint_position"  # joint_position, joint_velocity, ee_pose, etc.

    # 是否为 delta 动作
    is_delta: bool = True

    # 夹爪动作（如果独立于主动作）
    gripper: torch.Tensor | np.ndarray | None = None

    def to_tensor(self, device: str | torch.device = "cpu") -> Action:
        """将所有数据转为 torch.Tensor。"""
        def _convert(x: Any) -> torch.Tensor | None:
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x).to(device)
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        return Action(
            data=_convert(self.data),
            action_type=self.action_type,
            is_delta=self.is_delta,
            gripper=_convert(self.gripper),
        )


@dataclass
class Episode:
    """一条完整的演示轨迹。

    包含时间步序列的观测和动作，以及元信息。
    """

    # 观测序列
    observations: list[Observation] = field(default_factory=list)

    # 动作序列（长度通常 = len(observations) - 1 或 len(observations)）
    actions: list[Action] = field(default_factory=list)

    # 奖励序列（可选，用于 RL）
    rewards: list[float] = field(default_factory=list)

    # 终止标记
    dones: list[bool] = field(default_factory=list)

    # 元信息
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """轨迹长度。"""
        return len(self.observations)

    @property
    def success(self) -> bool:
        """是否成功完成任务。"""
        return self.metadata.get("success", False)
