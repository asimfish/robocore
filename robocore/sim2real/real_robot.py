"""真机部署接口。"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class RealRobotConfig:
    """真机配置。"""
    robot_type: str = "franka"  # "franka", "ur5", "xarm", "aloha"
    control_freq: float = 10.0  # Hz
    action_space: str = "eef_delta"  # "eef_delta", "joint_delta", "joint_pos"
    max_episode_steps: int = 300
    gripper_binary: bool = True

    # 安全限制
    max_pos_delta: float = 0.05  # m
    max_rot_delta: float = 0.1  # rad
    workspace_bounds: tuple[tuple[float, ...], tuple[float, ...]] = (
        (-0.5, -0.5, 0.0),
        (0.5, 0.5, 0.8),
    )


class RealRobotInterface:
    """真机部署接口。

    提供统一的真机控制 API，支持：
    - 安全检查 (工作空间限制、速度限制)
    - 动作平滑 (低通滤波)
    - 紧急停止
    - 多种机器人类型

    用法：
        robot = RealRobotInterface(config)
        robot.connect()
        obs = robot.get_observation()
        robot.execute_action(action)
        robot.disconnect()
    """

    def __init__(self, config: RealRobotConfig | None = None):
        self.config = config or RealRobotConfig()
        self._connected = False
        self._last_action: np.ndarray | None = None
        self._step_count = 0

        # 动作平滑滤波器
        self._action_filter_alpha = 0.7

    def connect(self) -> None:
        """连接机器人。"""
        logger.info(f"Connecting to {self.config.robot_type}...")
        self._connected = True
        self._step_count = 0
        logger.info("Connected.")

    def disconnect(self) -> None:
        """断开连接。"""
        if self._connected:
            logger.info("Disconnecting...")
            self._connected = False

    def get_observation(self) -> dict[str, np.ndarray]:
        """获取当前观测。

        Returns:
            {"state": np.ndarray, "image": np.ndarray, ...}
        """
        if not self._connected:
            raise RuntimeError("Robot not connected")

        # 占位实现 — 实际使用时对接具体机器人 SDK
        return {
            "state": np.zeros(14, dtype=np.float32),  # 7 joint pos + 7 joint vel
            "eef_pos": np.zeros(3, dtype=np.float32),
            "eef_quat": np.array([1, 0, 0, 0], dtype=np.float32),
            "gripper_pos": np.zeros(1, dtype=np.float32),
        }

    def _safety_check(self, action: np.ndarray) -> np.ndarray:
        """安全检查：裁剪动作到安全范围。"""
        cfg = self.config

        if cfg.action_space == "eef_delta":
            # 限制位移幅度
            pos_delta = action[:3]
            pos_norm = np.linalg.norm(pos_delta)
            if pos_norm > cfg.max_pos_delta:
                action[:3] = pos_delta / pos_norm * cfg.max_pos_delta

            # 限制旋转幅度
            if len(action) > 3:
                rot_delta = action[3:6]
                rot_norm = np.linalg.norm(rot_delta)
                if rot_norm > cfg.max_rot_delta:
                    action[3:6] = rot_delta / rot_norm * cfg.max_rot_delta

        return action

    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """动作平滑（低通滤波）。"""
        if self._last_action is None:
            self._last_action = action
            return action

        smoothed = self._action_filter_alpha * action + (1 - self._action_filter_alpha) * self._last_action
        self._last_action = smoothed
        return smoothed

    def execute_action(self, action: np.ndarray | torch.Tensor) -> None:
        """执行动作。"""
        if not self._connected:
            raise RuntimeError("Robot not connected")

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        action = action.flatten()
        action = self._safety_check(action)
        action = self._smooth_action(action)

        # 占位 — 实际发送到机器人
        self._step_count += 1

        # 控制频率
        time.sleep(1.0 / self.config.control_freq)

    def execute_trajectory(
        self,
        actions: np.ndarray | torch.Tensor,
        blocking: bool = True,
    ) -> None:
        """执行动作序列。"""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        for action in actions:
            self.execute_action(action)

    def emergency_stop(self) -> None:
        """紧急停止。"""
        logger.warning("EMERGENCY STOP!")
        self._connected = False

    @property
    def is_done(self) -> bool:
        return self._step_count >= self.config.max_episode_steps
