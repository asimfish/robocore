"""RoboMimic 数据生成脚本：CanTeacher 等 scripted policy。"""
from __future__ import annotations

from typing import Any

import numpy as np


class CanTeacher:
    """Can 任务的 scripted teacher policy。

    实现简单的抓取-放置策略：
    1. 移动到罐子上方
    2. 下降并抓取
    3. 移动到目标 bin
    4. 释放
    """

    def __init__(self, grasp_height: float = 0.86, lift_height: float = 1.0):
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self._target_pos: np.ndarray | None = None

    def reset(self, env: Any) -> None:
        """根据环境信息重置 teacher。"""
        object_id = getattr(env, "object_id", 0)
        placements = getattr(env, "target_bin_placements", None)
        if placements is not None and object_id < len(placements):
            self._target_pos = placements[object_id].copy()
        else:
            self._target_pos = getattr(env, "bin2_pos", np.zeros(3)).copy()
        self.target_bin = self._target_pos.copy()

    def act(
        self,
        obs: dict[str, np.ndarray],
        env: Any,
        grasped: bool = False,
    ) -> np.ndarray:
        """生成动作。

        Args:
            obs: 当前观测
            env: 环境实例
            grasped: 是否已抓取物体

        Returns:
            (7,) 动作向量 [dx, dy, dz, dax, day, daz, gripper]
        """
        eef_pos = obs.get("robot0_eef_pos", np.zeros(3))
        can_pos = obs.get("Can_pos", np.zeros(3))
        can_to_eef = obs.get("Can_to_robot0_eef_pos", can_pos - eef_pos)
        gripper_qpos = obs.get("robot0_gripper_qpos", np.zeros(2))

        action = np.zeros(7, dtype=np.float32)

        if not grasped:
            # 移向罐子
            delta = can_pos - eef_pos
            action[:3] = np.clip(delta * 5.0, -1.0, 1.0)
            # 张开夹爪
            action[6] = -1.0

            # 如果足够近，闭合夹爪
            if np.linalg.norm(delta[:2]) < 0.02 and abs(delta[2]) < 0.02:
                action[6] = 1.0
        else:
            # 移向目标
            if self._target_pos is not None:
                target = self._target_pos.copy()
                target[2] = self.lift_height
                delta = target - eef_pos
                action[:3] = np.clip(delta * 5.0, -1.0, 1.0)

                # 到达目标上方后释放
                if np.linalg.norm(delta[:2]) < 0.02:
                    action[6] = -1.0
                else:
                    action[6] = 1.0
            else:
                action[6] = 1.0

        return action
