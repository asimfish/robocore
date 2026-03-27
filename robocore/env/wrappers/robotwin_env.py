"""RobotWin 环境 wrapper。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)


@EnvRegistry.register("robotwin")
class RobotWinEnv(BaseEnv):
    """RobotWin benchmark 环境 wrapper。

    RobotWin 是双臂机器人操作 benchmark。
    """

    def __init__(
        self,
        task_name: str = "block_hammer_beat",
        obs_mode: str = "state",
        max_episode_steps: int = 600,
        **kwargs: Any,
    ):
        super().__init__(task_name=task_name)
        self.obs_mode = obs_mode
        self.max_episode_steps = max_episode_steps
        self._env = None
        self._step_count = 0

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        try:
            # RobotWin 使用 SAPIEN 引擎
            import gymnasium as gym
        except ImportError:
            raise ImportError("gymnasium not installed")

        logger.info(f"Initializing RobotWin env: {self.task_name}")
        # RobotWin 环境初始化（具体实现取决于 RobotWin 版本）
        # 这里提供接口框架，实际使用时需要根据 RobotWin 的 API 调整
        self._spec = EnvSpec(
            action_dim=14,  # 双臂 7+7
            task_name=self.task_name,
            max_episode_steps=self.max_episode_steps,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        self._step_count = 0
        if self._env is not None:
            obs, _ = self._env.reset(seed=seed)
            return self._process_obs(obs)
        return {"state": np.zeros(28, dtype=np.float32)}

    def step(self, action: np.ndarray) -> StepResult:
        self._step_count += 1
        if self._env is not None:
            obs, reward, terminated, truncated, info = self._env.step(action)
            return StepResult(
                obs=self._process_obs(obs),
                reward=float(reward),
                done=bool(terminated),
                truncated=bool(truncated) or self._step_count >= self.max_episode_steps,
                info=info,
            )
        return StepResult(
            obs={"state": np.zeros(28, dtype=np.float32)},
            done=self._step_count >= self.max_episode_steps,
        )

    def get_obs(self) -> dict[str, Any]:
        if self._env is not None:
            return self._process_obs(self._env.get_obs())
        return {"state": np.zeros(28, dtype=np.float32)}

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _process_obs(self, obs: Any) -> dict[str, Any]:
        if isinstance(obs, dict):
            result = {}
            state_parts = []
            for key in sorted(obs.keys()):
                val = obs[key]
                if isinstance(val, np.ndarray) and val.ndim == 1:
                    state_parts.append(val.astype(np.float32))
                elif isinstance(val, np.ndarray) and val.ndim == 3:
                    if val.shape[-1] in (1, 3):
                        val = np.transpose(val, (2, 0, 1))
                    result[f"image_{key}"] = val
            if state_parts:
                result["state"] = np.concatenate(state_parts)
            return result
        elif isinstance(obs, np.ndarray):
            return {"state": obs.flatten().astype(np.float32)}
        return {"state": np.zeros(28, dtype=np.float32)}
