"""向量化环境：支持并行评估。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from robocore.env.base import BaseEnv, StepResult


class VecEnv:
    """向量化环境包装器。

    支持两种模式：
    1. 串行：多个 BaseEnv 实例顺序执行
    2. GPU 并行：ManiSkill3 风格的 GPU 并行仿真
    """

    def __init__(self, envs: list[BaseEnv]):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """重置所有环境。"""
        obs_list = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs_list.append(env.reset(seed=env_seed))
        return self._stack_obs(obs_list)

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple[
        dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[dict]
    ]:
        """所有环境执行一步。

        Args:
            actions: (num_envs, action_dim)

        Returns:
            obs, rewards, dones, truncateds, infos
        """
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        results = []
        for i, env in enumerate(self.envs):
            results.append(env.step(actions[i]))

        obs = self._stack_obs([r.obs for r in results])
        rewards = np.array([r.reward for r in results])
        dones = np.array([r.done for r in results])
        truncateds = np.array([r.truncated for r in results])
        infos = [r.info for r in results]

        return obs, rewards, dones, truncateds, infos

    def close(self) -> None:
        for env in self.envs:
            env.close()

    @staticmethod
    def _stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        """堆叠多个环境的观测。"""
        result = {}
        for key in obs_list[0]:
            values = [o[key] for o in obs_list]
            if isinstance(values[0], (np.ndarray, torch.Tensor)):
                result[key] = np.stack(
                    [v.numpy() if isinstance(v, torch.Tensor) else v for v in values]
                )
            else:
                result[key] = values
        return result

    def __len__(self) -> int:
        return self.num_envs

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
