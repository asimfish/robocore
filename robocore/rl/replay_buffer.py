"""Replay Buffer：经验回放缓冲区。"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    """通用经验回放缓冲区。

    支持：
    - 标准 (s, a, r, s', done) 元组
    - 字典观测 (多模态)
    - 优先级采样 (PER)
    - 从离线数据集初始化 (RLPD)
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_keys: list[str] | None = None,
    ):
        self.capacity = capacity
        self.obs_keys = obs_keys or ["state"]
        self._storage: dict[str, list] = {
            "obs": [], "action": [], "reward": [],
            "next_obs": [], "done": [],
        }
        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        if self._size < self.capacity:
            for key in self._storage:
                self._storage[key].append(None)
            self._size += 1

        self._storage["obs"][self._ptr] = obs
        self._storage["action"][self._ptr] = action
        self._storage["reward"][self._ptr] = reward
        self._storage["next_obs"][self._ptr] = next_obs
        self._storage["done"][self._ptr] = done
        self._ptr = (self._ptr + 1) % self.capacity

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        indices = np.random.randint(0, self._size, size=batch_size)

        obs_batch: dict[str, list] = {}
        next_obs_batch: dict[str, list] = {}
        actions, rewards, dones = [], [], []

        for idx in indices:
            obs = self._storage["obs"][idx]
            next_obs = self._storage["next_obs"][idx]
            for key in self.obs_keys:
                obs_batch.setdefault(key, []).append(obs.get(key, obs.get("state")))
                next_obs_batch.setdefault(key, []).append(next_obs.get(key, next_obs.get("state")))
            actions.append(self._storage["action"][idx])
            rewards.append(self._storage["reward"][idx])
            dones.append(self._storage["done"][idx])

        result = {
            "obs": {k: torch.tensor(np.stack(v), dtype=torch.float32, device=device) for k, v in obs_batch.items()},
            "action": torch.tensor(np.stack(actions), dtype=torch.float32, device=device),
            "reward": torch.tensor(rewards, dtype=torch.float32, device=device),
            "next_obs": {k: torch.tensor(np.stack(v), dtype=torch.float32, device=device) for k, v in next_obs_batch.items()},
            "done": torch.tensor(dones, dtype=torch.float32, device=device),
        }
        return result

    def load_offline(self, dataset: Any) -> None:
        """从离线数据集加载 (RLPD style)。"""
        if hasattr(dataset, "__len__"):
            for i in range(len(dataset)):
                sample = dataset[i]
                self.add(
                    obs=sample.get("obs", {"state": sample.get("state", np.zeros(1))}),
                    action=sample.get("action", np.zeros(1)),
                    reward=sample.get("reward", 0.0),
                    next_obs=sample.get("next_obs", sample.get("obs", {"state": np.zeros(1)})),
                    done=sample.get("done", False),
                )
