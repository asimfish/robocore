"""统一数据集接口。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset

from robocore.data.episode import Action, Observation
from robocore.data.transforms import TransformPipeline


class BaseDataset(Dataset, ABC):
    """所有数据集的基类。

    提供统一的数据访问接口，支持：
    - 按 index 访问单个样本
    - 按 episode 访问完整轨迹
    - transform pipeline
    - lazy loading
    """

    def __init__(
        self,
        root: str,
        transform: TransformPipeline | None = None,
        obs_horizon: int = 1,
        action_horizon: int = 1,
        pred_horizon: int = 16,
    ):
        self.root = root
        self.transform = transform
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self._episode_index: list[dict[str, Any]] = []
        self._load_index()

    @abstractmethod
    def _load_index(self) -> None:
        """加载数据集索引（不加载实际数据）。

        填充 self._episode_index，每个元素包含：
        - episode_id: int
        - length: int
        - path: str (数据文件路径)
        - metadata: dict
        """

    @abstractmethod
    def _load_sample(self, episode_idx: int, step_idx: int) -> dict[str, Any]:
        """加载单个样本的原始数据。

        Returns:
            dict with keys: obs, action, and optionally reward, done, info
        """

    def __len__(self) -> int:
        """数据集总样本数（所有 episode 的有效步数之和）。"""
        return sum(
            max(0, ep["length"] - self.obs_horizon - self.pred_horizon + 1)
            for ep in self._episode_index
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """获取训练样本。

        Returns:
            dict with:
                obs: 观测数据（包含 obs_horizon 步历史）
                action: 动作序列（pred_horizon 步）
        """
        # 将全局 idx 映射到 (episode_idx, step_idx)
        episode_idx, step_idx = self._global_to_local(idx)

        # 加载 obs_horizon 步观测
        obs_steps = range(step_idx, step_idx + self.obs_horizon)
        # 加载 pred_horizon 步动作
        action_steps = range(step_idx, step_idx + self.pred_horizon)

        sample = {
            "obs": self._load_obs_sequence(episode_idx, obs_steps),
            "action": self._load_action_sequence(episode_idx, action_steps),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _global_to_local(self, idx: int) -> tuple[int, int]:
        """全局索引 → (episode_idx, step_idx)。"""
        cumsum = 0
        for ep_idx, ep in enumerate(self._episode_index):
            ep_len = max(0, ep["length"] - self.obs_horizon - self.pred_horizon + 1)
            if cumsum + ep_len > idx:
                return ep_idx, idx - cumsum
            cumsum += ep_len
        raise IndexError(f"Index {idx} out of range")

    def _load_obs_sequence(
        self, episode_idx: int, steps: range
    ) -> dict[str, torch.Tensor]:
        """加载观测序列并堆叠。"""
        obs_list = []
        for step in steps:
            sample = self._load_sample(episode_idx, step)
            obs_list.append(sample["obs"])
        return self._stack_obs(obs_list)

    def _load_action_sequence(
        self, episode_idx: int, steps: range
    ) -> torch.Tensor:
        """加载动作序列并堆叠。"""
        actions = []
        ep_len = self._episode_index[episode_idx]["length"]
        for step in steps:
            # 超出 episode 长度时 pad 最后一个动作
            actual_step = min(step, ep_len - 1)
            sample = self._load_sample(episode_idx, actual_step)
            actions.append(sample["action"])
        return torch.stack(actions, dim=0)

    @staticmethod
    def _stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """将观测列表堆叠为 batch 格式。"""
        if not obs_list:
            return {}
        result = {}
        for key in obs_list[0]:
            values = [o[key] for o in obs_list]
            if isinstance(values[0], torch.Tensor):
                result[key] = torch.stack(values, dim=0)
            else:
                result[key] = values
        return result

    @property
    def num_episodes(self) -> int:
        return len(self._episode_index)

    @property
    def obs_dim(self) -> int | None:
        """状态观测维度（如果有）。"""
        return self._episode_index[0].get("obs_dim") if self._episode_index else None

    @property
    def action_dim(self) -> int | None:
        """动作维度。"""
        return self._episode_index[0].get("action_dim") if self._episode_index else None


class EpisodeDataset(BaseDataset):
    """基于内存中 Episode 列表的数据集（用于小数据集或测试）。"""

    def __init__(
        self,
        episodes: list[Any],
        transform: TransformPipeline | None = None,
        obs_horizon: int = 1,
        action_horizon: int = 1,
        pred_horizon: int = 16,
    ):
        self._episodes = episodes
        super().__init__(
            root="memory://",
            transform=transform,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
        )

    def _load_index(self) -> None:
        for i, ep in enumerate(self._episodes):
            self._episode_index.append({
                "episode_id": i,
                "length": ep.length,
                "path": f"memory://{i}",
                "metadata": ep.metadata,
            })

    def _load_sample(self, episode_idx: int, step_idx: int) -> dict[str, Any]:
        ep = self._episodes[episode_idx]
        obs = ep.observations[step_idx]
        obs_dict: dict[str, Any] = {}
        if obs.state is not None:
            obs_dict["state"] = (
                torch.from_numpy(obs.state)
                if not isinstance(obs.state, torch.Tensor)
                else obs.state
            )
        for cam_name, img in obs.images.items():
            obs_dict[f"image_{cam_name}"] = (
                torch.from_numpy(img)
                if not isinstance(img, torch.Tensor)
                else img
            )

        action_data = ep.actions[min(step_idx, len(ep.actions) - 1)].data
        action = (
            torch.from_numpy(action_data)
            if not isinstance(action_data, torch.Tensor)
            else action_data
        )

        return {"obs": obs_dict, "action": action}
