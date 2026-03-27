"""RLDS (Reinforcement Learning Datasets) 适配器。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from robocore.data.adapters.registry import DatasetRegistry
from robocore.data.dataset import BaseDataset
from robocore.data.transforms import TransformPipeline

logger = logging.getLogger(__name__)


@DatasetRegistry.register("rlds")
class RLDSDataset(BaseDataset):
    """RLDS 格式数据集。

    RLDS 是 Google 的标准化 RL 数据格式，基于 TensorFlow Datasets。
    广泛用于 Open X-Embodiment 等大规模数据集。

    数据结构：
    - 每个 episode 是一个 dict，包含 steps
    - 每个 step 包含 observation, action, reward, is_terminal 等
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = "",
        obs_keys: list[str] | None = None,
        image_keys: list[str] | None = None,
        transform: TransformPipeline | None = None,
        obs_horizon: int = 2,
        action_horizon: int = 1,
        pred_horizon: int = 16,
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.obs_keys = obs_keys or ["state"]
        self.image_keys = image_keys or []
        self.split = split
        self._tf_dataset = None
        self._episodes_cache: list[dict] = []

        super().__init__(
            root=root,
            transform=transform,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
        )

    def _load_index(self) -> None:
        """加载 RLDS 数据集索引。"""
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow-datasets not installed. "
                "Install with: pip install tensorflow-datasets tensorflow"
            )

        builder = tfds.builder(self.dataset_name, data_dir=self.root)
        ds = builder.as_dataset(split=self.split)

        # 遍历 episodes 构建索引
        for ep_idx, episode in enumerate(ds):
            steps = list(episode["steps"])
            ep_len = len(steps)

            # 缓存 episode 数据
            self._episodes_cache.append({
                "steps": steps,
                "length": ep_len,
            })

            # 推断维度
            first_step = steps[0]
            action_dim = first_step["action"].numpy().shape[0] if "action" in first_step else 0

            self._episode_index.append({
                "episode_id": ep_idx,
                "length": ep_len,
                "path": self.root,
                "action_dim": action_dim,
                "metadata": {},
            })

        logger.info(
            f"Loaded RLDS dataset '{self.dataset_name}': "
            f"{len(self._episode_index)} episodes"
        )

    def _load_sample(self, episode_idx: int, step_idx: int) -> dict[str, Any]:
        """加载单个样本。"""
        step = self._episodes_cache[episode_idx]["steps"][step_idx]

        obs: dict[str, torch.Tensor] = {}

        # 观测
        if "observation" in step:
            observation = step["observation"]
            for key in self.obs_keys:
                if key in observation:
                    val = observation[key].numpy().astype(np.float32)
                    obs["state"] = torch.from_numpy(val)

            for key in self.image_keys:
                if key in observation:
                    img = observation[key].numpy()
                    if img.ndim == 3 and img.shape[-1] in (1, 3):
                        img = np.transpose(img, (2, 0, 1))
                    obs[f"image_{key}"] = torch.from_numpy(img)

        # 动作
        action = torch.from_numpy(step["action"].numpy().astype(np.float32))

        return {"obs": obs, "action": action}
