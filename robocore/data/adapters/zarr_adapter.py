"""Zarr 数据集适配器。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robocore.data.adapters.registry import DatasetRegistry
from robocore.data.dataset import BaseDataset
from robocore.data.transforms import TransformPipeline

logger = logging.getLogger(__name__)


@DatasetRegistry.register("zarr")
class ZarrDataset(BaseDataset):
    """Zarr 格式数据集。

    适用于大规模数据集，支持：
    - 分块存储和压缩
    - 并行读取
    - 云存储后端

    数据结构：
    - /data/action: (total_steps, action_dim)
    - /data/state: (total_steps, state_dim)
    - /data/img: (total_steps, C, H, W)
    - /meta/episode_ends: episode 结束索引
    """

    def __init__(
        self,
        root: str,
        obs_keys: list[str] | None = None,
        image_keys: list[str] | None = None,
        transform: TransformPipeline | None = None,
        obs_horizon: int = 2,
        action_horizon: int = 1,
        pred_horizon: int = 16,
    ):
        self.obs_keys = obs_keys or ["state"]
        self.image_keys = image_keys or []
        self._zarr_store = None

        super().__init__(
            root=root,
            transform=transform,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
        )

    def _load_index(self) -> None:
        """加载 Zarr 数据集索引。"""
        try:
            import zarr
        except ImportError:
            raise ImportError("zarr not installed. Install with: pip install zarr")

        self._zarr_store = zarr.open(self.root, mode="r")

        # 读取 episode 边界
        if "meta" in self._zarr_store and "episode_ends" in self._zarr_store["meta"]:
            episode_ends = np.array(self._zarr_store["meta"]["episode_ends"])
        else:
            # 假设单 episode
            total_steps = self._zarr_store["data"]["action"].shape[0]
            episode_ends = np.array([total_steps])

        # 推断维度
        action_dim = self._zarr_store["data"]["action"].shape[1]
        obs_dim = 0
        if "state" in self._zarr_store["data"]:
            obs_dim = self._zarr_store["data"]["state"].shape[1]

        prev_end = 0
        for ep_idx, end in enumerate(episode_ends):
            ep_len = int(end) - prev_end
            self._episode_index.append({
                "episode_id": ep_idx,
                "length": ep_len,
                "path": self.root,
                "start_idx": prev_end,
                "action_dim": action_dim,
                "obs_dim": obs_dim,
                "metadata": {},
            })
            prev_end = int(end)

        logger.info(
            f"Loaded Zarr dataset: {len(self._episode_index)} episodes, "
            f"action_dim={action_dim}, obs_dim={obs_dim}"
        )

    def _load_sample(self, episode_idx: int, step_idx: int) -> dict[str, Any]:
        """加载单个样本。"""
        ep_info = self._episode_index[episode_idx]
        global_idx = ep_info["start_idx"] + step_idx
        data = self._zarr_store["data"]

        obs: dict[str, torch.Tensor] = {}

        # 状态
        if "state" in data:
            obs["state"] = torch.from_numpy(
                np.array(data["state"][global_idx], dtype=np.float32)
            )

        # 图像
        for key in self.image_keys:
            if key in data:
                img = np.array(data[key][global_idx])
                if img.dtype == np.uint8 and img.ndim == 3 and img.shape[-1] in (1, 3):
                    img = np.transpose(img, (2, 0, 1))
                obs[f"image_{key}"] = torch.from_numpy(img)

        # 动作
        action = torch.from_numpy(
            np.array(data["action"][global_idx], dtype=np.float32)
        )

        return {"obs": obs, "action": action}

    def close(self) -> None:
        self._zarr_store = None
