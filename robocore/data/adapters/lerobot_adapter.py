"""LeRobot 数据集适配器。"""
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


@DatasetRegistry.register("lerobot")
class LeRobotDataset(BaseDataset):
    """LeRobot 格式数据集。

    LeRobot 使用 HuggingFace datasets 格式：
    - Parquet 文件存储结构化数据
    - MP4 视频存储图像观测
    - metadata.json 描述数据集结构

    支持从 HuggingFace Hub 或本地路径加载。
    """

    def __init__(
        self,
        root: str,
        repo_id: str | None = None,
        obs_keys: list[str] | None = None,
        image_keys: list[str] | None = None,
        transform: TransformPipeline | None = None,
        obs_horizon: int = 2,
        action_horizon: int = 1,
        pred_horizon: int = 16,
        delta_timestamps: dict[str, list[float]] | None = None,
    ):
        self.repo_id = repo_id
        self.obs_keys = obs_keys or ["observation.state"]
        self.image_keys = image_keys or ["observation.images.top"]
        self.delta_timestamps = delta_timestamps
        self._hf_dataset = None

        super().__init__(
            root=root,
            transform=transform,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
        )

    def _load_index(self) -> None:
        """加载 LeRobot 数据集索引。"""
        try:
            from datasets import load_dataset, load_from_disk
        except ImportError:
            raise ImportError(
                "datasets not installed. Install with: pip install datasets"
            )

        # 从 Hub 或本地加载
        if self.repo_id:
            self._hf_dataset = load_dataset(self.repo_id, split="train")
        else:
            path = Path(self.root)
            if (path / "dataset_info.json").exists():
                self._hf_dataset = load_from_disk(str(path))
            else:
                raise FileNotFoundError(
                    f"No LeRobot dataset found at {self.root}. "
                    "Provide repo_id for HuggingFace Hub datasets."
                )

        # 构建 episode 索引
        if "episode_index" in self._hf_dataset.column_names:
            episode_ids = set(self._hf_dataset["episode_index"])
            for ep_id in sorted(episode_ids):
                # 找到该 episode 的所有行
                indices = [
                    i for i, eid in enumerate(self._hf_dataset["episode_index"])
                    if eid == ep_id
                ]
                ep_len = len(indices)

                # 推断维度
                first_row = self._hf_dataset[indices[0]]
                action_dim = len(first_row.get("action", []))

                self._episode_index.append({
                    "episode_id": ep_id,
                    "length": ep_len,
                    "path": self.root,
                    "start_idx": indices[0],
                    "action_dim": action_dim,
                    "metadata": {},
                })
        else:
            # 单 episode 数据集
            ep_len = len(self._hf_dataset)
            first_row = self._hf_dataset[0]
            action_dim = len(first_row.get("action", []))
            self._episode_index.append({
                "episode_id": 0,
                "length": ep_len,
                "path": self.root,
                "start_idx": 0,
                "action_dim": action_dim,
                "metadata": {},
            })

        logger.info(
            f"Loaded LeRobot dataset: {len(self._episode_index)} episodes"
        )

    def _load_sample(self, episode_idx: int, step_idx: int) -> dict[str, Any]:
        """加载单个样本。"""
        ep_info = self._episode_index[episode_idx]
        global_idx = ep_info["start_idx"] + step_idx
        row = self._hf_dataset[global_idx]

        obs: dict[str, torch.Tensor] = {}

        # 状态观测
        for key in self.obs_keys:
            if key in row:
                val = row[key]
                if isinstance(val, list):
                    obs["state"] = torch.tensor(val, dtype=torch.float32)
                elif isinstance(val, np.ndarray):
                    obs["state"] = torch.from_numpy(val.astype(np.float32))

        # 图像观测
        for key in self.image_keys:
            if key in row:
                img = row[key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                if isinstance(img, np.ndarray):
                    if img.ndim == 3 and img.shape[-1] in (1, 3):
                        img = np.transpose(img, (2, 0, 1))
                    obs[f"image_{key.split('.')[-1]}"] = torch.from_numpy(img)

        # 动作
        action_data = row.get("action", [])
        if isinstance(action_data, list):
            action = torch.tensor(action_data, dtype=torch.float32)
        else:
            action = torch.from_numpy(np.array(action_data, dtype=np.float32))

        return {"obs": obs, "action": action}
