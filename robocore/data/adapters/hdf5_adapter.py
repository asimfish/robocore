"""HDF5 数据集适配器（robomimic 风格）。"""
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


@DatasetRegistry.register("hdf5")
class HDF5Dataset(BaseDataset):
    """HDF5 格式数据集。

    兼容 robomimic 的 HDF5 数据格式：
    - /data/demo_0/obs/{key}: 观测数据
    - /data/demo_0/actions: 动作数据
    - /data/demo_0/rewards: 奖励（可选）
    - /data/demo_0/dones: 终止标记（可选）

    支持 lazy loading：只在需要时读取数据。
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
        cache_mode: str = "none",  # none, index, full
    ):
        self.obs_keys = obs_keys or ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
        self.image_keys = image_keys or []
        self.cache_mode = cache_mode
        self._file_handle = None
        self._cache: dict[str, Any] = {}

        super().__init__(
            root=root,
            transform=transform,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            pred_horizon=pred_horizon,
        )

    def _load_index(self) -> None:
        """扫描 HDF5 文件构建索引。"""
        import h5py

        hdf5_path = Path(self.root)
        if hdf5_path.is_dir():
            # 查找目录下的 HDF5 文件
            files = sorted(hdf5_path.glob("*.hdf5")) + sorted(hdf5_path.glob("*.h5"))
            if not files:
                raise FileNotFoundError(f"No HDF5 files found in {self.root}")
            hdf5_path = files[0]

        with h5py.File(str(hdf5_path), "r") as f:
            data_group = f["data"]
            demo_keys = sorted(
                [k for k in data_group.keys() if k.startswith("demo")],
                key=lambda x: int(x.split("_")[1]),
            )

            for demo_key in demo_keys:
                demo = data_group[demo_key]
                ep_len = demo["actions"].shape[0]
                action_dim = demo["actions"].shape[1]

                # 推断 obs_dim
                obs_dim = 0
                if "obs" in demo:
                    for key in self.obs_keys:
                        if key in demo["obs"]:
                            obs_dim += demo["obs"][key].shape[-1]

                self._episode_index.append({
                    "episode_id": len(self._episode_index),
                    "length": ep_len,
                    "path": str(hdf5_path),
                    "demo_key": demo_key,
                    "action_dim": action_dim,
                    "obs_dim": obs_dim,
                    "metadata": dict(demo.attrs) if demo.attrs else {},
                })

        logger.info(
            f"Loaded HDF5 dataset: {len(self._episode_index)} episodes, "
            f"action_dim={self._episode_index[0].get('action_dim', '?')}"
        )

    def _get_file(self):
        """获取 HDF5 文件句柄（lazy open）。"""
        if self._file_handle is None:
            import h5py
            path = self._episode_index[0]["path"]
            self._file_handle = h5py.File(path, "r")
        return self._file_handle

    def _load_sample(self, episode_idx: int, step_idx: int) -> dict[str, Any]:
        """加载单个样本。"""
        ep_info = self._episode_index[episode_idx]
        demo_key = ep_info["demo_key"]
        f = self._get_file()
        demo = f["data"][demo_key]

        # 加载观测
        obs: dict[str, torch.Tensor] = {}

        # 状态观测
        state_parts = []
        if "obs" in demo:
            for key in self.obs_keys:
                if key in demo["obs"]:
                    val = demo["obs"][key][step_idx]
                    state_parts.append(torch.from_numpy(np.array(val, dtype=np.float32)))

        if state_parts:
            obs["state"] = torch.cat(state_parts, dim=-1)

        # 图像观测
        if "obs" in demo:
            for key in self.image_keys:
                if key in demo["obs"]:
                    img = np.array(demo["obs"][key][step_idx], dtype=np.uint8)
                    # HWC -> CHW
                    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                        img = np.transpose(img, (2, 0, 1))
                    obs[f"image_{key}"] = torch.from_numpy(img)

        # 加载动作
        action = torch.from_numpy(
            np.array(demo["actions"][step_idx], dtype=np.float32)
        )

        return {"obs": obs, "action": action}

    def close(self) -> None:
        """关闭文件句柄。"""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def __del__(self):
        self.close()
