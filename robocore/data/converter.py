"""数据格式转换器。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DataConverter:
    """数据格式转换器。

    支持在 HDF5, LeRobot, RLDS, Zarr 之间互相转换。

    用法：
        converter = DataConverter()
        converter.convert(
            src="data/demo.hdf5",
            dst="data/demo_lerobot",
            src_format="hdf5",
            dst_format="lerobot",
        )
    """

    def convert(
        self,
        src: str,
        dst: str,
        src_format: str,
        dst_format: str,
        **kwargs: Any,
    ) -> None:
        """执行格式转换。"""
        logger.info(f"Converting {src} ({src_format}) -> {dst} ({dst_format})")

        # 加载源数据为统一中间格式
        episodes = self._load_episodes(src, src_format, **kwargs)
        logger.info(f"Loaded {len(episodes)} episodes")

        # 保存为目标格式
        self._save_episodes(episodes, dst, dst_format, **kwargs)
        logger.info(f"Saved to {dst}")

    def _load_episodes(
        self, path: str, fmt: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """加载为统一中间格式。

        每个 episode 是一个 dict：
        - actions: np.ndarray (T, action_dim)
        - observations: dict[str, np.ndarray]
        - rewards: np.ndarray (T,) (optional)
        - metadata: dict
        """
        if fmt == "hdf5":
            return self._load_hdf5(path, **kwargs)
        elif fmt == "zarr":
            return self._load_zarr(path, **kwargs)
        elif fmt == "lerobot":
            return self._load_lerobot(path, **kwargs)
        else:
            raise ValueError(f"Unsupported source format: {fmt}")

    def _save_episodes(
        self, episodes: list[dict[str, Any]], path: str, fmt: str, **kwargs: Any
    ) -> None:
        """保存为目标格式。"""
        if fmt == "hdf5":
            self._save_hdf5(episodes, path, **kwargs)
        elif fmt == "zarr":
            self._save_zarr(episodes, path, **kwargs)
        elif fmt == "lerobot":
            self._save_lerobot(episodes, path, **kwargs)
        else:
            raise ValueError(f"Unsupported target format: {fmt}")

    def _load_hdf5(self, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        """从 HDF5 加载。"""
        import h5py

        episodes = []
        with h5py.File(path, "r") as f:
            data = f["data"]
            for demo_key in sorted(data.keys()):
                demo = data[demo_key]
                ep: dict[str, Any] = {
                    "actions": np.array(demo["actions"]),
                    "observations": {},
                    "metadata": dict(demo.attrs) if demo.attrs else {},
                }
                if "obs" in demo:
                    for key in demo["obs"]:
                        ep["observations"][key] = np.array(demo["obs"][key])
                if "rewards" in demo:
                    ep["rewards"] = np.array(demo["rewards"])
                episodes.append(ep)
        return episodes

    def _save_hdf5(
        self, episodes: list[dict[str, Any]], path: str, **kwargs: Any
    ) -> None:
        """保存为 HDF5。"""
        import h5py

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            data_grp = f.create_group("data")
            for i, ep in enumerate(episodes):
                demo = data_grp.create_group(f"demo_{i}")
                demo.create_dataset("actions", data=ep["actions"])
                if "observations" in ep:
                    obs_grp = demo.create_group("obs")
                    for key, val in ep["observations"].items():
                        obs_grp.create_dataset(key, data=val)
                if "rewards" in ep:
                    demo.create_dataset("rewards", data=ep["rewards"])
                for k, v in ep.get("metadata", {}).items():
                    demo.attrs[k] = v

    def _load_zarr(self, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        """从 Zarr 加载。"""
        import zarr

        store = zarr.open(path, mode="r")
        data = store["data"]

        # 获取 episode 边界
        if "meta" in store and "episode_ends" in store["meta"]:
            episode_ends = np.array(store["meta"]["episode_ends"])
        else:
            episode_ends = np.array([data["action"].shape[0]])

        episodes = []
        prev_end = 0
        for end in episode_ends:
            end = int(end)
            ep: dict[str, Any] = {
                "actions": np.array(data["action"][prev_end:end]),
                "observations": {},
                "metadata": {},
            }
            for key in data.keys():
                if key != "action":
                    ep["observations"][key] = np.array(data[key][prev_end:end])
            episodes.append(ep)
            prev_end = end

        return episodes

    def _save_zarr(
        self, episodes: list[dict[str, Any]], path: str, **kwargs: Any
    ) -> None:
        """保存为 Zarr。"""
        import zarr

        store = zarr.open(path, mode="w")
        data_grp = store.create_group("data")
        meta_grp = store.create_group("meta")

        # 拼接所有 episodes
        all_actions = np.concatenate([ep["actions"] for ep in episodes])
        data_grp.create_dataset("action", data=all_actions, chunks=(1000, -1))

        # 观测
        obs_keys = set()
        for ep in episodes:
            obs_keys.update(ep.get("observations", {}).keys())

        for key in obs_keys:
            all_data = np.concatenate([
                ep["observations"][key] for ep in episodes if key in ep.get("observations", {})
            ])
            data_grp.create_dataset(key, data=all_data, chunks=True)

        # Episode 边界
        episode_ends = np.cumsum([ep["actions"].shape[0] for ep in episodes])
        meta_grp.create_dataset("episode_ends", data=episode_ends)

    def _load_lerobot(self, path: str, **kwargs: Any) -> list[dict[str, Any]]:
        """从 LeRobot 格式加载。"""
        from datasets import load_from_disk

        ds = load_from_disk(path)
        episodes: dict[int, dict[str, list]] = {}

        for row in ds:
            ep_id = row.get("episode_index", 0)
            if ep_id not in episodes:
                episodes[ep_id] = {"actions": [], "observations": {}}

            episodes[ep_id]["actions"].append(row.get("action", []))

            for key in row:
                if key.startswith("observation."):
                    obs_key = key.replace("observation.", "")
                    if obs_key not in episodes[ep_id]["observations"]:
                        episodes[ep_id]["observations"][obs_key] = []
                    episodes[ep_id]["observations"][obs_key].append(row[key])

        result = []
        for ep_id in sorted(episodes.keys()):
            ep = episodes[ep_id]
            ep_dict: dict[str, Any] = {
                "actions": np.array(ep["actions"], dtype=np.float32),
                "observations": {
                    k: np.array(v) for k, v in ep["observations"].items()
                },
                "metadata": {"episode_id": ep_id},
            }
            result.append(ep_dict)

        return result

    def _save_lerobot(
        self, episodes: list[dict[str, Any]], path: str, **kwargs: Any
    ) -> None:
        """保存为 LeRobot 格式。"""
        from datasets import Dataset

        rows = []
        for ep_idx, ep in enumerate(episodes):
            actions = ep["actions"]
            for step_idx in range(len(actions)):
                row = {
                    "episode_index": ep_idx,
                    "frame_index": step_idx,
                    "timestamp": step_idx / 10.0,  # 假设 10Hz
                    "action": actions[step_idx].tolist(),
                }
                for key, val in ep.get("observations", {}).items():
                    row[f"observation.{key}"] = val[step_idx].tolist()
                rows.append(row)

        ds = Dataset.from_list(rows)
        ds.save_to_disk(path)
        logger.info(f"Saved LeRobot dataset: {len(rows)} steps, {len(episodes)} episodes")
