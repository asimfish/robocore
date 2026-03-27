"""LIBERO 环境 wrapper。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)


@EnvRegistry.register("libero")
class LiberoEnv(BaseEnv):
    """LIBERO benchmark 环境 wrapper。

    LIBERO 提供 4 个 benchmark suite：
    - LIBERO-Spatial: 空间关系任务
    - LIBERO-Object: 物体操作任务
    - LIBERO-Goal: 目标导向任务
    - LIBERO-Long: 长序列任务
    """

    def __init__(
        self,
        task_name: str = "libero_spatial",
        task_id: int = 0,
        camera_names: list[str] | None = None,
        camera_height: int = 128,
        camera_width: int = 128,
        max_episode_steps: int = 300,
        **kwargs: Any,
    ):
        super().__init__(task_name=task_name)
        self.task_id = task_id
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_episode_steps = max_episode_steps
        self._env = None
        self._step_count = 0

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        try:
            from libero.libero import benchmark
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError:
            raise ImportError(
                "libero not installed. Install with: pip install libero"
            )

        bench = benchmark.get_benchmark(self.task_name)
        task = bench.get_task(self.task_id)
        task_description = bench.get_task_description(self.task_id)

        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": self.camera_height,
            "camera_widths": self.camera_width,
            "camera_names": self.camera_names,
        }

        self._env = OffScreenRenderEnv(**env_args)
        self._task_description = task_description

        obs = self._env.reset()
        action_dim = self._env.action_spec[0].shape[0]

        self._spec = EnvSpec(
            obs_keys=["state"] + [f"image_{c}" for c in self.camera_names],
            action_dim=action_dim,
            task_name=f"{self.task_name}_{self.task_id}",
            max_episode_steps=self.max_episode_steps,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        if seed is not None:
            np.random.seed(seed)
        raw_obs = self._env.reset()
        self._step_count = 0
        return self._process_obs(raw_obs)

    def step(self, action: np.ndarray) -> StepResult:
        raw_obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps

        return StepResult(
            obs=self._process_obs(raw_obs),
            reward=float(reward),
            done=bool(done),
            truncated=truncated,
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        return self._process_obs(self._env._get_observations())

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _process_obs(self, raw_obs: dict) -> dict[str, Any]:
        obs = {}
        state_parts = []
        for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]:
            if key in raw_obs:
                state_parts.append(raw_obs[key].astype(np.float32))
        if state_parts:
            obs["state"] = np.concatenate(state_parts)

        for cam in self.camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                img = raw_obs[key].astype(np.uint8)
                if img.ndim == 3 and img.shape[-1] in (1, 3):
                    img = np.transpose(img, (2, 0, 1))
                obs[f"image_{cam}"] = img

        return obs
