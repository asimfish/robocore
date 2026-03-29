"""ManiSkill 3 环境 wrapper。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)


@EnvRegistry.register("maniskill3")
class ManiSkill3Env(BaseEnv):
    """ManiSkill 3 环境 wrapper。"""

    def __init__(
        self,
        task_name: str = "PickCube-v1",
        obs_mode: str = "state",  # state, rgbd, pointcloud
        control_mode: str = "pd_joint_delta_pos",
        num_envs: int = 1,
        max_episode_steps: int = 200,
        render_mode: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(task_name=task_name)
        self.obs_mode = obs_mode
        self.control_mode = control_mode
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._env = None
        self._step_count = 0
        self._last_obs: Any | None = None

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        try:
            import mani_skill.envs  # noqa: F401
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "mani_skill not installed. Recommended install: pip install mani_skill"
            ) from exc

        # ManiSkill 3 当前通过环境注册时绑定 episode limit，gym.make 不再接收
        # max_episode_steps 覆盖参数，因此这里由 wrapper 自行维护 truncated。
        env_kwargs = {
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "num_envs": self.num_envs,
        }
        if self.render_mode:
            env_kwargs["render_mode"] = self.render_mode

        self._env = gym.make(self.task_name, **env_kwargs)

        obs, _ = self._env.reset()
        self._last_obs = obs
        action_space = getattr(self._env, "single_action_space", self._env.action_space)
        action_shape = getattr(action_space, "shape", ())
        action_dim = int(action_shape[-1]) if action_shape else 0

        self._spec = EnvSpec(
            action_dim=action_dim,
            task_name=self.task_name,
            max_episode_steps=self.max_episode_steps,
            num_envs=self.num_envs,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        obs, _ = self._env.reset(seed=seed)
        self._last_obs = obs
        self._step_count = 0
        return self._process_obs(obs)

    def step(self, action: np.ndarray) -> StepResult:
        import torch as th

        self._lazy_init()

        if isinstance(action, np.ndarray):
            action_input = th.from_numpy(action)
        elif isinstance(action, th.Tensor):
            action_input = action
        else:
            action_input = th.as_tensor(action)

        action_input = action_input.float()
        if self.num_envs == 1 and action_input.ndim == 1:
            action_input = action_input.unsqueeze(0)

        obs, reward, terminated, truncated, info = self._env.step(action_input)
        self._last_obs = obs
        self._step_count += 1

        reward_value = self._to_scalar_or_array(reward, np.float32)
        done_value = self._to_scalar_or_array(terminated, np.bool_)
        truncated_value = self._to_scalar_or_array(truncated, np.bool_)

        if self.num_envs == 1:
            truncated_value = bool(truncated_value) or self._step_count >= self.max_episode_steps
        else:
            truncated_value = np.asarray(truncated_value, dtype=bool) | (
                self._step_count >= self.max_episode_steps
            )

        return StepResult(
            obs=self._process_obs(obs),
            reward=float(reward_value) if self.num_envs == 1 else reward_value,
            done=bool(done_value) if self.num_envs == 1 else done_value,
            truncated=truncated_value,
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        self._lazy_init()
        if hasattr(self._env, "get_obs"):
            self._last_obs = self._env.get_obs()
        if self._last_obs is None:
            raise RuntimeError("Observation not available before reset().")
        return self._process_obs(self._last_obs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._last_obs = None

    def _process_obs(self, obs: Any) -> dict[str, Any]:
        """处理 ManiSkill 3 观测。"""
        result: dict[str, Any] = {}

        if isinstance(obs, dict):
            if "agent" in obs:
                agent_obs = obs["agent"]
                parts = []
                for key in ["qpos", "qvel", "base_pose"]:
                    if key in agent_obs:
                        parts.append(self._flatten_feature(agent_obs[key]))
                if parts:
                    result["state"] = np.concatenate(parts, axis=0 if self.num_envs == 1 else -1)

            if "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    if "rgb" not in cam_data:
                        continue
                    img = self._strip_single_env_batch(self._to_numpy(cam_data["rgb"]))
                    if self.num_envs == 1:
                        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                            img = np.transpose(img, (2, 0, 1))
                    elif img.ndim == 4 and img.shape[-1] in (1, 3, 4):
                        img = np.transpose(img, (0, 3, 1, 2))
                    result[f"image_{cam_name}"] = img

            if "pointcloud" in obs and "xyzw" in obs["pointcloud"]:
                pointcloud = self._strip_single_env_batch(
                    self._to_numpy(obs["pointcloud"]["xyzw"])
                )
                result["pointcloud"] = pointcloud.astype(np.float32)

            return result

        state = self._strip_single_env_batch(self._to_numpy(obs)).astype(np.float32)
        result["state"] = state.reshape(-1) if self.num_envs == 1 else state
        return result

    def _flatten_feature(self, value: Any) -> np.ndarray:
        array = self._strip_single_env_batch(self._to_numpy(value)).astype(np.float32)
        if self.num_envs == 1:
            return array.reshape(-1)
        return array.reshape(array.shape[0], -1)

    def _to_scalar_or_array(self, value: Any, dtype: type[np.float32] | type[np.bool_]) -> Any:
        array = self._strip_single_env_batch(self._to_numpy(value))
        if array.shape == ():
            scalar = array.item()
            return bool(scalar) if dtype is np.bool_ else float(scalar)
        return array.astype(dtype)

    def _strip_single_env_batch(self, value: np.ndarray) -> np.ndarray:
        if self.num_envs == 1 and value.ndim > 0 and value.shape[0] == 1:
            return value[0]
        return value

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        import torch as th

        if isinstance(value, th.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.asarray(value)
