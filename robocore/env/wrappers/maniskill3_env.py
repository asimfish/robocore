"""ManiSkill3 环境 wrapper。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)


@EnvRegistry.register("maniskill3")
class ManiSkill3Env(BaseEnv):
    """ManiSkill3 环境 wrapper。

    ManiSkill3 特点：
    - GPU 并行仿真（SAPIEN）
    - 丰富的操作任务
    - 支持点云观测
    """

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

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        try:
            import mani_skill.envs  # noqa: F401
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "mani_skill not installed. Install with: pip install mani-skill"
            )

        env_kwargs = {
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "max_episode_steps": self.max_episode_steps,
        }
        if self.num_envs > 1:
            env_kwargs["num_envs"] = self.num_envs

        if self.render_mode:
            env_kwargs["render_mode"] = self.render_mode

        self._env = gym.make(self.task_name, **env_kwargs)

        obs, _ = self._env.reset()
        action_dim = self._env.action_space.shape[-1]

        self._spec = EnvSpec(
            action_dim=action_dim,
            task_name=self.task_name,
            max_episode_steps=self.max_episode_steps,
            num_envs=self.num_envs,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        obs, info = self._env.reset(seed=seed)
        self._step_count = 0
        return self._process_obs(obs)

    def step(self, action: np.ndarray) -> StepResult:
        import torch as th

        if isinstance(action, np.ndarray):
            action_input = th.from_numpy(action).float()
        else:
            action_input = action

        obs, reward, terminated, truncated, info = self._env.step(action_input)
        self._step_count += 1

        return StepResult(
            obs=self._process_obs(obs),
            reward=float(reward) if np.isscalar(reward) else reward,
            done=bool(terminated) if np.isscalar(terminated) else terminated,
            truncated=bool(truncated) if np.isscalar(truncated) else truncated,
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        return self._process_obs(self._env.get_obs())

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _process_obs(self, obs: Any) -> dict[str, Any]:
        """处理 ManiSkill3 观测。"""
        import torch as th

        result = {}

        if isinstance(obs, dict):
            # state 模式
            if "agent" in obs:
                agent_obs = obs["agent"]
                parts = []
                for key in ["qpos", "qvel", "base_pose"]:
                    if key in agent_obs:
                        val = agent_obs[key]
                        if isinstance(val, th.Tensor):
                            val = val.cpu().numpy()
                        parts.append(val.flatten().astype(np.float32))
                if parts:
                    result["state"] = np.concatenate(parts)

            # rgbd 模式
            if "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    if "rgb" in cam_data:
                        img = cam_data["rgb"]
                        if isinstance(img, th.Tensor):
                            img = img.cpu().numpy()
                        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                            img = np.transpose(img, (2, 0, 1))
                        result[f"image_{cam_name}"] = img

            # pointcloud 模式
            if "pointcloud" in obs:
                pc = obs["pointcloud"]
                if "xyzw" in pc:
                    val = pc["xyzw"]
                    if isinstance(val, th.Tensor):
                        val = val.cpu().numpy()
                    result["pointcloud"] = val

        elif isinstance(obs, (np.ndarray, th.Tensor)):
            # 纯向量观测
            if isinstance(obs, th.Tensor):
                obs = obs.cpu().numpy()
            result["state"] = obs.flatten().astype(np.float32)

        return result
