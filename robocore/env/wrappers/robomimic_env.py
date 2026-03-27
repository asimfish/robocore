"""RoboMimic 环境 wrapper。"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)


@EnvRegistry.register("robomimic")
class RoboMimicEnv(BaseEnv):
    """RoboMimic 环境 wrapper。

    封装 robosuite 环境，提供统一接口。
    """

    def __init__(
        self,
        task_name: str = "Lift",
        robot: str = "Panda",
        camera_names: list[str] | None = None,
        camera_height: int = 84,
        camera_width: int = 84,
        use_image: bool = True,
        max_episode_steps: int = 400,
        **kwargs: Any,
    ):
        super().__init__(task_name=task_name)
        self.robot = robot
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.use_image = use_image
        self.max_episode_steps = max_episode_steps
        self._env = None
        self._step_count = 0

    def _lazy_init(self) -> None:
        """延迟初始化 robosuite 环境。"""
        if self._env is not None:
            return

        try:
            import robosuite as suite
            from robosuite.wrappers import GymWrapper
        except ImportError:
            raise ImportError(
                "robosuite not installed. Install with: pip install robosuite"
            )

        env_kwargs = {
            "env_name": self.task_name,
            "robots": self.robot,
            "has_renderer": False,
            "has_offscreen_renderer": self.use_image,
            "use_camera_obs": self.use_image,
            "camera_names": self.camera_names if self.use_image else [],
            "camera_heights": self.camera_height,
            "camera_widths": self.camera_width,
            "horizon": self.max_episode_steps,
            "control_freq": 20,
        }

        self._env = suite.make(**env_kwargs)

        # 构建 EnvSpec
        obs = self._env.reset()
        action_dim = self._env.action_spec[0].shape[0]

        obs_keys = []
        obs_shapes = {}
        if "robot0_eef_pos" in obs:
            state_dim = sum(
                obs[k].shape[0]
                for k in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
                if k in obs
            )
            obs_keys.append("state")
            obs_shapes["state"] = (state_dim,)

        if self.use_image:
            for cam in self.camera_names:
                key = f"{cam}_image"
                if key in obs:
                    obs_keys.append(f"image_{cam}")
                    obs_shapes[f"image_{cam}"] = (3, self.camera_height, self.camera_width)

        self._spec = EnvSpec(
            obs_keys=obs_keys,
            obs_shapes=obs_shapes,
            action_dim=action_dim,
            action_type="joint_velocity",
            action_range=(-1.0, 1.0),
            task_name=self.task_name,
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
            info={"success": bool(self._env._check_success())} if hasattr(self._env, "_check_success") else info,
        )

    def get_obs(self) -> dict[str, Any]:
        raw_obs = self._env._get_observations()
        return self._process_obs(raw_obs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _process_obs(self, raw_obs: dict) -> dict[str, Any]:
        """将 robosuite 观测转为统一格式。"""
        obs = {}

        # 状态
        state_parts = []
        for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
            if key in raw_obs:
                state_parts.append(raw_obs[key].astype(np.float32))
        if state_parts:
            obs["state"] = np.concatenate(state_parts)

        # 图像
        if self.use_image:
            for cam in self.camera_names:
                key = f"{cam}_image"
                if key in raw_obs:
                    img = raw_obs[key].astype(np.uint8)
                    if img.ndim == 3 and img.shape[-1] in (1, 3):
                        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
                    obs[f"image_{cam}"] = img

        return obs
