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
        self._extra_kwargs = kwargs
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

        env_kwargs: dict[str, Any] = {
            "obs_mode": self.obs_mode,
            "control_mode": self.control_mode,
            "num_envs": self.num_envs,
        }
        if self.render_mode:
            env_kwargs["render_mode"] = self.render_mode
        # 传递额外参数（如 robot_uids）
        env_kwargs.update(self._extra_kwargs)

        self._env = gym.make(self.task_name, **env_kwargs)

        # 获取动作维度
        action_space = getattr(self._env, "single_action_space", self._env.action_space)
        action_dim = action_space.shape[-1] if hasattr(action_space, "shape") else 7

        # 获取动作范围
        action_low = float(getattr(action_space, "low", np.array([-1.0])).flat[0])
        action_high = float(getattr(action_space, "high", np.array([1.0])).flat[0])

        self._spec = EnvSpec(
            action_dim=action_dim,
            action_range=(action_low, action_high),
            task_name=self.task_name,
            max_episode_steps=self.max_episode_steps,
            num_envs=self.num_envs,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        self._step_count = 0
        obs, info = self._env.reset(seed=seed)
        self._last_obs = obs
        processed = self._process_obs(obs)
        self._update_spec_from_obs(processed)
        return processed

    def step(self, action: np.ndarray) -> StepResult:
        self._lazy_init()
        self._step_count += 1

        # ManiSkill 3 期望 (num_envs, action_dim) 的 tensor
        import torch
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action).float()
        else:
            action_t = action
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)

        obs, reward, terminated, truncated, info = self._env.step(action_t)
        self._last_obs = obs

        # 处理标量
        reward_val = float(self._to_numpy(reward).flat[0]) if hasattr(reward, '__len__') else float(reward)
        term_val = bool(self._to_numpy(terminated).flat[0]) if hasattr(terminated, '__len__') else bool(terminated)
        trunc_val = bool(self._to_numpy(truncated).flat[0]) if hasattr(truncated, '__len__') else bool(truncated)

        # 手动 truncation
        if self._step_count >= self.max_episode_steps and not term_val:
            trunc_val = True

        return StepResult(
            obs=self._process_obs(obs),
            reward=reward_val,
            done=term_val,
            truncated=trunc_val,
            info=self._process_info(info),
        )

    def get_obs(self) -> dict[str, Any]:
        self._lazy_init()
        obs = self._env.get_obs()
        return self._process_obs(obs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _update_spec_from_obs(self, processed_obs: dict[str, Any]) -> None:
        """根据实际观测更新 spec 的 obs_keys/shapes。"""
        if self._spec is None:
            return
        self._spec.obs_keys = list(processed_obs.keys())
        self._spec.obs_shapes = {k: v.shape for k, v in processed_obs.items() if hasattr(v, "shape")}

    def _process_obs(self, obs: Any) -> dict[str, np.ndarray]:
        """将 ManiSkill 观测转为统一格式。"""
        if isinstance(obs, dict):
            state_parts: list[np.ndarray] = []
            result: dict[str, np.ndarray] = {}

            # agent 状态
            if "agent" in obs:
                agent = obs["agent"]
                if isinstance(agent, dict):
                    for v in agent.values():
                        arr = self._strip_single_env_batch(self._to_numpy(v))
                        state_parts.append(arr.flatten().astype(np.float32))
                else:
                    arr = self._strip_single_env_batch(self._to_numpy(agent))
                    state_parts.append(arr.flatten().astype(np.float32))

            # extra 状态（goal_pos 等）
            if "extra" in obs:
                extra = obs["extra"]
                if isinstance(extra, dict):
                    for v in extra.values():
                        arr = self._strip_single_env_batch(self._to_numpy(v))
                        state_parts.append(arr.flatten().astype(np.float32))
                else:
                    arr = self._strip_single_env_batch(self._to_numpy(extra))
                    state_parts.append(arr.flatten().astype(np.float32))

            # 传感器数据（图像/点云）
            if "sensor_data" in obs:
                for cam_name, cam_data in obs["sensor_data"].items():
                    if isinstance(cam_data, dict):
                        # 处理 Color (RGBA → RGB, HWC → CHW)
                        if "Color" in cam_data:
                            rgba = self._strip_single_env_batch(self._to_numpy(cam_data["Color"]))
                            # 取 RGB 通道（去掉 alpha）
                            if rgba.ndim == 3 and rgba.shape[-1] == 4:
                                rgb = rgba[:, :, :3]
                            elif rgba.ndim == 3 and rgba.shape[-1] == 3:
                                rgb = rgba
                            else:
                                rgb = rgba
                            # HWC → CHW
                            if rgb.ndim == 3:
                                rgb = np.transpose(rgb, (2, 0, 1))
                            result[f"image_{cam_name}"] = rgb
                        # 处理 depth
                        if "depth" in cam_data:
                            depth = self._strip_single_env_batch(self._to_numpy(cam_data["depth"]))
                            result[f"depth_{cam_name}"] = depth
                        # 处理 rgb（小写）
                        if "rgb" in cam_data:
                            rgb = self._strip_single_env_batch(self._to_numpy(cam_data["rgb"]))
                            if rgb.ndim == 3 and rgb.shape[-1] in (3, 4):
                                rgb = np.transpose(rgb[:, :, :3], (2, 0, 1))
                            result[f"image_{cam_name}"] = rgb
                    else:
                        arr = self._strip_single_env_batch(self._to_numpy(cam_data))
                        result[f"sensor_{cam_name}"] = arr

            if state_parts:
                result["state"] = np.concatenate(state_parts)
            return result

        # 纯 tensor/array 观测
        arr = self._strip_single_env_batch(self._to_numpy(obs))
        return {"state": arr.flatten().astype(np.float32)}

    def _process_info(self, info: dict[str, Any]) -> dict[str, Any]:
        """递归将 info 中的 tensor 转为 Python 标量。"""
        result: dict[str, Any] = {}
        for k, v in info.items():
            if isinstance(v, dict):
                result[k] = self._process_info(v)
            else:
                arr = self._to_numpy(v) if hasattr(v, '__len__') or hasattr(v, 'item') else v
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 0:
                        result[k] = arr.item()
                    elif arr.size == 1:
                        result[k] = arr.flat[0]
                        # 转为 Python 原生类型
                        if isinstance(result[k], (np.bool_,)):
                            result[k] = bool(result[k])
                        elif isinstance(result[k], (np.integer,)):
                            result[k] = int(result[k])
                        elif isinstance(result[k], (np.floating,)):
                            result[k] = float(result[k])
                    else:
                        result[k] = arr
                else:
                    result[k] = v
        return result

    @staticmethod
    def _safe_dtype(dtype: Any) -> str:
        return str(dtype)

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
