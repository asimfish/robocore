"""IsaacLab 环境 wrapper。

通过 NVIDIA IsaacSim + IsaacLab 提供 GPU 并行仿真环境。
安装方式：uv sync --extra isaaclab
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)

# 常用 IsaacLab 任务别名 → 完整 task id
ISAACLAB_TASK_ALIASES: dict[str, str] = {
    "G1-Locomotion": "Isaac-Velocity-Rough-G1-v0",
    "H1-Locomotion": "Isaac-Velocity-Rough-H1-v0",
    "AnymalD-Locomotion": "Isaac-Velocity-Rough-Anymal-D-v0",
    "Go2-Locomotion": "Isaac-Velocity-Rough-Unitree-Go2-v0",
}


def resolve_isaaclab_task(task_name: str) -> str:
    """将别名映射到 IsaacLab 完整 task id。"""
    return ISAACLAB_TASK_ALIASES.get(task_name, task_name)


@EnvRegistry.register("isaaclab")
class IsaacLabEnv(BaseEnv):
    """IsaacLab GPU 并行环境 wrapper。

    封装 IsaacLab 的 ManagerBasedRLEnv / DirectRLEnv，
    提供与 robocore 统一的 BaseEnv 接口。
    """

    def __init__(
        self,
        task_name: str = "Isaac-Velocity-Rough-H1-v0",
        num_envs: int = 64,
        device: str = "cuda:0",
        headless: bool = True,
        use_image: bool = False,
        camera_names: list[str] | None = None,
        camera_height: int = 84,
        camera_width: int = 84,
        max_episode_steps: int = 1000,
        **kwargs: Any,
    ):
        real_task = resolve_isaaclab_task(task_name)
        super().__init__(task_name=real_task)
        self.num_envs = num_envs
        self.device = device
        self.headless = headless
        self.use_image = use_image
        self.camera_names = camera_names or []
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_episode_steps = max_episode_steps
        self._extra_kwargs = kwargs
        self._env = None
        self._step_count: torch.Tensor | None = None

    def _lazy_init(self) -> None:
        """延迟初始化 IsaacLab 环境。"""
        if self._env is not None:
            return

        try:
            from omni.isaac.lab.app import AppLauncher
        except ImportError:
            try:
                from isaaclab.app import AppLauncher
            except ImportError:
                raise ImportError(
                    "IsaacLab not installed. Install with:\n"
                    "  1. git submodule update --init --recursive\n"
                    "  2. uv sync --extra isaaclab"
                )

        # 启动 IsaacSim 应用（如果尚未启动）
        import builtins
        if not hasattr(builtins, "ISAAC_LAUNCHED"):
            launcher = AppLauncher(headless=self.headless)
            builtins.ISAAC_LAUNCHED = True  # noqa: B010
            logger.info("IsaacSim AppLauncher started (headless=%s)", self.headless)

        import gymnasium as gym
        try:
            import isaaclab_tasks  # noqa: F401 — 注册 IsaacLab 任务
        except ImportError:
            pass

        self._env = gym.make(
            self.task_name,
            num_envs=self.num_envs,
            device=self.device,
            **self._extra_kwargs,
        )

        # 构建 EnvSpec
        obs_space = self._env.observation_space
        act_space = self._env.action_space

        action_dim = (
            act_space.shape[0] if hasattr(act_space, "shape") and act_space.shape
            else int(act_space.n) if hasattr(act_space, "n")
            else 1
        )

        obs_keys = ["state"]
        obs_shapes: dict[str, tuple[int, ...]] = {}

        if hasattr(obs_space, "shape") and obs_space.shape:
            obs_shapes["state"] = tuple(obs_space.shape)
        elif hasattr(obs_space, "spaces"):
            # Dict observation space
            total_dim = 0
            for k, v in obs_space.spaces.items():
                if v.shape and len(v.shape) == 1:
                    total_dim += v.shape[0]
            obs_shapes["state"] = (total_dim,) if total_dim > 0 else (1,)

        if self.use_image:
            for cam in self.camera_names:
                key = f"image_{cam}"
                obs_keys.append(key)
                obs_shapes[key] = (3, self.camera_height, self.camera_width)

        self._spec = EnvSpec(
            obs_keys=obs_keys,
            obs_shapes=obs_shapes,
            action_dim=action_dim,
            action_type="joint_position",
            action_range=(-1.0, 1.0),
            task_name=self.task_name,
            max_episode_steps=self.max_episode_steps,
            num_envs=self.num_envs,
        )

        self._step_count = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)

        obs, info = self._env.reset()
        self._step_count.zero_()
        return self._process_obs(obs)

    def step(self, action: np.ndarray | torch.Tensor) -> StepResult:
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action).to(self.device)

        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1

        # 合并 done 信号
        done = terminated | truncated

        return StepResult(
            obs=self._process_obs(obs),
            reward=reward.cpu().numpy() if isinstance(reward, torch.Tensor) else reward,
            done=done.cpu().numpy() if isinstance(done, torch.Tensor) else done,
            truncated=truncated.cpu().numpy() if isinstance(truncated, torch.Tensor) else truncated,
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        obs = self._env.unwrapped.obs_buf if hasattr(self._env.unwrapped, "obs_buf") else {}
        return self._process_obs(obs)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._step_count = None

    def render(self) -> np.ndarray | None:
        if self._env is not None and hasattr(self._env, "render"):
            return self._env.render()
        return None

    def _process_obs(self, raw_obs: dict | torch.Tensor) -> dict[str, Any]:
        """将 IsaacLab 观测转为统一格式。"""
        obs: dict[str, Any] = {}

        if isinstance(raw_obs, torch.Tensor):
            obs["state"] = raw_obs.cpu().numpy()
        elif isinstance(raw_obs, dict):
            # 拼接所有 1D 向量作为 state
            parts = []
            for k, v in raw_obs.items():
                if isinstance(v, torch.Tensor):
                    v_np = v.cpu().numpy()
                elif isinstance(v, np.ndarray):
                    v_np = v
                else:
                    continue
                if v_np.ndim <= 2:
                    parts.append(v_np.reshape(v_np.shape[0], -1) if v_np.ndim == 2 else v_np.reshape(1, -1))
            if parts:
                obs["state"] = np.concatenate(parts, axis=-1)
            else:
                obs["state"] = np.zeros((self.num_envs, 1), dtype=np.float32)
        else:
            obs["state"] = np.array(raw_obs)

        return obs
