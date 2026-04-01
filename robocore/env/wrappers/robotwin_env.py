"""RobotWin 环境 wrapper。"""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)


@EnvRegistry.register("robotwin")
class RobotWinEnv(BaseEnv):
    """RobotWin benchmark 环境 wrapper。

    桥接 RobotWin 官方任务 API（setup_demo / take_action / get_obs）
    到 robocore 统一的 reset / step / get_obs 接口。
    """

    def __init__(
        self,
        task_name: str = "block_hammer_beat",
        obs_mode: str = "state",
        max_episode_steps: int = 600,
        task_config: str = "demo_clean",
        robotwin_root: str | None = None,
        action_type: str = "qpos",
        instruction: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(task_name=task_name)
        self.obs_mode = obs_mode
        self._user_max_episode_steps = max_episode_steps
        self.task_config = task_config
        self.robotwin_root = robotwin_root
        self.action_type = action_type
        self.instruction = instruction
        self._env = None
        self._step_count = 0
        self._episode_num = 0
        self._camera_cfg: dict[str, Any] = {}

    def _load_configs(self) -> dict[str, Any]:
        """加载 RobotWin 的 YAML 配置。"""
        import yaml

        root = Path(self.robotwin_root) if self.robotwin_root else Path(".")
        cfg_dir = root / "task_config"

        result: dict[str, Any] = {}

        # 主配置
        main_cfg_path = cfg_dir / f"{self.task_config}.yml"
        if main_cfg_path.exists():
            with open(main_cfg_path) as f:
                result.update(yaml.safe_load(f) or {})

        # 相机配置
        cam_cfg_path = cfg_dir / "_camera_config.yml"
        if cam_cfg_path.exists():
            with open(cam_cfg_path) as f:
                self._camera_cfg = yaml.safe_load(f) or {}

        # embodiment 配置
        emb_cfg_path = cfg_dir / "_embodiment_config.yml"
        if emb_cfg_path.exists():
            with open(emb_cfg_path) as f:
                result["embodiment_config"] = yaml.safe_load(f) or {}

        return result

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        configs = self._load_configs()

        # 动态导入任务类
        root = Path(self.robotwin_root) if self.robotwin_root else Path(".")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        task_module = importlib.import_module(f"envs.{self.task_name}")
        task_cls = getattr(task_module, self.task_name)
        self._env = task_cls()

        # 从任务实例获取 step_lim 作为 max_episode_steps
        step_lim = getattr(self._env, "step_lim", self._user_max_episode_steps)

        self._spec = EnvSpec(
            action_dim=14,  # 双臂 7+7
            action_type="joint_position",
            task_name=self.task_name,
            max_episode_steps=step_lim,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        self._step_count = 0
        self._episode_num += 1

        # 构建 setup_demo 的额外参数（相机分辨率等）
        setup_kwargs: dict[str, Any] = {}
        for cam_name, cam_cfg in self._camera_cfg.items():
            if isinstance(cam_cfg, dict):
                for k, v in cam_cfg.items():
                    # 展平为 head_camera_h, head_camera_w 等
                    # 需要找到哪个相机名匹配
                    pass

        # 从配置中提取相机参数
        if self._camera_cfg:
            # RobotWin 的相机配置格式：{D435: {h: 240, w: 320}}
            # 需要映射到 setup_demo 的参数
            for cam_type, cam_params in self._camera_cfg.items():
                if isinstance(cam_params, dict):
                    for param_key, param_val in cam_params.items():
                        # 生成如 head_camera_h, head_camera_w 的参数名
                        setup_kwargs[f"head_camera_{param_key}"] = param_val

        self._env.setup_demo(
            now_ep_num=self._episode_num,
            seed=seed,
            is_test=True,
            **setup_kwargs,
        )

        # 设置指令
        if self.instruction is not None and hasattr(self._env, "set_instruction"):
            self._env.set_instruction(instruction=self.instruction)

        obs = self._env.get_obs()
        return self._process_obs(obs)

    def step(self, action: np.ndarray) -> StepResult:
        self._lazy_init()
        self._step_count += 1

        if not isinstance(action, np.ndarray):
            action = np.asarray(action, dtype=np.float32)

        # 调用 RobotWin 的 take_action
        self._env.take_action(action, action_type=self.action_type)

        obs = self._env.get_obs()
        reward = getattr(self._env, "episode_score", 0.0) - sum(
            getattr(self, "_prev_scores", [0.0])
        )
        # 简化：直接用 episode_score 的增量
        prev_score = getattr(self, "_prev_score", 0.0)
        current_score = getattr(self._env, "episode_score", 0.0)
        reward = current_score - prev_score
        self._prev_score = current_score

        # 检查是否完成
        done = getattr(self._env, "eval_success", False)
        truncated = False
        if not done and self._step_count >= self.spec.max_episode_steps:
            truncated = True
            done = True

        # 构建 info
        info: dict[str, Any] = {}
        if self.instruction is not None:
            info["instruction"] = self.instruction
        info["eval_success"] = getattr(self._env, "eval_success", False)
        info["episode_score"] = current_score

        return StepResult(
            obs=self._process_obs(obs),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        self._lazy_init()
        obs = self._env.get_obs()
        return self._process_obs(obs)

    def close(self) -> None:
        if self._env is not None:
            if hasattr(self._env, "close_env"):
                self._env.close_env(clear_cache=True)
            self._env = None

    def _process_obs(self, obs: Any) -> dict[str, np.ndarray]:
        """将 RobotWin 观测转为统一格式。"""
        if isinstance(obs, dict):
            state_parts: list[np.ndarray] = []
            result: dict[str, np.ndarray] = {}

            for key, val in obs.items():
                if not isinstance(val, np.ndarray):
                    val = np.asarray(val)

                if "pointcloud" in key:
                    result[key] = val
                elif val.ndim == 1:
                    state_parts.append(val.astype(np.float32))
                elif val.ndim == 3 and val.shape[-1] in (1, 3):
                    # HWC → CHW 图像
                    result[f"image_{key}"] = np.transpose(val, (2, 0, 1))
                elif val.ndim == 3:
                    result[f"image_{key}"] = val
                else:
                    state_parts.append(val.flatten().astype(np.float32))

            if state_parts:
                result["state"] = np.concatenate(state_parts)
            return result
        elif isinstance(obs, np.ndarray):
            return {"state": obs.flatten().astype(np.float32)}
        return {"state": np.zeros(28, dtype=np.float32)}
