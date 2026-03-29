"""LIBERO 环境 wrapper。"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robocore.env.base import BaseEnv, EnvSpec, StepResult
from robocore.env.wrappers.registry import EnvRegistry

logger = logging.getLogger(__name__)



def _maybe_bootstrap_libero_config() -> None:
    """为首次安装的 LIBERO 创建默认配置，避免交互式 prompt。"""
    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero")).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return

    try:
        spec = importlib.util.find_spec("libero")
    except (ImportError, AttributeError, ValueError):
        return

    if spec is None or not spec.submodule_search_locations:
        return

    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    benchmark_root = package_root / "libero"
    if not benchmark_root.exists():
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    default_paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(benchmark_root.parent / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(default_paths, f, sort_keys=False)
    logger.info("Initialized LIBERO config at %s", config_file)


@EnvRegistry.register("libero")
class LiberoEnv(BaseEnv):
    """LIBERO benchmark 环境 wrapper。

    当前上游 LIBERO 主要提供以下 suite：
    - ``libero_spatial``
    - ``libero_object``
    - ``libero_goal``
    - ``libero_10`` / ``libero_90`` / ``libero_100``
    """

    def __init__(
        self,
        task_name: str = "libero_spatial",
        task_id: int = 0,
        camera_names: list[str] | None = None,
        camera_height: int = 128,
        camera_width: int = 128,
        max_episode_steps: int = 300,
        use_init_states: bool = True,
        **kwargs: Any,
    ):
        super().__init__(task_name=task_name)
        self.task_id = task_id
        self.camera_names = camera_names or ["agentview", "robot0_eye_in_hand"]
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.max_episode_steps = max_episode_steps
        self.use_init_states = use_init_states
        self._env = None
        self._step_count = 0
        self._task_description = ""
        self._init_states: np.ndarray | None = None
        self._init_state_id: int = 0

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        _maybe_bootstrap_libero_config()

        try:
            from libero.libero import benchmark, get_libero_path
            from libero.libero.envs import OffScreenRenderEnv
        except ImportError as exc:
            raise ImportError(
                "libero not installed. Prefer the upstream source install "
                "(`pip install -e <LIBERO_repo>`) or install this project's "
                "optional dependency with `uv sync --extra libero`."
            ) from exc

        bench_or_factory = benchmark.get_benchmark(self.task_name)
        if isinstance(bench_or_factory, type):
            bench = bench_or_factory()
        elif callable(bench_or_factory) and not hasattr(bench_or_factory, "tasks"):
            bench = bench_or_factory()
        else:
            bench = bench_or_factory

        task = bench.get_task(self.task_id)
        task_description = getattr(task, "language", None)
        if task_description is None and hasattr(bench, "get_task_description"):
            task_description = bench.get_task_description(self.task_id)
        if task_description is None:
            task_description = getattr(task, "name", f"{self.task_name}_{self.task_id}")

        task_bddl_file = self._resolve_bddl_file(task, get_libero_path)
        env_args = {
            "bddl_file_name": str(task_bddl_file),
            "camera_heights": self.camera_height,
            "camera_widths": self.camera_width,
            "camera_names": self.camera_names,
            "horizon": self.max_episode_steps,
        }

        self._env = OffScreenRenderEnv(**env_args)
        self._task_description = str(task_description)

        # 加载 init states（上游评估链路必需）
        if self.use_init_states and hasattr(bench, "get_task_init_states"):
            try:
                init_states = bench.get_task_init_states(self.task_id)
                self._init_states = np.asarray(init_states)
            except Exception:
                logger.debug(
                    "Could not load init states for task %s_%d",
                    self.task_name, self.task_id,
                )

        raw_obs = self._env.reset()
        action_low, action_high = self._get_action_bounds()
        processed_obs = self._process_obs(raw_obs)

        self._spec = EnvSpec(
            obs_keys=list(processed_obs.keys()),
            obs_shapes={key: tuple(value.shape) for key, value in processed_obs.items()},
            obs_dtypes={key: str(value.dtype) for key, value in processed_obs.items()},
            action_dim=int(action_low.shape[0]),
            action_type="eef_delta_pose",
            action_range=(float(action_low.min()), float(action_high.max())),
            task_name=f"{self.task_name}_{self.task_id}",
            max_episode_steps=self.max_episode_steps,
        )

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        self._lazy_init()
        if seed is not None:
            np.random.seed(seed)
            if hasattr(self._env, "seed"):
                self._env.seed(seed)

        # 使用 init_states 重置（与上游评估链路对齐）
        if self._init_states is not None and hasattr(self._env, "set_init_state"):
            idx = self._init_state_id % len(self._init_states)
            raw_obs = self._env.set_init_state(self._init_states[idx])
            self._init_state_id += 1
        else:
            raw_obs = self._env.reset()

        self._step_count = 0
        return self._process_obs(raw_obs)

    def step(self, action: np.ndarray) -> StepResult:
        raw_obs, reward, done, info = self._env.step(action)
        self._step_count += 1
        truncated = self._step_count >= self.max_episode_steps
        info = dict(info or {})
        if hasattr(self._env, "check_success"):
            info.setdefault("success", bool(self._env.check_success()))

        return StepResult(
            obs=self._process_obs(raw_obs),
            reward=float(reward),
            done=bool(done),
            truncated=truncated,
            info=info,
        )

    def get_obs(self) -> dict[str, Any]:
        self._lazy_init()
        return self._process_obs(self._get_raw_observations())

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    def _get_raw_observations(self) -> dict[str, Any]:
        if hasattr(self._env, "_get_observations"):
            return self._env._get_observations()
        inner_env = getattr(self._env, "env", None)
        if inner_env is not None and hasattr(inner_env, "_get_observations"):
            return inner_env._get_observations()
        raise AttributeError("LIBERO env does not expose _get_observations().")

    def _get_action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """获取动作空间上下界，兼容多种上游 API。"""
        # 优先使用 action_spec（ControlEnv 直接暴露的属性）
        if hasattr(self._env, "action_spec"):
            low = np.asarray(self._env.action_spec[0], dtype=np.float32).reshape(-1)
            high = np.asarray(self._env.action_spec[1], dtype=np.float32).reshape(-1)
            return low, high
        # 回退到内部 robosuite env 的 action_spec
        inner = getattr(self._env, "env", None)
        if inner is not None and hasattr(inner, "action_spec"):
            low = np.asarray(inner.action_spec[0], dtype=np.float32).reshape(-1)
            high = np.asarray(inner.action_spec[1], dtype=np.float32).reshape(-1)
            return low, high
        # 最终回退：7-DoF 默认范围
        logger.warning("Cannot determine action_spec; falling back to 7-DoF [-1, 1].")
        return -np.ones(7, dtype=np.float32), np.ones(7, dtype=np.float32)

    def _resolve_bddl_file(self, task: Any, get_libero_path: Any) -> Path:
        task_bddl = Path(getattr(task, "bddl_file", ""))
        if task_bddl.is_absolute() and task_bddl.exists():
            return task_bddl

        bddl_root = Path(get_libero_path("bddl_files"))
        problem_folder = getattr(task, "problem_folder", "")
        candidates = []
        if task_bddl:
            candidates.append(bddl_root / problem_folder / task_bddl)
            candidates.append(bddl_root / task_bddl)
            candidates.append(task_bddl)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Unable to locate LIBERO BDDL file for task "
            f"'{getattr(task, 'name', self.task_name)}'. "
            f"Tried: {[str(c) for c in candidates]}"
        )

    def _process_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        obs: dict[str, Any] = {}
        state_parts = []
        for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]:
            if key in raw_obs:
                state_parts.append(np.asarray(raw_obs[key], dtype=np.float32))
        if state_parts:
            obs["state"] = np.concatenate(state_parts)

        for cam in self.camera_names:
            key = f"{cam}_image"
            if key in raw_obs:
                img = np.asarray(raw_obs[key], dtype=np.uint8)
                if img.ndim == 3 and img.shape[-1] in (1, 3):
                    img = np.transpose(img, (2, 0, 1))
                obs[f"image_{cam}"] = img

        return obs
