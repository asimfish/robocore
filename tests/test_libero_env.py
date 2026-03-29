from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from robocore.env.wrappers.libero_env import LiberoEnv


def _install_fake_libero(monkeypatch, tmp_path: Path) -> Path:
    suite_dir = tmp_path / "libero_spatial"
    suite_dir.mkdir(parents=True, exist_ok=True)
    bddl_path = suite_dir / "fake_task.bddl"
    bddl_path.write_text("(define (problem fake_task))", encoding="utf-8")

    def make_raw_obs() -> dict[str, np.ndarray]:
        return {
            "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
            "robot0_joint_pos": np.arange(7, dtype=np.float32),
            "agentview_image": np.ones((4, 5, 3), dtype=np.uint8),
        }

    class FakeBenchmark:
        def __init__(self) -> None:
            self._task = types.SimpleNamespace(
                name="fake_task",
                language="pick up the mug",
                problem_folder="libero_spatial",
                bddl_file="fake_task.bddl",
            )

        def get_task(self, task_id: int):
            assert task_id == 0
            return self._task

    benchmark_module = types.ModuleType("libero.libero.benchmark")
    benchmark_module.get_benchmark = lambda name: FakeBenchmark

    class FakeInnerEnv:
        def _get_observations(self) -> dict[str, np.ndarray]:
            return make_raw_obs()

    class FakeOffScreenRenderEnv:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.action_spec = (
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
            )
            self.env = FakeInnerEnv()
            self.seed_value = None
            self.closed = False

        def seed(self, seed: int) -> None:
            self.seed_value = seed

        def reset(self) -> dict[str, np.ndarray]:
            return make_raw_obs()

        def step(self, action):
            return make_raw_obs(), 0.5, False, {"raw_action": action}

        def close(self) -> None:
            self.closed = True

    envs_module = types.ModuleType("libero.libero.envs")
    envs_module.OffScreenRenderEnv = FakeOffScreenRenderEnv

    libero_pkg = types.ModuleType("libero")
    libero_pkg.__path__ = []

    libero_core_module = types.ModuleType("libero.libero")
    libero_core_module.__path__ = []
    libero_core_module.benchmark = benchmark_module
    libero_core_module.get_libero_path = lambda key: str(tmp_path)
    libero_core_module.envs = envs_module

    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_core_module)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)

    return bddl_path


def test_libero_env_supports_current_benchmark_api(monkeypatch, tmp_path: Path) -> None:
    bddl_path = _install_fake_libero(monkeypatch, tmp_path)

    env = LiberoEnv(
        task_name="libero_spatial",
        task_id=0,
        camera_names=["agentview"],
        camera_height=4,
        camera_width=5,
        max_episode_steps=12,
    )

    obs = env.reset(seed=7)

    assert env.spec.action_dim == 7
    assert env._task_description == "pick up the mug"
    assert env._env.kwargs["bddl_file_name"] == str(bddl_path)
    assert env._env.kwargs["horizon"] == 12
    assert env._env.seed_value == 7
    assert obs["state"].shape == (16,)
    assert obs["image_agentview"].shape == (3, 4, 5)

    current_obs = env.get_obs()
    assert current_obs["state"].shape == (16,)
    assert current_obs["image_agentview"].shape == (3, 4, 5)

    step_result = env.step(np.zeros(7, dtype=np.float32))
    assert step_result.reward == 0.5
    assert step_result.done is False

    env.close()


def test_libero_env_applies_benchmark_init_states_on_reset(monkeypatch, tmp_path: Path) -> None:
    bddl_path = _install_fake_libero(monkeypatch, tmp_path)
    init_states = np.asarray(
        [
            np.arange(7, dtype=np.float32),
            np.arange(7, dtype=np.float32) + 10,
        ]
    )

    class FakeBenchmarkWithInitStates:
        def __init__(self) -> None:
            self._task = types.SimpleNamespace(
                name="fake_task",
                language="pick up the mug",
                problem_folder="libero_spatial",
                bddl_file="fake_task.bddl",
                init_states_file="fake_task.pruned_init",
            )

        def get_task(self, task_id: int):
            assert task_id == 0
            return self._task

        def get_task_init_states(self, task_id: int):
            assert task_id == 0
            return init_states

    benchmark_module = types.ModuleType("libero.libero.benchmark")
    benchmark_module.get_benchmark = lambda name: FakeBenchmarkWithInitStates

    class FakeInnerEnv:
        def _get_observations(self) -> dict[str, np.ndarray]:
            return {
                "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
                "robot0_gripper_qpos": np.array([0.4, 0.5], dtype=np.float32),
                "robot0_joint_pos": np.arange(7, dtype=np.float32),
                "agentview_image": np.ones((4, 5, 3), dtype=np.uint8),
            }

    class FakeOffScreenRenderEnv:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.action_spec = (
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
            )
            self.env = FakeInnerEnv()
            self.init_state_calls: list[np.ndarray] = []

        def seed(self, seed: int) -> None:
            self.seed_value = seed

        def reset(self):
            return self.env._get_observations()

        def set_init_state(self, init_state):
            init_state = np.asarray(init_state, dtype=np.float32)
            self.init_state_calls.append(init_state)
            obs = self.env._get_observations().copy()
            obs["robot0_joint_pos"] = init_state
            return obs

        def step(self, action):
            return self.env._get_observations(), 0.5, False, {"raw_action": action}

        def close(self) -> None:
            self.closed = True

    envs_module = types.ModuleType("libero.libero.envs")
    envs_module.OffScreenRenderEnv = FakeOffScreenRenderEnv

    libero_pkg = types.ModuleType("libero")
    libero_pkg.__path__ = []

    libero_core_module = types.ModuleType("libero.libero")
    libero_core_module.__path__ = []
    libero_core_module.benchmark = benchmark_module
    libero_core_module.get_libero_path = lambda key: str(tmp_path)
    libero_core_module.envs = envs_module

    monkeypatch.setitem(sys.modules, "libero", libero_pkg)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_core_module)
    monkeypatch.setitem(sys.modules, "libero.libero.benchmark", benchmark_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)

    env = LiberoEnv(
        task_name="libero_spatial",
        task_id=0,
        camera_names=["agentview"],
        camera_height=4,
        camera_width=5,
        max_episode_steps=12,
    )

    first_obs = env.reset(seed=7)
    second_obs = env.reset(seed=7)

    assert env._env.kwargs["bddl_file_name"] == str(bddl_path)
    np.testing.assert_allclose(env._env.init_state_calls[0], init_states[0])
    np.testing.assert_allclose(env._env.init_state_calls[1], init_states[1])
    np.testing.assert_allclose(first_obs["state"][-7:], init_states[0])
    np.testing.assert_allclose(second_obs["state"][-7:], init_states[1])

    env.close()
