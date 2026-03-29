from __future__ import annotations

import sys
import types

import numpy as np
import torch

from robocore.env.wrappers.maniskill3_env import ManiSkill3Env


def _install_fake_maniskill(monkeypatch):
    captured: dict[str, object] = {}

    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = types.SimpleNamespace(shape=(1, 7))
            self.single_action_space = types.SimpleNamespace(shape=(7,))
            self.last_action = None
            self.closed = False

        def reset(self, seed: int | None = None):
            captured["seed"] = seed
            return torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32), {"seed": seed}

        def step(self, action):
            self.last_action = action
            return (
                torch.tensor([[5.0, 6.0, 7.0, 8.0]], dtype=torch.float32),
                torch.tensor([1.25], dtype=torch.float32),
                torch.tensor([False]),
                torch.tensor([False]),
                {"success": torch.tensor([True])},
            )

        def get_obs(self):
            return torch.tensor([[9.0, 10.0]], dtype=torch.float32)

        def close(self) -> None:
            self.closed = True

    fake_env = FakeEnv()

    def make(task_name: str, **kwargs):
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs
        return fake_env

    gym_module = types.ModuleType("gymnasium")
    gym_module.make = make

    mani_skill_module = types.ModuleType("mani_skill")
    mani_skill_module.__path__ = []
    mani_skill_envs_module = types.ModuleType("mani_skill.envs")

    monkeypatch.setitem(sys.modules, "gymnasium", gym_module)
    monkeypatch.setitem(sys.modules, "mani_skill", mani_skill_module)
    monkeypatch.setitem(sys.modules, "mani_skill.envs", mani_skill_envs_module)

    return fake_env, captured


def _install_fake_maniskill_dict_obs(monkeypatch):
    captured: dict[str, object] = {}

    def make_obs() -> dict[str, object]:
        return {
            "agent": {
                "qpos": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                "qvel": torch.tensor([[3.0, 4.0]], dtype=torch.float32),
            },
            "extra": {
                "goal_pos": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
            },
            "sensor_data": {
                "base_camera": {
                    "Color": torch.tensor(
                        [
                            [
                                [[10, 20, 30, 255], [40, 50, 60, 255]],
                                [[70, 80, 90, 255], [100, 110, 120, 255]],
                            ]
                        ],
                        dtype=torch.uint8,
                    )
                }
            },
        }

    class FakeEnv:
        def __init__(self) -> None:
            self.action_space = types.SimpleNamespace(
                shape=(1, 4),
                low=np.full((1, 4), -1.0, dtype=np.float32),
                high=np.full((1, 4), 1.0, dtype=np.float32),
            )
            self.single_action_space = types.SimpleNamespace(
                shape=(4,),
                low=np.full((4,), -1.0, dtype=np.float32),
                high=np.full((4,), 1.0, dtype=np.float32),
            )
            self.last_action = None
            self.closed = False

        def reset(self, seed: int | None = None):
            captured["seed"] = seed
            return make_obs(), {"seed": seed}

        def step(self, action):
            self.last_action = action
            return (
                make_obs(),
                torch.tensor([0.5], dtype=torch.float32),
                torch.tensor([True]),
                torch.tensor([False]),
                {
                    "success": torch.tensor([True]),
                    "metric": torch.tensor([2.0], dtype=torch.float32),
                    "nested": {"flag": torch.tensor([False])},
                },
            )

        def get_obs(self):
            return make_obs()

        def close(self) -> None:
            self.closed = True

    fake_env = FakeEnv()

    def make(task_name: str, **kwargs):
        captured["task_name"] = task_name
        captured["kwargs"] = kwargs
        return fake_env

    gym_module = types.ModuleType("gymnasium")
    gym_module.make = make

    mani_skill_module = types.ModuleType("mani_skill")
    mani_skill_module.__path__ = []
    mani_skill_envs_module = types.ModuleType("mani_skill.envs")

    monkeypatch.setitem(sys.modules, "gymnasium", gym_module)
    monkeypatch.setitem(sys.modules, "mani_skill", mani_skill_module)
    monkeypatch.setitem(sys.modules, "mani_skill.envs", mani_skill_envs_module)

    return fake_env, captured


def test_maniskill3_env_supports_current_gym_api(monkeypatch) -> None:
    fake_env, captured = _install_fake_maniskill(monkeypatch)

    env = ManiSkill3Env(
        task_name="PickCube-v1",
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
        max_episode_steps=1,
        robot_uids="fetch",
    )

    obs = env.reset(seed=7)

    assert captured["task_name"] == "PickCube-v1"
    assert captured["kwargs"] == {
        "obs_mode": "state",
        "control_mode": "pd_joint_delta_pos",
        "num_envs": 1,
        "robot_uids": "fetch",
    }
    assert env.spec.action_dim == 7
    np.testing.assert_allclose(obs["state"], np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    current_obs = env.get_obs()
    np.testing.assert_allclose(current_obs["state"], np.array([9.0, 10.0], dtype=np.float32))

    step_result = env.step(np.zeros(7, dtype=np.float32))
    assert tuple(fake_env.last_action.shape) == (1, 7)
    np.testing.assert_allclose(step_result.obs["state"], np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32))
    assert step_result.reward == 1.25
    assert step_result.done is False
    assert step_result.truncated is True

    env.close()
    assert fake_env.closed is True


def test_maniskill3_env_processes_state_dict_color_obs_and_info(monkeypatch) -> None:
    fake_env, _ = _install_fake_maniskill_dict_obs(monkeypatch)

    env = ManiSkill3Env(
        task_name="PickCube-v1",
        obs_mode="state_dict+rgb",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
        max_episode_steps=5,
    )

    obs = env.reset(seed=3)

    np.testing.assert_allclose(obs["state"], np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32))
    assert obs["image_base_camera"].shape == (3, 2, 2)
    np.testing.assert_array_equal(
        obs["image_base_camera"][:, 0, 0],
        np.array([10, 20, 30], dtype=np.uint8),
    )
    assert set(env.spec.obs_keys) == {"state", "image_base_camera"}
    assert env.spec.obs_shapes["state"] == (6,)
    assert env.spec.obs_shapes["image_base_camera"] == (3, 2, 2)
    assert env.spec.action_range == (-1.0, 1.0)

    step_result = env.step(np.zeros(4, dtype=np.float32))

    assert tuple(fake_env.last_action.shape) == (1, 4)
    assert step_result.done is True
    assert step_result.truncated is False
    assert step_result.info == {
        "success": True,
        "metric": 2.0,
        "nested": {"flag": False},
    }

    env.close()
    assert fake_env.closed is True
