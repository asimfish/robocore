from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from robocore.env.wrappers.robotwin_env import RobotWinEnv


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _create_fake_robotwin_root(tmp_path: Path, task_name: str = "beat_block_hammer") -> Path:
    root = tmp_path / "RoboTwin"

    _write_text(
        root / "collect_data.sh",
        "#!/bin/bash\npython script/collect_data.py\n",
    )
    _write_text(root / "envs" / "__init__.py", "")
    _write_text(
        root / "envs" / f"{task_name}.py",
        f"""
from __future__ import annotations

import numpy as np


class {task_name}:
    def __init__(self):
        self.episode_score = 0.0
        self.step_lim = 2
        self.take_action_cnt = 0
        self.eval_success = False
        self.last_setup = None
        self.last_action = None
        self.instruction = None
        self.closed = False

    def setup_demo(self, now_ep_num, seed, is_test=True, **kwargs):
        self.last_setup = {{
            "now_ep_num": now_ep_num,
            "seed": seed,
            "is_test": is_test,
            "kwargs": kwargs,
        }}
        self.episode_score = 0.0
        self.take_action_cnt = 0
        self.eval_success = False
        self.closed = False

    def get_obs(self):
        return {{
            "joint_state": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "head_camera": np.ones((4, 5, 3), dtype=np.uint8),
            "scene_pointcloud": np.ones((6, 4), dtype=np.float32),
        }}

    def take_action(self, action, action_type="qpos"):
        self.last_action = (np.asarray(action, dtype=np.float32), action_type)
        self.take_action_cnt += 1
        self.episode_score += 0.5
        if self.take_action_cnt >= self.step_lim:
            self.eval_success = True

    def close_env(self, clear_cache=False):
        self.closed = True

    def set_instruction(self, instruction=None):
        self.instruction = instruction

    def get_instruction(self):
        return self.instruction
""",
    )

    _write_text(
        root / "task_config" / "demo_clean.yml",
        """
render_freq: 0
episode_num: 50
use_seed: false
save_freq: 15
embodiment: [aloha-agilex]
language_num: 100
domain_randomization:
  random_background: false
  cluttered_table: false
camera:
  head_camera_type: D435
  wrist_camera_type: D435
  collect_head_camera: true
  collect_wrist_camera: true
data_type:
  rgb: true
  qpos: true
collect_data: false
eval_video_log: false
""".strip()
        + "\n",
    )
    _write_text(
        root / "task_config" / "_embodiment_config.yml",
        """
aloha-agilex:
  file_path: "./assets/embodiments/aloha-agilex"
""".strip()
        + "\n",
    )
    _write_text(
        root / "task_config" / "_camera_config.yml",
        """
D435:
  h: 240
  w: 320
""".strip()
        + "\n",
    )
    _write_text(
        root / "assets" / "embodiments" / "aloha-agilex" / "config.yml",
        """
arm_joints_name:
  - [left_1, left_2, left_3, left_4, left_5, left_6]
  - [right_1, right_2, right_3, right_4, right_5, right_6]
""".strip()
        + "\n",
    )
    return root


def test_robotwin_env_bridges_official_task_api(tmp_path: Path) -> None:
    root = _create_fake_robotwin_root(tmp_path)

    env = RobotWinEnv(
        task_name="beat_block_hammer",
        task_config="demo_clean",
        robotwin_root=str(root),
        action_type="qpos",
        instruction="beat the block",
        max_episode_steps=5,
    )

    try:
        obs = env.reset(seed=7)

        assert env.spec.action_dim == 14
        assert env.spec.max_episode_steps == 2
        assert obs["state"].shape == (3,)
        assert obs["image_head_camera"].shape == (3, 4, 5)
        assert obs["scene_pointcloud"].shape == (6, 4)
        assert env._env.last_setup["seed"] == 7
        assert env._env.last_setup["kwargs"]["head_camera_h"] == 240
        assert env._env.instruction == "beat the block"

        step_1 = env.step(np.zeros(14, dtype=np.float32))
        assert step_1.reward == 0.5
        assert step_1.done is False
        assert step_1.truncated is False
        assert step_1.info["instruction"] == "beat the block"
        np.testing.assert_allclose(env._env.last_action[0], np.zeros(14, dtype=np.float32))
        assert env._env.last_action[1] == "qpos"

        step_2 = env.step(np.ones(14, dtype=np.float32))
        assert step_2.reward == 0.5
        assert step_2.done is True
        assert step_2.truncated is False
    finally:
        env.close()
        importlib.invalidate_caches()
        sys.modules.pop("envs.beat_block_hammer", None)
        sys.modules.pop("envs", None)
