"""测试核心数据结构。"""
import numpy as np
import torch

from robocore.data.episode import Action, Episode, Observation
from robocore.data.transforms import ImageTransform, Normalize, TransformPipeline


def test_observation():
    obs = Observation(
        state=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        images={"front": np.random.randint(0, 255, (3, 84, 84), dtype=np.uint8)},
        language="pick up the red block",
    )
    assert obs.state is not None
    assert "front" in obs.images
    assert obs.language == "pick up the red block"

    # to_tensor
    obs_t = obs.to_tensor("cpu")
    assert isinstance(obs_t.state, torch.Tensor)
    assert isinstance(obs_t.images["front"], torch.Tensor)


def test_action():
    action = Action(
        data=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32),
        action_type="joint_position",
        is_delta=True,
    )
    assert action.data is not None
    assert action.action_type == "joint_position"

    action_t = action.to_tensor("cpu")
    assert isinstance(action_t.data, torch.Tensor)


def test_episode():
    obs_list = [
        Observation(state=np.random.randn(10).astype(np.float32))
        for _ in range(50)
    ]
    act_list = [
        Action(data=np.random.randn(7).astype(np.float32))
        for _ in range(49)
    ]
    ep = Episode(
        observations=obs_list,
        actions=act_list,
        metadata={"success": True, "task": "lift"},
    )
    assert ep.length == 50
    assert ep.success is True


def test_transform_pipeline():
    pipeline = TransformPipeline([
        ImageTransform(image_keys=["image_front"], normalize=True),
    ])

    sample = {
        "obs": {
            "state": torch.randn(10),
            "image_front": torch.randint(0, 255, (3, 84, 84), dtype=torch.uint8),
        },
        "action": torch.randn(16, 7),
    }

    result = pipeline(sample)
    assert result["obs"]["image_front"].dtype == torch.float32
    assert result["obs"]["image_front"].max() <= 1.0


def test_normalize():
    norm = Normalize(keys=["action"], mode="minmax")
    sample = {
        "obs": {"state": torch.randn(10)},
        "action": torch.randn(7),
    }
    result = norm(sample)
    assert "action" in result


def test_resolve_robomimic_env_name_supports_can_alias() -> None:
    from robocore.env.wrappers.robomimic_env import resolve_robomimic_env_name

    assert resolve_robomimic_env_name("Lift") == "Lift"
    assert resolve_robomimic_env_name("Can") == "PickPlaceCan"
    assert resolve_robomimic_env_name("PickPlaceCan") == "PickPlaceCan"


def test_can_teacher_contract_placeholder() -> None:
    from scripts.generate_robomimic_data import CanTeacher

    class DummyCanEnv:
        object_id = 3
        target_bin_placements = np.array(
            [
                [0.0025, 0.1575, 0.8],
                [0.1975, 0.1575, 0.8],
                [0.0025, 0.4025, 0.8],
                [0.1975, 0.4025, 0.8],
            ],
            dtype=np.float32,
        )
        bin2_pos = np.array([0.1, 0.28, 0.8], dtype=np.float32)

    env = DummyCanEnv()
    teacher = CanTeacher()
    teacher.reset(env)

    obs = {
        "robot0_eef_pos": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "Can_pos": np.array([0.05, -0.1, 0.86], dtype=np.float32),
        "Can_to_robot0_eef_pos": np.array([0.05, -0.1, -0.14], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
    }

    action = teacher.act(obs, env, grasped=False)

    assert action.shape == (7,)
    assert np.all(np.isfinite(action))
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    assert np.allclose(teacher.target_bin, env.target_bin_placements[env.object_id])
