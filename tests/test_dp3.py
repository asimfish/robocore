"""测试 DP3 策略。"""
import torch

from robocore.policy.algos.dp3 import DP3Policy
from robocore.policy.registry import PolicyRegistry


def test_dp3_registered():
    assert "dp3" in PolicyRegistry.list()


def test_dp3_forward():
    policy = DP3Policy(
        obs_dim=10,
        action_dim=7,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
        point_dim=3,
        num_points=256,
        num_diffusion_steps=10,
        num_inference_steps=5,
    )

    obs = {
        "pointcloud": torch.randn(2, 256, 3),
        "state": torch.randn(2, 2, 10),
    }
    action = torch.randn(2, 16, 7)

    # 训练
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert losses["loss"].requires_grad

    # 推理
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (2, 16, 7)


def test_dp3_pointcloud_only():
    """测试仅使用点云（无状态）。"""
    policy = DP3Policy(
        obs_dim=None,
        action_dim=7,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
        num_diffusion_steps=10,
        num_inference_steps=5,
    )

    obs = {"pointcloud": torch.randn(2, 512, 3)}
    action = torch.randn(2, 16, 7)

    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
