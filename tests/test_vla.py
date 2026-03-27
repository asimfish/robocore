"""测试 VLA 策略 (OpenVLA, Pi0)。"""
import torch

from robocore.policy.algos.openvla import OpenVLAPolicy
from robocore.policy.algos.pi0 import Pi0Policy
from robocore.policy.registry import PolicyRegistry


def test_vla_registered():
    available = PolicyRegistry.list()
    assert "openvla" in available
    assert "pi0" in available


def test_openvla_forward():
    policy = OpenVLAPolicy(
        obs_dim=10,
        action_dim=7,
        device="cpu",
        action_bins=32,
    )

    obs = {"state": torch.randn(2, 10)}
    action = torch.randn(2, 1, 7)

    # 训练
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert losses["loss"].requires_grad

    # 推理
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (2, 1, 7)


def test_pi0_forward():
    policy = Pi0Policy(
        obs_dim=10,
        action_dim=7,
        pred_horizon=16,
        action_horizon=16,
        device="cpu",
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_inference_steps=5,
    )

    obs = {"state": torch.randn(2, 10)}
    action = torch.randn(2, 16, 7)

    # 训练
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert losses["loss"].requires_grad

    # 推理
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (2, 16, 7)


def test_pi0_action_chunking():
    """测试 Pi0 的 action chunking。"""
    policy = Pi0Policy(
        obs_dim=10,
        action_dim=7,
        pred_horizon=32,
        action_horizon=16,
        device="cpu",
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        num_inference_steps=3,
    )

    obs = {"state": torch.randn(4, 10)}
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (4, 32, 7)

    exec_action = policy.get_action(output)
    assert exec_action.shape == (4, 16, 7)
