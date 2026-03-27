"""测试策略模块。"""
import torch

from robocore.policy.algos.act import ACTPolicy
from robocore.policy.algos.diffusion_policy import DiffusionPolicy
from robocore.policy.algos.flow_policy import FlowPolicy
from robocore.policy.registry import PolicyRegistry


def test_policy_registry():
    """测试策略注册表。"""
    available = PolicyRegistry.list()
    assert "dp" in available
    assert "act" in available
    assert "flow_policy" in available


def test_diffusion_policy_forward():
    """测试 Diffusion Policy 前向传播。"""
    policy = DiffusionPolicy(
        obs_dim=10,
        action_dim=7,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
        num_diffusion_steps=10,
        num_inference_steps=5,
    )

    obs = {"state": torch.randn(2, 2, 10)}
    action = torch.randn(2, 16, 7)

    # 训练
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert losses["loss"].requires_grad

    # 推理
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (2, 16, 7)

    # get_action
    exec_action = policy.get_action(output)
    assert exec_action.shape == (2, 8, 7)


def test_act_policy_forward():
    """测试 ACT 前向传播。"""
    policy = ACTPolicy(
        obs_dim=10,
        action_dim=7,
        obs_horizon=1,
        pred_horizon=16,
        action_horizon=16,
        device="cpu",
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        latent_dim=16,
    )

    obs = {"state": torch.randn(2, 1, 10)}
    action = torch.randn(2, 16, 7)

    # 训练
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert "recon_loss" in losses
    assert "kl_loss" in losses

    # 推理
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (2, 16, 7)


def test_flow_policy_forward():
    """测试 Flow Policy 前向传播。"""
    policy = FlowPolicy(
        obs_dim=10,
        action_dim=7,
        obs_horizon=2,
        pred_horizon=16,
        action_horizon=8,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
        num_inference_steps=5,
    )

    obs = {"state": torch.randn(2, 2, 10)}
    action = torch.randn(2, 16, 7)

    # 训练
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses

    # 推理
    policy.train(False)
    output = policy.predict(obs)
    assert output.action.shape == (2, 16, 7)


def test_policy_save_load(tmp_path):
    """测试策略保存和加载。"""
    policy = DiffusionPolicy(
        obs_dim=10,
        action_dim=7,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
        num_diffusion_steps=10,
        num_inference_steps=5,
    )

    save_path = tmp_path / "test_policy"
    policy.save(save_path)

    policy2 = DiffusionPolicy(
        obs_dim=10,
        action_dim=7,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
        num_diffusion_steps=10,
        num_inference_steps=5,
    )
    policy2.load(save_path)

    for p1, p2 in zip(policy.parameters(), policy2.parameters()):
        assert torch.allclose(p1, p2)


def test_policy_num_params():
    """测试参数计数。"""
    policy = DiffusionPolicy(
        obs_dim=10,
        action_dim=7,
        device="cpu",
        hidden_dim=64,
        down_dims=(64, 128),
    )
    assert policy.num_params > 0
