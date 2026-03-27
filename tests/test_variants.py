"""测试所有新增算法变种。"""
import torch
import pytest

from robocore.policy.registry import PolicyRegistry


# 小型参数，加速测试
SMALL = dict(
    obs_dim=10, action_dim=7, obs_horizon=2, pred_horizon=16,
    action_horizon=8, device="cpu", hidden_dim=64,
)


def _make_obs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {"state": torch.randn(batch, 2, 10)}


def _make_action(batch: int = 2) -> torch.Tensor:
    return torch.randn(batch, 16, 7)


# ============================================================
# Pi0.5
# ============================================================

def test_pi05_registered():
    assert "pi05" in PolicyRegistry.list()


def test_pi05_forward():
    from robocore.policy.algos.pi05 import Pi05Policy
    policy = Pi05Policy(
        **SMALL, num_layers=2, num_heads=4,
        num_inference_steps=3, max_reasoning_tokens=8,
    )
    obs, action = _make_obs(), _make_action()

    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert "system1_loss" in losses
    assert "system2_loss" in losses
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)
    assert "route_score" in out.info


def test_pi05_no_reasoning():
    from robocore.policy.algos.pi05 import Pi05Policy
    policy = Pi05Policy(
        **SMALL, num_layers=2, num_heads=4,
        num_inference_steps=3, use_reasoning=False,
    )
    obs = _make_obs()
    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


# ============================================================
# DiT Policy
# ============================================================

def test_dit_registered():
    assert "dit" in PolicyRegistry.list()


def test_dit_forward():
    from robocore.policy.algos.dit_policy import DiTPolicy
    policy = DiTPolicy(
        **SMALL, num_layers=2, num_heads=4,
        num_diffusion_steps=10, num_inference_steps=5,
    )
    obs, action = _make_obs(), _make_action()

    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


# ============================================================
# DP 变种
# ============================================================

def test_dp_variants_registered():
    names = PolicyRegistry.list()
    for name in ["dp_t", "dp_cnn", "dp_be", "dp_cfg"]:
        assert name in names, f"{name} not registered"


def test_dp_transformer():
    from robocore.policy.algos.dp_variants import DPTransformer
    policy = DPTransformer(
        **SMALL, num_diffusion_steps=10, num_inference_steps=5,
        num_layers=2, num_heads=4,
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


def test_dp_cnn():
    from robocore.policy.algos.dp_variants import DPCNN
    policy = DPCNN(
        **SMALL, num_diffusion_steps=10, num_inference_steps=5,
        down_dims=(64, 128),
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad


def test_dp_behavior_ensemble():
    from robocore.policy.algos.dp_variants import DPBehaviorEnsemble
    policy = DPBehaviorEnsemble(
        **SMALL, num_diffusion_steps=10, num_inference_steps=3,
        num_ensemble_heads=2, down_dims=(64, 128),
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)
    assert "ensemble_actions" in out.info


def test_dp_cfg():
    from robocore.policy.algos.dp_variants import DPClassifierFreeGuidance
    policy = DPClassifierFreeGuidance(
        **SMALL, num_diffusion_steps=10, num_inference_steps=5,
        p_uncond=0.1, guidance_scale=1.5, down_dims=(64, 128),
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


# ============================================================
# Flow 变种
# ============================================================

def test_flow_variants_registered():
    names = PolicyRegistry.list()
    for name in ["mean_flow", "consistency_flow", "rectified_flow", "flow_dit"]:
        assert name in names, f"{name} not registered"


def test_mean_flow():
    from robocore.policy.algos.flow_variants import MeanFlowPolicy
    policy = MeanFlowPolicy(
        **SMALL, num_inference_steps=1, down_dims=(64, 128),
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


def test_mean_flow_multistep():
    from robocore.policy.algos.flow_variants import MeanFlowPolicy
    policy = MeanFlowPolicy(
        **SMALL, num_inference_steps=5, down_dims=(64, 128),
    )
    policy.train(False)
    out = policy.predict(_make_obs())
    assert out.action.shape == (2, 16, 7)


def test_consistency_flow():
    from robocore.policy.algos.flow_variants import ConsistencyFlowPolicy
    policy = ConsistencyFlowPolicy(
        **SMALL, num_inference_steps=3, ema_decay=0.99,
        down_dims=(64, 128),
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert "loss" in losses
    assert "flow_loss" in losses
    assert "consistency_loss" in losses

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


def test_rectified_flow():
    from robocore.policy.algos.flow_variants import RectifiedFlowPolicy
    policy = RectifiedFlowPolicy(
        **SMALL, num_inference_steps=5, down_dims=(64, 128),
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


def test_flow_dit():
    from robocore.policy.algos.flow_variants import FlowDiTPolicy
    policy = FlowDiTPolicy(
        **SMALL, num_inference_steps=3,
        num_layers=2, num_heads=4,
    )
    obs, action = _make_obs(), _make_action()
    losses = policy.compute_loss(obs, action)
    assert losses["loss"].requires_grad

    policy.train(False)
    out = policy.predict(obs)
    assert out.action.shape == (2, 16, 7)


# ============================================================
# DiT 网络单元测试
# ============================================================

def test_dit_action_model():
    from robocore.policy.networks.dit import DiTActionModel
    model = DiTActionModel(
        action_dim=7, cond_dim=64, hidden_dim=64,
        num_layers=2, num_heads=4,
    )
    x = torch.randn(2, 16, 7)
    t = torch.randint(0, 100, (2,))
    cond = torch.randn(2, 64)
    out = model(x, t, cond)
    assert out.shape == (2, 16, 7)


def test_ada_layer_norm():
    from robocore.policy.networks.dit import AdaLayerNorm
    norm = AdaLayerNorm(64, 64)
    x = torch.randn(2, 16, 64)
    cond = torch.randn(2, 64)
    out = norm(x, cond)
    assert out.shape == (2, 16, 64)
