"""测试模仿学习算法。"""
import torch
from robocore.policy.registry import PolicyRegistry


S = dict(obs_dim=10, action_dim=7, device="cpu", hidden_dim=64)


def test_il_registered():
    names = PolicyRegistry.list()
    for n in ["bc", "bc_rnn", "ibc", "dagger"]:
        assert n in names


def test_bc():
    from robocore.policy.algos.imitation import BCPolicy
    p = BCPolicy(**S, num_layers=2)
    obs = {"state": torch.randn(2, 10)}
    losses = p.compute_loss(obs, torch.randn(2, 7))
    assert losses["loss"].requires_grad
    out = p.predict(obs)
    assert out.action.shape == (2, 1, 7)


def test_bc_rnn():
    from robocore.policy.algos.imitation import BCRNNPolicy
    p = BCRNNPolicy(**S, obs_horizon=5, rnn_layers=1)
    obs = {"state": torch.randn(2, 5, 10)}
    losses = p.compute_loss(obs, torch.randn(2, 1, 7))
    assert losses["loss"].requires_grad
    out = p.predict(obs)
    assert out.action.shape == (2, 1, 7)


def test_ibc():
    from robocore.policy.algos.imitation import IBCPolicy
    p = IBCPolicy(**S, num_neg_samples=16, langevin_steps=5)
    obs = {"state": torch.randn(2, 10)}
    losses = p.compute_loss(obs, torch.randn(2, 7))
    assert losses["loss"].requires_grad
    p.train(False)
    out = p.predict(obs)
    assert out.action.shape == (2, 1, 7)


def test_dagger():
    from robocore.policy.algos.imitation import DAggerPolicy
    p = DAggerPolicy(**S)
    assert p.beta == 1.0
    p.step_iteration()
    assert p.beta < 1.0
    obs = {"state": torch.randn(2, 10)}
    losses = p.compute_loss(obs, torch.randn(2, 7))
    assert losses["loss"].requires_grad
