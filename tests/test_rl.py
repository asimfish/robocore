"""测试强化学习模块。"""
import numpy as np
import torch
from robocore.rl.replay_buffer import ReplayBuffer
from robocore.rl.sac import SACAgent


def test_replay_buffer():
    buf = ReplayBuffer(capacity=100)
    for i in range(10):
        buf.add(
            obs={"state": np.random.randn(10).astype(np.float32)},
            action=np.random.randn(7).astype(np.float32),
            reward=float(i),
            next_obs={"state": np.random.randn(10).astype(np.float32)},
            done=False,
        )
    assert len(buf) == 10
    batch = buf.sample(4)
    assert batch["action"].shape == (4, 7)
    assert batch["reward"].shape == (4,)


def test_sac_update():
    agent = SACAgent(obs_dim=10, action_dim=7, hidden_dim=32)
    batch = {
        "obs": {"state": torch.randn(8, 10)},
        "action": torch.randn(8, 7),
        "reward": torch.randn(8),
        "next_obs": {"state": torch.randn(8, 10)},
        "done": torch.zeros(8),
    }
    info = agent.update(batch)
    assert "critic_loss" in info
    assert "actor_loss" in info
    assert "alpha" in info


def test_sac_get_action():
    agent = SACAgent(obs_dim=10, action_dim=7, hidden_dim=32)
    obs = torch.randn(1, 10)
    a = agent.get_action(obs)
    assert a.shape == (1, 7)
    assert (a.abs() <= 1).all()
