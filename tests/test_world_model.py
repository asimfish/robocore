"""测试世界模型。"""
import torch
from robocore.world_model.latent_dynamics import LatentDynamicsModel
from robocore.world_model.video_predictor import VideoPredictor
from robocore.world_model.reward_predictor import RewardPredictor


def test_latent_dynamics_deterministic():
    model = LatentDynamicsModel(obs_dim=16, action_dim=7, latent_dim=32, hidden_dim=64)
    obs_seq = torch.randn(2, 6, 16)
    act_seq = torch.randn(2, 5, 7)
    losses = model.compute_loss(obs_seq, act_seq)
    assert "loss" in losses and losses["loss"].requires_grad


def test_latent_dynamics_stochastic():
    model = LatentDynamicsModel(obs_dim=16, action_dim=7, latent_dim=32, hidden_dim=64, stochastic=True, stoch_dim=8)
    obs_seq = torch.randn(2, 6, 16)
    act_seq = torch.randn(2, 5, 7)
    losses = model.compute_loss(obs_seq, act_seq)
    assert "kl_loss" in losses


def test_latent_dynamics_rollout():
    model = LatentDynamicsModel(obs_dim=16, action_dim=7, latent_dim=32, hidden_dim=64)
    out = model.rollout(torch.randn(2, 16), torch.randn(2, 10, 7))
    assert out["obs_pred"].shape == (2, 10, 16)
    assert out["reward_pred"].shape == (2, 10)


def test_video_predictor():
    model = VideoPredictor(image_channels=3, image_size=32, action_dim=7, latent_dim=64, hidden_dim=128)
    imgs = torch.randn(2, 4, 3, 32, 32)
    acts = torch.randn(2, 3, 7)
    losses = model.compute_loss(imgs, acts)
    assert "loss" in losses and losses["loss"].requires_grad


def test_video_predictor_rollout():
    model = VideoPredictor(image_channels=3, image_size=32, action_dim=7, latent_dim=64, hidden_dim=128)
    pred = model.predict_video(torch.randn(2, 3, 32, 32), torch.randn(2, 5, 7))
    assert pred.shape == (2, 5, 3, 32, 32)


def test_reward_predictor_classify():
    model = RewardPredictor(input_dim=16, hidden_dim=32, mode="classify")
    pred = model(torch.randn(4, 16))
    assert pred.shape == (4,) and (pred >= 0).all() and (pred <= 1).all()
    losses = model.compute_loss(torch.randn(4, 16), torch.ones(4))
    assert losses["loss"].requires_grad


def test_reward_predictor_regress():
    model = RewardPredictor(input_dim=16, hidden_dim=32, mode="regress")
    losses = model.compute_loss(torch.randn(4, 16), torch.randn(4))
    assert losses["loss"].requires_grad
