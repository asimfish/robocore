"""测试表征学习模块。"""
import torch
from robocore.representations.pretrained import PretrainedEncoder
from robocore.representations.contrastive import ContrastiveLearner


def test_pretrained_encoder():
    enc = PretrainedEncoder(name="fallback", output_dim=64, pretrained=False)
    img = torch.randn(2, 3, 64, 64)
    out = enc(img)
    assert out.shape == (2, 64)


def test_pretrained_frozen():
    enc = PretrainedEncoder(name="fallback", output_dim=64, freeze=True, pretrained=False)
    for p in enc.backbone.parameters():
        assert not p.requires_grad
    for p in enc.projector.parameters():
        assert p.requires_grad


def test_contrastive_simclr():
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 4, 2, 1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
    )
    learner = ContrastiveLearner(backbone, projection_dim=32, feature_dim=32, mode="simclr")
    v1 = torch.randn(4, 3, 32, 32)
    v2 = torch.randn(4, 3, 32, 32)
    losses = learner.compute_loss(v1, v2)
    assert losses["loss"].requires_grad


def test_contrastive_tcn():
    backbone = torch.nn.Sequential(
        torch.nn.Conv2d(3, 32, 4, 2, 1), torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
    )
    learner = ContrastiveLearner(backbone, projection_dim=32, feature_dim=32, mode="tcn")
    anchor = torch.randn(4, 3, 32, 32)
    pos = torch.randn(4, 3, 32, 32)
    neg = torch.randn(4, 3, 32, 32)
    losses = learner.compute_loss(anchor, pos, neg)
    assert losses["loss"].requires_grad
