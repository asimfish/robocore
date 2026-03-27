"""测试多模态融合模块。"""
import torch
from robocore.multimodal.language import LanguageConditioner
from robocore.multimodal.goal_image import GoalImageConditioner
from robocore.multimodal.tactile import TactileEncoder
from robocore.multimodal.fusion import MultiModalFusion


def test_language_lstm():
    enc = LanguageConditioner(mode="lstm", vocab_size=256, embed_dim=32, hidden_dim=64, output_dim=64)
    ids = torch.randint(0, 256, (2, 20))
    out = enc(ids)
    assert out.shape == (2, 64)


def test_language_encode_text():
    enc = LanguageConditioner(mode="lstm", vocab_size=256, embed_dim=32, hidden_dim=64, output_dim=64)
    out = enc.encode_text(["pick up the cup", "place on table"])
    assert out.shape == (2, 64)


def test_goal_image_absolute():
    enc = GoalImageConditioner(output_dim=64, mode="absolute")
    goal = torch.randn(2, 3, 64, 64)
    out = enc(goal)
    assert out.shape == (2, 64)


def test_goal_image_relative():
    enc = GoalImageConditioner(output_dim=64, mode="relative")
    goal = torch.randn(2, 3, 64, 64)
    curr = torch.randn(2, 3, 64, 64)
    out = enc(goal, curr)
    assert out.shape == (2, 64)


def test_tactile_image():
    enc = TactileEncoder(sensor_type="image", output_dim=64)
    x = torch.randn(2, 3, 32, 32)
    assert enc(x).shape == (2, 64)


def test_tactile_force():
    enc = TactileEncoder(sensor_type="force", output_dim=64, force_dim=6)
    x = torch.randn(2, 6)
    assert enc(x).shape == (2, 64)


def test_tactile_array():
    enc = TactileEncoder(sensor_type="array", output_dim=64)
    x = torch.randn(2, 16, 16)
    assert enc(x).shape == (2, 64)


def test_fusion_concat():
    fuse = MultiModalFusion({"vision": 64, "language": 32, "tactile": 16}, output_dim=64, fusion_type="concat")
    feats = {"vision": torch.randn(2, 64), "language": torch.randn(2, 32), "tactile": torch.randn(2, 16)}
    assert fuse(feats).shape == (2, 64)


def test_fusion_attention():
    fuse = MultiModalFusion({"vision": 64, "language": 32}, output_dim=64, fusion_type="attention")
    feats = {"vision": torch.randn(2, 64), "language": torch.randn(2, 32)}
    assert fuse(feats).shape == (2, 64)


def test_fusion_film():
    fuse = MultiModalFusion({"vision": 64, "language": 32}, output_dim=64, fusion_type="film")
    feats = {"vision": torch.randn(2, 64), "language": torch.randn(2, 32)}
    assert fuse(feats).shape == (2, 64)
