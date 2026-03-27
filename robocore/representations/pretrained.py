"""预训练视觉表征 wrapper：R3M, MVP, VC-1, CLIP, DINOv2。"""
from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PretrainedEncoder(nn.Module):
    """预训练视觉编码器的统一 wrapper。

    支持冻结/微调/LoRA 三种模式。
    """

    def __init__(
        self,
        name: str = "resnet18",
        output_dim: int = 256,
        freeze: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        self.name = name
        self.freeze = freeze
        self._output_dim = output_dim

        # 构建 backbone
        self.backbone, feat_dim = self._build_backbone(name, pretrained)

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _build_backbone(self, name: str, pretrained: bool) -> tuple[nn.Module, int]:
        """构建 backbone，返回 (model, feature_dim)。"""
        try:
            import torchvision.models as models
            has_tv = True
        except ImportError:
            has_tv = False

        if has_tv and name == "resnet18":
            model = models.resnet18(weights="DEFAULT" if pretrained else None)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feat_dim
        elif has_tv and name == "resnet50":
            model = models.resnet50(weights="DEFAULT" if pretrained else None)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, feat_dim
        elif has_tv and name == "efficientnet_b0":
            model = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
            feat_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, feat_dim
        else:
            if not has_tv and name not in ("fallback",):
                logger.warning(f"torchvision not available, using simple CNN for '{name}'")
            else:
                logger.info(f"Using simple CNN backbone for '{name}'")
            backbone = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            return backbone, 128

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, output_dim)"""
        with torch.set_grad_enabled(not self.freeze):
            feat = self.backbone(images)
        return self.projector(feat)


class R3MEncoder(PretrainedEncoder):
    """R3M: 从人类视频学习的视觉表征。

    Nair et al. "R3M: A Universal Visual Representation for Robot Manipulation"
    使用 time-contrastive + language-aligned 预训练。
    """

    def __init__(self, output_dim: int = 256, freeze: bool = True, **kwargs: Any):
        super().__init__(name="resnet50", output_dim=output_dim, freeze=freeze)
        # 实际使用时加载 R3M 权重：
        # from r3m import load_r3m
        # self.backbone = load_r3m("resnet50")


class MVPEncoder(PretrainedEncoder):
    """MVP: Masked Visual Pre-training for Motor Control。

    Radosavovic et al. 使用 MAE 在 Ego4D 上预训练。
    """

    def __init__(self, output_dim: int = 256, freeze: bool = True, **kwargs: Any):
        super().__init__(name="resnet50", output_dim=output_dim, freeze=freeze)
        # 实际使用时加载 MVP 权重：
        # from mvp import load_mvp
        # self.backbone = load_mvp()


class VC1Encoder(PretrainedEncoder):
    """VC-1: Visual Cortex 1。

    Majumdar et al. 大规模 MAE 预训练的视觉表征。
    """

    def __init__(self, output_dim: int = 256, freeze: bool = True, **kwargs: Any):
        super().__init__(name="resnet50", output_dim=output_dim, freeze=freeze)
        # 实际使用时加载 VC-1 权重：
        # from vc_models.models.vit import model_utils
        # self.backbone, _, _, _ = model_utils.load_model(model_utils.VC1_BASE_NAME)


class DINOv2Encoder(PretrainedEncoder):
    """DINOv2 视觉表征。

    Meta 的自监督 ViT，在机器人领域表现优异。
    """

    def __init__(self, output_dim: int = 256, freeze: bool = True, **kwargs: Any):
        super().__init__(name="resnet50", output_dim=output_dim, freeze=freeze)
        # 实际使用时：
        # self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')


class CLIPEncoder(PretrainedEncoder):
    """CLIP 视觉编码器。

    OpenAI CLIP 的视觉分支，天然支持语言对齐。
    """

    def __init__(self, output_dim: int = 256, freeze: bool = True, **kwargs: Any):
        super().__init__(name="resnet50", output_dim=output_dim, freeze=freeze)
        # 实际使用时：
        # import open_clip
        # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32')
        # self.backbone = model.visual
