"""观测编码器。"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseEncoder(nn.Module, ABC):
    """编码器基类。"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入。"""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """输出特征维度。"""


class StateEncoder(BaseEncoder):
    """状态向量编码器：简单 MLP。"""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256):
        super().__init__()
        self._output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class ImageEncoder(BaseEncoder):
    """图像编码器：支持 ResNet 系列。"""

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        output_dim: int = 512,
        freeze: bool = False,
    ):
        super().__init__()
        self._output_dim = output_dim

        import torchvision.models as models

        if backbone == "resnet18":
            resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(
                weights=models.ResNet34_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 512
        elif backbone == "resnet50":
            resnet = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # 去掉最后的 fc 层
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(feat_dim, output_dim) if feat_dim != output_dim else nn.Identity()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, H, W) or (batch, T, C, H, W)
        Returns:
            (batch, output_dim) or (batch, T, output_dim)
        """
        has_time = x.ndim == 5
        if has_time:
            b, t, c, h, w = x.shape
            x = x.reshape(b * t, c, h, w)

        feat = self.backbone(x).flatten(1)  # (B*T, feat_dim)
        feat = self.projection(feat)

        if has_time:
            feat = feat.reshape(b, t, -1)

        return feat

    @property
    def output_dim(self) -> int:
        return self._output_dim


class MultiViewEncoder(BaseEncoder):
    """多视角图像编码器：共享权重编码多个相机视角。"""

    def __init__(
        self,
        camera_names: list[str],
        backbone: str = "resnet18",
        pretrained: bool = True,
        per_view_dim: int = 512,
        freeze: bool = False,
    ):
        super().__init__()
        self.camera_names = camera_names
        self._per_view_dim = per_view_dim
        self._output_dim = per_view_dim * len(camera_names)

        # 共享编码器
        self.encoder = ImageEncoder(
            backbone=backbone,
            pretrained=pretrained,
            output_dim=per_view_dim,
            freeze=freeze,
        )

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images: {camera_name: (batch, C, H, W) or (batch, T, C, H, W)}
        Returns:
            (batch, num_views * per_view_dim) or (batch, T, num_views * per_view_dim)
        """
        features = []
        for name in self.camera_names:
            if name in images:
                features.append(self.encoder(images[name]))
        return torch.cat(features, dim=-1)

    @property
    def output_dim(self) -> int:
        return self._output_dim
