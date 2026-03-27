"""触觉传感器编码模块。"""
from __future__ import annotations

import torch
import torch.nn as nn


class TactileEncoder(nn.Module):
    """触觉传感器编码器。

    支持多种触觉传感器格式：
    - GelSight / DIGIT: 触觉图像 (H, W, 3)
    - 力/力矩传感器: 6D 向量
    - 触觉阵列: (rows, cols) 压力矩阵

    将触觉信号编码为固定维度的特征向量。
    """

    def __init__(
        self,
        sensor_type: str = "image",  # "image", "force", "array"
        output_dim: int = 128,
        # image 参数
        image_channels: int = 3,
        # force 参数
        force_dim: int = 6,
        # array 参数
        array_shape: tuple[int, int] = (16, 16),
    ):
        super().__init__()
        self.sensor_type = sensor_type
        self.output_dim = output_dim

        if sensor_type == "image":
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, output_dim),
            )
        elif sensor_type == "force":
            self.encoder = nn.Sequential(
                nn.Linear(force_dim, 64), nn.ReLU(),
                nn.Linear(64, output_dim),
            )
        elif sensor_type == "array":
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, output_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sensor_type == "array" and x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        return self.encoder(x)
