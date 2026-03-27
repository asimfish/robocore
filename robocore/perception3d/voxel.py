"""体素化编码器。"""
from __future__ import annotations

import torch
import torch.nn as nn


class VoxelEncoder(nn.Module):
    """3D 体素编码器。

    将点云体素化后用 3D CNN 编码。
    适用于需要空间结构信息的任务。
    """

    def __init__(
        self,
        voxel_size: int = 32,
        output_dim: int = 256,
        channels: tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        self.voxel_size = voxel_size

        layers: list[nn.Module] = []
        in_ch = 1  # 占据网格
        for ch in channels:
            layers.extend([
                nn.Conv3d(in_ch, ch, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool3d(2),
            ])
            in_ch = ch

        self.conv = nn.Sequential(*layers)

        # 计算 flatten 后的维度
        final_size = voxel_size // (2 ** len(channels))
        flat_dim = channels[-1] * (final_size ** 3) if final_size > 0 else channels[-1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, output_dim),
            nn.ReLU(),
        )

    def voxelize(self, points: torch.Tensor) -> torch.Tensor:
        """将点云转为体素网格。

        Args:
            points: (B, N, 3) 归一化到 [-1, 1] 的点云
        Returns:
            (B, 1, V, V, V) 占据网格
        """
        B = points.shape[0]
        V = self.voxel_size
        device = points.device

        grid = torch.zeros(B, 1, V, V, V, device=device)

        # 将 [-1, 1] 映射到 [0, V-1]
        coords = ((points[:, :, :3] + 1) / 2 * (V - 1)).long()
        coords = coords.clamp(0, V - 1)

        for b in range(B):
            x, y, z = coords[b, :, 0], coords[b, :, 1], coords[b, :, 2]
            grid[b, 0, x, y, z] = 1.0

        return grid

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """(B, N, 3) → (B, output_dim)"""
        grid = self.voxelize(points)
        feat = self.conv(grid)
        return self.fc(feat)
