"""深度图处理模块。"""
from __future__ import annotations

import torch
import torch.nn as nn


class DepthToPointCloud(nn.Module):
    """深度图转点云。

    给定深度图和相机内参，反投影为 3D 点云。
    """

    def __init__(
        self,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 240.0,
        depth_scale: float = 1000.0,
        max_depth: float = 3.0,
    ):
        super().__init__()
        self.register_buffer("intrinsics", torch.tensor([fx, fy, cx, cy]))
        self.depth_scale = depth_scale
        self.max_depth = max_depth

    def forward(
        self,
        depth: torch.Tensor,
        rgb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            depth: (B, H, W) or (B, 1, H, W) 深度图
            rgb: (B, 3, H, W) 可选 RGB 图像
        Returns:
            (B, H*W, 3) or (B, H*W, 6) 点云 (xyz or xyzrgb)
        """
        if depth.ndim == 4:
            depth = depth.squeeze(1)

        B, H, W = depth.shape
        fx, fy, cx, cy = self.intrinsics

        # 像素坐标网格
        v, u = torch.meshgrid(
            torch.arange(H, device=depth.device, dtype=torch.float32),
            torch.arange(W, device=depth.device, dtype=torch.float32),
            indexing="ij",
        )

        # 反投影
        z = depth / self.depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        points = torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)
        points = points.reshape(B, H * W, 3)

        # 过滤无效点
        valid = (z > 0) & (z < self.max_depth)
        valid = valid.reshape(B, H * W)
        points = points * valid.unsqueeze(-1).float()

        # 拼接 RGB
        if rgb is not None:
            rgb_flat = rgb.permute(0, 2, 3, 1).reshape(B, H * W, 3)
            rgb_flat = rgb_flat.float() / 255.0 if rgb_flat.max() > 1 else rgb_flat
            points = torch.cat([points, rgb_flat], dim=-1)

        return points
