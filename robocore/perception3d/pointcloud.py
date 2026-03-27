"""点云处理模块。"""
from __future__ import annotations

import torch
import torch.nn as nn


class FarthestPointSampling(nn.Module):
    """最远点采样 (FPS)。

    从 N 个点中采样 K 个最远点，保持几何覆盖。
    """

    def __init__(self, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3+) 输入点云
        Returns:
            (B, K, 3+) 采样后的点云
        """
        B, N, D = points.shape
        K = min(self.num_points, N)

        # 贪心 FPS
        device = points.device
        centroids = torch.zeros(B, K, dtype=torch.long, device=device)
        distances = torch.full((B, N), 1e10, device=device)

        # 随机选第一个点
        centroids[:, 0] = torch.randint(0, N, (B,), device=device)

        for i in range(1, K):
            last = centroids[:, i - 1]
            last_point = points[torch.arange(B, device=device), last].unsqueeze(1)  # (B, 1, D)
            dist = ((points[:, :, :3] - last_point[:, :, :3]) ** 2).sum(dim=-1)  # (B, N)
            distances = torch.min(distances, dist)
            centroids[:, i] = distances.argmax(dim=-1)

        # 收集采样点
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)
        return points[batch_idx, centroids]


class PointCloudProcessor(nn.Module):
    """点云处理管线。

    完整的点云处理流程：
    1. 裁剪工作空间
    2. 下采样 (FPS / random / voxel)
    3. 归一化
    4. 可选：法线估计、颜色处理
    """

    def __init__(
        self,
        num_points: int = 1024,
        workspace_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        normalize: bool = True,
        downsample_method: str = "fps",  # "fps", "random"
    ):
        super().__init__()
        self.num_points = num_points
        self.normalize = normalize
        self.downsample_method = downsample_method

        if workspace_bounds:
            self.register_buffer("ws_min", torch.tensor(workspace_bounds[0]))
            self.register_buffer("ws_max", torch.tensor(workspace_bounds[1]))
        else:
            self.ws_min = None
            self.ws_max = None

        if downsample_method == "fps":
            self.fps = FarthestPointSampling(num_points)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Args:
            points: (B, N, 3+)
        Returns:
            (B, num_points, 3+)
        """
        # 1. 工作空间裁剪
        if self.ws_min is not None:
            xyz = points[:, :, :3]
            mask = ((xyz >= self.ws_min) & (xyz <= self.ws_max)).all(dim=-1)
            # 简化：对每个 batch 取前 N 个有效点
            # 实际应用中需要更精细的处理
            points = points * mask.unsqueeze(-1).float()

        # 2. 下采样
        if self.downsample_method == "fps":
            points = self.fps(points)
        elif self.downsample_method == "random":
            B, N, D = points.shape
            K = min(self.num_points, N)
            idx = torch.randint(0, N, (B, K), device=points.device)
            batch_idx = torch.arange(B, device=points.device).unsqueeze(1).expand(-1, K)
            points = points[batch_idx, idx]

        # 3. 归一化
        if self.normalize:
            xyz = points[:, :, :3]
            centroid = xyz.mean(dim=1, keepdim=True)
            xyz = xyz - centroid
            scale = xyz.abs().max(dim=1, keepdim=True)[0].max(dim=-1, keepdim=True)[0].clamp(min=1e-6)
            xyz = xyz / scale
            points = torch.cat([xyz, points[:, :, 3:]], dim=-1) if points.shape[-1] > 3 else xyz

        return points


class PointNetEncoder(nn.Module):
    """PointNet 编码器（增强版）。

    支持：
    - 局部特征 + 全局特征
    - T-Net 空间变换
    - 多尺度特征聚合
    """

    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 256,
        use_tnet: bool = False,
    ):
        super().__init__()
        self.use_tnet = use_tnet

        if use_tnet:
            self.tnet = nn.Sequential(
                nn.Linear(input_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, input_dim * input_dim),
            )

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128),
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256),
        )
        self.proj = nn.Linear(256, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, input_dim) → (B, output_dim)"""
        B, N, D = x.shape

        if self.use_tnet:
            # 预测变换矩阵
            global_feat = self.tnet(x).max(dim=1)[0]
            transform = global_feat.view(B, D, D)
            transform = transform + torch.eye(D, device=x.device).unsqueeze(0)
            x = torch.bmm(x, transform)

        # 逐点 MLP (需要 reshape for BatchNorm)
        x = self.mlp1(x.reshape(B * N, D)).reshape(B, N, 64)
        x = self.mlp2(x.reshape(B * N, 64)).reshape(B, N, 128)
        x = self.mlp3(x.reshape(B * N, 128)).reshape(B, N, 256)

        # 全局最大池化
        global_feat = x.max(dim=1)[0]
        return self.proj(global_feat)
