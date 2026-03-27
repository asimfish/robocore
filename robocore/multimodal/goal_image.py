"""目标图像条件模块。"""
from __future__ import annotations

import torch
import torch.nn as nn


class GoalImageConditioner(nn.Module):
    """目标图像条件编码器。

    将目标图像编码为条件向量，用于：
    - Goal-conditioned policy
    - 视觉目标达成任务
    - 图像子目标规划

    支持：
    - 绝对目标编码
    - 相对编码 (当前图像 vs 目标图像的差异)
    """

    def __init__(
        self,
        output_dim: int = 256,
        mode: str = "absolute",  # "absolute" or "relative"
    ):
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, output_dim),
        )

        if mode == "relative":
            self.diff_proj = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )

    def forward(
        self,
        goal_image: torch.Tensor,
        current_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            goal_image: (B, C, H, W)
            current_image: (B, C, H, W) 仅 relative 模式需要
        Returns:
            (B, output_dim)
        """
        goal_feat = self.encoder(goal_image)

        if self.mode == "relative" and current_image is not None:
            curr_feat = self.encoder(current_image)
            combined = torch.cat([curr_feat, goal_feat], dim=-1)
            return self.diff_proj(combined)

        return goal_feat
