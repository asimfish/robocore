"""奖励预测器：从观测预测任务奖励/成功信号。"""
from __future__ import annotations

import torch
import torch.nn as nn


class RewardPredictor(nn.Module):
    """奖励/成功预测器。

    用于：
    - 无奖励环境中的奖励学习
    - 目标条件 RL 的成功检测
    - 世界模型中的奖励头

    支持：
    - 状态输入
    - 图像输入
    - 分类 (success/fail) 或回归 (continuous reward)
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        mode: str = "classify",  # "classify" or "regress"
        image_input: bool = False,
        image_channels: int = 3,
    ):
        super().__init__()
        self.mode = mode

        if image_input:
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, 32, 4, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, input_dim),
                nn.ReLU(),
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim if not image_input else input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        logit = self.head(feat).squeeze(-1)
        if self.mode == "classify":
            return torch.sigmoid(logit)
        return logit

    def compute_loss(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pred = self.forward(x)
        if self.mode == "classify":
            loss = nn.functional.binary_cross_entropy(pred, target.float())
        else:
            loss = nn.functional.mse_loss(pred, target.float())
        return {"loss": loss}
