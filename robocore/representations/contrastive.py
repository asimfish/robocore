"""对比学习：用于机器人视觉表征的自监督学习。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLearner(nn.Module):
    """对比学习框架。

    支持多种对比学习目标：
    - Time Contrastive (TCN): 时间上相近的帧应该相似
    - SimCLR-style: 同一帧的不同增强应该相似
    - Action-conditioned: 相同动作导致的状态变化应该相似

    用于：
    - 预训练视觉编码器
    - 学习任务无关的表征
    - 辅助训练损失
    """

    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        feature_dim: int = 256,
        temperature: float = 0.07,
        mode: str = "simclr",  # "simclr", "tcn", "byol"
    ):
        super().__init__()
        self.encoder = encoder
        self.temperature = temperature
        self.mode = mode

        # 投影头 (MLP)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, projection_dim),
        )

        if mode == "byol":
            # BYOL 需要 predictor
            self.predictor = nn.Sequential(
                nn.Linear(projection_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, projection_dim),
            )
            # EMA target encoder
            self.target_encoder = None  # 延迟初始化
            self.target_projector = None

    def _get_features(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.projector(feat)

    def info_nce_loss(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> torch.Tensor:
        """InfoNCE 对比损失。"""
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        B = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2B, 2B)

        # 正样本对的 mask
        labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)

        # 去掉对角线
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, float("-inf"))

        return F.cross_entropy(sim, labels)

    def time_contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """时间对比损失 (TCN)。

        anchor 和 positive 是时间上相近的帧，negative 是远的。
        """
        z_a = F.normalize(self._get_features(anchor), dim=-1)
        z_p = F.normalize(self._get_features(positive), dim=-1)
        z_n = F.normalize(self._get_features(negative), dim=-1)

        pos_sim = (z_a * z_p).sum(dim=-1) / self.temperature
        neg_sim = (z_a * z_n).sum(dim=-1) / self.temperature

        logits = torch.stack([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(z_a.shape[0], dtype=torch.long, device=z_a.device)

        return F.cross_entropy(logits, labels)

    def compute_loss(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        negative: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """计算对比学习损失。"""
        if self.mode == "tcn" and negative is not None:
            loss = self.time_contrastive_loss(view1, view2, negative)
        else:
            z1 = self._get_features(view1)
            z2 = self._get_features(view2)
            loss = self.info_nce_loss(z1, z2)

        return {"loss": loss}
