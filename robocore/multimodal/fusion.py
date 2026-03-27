"""多模态融合模块：将不同模态的特征融合为统一条件。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class MultiModalFusion(nn.Module):
    """多模态特征融合。

    支持多种融合策略：
    - concat: 简单拼接 + MLP
    - attention: Cross-attention 融合
    - film: FiLM (Feature-wise Linear Modulation)
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 256,
        fusion_type: str = "concat",  # "concat", "attention", "film"
        num_heads: int = 4,
    ):
        super().__init__()
        self.modality_dims = modality_dims
        self.output_dim = output_dim
        self.fusion_type = fusion_type

        total_dim = sum(modality_dims.values())

        if fusion_type == "concat":
            self.fuse = nn.Sequential(
                nn.Linear(total_dim, output_dim), nn.ReLU(),
                nn.Linear(output_dim, output_dim),
            )
        elif fusion_type == "attention":
            # 每个模态投影到相同维度
            self.projectors = nn.ModuleDict({
                name: nn.Linear(dim, output_dim)
                for name, dim in modality_dims.items()
            })
            self.cross_attn = nn.MultiheadAttention(
                output_dim, num_heads, batch_first=True
            )
            self.out_proj = nn.Linear(output_dim, output_dim)
        elif fusion_type == "film":
            # 主模态 + 条件模态
            modality_names = list(modality_dims.keys())
            main_dim = modality_dims[modality_names[0]]
            cond_dim = sum(modality_dims[n] for n in modality_names[1:])
            self.main_proj = nn.Linear(main_dim, output_dim)
            self.film_gen = nn.Linear(cond_dim, output_dim * 2)  # gamma + beta

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {modality_name: (B, dim)} 各模态特征
        Returns:
            (B, output_dim) 融合后的特征
        """
        if self.fusion_type == "concat":
            cat = torch.cat([features[k] for k in self.modality_dims], dim=-1)
            return self.fuse(cat)

        elif self.fusion_type == "attention":
            tokens = []
            for name in self.modality_dims:
                if name in features:
                    tokens.append(self.projectors[name](features[name]).unsqueeze(1))
            tokens = torch.cat(tokens, dim=1)  # (B, num_modalities, output_dim)
            attn_out, _ = self.cross_attn(tokens, tokens, tokens)
            pooled = attn_out.mean(dim=1)
            return self.out_proj(pooled)

        elif self.fusion_type == "film":
            names = list(self.modality_dims.keys())
            main_feat = self.main_proj(features[names[0]])
            cond_feats = torch.cat([features[n] for n in names[1:]], dim=-1)
            film_params = self.film_gen(cond_feats)
            gamma, beta = film_params.chunk(2, dim=-1)
            return main_feat * (1 + gamma) + beta

        raise ValueError(f"Unknown fusion type: {self.fusion_type}")
