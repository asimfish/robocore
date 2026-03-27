"""DiT (Diffusion Transformer) 网络：用于替代 U-Net 的 Transformer backbone。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Norm：用 timestep 条件调制 scale 和 shift。"""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale_shift = self.proj(cond)
        if scale_shift.ndim == 2:
            scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT Transformer Block：Self-Attention + Cross-Attention + FFN。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = AdaLayerNorm(hidden_dim, cond_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = AdaLayerNorm(hidden_dim, cond_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm3 = AdaLayerNorm(hidden_dim, cond_dim)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond_tokens: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention
        h = self.norm1(x, t_emb)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention with condition
        h = self.norm2(x, t_emb)
        h, _ = self.cross_attn(h, cond_tokens, cond_tokens)
        x = x + h

        # FFN
        x = x + self.ffn(self.norm3(x, t_emb))
        return x


class DiTActionModel(nn.Module):
    """DiT 动作生成模型。

    输入：噪声动作序列 + 时间步 + 条件
    输出：预测噪声/速度（与输入同形状）

    可用于：
    - Diffusion Policy (DiT backbone)
    - Flow Policy (DiT backbone)
    - Pi0 action expert
    """

    def __init__(
        self,
        action_dim: int = 7,
        cond_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        max_horizon: int = 64,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 动作投影
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.action_out = nn.Linear(hidden_dim, action_dim)

        # 位置编码
        self.pos_embed = nn.Embedding(max_horizon, hidden_dim)

        # 时间步编码
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # 条件投影
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, hidden_dim, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # 最终 norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init 输出层（DiT 论文推荐）
        nn.init.zeros_(self.action_out.weight)
        nn.init.zeros_(self.action_out.bias)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (B, T, action_dim)
            timestep: (B,) 整数或浮点时间步
            cond: (B, cond_dim) 全局条件
        Returns:
            (B, T, action_dim) 预测噪声/速度
        """
        B, T, _ = noisy_action.shape

        # 时间步编码
        t_emb = self.time_embed(timestep.float())  # (B, hidden)

        # 条件 token
        cond_tokens = self.cond_proj(cond).unsqueeze(1)  # (B, 1, hidden)

        # 动作 token + 位置编码
        x = self.action_proj(noisy_action)  # (B, T, hidden)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions).unsqueeze(0)

        # DiT blocks
        for block in self.blocks:
            x = block(x, cond_tokens, t_emb)

        # 输出
        x = self.final_norm(x)
        return self.action_out(x)
