"""1D 条件 U-Net：Diffusion Policy 的核心网络。"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    """正弦位置编码（用于 diffusion timestep）。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalResBlock1d(nn.Module):
    """条件残差块（1D 卷积）。"""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.Mish(),
        )
        self.cond_proj = nn.Linear(cond_dim, out_channels)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, horizon)
            cond: (batch, cond_dim)
        """
        h = self.blocks[0](x)  # conv
        h = self.blocks[1](h)  # norm
        h = self.blocks[2](h)  # mish

        # 注入条件
        cond_emb = self.cond_proj(cond).unsqueeze(-1)
        h = h + cond_emb

        h = self.blocks[3](h)  # conv
        h = self.blocks[4](h)  # norm
        h = self.blocks[5](h)  # mish

        return h + self.residual(x)


class ConditionalUNet1d(nn.Module):
    """条件 1D U-Net。

    输入：噪声动作序列 + 条件（观测编码 + timestep）
    输出：预测噪声（或速度场）
    """

    def __init__(
        self,
        action_dim: int = 7,
        cond_dim: int = 256,
        diffusion_step_embed_dim: int = 128,
        down_dims: tuple[int, ...] = (256, 512, 1024),
    ):
        super().__init__()

        # Timestep 编码
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(diffusion_step_embed_dim),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_step_embed_dim * 4, diffusion_step_embed_dim),
        )

        total_cond_dim = cond_dim + diffusion_step_embed_dim

        # 输入投影
        self.input_proj = nn.Conv1d(action_dim, down_dims[0], 1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = down_dims[0]
        for out_ch in down_dims[1:]:
            self.down_blocks.append(ConditionalResBlock1d(in_ch, in_ch, total_cond_dim))
            self.down_samples.append(nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        # 中间块
        self.mid_block = ConditionalResBlock1d(in_ch, in_ch, total_cond_dim)

        # 上采样路径
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for out_ch in reversed(down_dims[:-1]):
            self.up_samples.append(nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1))
            self.up_blocks.append(
                ConditionalResBlock1d(out_ch * 2, out_ch, total_cond_dim)
            )
            in_ch = out_ch

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv1d(down_dims[0], down_dims[0], 3, padding=1),
            nn.GroupNorm(8, down_dims[0]),
            nn.Mish(),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (batch, horizon, action_dim)
            timestep: (batch,) diffusion timestep
            cond: (batch, cond_dim) 条件编码

        Returns:
            (batch, horizon, action_dim) 预测噪声
        """
        # (batch, horizon, action_dim) -> (batch, action_dim, horizon)
        x = noisy_action.permute(0, 2, 1)

        # 时间步编码
        t_emb = self.time_embed(timestep.float())
        global_cond = torch.cat([cond, t_emb], dim=-1)

        # 输入投影
        x = self.input_proj(x)

        # 下采样
        skip_connections = [x]
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            x = down_block(x, global_cond)
            skip_connections.append(x)
            x = down_sample(x)

        # 中间
        x = self.mid_block(x, global_cond)

        # 上采样
        for up_sample, up_block in zip(self.up_samples, self.up_blocks):
            x = up_sample(x)
            skip = skip_connections.pop()
            # 处理尺寸不匹配
            if x.shape[-1] != skip.shape[-1]:
                x = x[..., : skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, global_cond)

        # 输出
        x = self.output_proj(x)

        # (batch, action_dim, horizon) -> (batch, horizon, action_dim)
        return x.permute(0, 2, 1)
