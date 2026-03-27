"""OpenVLA 策略实现。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.registry import PolicyRegistry


@PolicyRegistry.register("openvla")
class OpenVLAPolicy(BasePolicy):
    """OpenVLA: Vision-Language-Action 模型。

    基于预训练 VLM（如 Prismatic VLM）微调的机器人策略。
    核心思路：
    1. 将图像 + 语言指令输入 VLM
    2. VLM 输出离散化的动作 token
    3. 解码 token 为连续动作

    支持 LoRA 高效微调。
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 1,
        action_horizon: int = 1,
        device: str | torch.device = "cuda",
        # VLA 参数
        model_name: str = "openvla/openvla-7b",
        use_lora: bool = True,
        lora_rank: int = 32,
        action_bins: int = 256,
        action_token_begin_idx: int = 32000,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.action_bins = action_bins
        self.action_token_begin_idx = action_token_begin_idx

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_model(self) -> None:
        """构建 VLA 模型。

        实际使用时会加载预训练的 VLM。
        这里提供轻量级替代实现用于测试和开发。
        """
        # 轻量级替代：图像编码 + 语言编码 + 动作解码
        self.image_proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )

        self.language_embed = nn.Embedding(32000, 256)
        self.language_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256, nhead=4, dim_feedforward=512, batch_first=True
            ),
            num_layers=2,
        )

        # 动作解码器：输出离散 bin 的 logits
        self.action_decoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim * self.action_bins),
        )

        # 动作归一化参数
        self.register_buffer(
            "action_mean", torch.zeros(self.action_dim)
        )
        self.register_buffer(
            "action_std", torch.ones(self.action_dim)
        )

    def _discretize_action(self, action: torch.Tensor) -> torch.Tensor:
        """将连续动作离散化为 bin 索引。"""
        # 归一化到 [0, 1]
        normalized = (action - self.action_mean) / (self.action_std + 1e-8)
        normalized = torch.clamp(normalized, -1, 1)
        normalized = (normalized + 1) / 2  # [-1, 1] -> [0, 1]

        # 量化到 bins
        bins = (normalized * (self.action_bins - 1)).long()
        bins = torch.clamp(bins, 0, self.action_bins - 1)
        return bins

    def _continuous_action(self, bins: torch.Tensor) -> torch.Tensor:
        """将 bin 索引转回连续动作。"""
        normalized = bins.float() / (self.action_bins - 1)  # [0, 1]
        normalized = normalized * 2 - 1  # [-1, 1]
        return normalized * self.action_std + self.action_mean

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """推理：从图像+语言生成动作。"""
        # 简化版：使用状态代替图像特征
        if "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state[:, -1]
            # 投影到特征空间
            feat = torch.zeros(state.shape[0], 512, device=state.device)
            feat[:, :state.shape[-1]] = state
        else:
            batch_size = 1
            feat = torch.zeros(batch_size, 512, device=self._device)

        feat = self.image_proj(feat)  # (B, 256)

        # 解码动作
        logits = self.action_decoder(feat)  # (B, action_dim * bins)
        logits = logits.view(-1, self.action_dim, self.action_bins)

        # 取 argmax
        bins = logits.argmax(dim=-1)  # (B, action_dim)
        action = self._continuous_action(bins)

        return PolicyOutput(action=action.unsqueeze(1))  # (B, 1, action_dim)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """计算交叉熵损失。"""
        if "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state[:, -1]
            feat = torch.zeros(state.shape[0], 512, device=state.device)
            feat[:, :state.shape[-1]] = state
        else:
            feat = torch.zeros(action.shape[0], 512, device=action.device)

        feat = self.image_proj(feat)
        logits = self.action_decoder(feat)
        logits = logits.view(-1, self.action_dim, self.action_bins)

        # 目标：离散化的动作
        if action.ndim == 3:
            action = action[:, 0]  # 取第一步
        target_bins = self._discretize_action(action)

        # 交叉熵损失
        loss = nn.functional.cross_entropy(
            logits.permute(0, 2, 1),  # (B, bins, action_dim)
            target_bins,  # (B, action_dim)
        )

        return {"loss": loss}
