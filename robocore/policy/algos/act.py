"""ACT (Action Chunking with Transformers) 实现。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.encoders import ImageEncoder, StateEncoder
from robocore.policy.registry import PolicyRegistry


@PolicyRegistry.register("act")
class ACTPolicy(BasePolicy):
    """ACT: Action Chunking with Transformers。

    核心思路：
    1. CVAE 架构：编码器编码 (obs, action) → latent z；解码器从 (obs, z) 生成动作
    2. 训练时用编码器采样 z；推理时从先验采样 z
    3. Transformer 解码器自回归生成 action chunk
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 16,
        action_horizon: int = 16,
        device: str | torch.device = "cuda",
        # 网络参数
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        latent_dim: int = 32,
        dropout: float = 0.1,
        kl_weight: float = 10.0,
        # 图像编码器
        image_encoder: str = "resnet18",
        image_keys: list[str] | None = None,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.kl_weight = kl_weight
        self.image_encoder_name = image_encoder
        self.image_keys = image_keys or []

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_model(self) -> None:
        """构建 CVAE 网络。"""
        # 状态编码器
        if self.obs_dim is not None and self.obs_dim > 0:
            self.state_encoder = StateEncoder(
                input_dim=self.obs_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
            )
            obs_feat_dim = self.hidden_dim
        else:
            self.state_encoder = None
            obs_feat_dim = self.hidden_dim

        # CVAE 编码器（训练时使用）
        self.action_encoder = nn.Linear(self.action_dim, self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.cvae_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.latent_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.latent_logvar = nn.Linear(self.hidden_dim, self.latent_dim)

        # 解码器
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.obs_proj = nn.Linear(obs_feat_dim, self.hidden_dim)

        # 位置编码
        self.pos_embed = nn.Embedding(self.pred_horizon, self.hidden_dim)

        # Transformer 解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.num_layers
        )

        # 动作输出头
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """编码观测。"""
        if self.state_encoder is not None and "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state[:, -1]  # 取最后一帧
            return self.state_encoder(state)
        return torch.zeros(1, self.hidden_dim, device=self._device)

    def _encode_posterior(
        self, obs_feat: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CVAE 后验编码。"""
        action_feat = self.action_encoder(actions)  # (B, T, hidden)
        obs_token = obs_feat.unsqueeze(1)  # (B, 1, hidden)
        encoder_input = torch.cat([obs_token, action_feat], dim=1)

        encoded = self.cvae_encoder(encoder_input)
        cls_token = encoded[:, 0]  # 取第一个 token

        mu = self.latent_mu(cls_token)
        logvar = self.latent_logvar(cls_token)

        # 重参数化
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, logvar

    def _decode(self, obs_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """从 (obs, z) 解码动作序列。"""
        batch_size = obs_feat.shape[0]

        # 条件 = obs + latent
        z_feat = self.latent_proj(z)  # (B, hidden)
        obs_proj = self.obs_proj(obs_feat)  # (B, hidden)
        memory = torch.stack([obs_proj, z_feat], dim=1)  # (B, 2, hidden)

        # 查询 = 位置编码
        positions = torch.arange(self.pred_horizon, device=obs_feat.device)
        query = self.pos_embed(positions).unsqueeze(0).expand(batch_size, -1, -1)

        # Transformer 解码
        decoded = self.transformer_decoder(query, memory)
        actions = self.action_head(decoded)

        return actions

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        """推理：从先验采样 z，解码动作。"""
        obs_feat = self._encode_obs(obs)
        batch_size = obs_feat.shape[0]

        # 从先验采样
        z = torch.randn(batch_size, self.latent_dim, device=obs_feat.device)

        actions = self._decode(obs_feat, z)
        return PolicyOutput(action=actions)

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """计算 CVAE 损失 = 重建损失 + KL 散度。"""
        obs_feat = self._encode_obs(obs)

        # 后验编码
        z, mu, logvar = self._encode_posterior(obs_feat, action)

        # 解码
        pred_action = self._decode(obs_feat, z)

        # 重建损失
        recon_loss = nn.functional.l1_loss(pred_action, action)

        # KL 散度
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.kl_weight * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
