"""Pi0.5 策略实现：双系统 VLA (快思考 + 慢思考)。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.networks.dit import DiTActionModel
from robocore.policy.registry import PolicyRegistry


@PolicyRegistry.register("pi05")
class Pi05Policy(BasePolicy):
    """Pi0.5: Physical Intelligence 的双系统 VLA。

    相比 Pi0 的核心改进：
    1. 双系统架构：
       - System 1 (快思考): 低延迟反应式控制，直接输出动作
       - System 2 (慢思考): 高层推理规划，输出子目标/语言计划
    2. 语言推理链 (Chain-of-Thought for Robotics):
       - VLM 先生成文本推理（分析场景、规划步骤）
       - 再基于推理结果生成动作
    3. Web-scale 预训练 + 机器人数据微调
    4. 支持更长的 action horizon 和多任务泛化

    架构：
    - VLM backbone: 处理图像+语言，生成推理 token
    - Reasoning head: 生成文本推理链
    - Action expert: DiT-based flow matching 动作生成
    - Router: 根据任务复杂度选择 System 1/2
    """

    def __init__(
        self,
        obs_dim: int | None = None,
        action_dim: int = 7,
        obs_horizon: int = 1,
        pred_horizon: int = 32,
        action_horizon: int = 16,
        device: str | torch.device = "cuda",
        # 网络参数
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        # Flow matching
        num_inference_steps: int = 10,
        # 双系统
        use_reasoning: bool = True,
        reasoning_dim: int = 256,
        max_reasoning_tokens: int = 64,
        # 路由
        system1_threshold: float = 0.5,
        **kwargs: Any,
    ):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_inference_steps = num_inference_steps
        self.use_reasoning = use_reasoning
        self.reasoning_dim = reasoning_dim
        self.max_reasoning_tokens = max_reasoning_tokens
        self.system1_threshold = system1_threshold

        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            device=device,
        )

    def _build_model(self) -> None:
        obs_input_dim = self.obs_dim if self.obs_dim else 256

        # === 共享观测编码器 ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # === System 1: 快速反应式控制 ===
        # 轻量 MLP 直接映射 obs -> action
        self.system1_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.pred_horizon * self.action_dim),
        )

        # === System 2: 推理式控制 ===
        if self.use_reasoning:
            # 推理模块：生成 reasoning embedding
            self.reasoning_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.hidden_dim,
                    nhead=self.num_heads,
                    dim_feedforward=self.hidden_dim * 4,
                    batch_first=True,
                ),
                num_layers=2,
            )
            self.reasoning_tokens = nn.Parameter(
                torch.randn(1, self.max_reasoning_tokens, self.hidden_dim) * 0.02
            )

        # Action expert (DiT-based flow matching)
        cond_dim = self.hidden_dim * 2 if self.use_reasoning else self.hidden_dim
        self.action_expert = DiTActionModel(
            action_dim=self.action_dim,
            cond_dim=cond_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            max_horizon=self.pred_horizon,
        )

        # === 路由器：决定用 System 1 还是 System 2 ===
        self.router = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _encode_obs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        if "state" in obs:
            state = obs["state"]
            if state.ndim == 3:
                state = state[:, -1]
            return self.obs_encoder(state)
        batch_size = 1
        return torch.zeros(batch_size, self.hidden_dim, device=self._device)

    def _get_reasoning_embedding(self, obs_feat: torch.Tensor) -> torch.Tensor:
        """System 2: 生成推理 embedding。"""
        B = obs_feat.shape[0]
        # 拼接 obs token + learnable reasoning tokens
        obs_token = obs_feat.unsqueeze(1)  # (B, 1, hidden)
        reason_tokens = self.reasoning_tokens.expand(B, -1, -1)
        tokens = torch.cat([obs_token, reason_tokens], dim=1)

        # Transformer 推理
        output = self.reasoning_encoder(tokens)

        # 取推理 token 的均值作为 reasoning embedding
        reasoning_emb = output[:, 1:].mean(dim=1)  # (B, hidden)
        return reasoning_emb

    def _flow_matching_sample(
        self, cond: torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Flow matching ODE 采样。"""
        x = torch.randn(batch_size, self.pred_horizon, self.action_dim, device=device)
        dt = 1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t = torch.full((batch_size,), step / self.num_inference_steps, device=device)
            velocity = self.action_expert(x, t, cond)
            x = x + velocity * dt

        return x

    def predict(self, obs: dict[str, torch.Tensor]) -> PolicyOutput:
        obs_feat = self._encode_obs(obs)
        B = obs_feat.shape[0]
        device = obs_feat.device

        # 路由决策
        route_score = self.router(obs_feat)  # (B, 1)

        # System 1: 快速路径
        s1_action = self.system1_head(obs_feat)
        s1_action = s1_action.view(B, self.pred_horizon, self.action_dim)

        # System 2: 推理路径
        if self.use_reasoning:
            reasoning_emb = self._get_reasoning_embedding(obs_feat)
            cond = torch.cat([obs_feat, reasoning_emb], dim=-1)
        else:
            cond = obs_feat

        s2_action = self._flow_matching_sample(cond, B, device)

        # 混合：根据路由分数加权
        weight = route_score.unsqueeze(-1)  # (B, 1, 1)
        action = weight * s2_action + (1 - weight) * s1_action

        return PolicyOutput(
            action=action,
            info={
                "route_score": route_score,
                "system1_action": s1_action,
                "system2_action": s2_action,
            },
        )

    def compute_loss(
        self, obs: dict[str, torch.Tensor], action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        obs_feat = self._encode_obs(obs)
        B = action.shape[0]
        device = action.device

        # === System 1 损失 ===
        s1_pred = self.system1_head(obs_feat)
        s1_pred = s1_pred.view(B, self.pred_horizon, self.action_dim)
        s1_loss = nn.functional.mse_loss(s1_pred, action)

        # === System 2 损失 (flow matching) ===
        if self.use_reasoning:
            reasoning_emb = self._get_reasoning_embedding(obs_feat)
            cond = torch.cat([obs_feat, reasoning_emb], dim=-1)
        else:
            cond = obs_feat

        t = torch.rand(B, device=device)
        noise = torch.randn_like(action)
        t_expand = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t_expand) * noise + t_expand * action
        target_velocity = action - noise

        pred_velocity = self.action_expert(x_t, t, cond)
        s2_loss = nn.functional.mse_loss(pred_velocity, target_velocity)

        # === 路由损失 ===
        # 鼓励简单任务走 System 1，复杂任务走 System 2
        route_score = self.router(obs_feat.detach())
        # 用 System 1 的误差作为复杂度信号
        with torch.no_grad():
            complexity = s1_loss.detach().clamp(0, 1)
        route_loss = nn.functional.binary_cross_entropy(
            route_score.squeeze(-1),
            (complexity > 0.1).float().expand(B),
        )

        loss = s1_loss + s2_loss + 0.1 * route_loss

        return {
            "loss": loss,
            "system1_loss": s1_loss,
            "system2_loss": s2_loss,
            "route_loss": route_loss,
        }
