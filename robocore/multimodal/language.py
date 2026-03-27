"""语言条件模块：将自然语言指令编码为策略条件。"""
from __future__ import annotations

import torch
import torch.nn as nn


class LanguageConditioner(nn.Module):
    """语言条件编码器。

    将自然语言指令编码为条件向量，用于：
    - 语言条件策略 (language-conditioned policy)
    - VLA 模型的语言输入
    - 多任务学习的任务描述

    支持：
    - 简单 embedding + LSTM (无需大模型)
    - CLIP text encoder wrapper
    - SentenceTransformer wrapper
    """

    def __init__(
        self,
        mode: str = "lstm",  # "lstm", "embed"
        vocab_size: int = 10000,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 256,
        max_len: int = 77,
    ):
        super().__init__()
        self.mode = mode
        self.output_dim = output_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if mode == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, seq_len) 整数 token
        Returns:
            (B, output_dim) 语言条件向量
        """
        emb = self.embedding(token_ids)  # (B, L, embed_dim)

        if self.mode == "lstm":
            output, (h, _) = self.rnn(emb)
            # 取最后一层双向拼接
            feat = torch.cat([h[-2], h[-1]], dim=-1)
            return self.proj(feat)
        else:
            # 均值池化
            feat = emb.mean(dim=1)
            return self.proj(feat)

    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """便捷方法：直接从文本字符串编码。

        注意：这里用简单的字符级 tokenization。
        实际使用时应替换为 proper tokenizer。
        """
        if isinstance(text, str):
            text = [text]

        batch_ids = []
        for t in text:
            ids = [ord(c) % self.embedding.num_embeddings for c in t[:77]]
            ids = ids + [0] * (77 - len(ids))
            batch_ids.append(ids)

        token_ids = torch.tensor(batch_ids, device=next(self.parameters()).device)
        return self.forward(token_ids)
