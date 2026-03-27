"""训练器基类。"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from robocore.config.base import ExperimentConfig
from robocore.env.evaluator import Evaluator
from robocore.policy.base import BasePolicy

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """训练器基类。

    提供标准训练循环：
    1. 数据加载
    2. 前向 + 反向
    3. 梯度裁剪 + 优化器更新
    4. 日志记录
    5. 定期评估
    6. Checkpoint 保存/恢复
    """

    def __init__(
        self,
        config: ExperimentConfig,
        policy: BasePolicy,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        evaluator: Evaluator | None = None,
    ):
        self.config = config
        self.policy = policy.to(config.train.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = evaluator

        # 优化器
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # EMA
        self.ema_policy: BasePolicy | None = None
        if config.train.use_ema:
            self.ema_policy = deepcopy(policy).to(config.train.device)
            self.ema_policy.requires_grad_(False)

        # AMP
        self.scaler = torch.amp.GradScaler("cuda") if config.train.use_amp else None
        self.amp_dtype = (
            torch.bfloat16 if config.train.amp_dtype == "bf16" else torch.float16
        )

        # 状态
        self.global_step = 0
        self.epoch = 0
        self.best_metric = -float("inf")

        # 输出目录
        self.output_dir = Path(config.train.output_dir) / config.train.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self.loggers: list[Any] = []
        self._setup_loggers()

    def train(self) -> dict[str, Any]:
        """完整训练循环。"""
        logger.info(f"Starting training: {self.config.train.num_epochs} epochs")
        logger.info(f"Policy params: {self.policy.num_params:,}")
        logger.info(f"Output dir: {self.output_dir}")

        # 保存配置
        self.config.to_yaml(self.output_dir / "config.yaml")

        train_metrics: dict[str, Any] = {}

        for epoch in range(self.epoch, self.config.train.num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # 训练一个 epoch
            train_metrics = self._train_epoch()

            epoch_time = time.time() - epoch_start
            train_metrics["epoch_time"] = epoch_time

            # 日志
            if (epoch + 1) % self.config.train.log_interval == 0:
                self._log_metrics(train_metrics, prefix="train")

            # 评估
            if (
                self.evaluator is not None
                and (epoch + 1) % self.config.train.eval_interval == 0
            ):
                eval_policy = self.ema_policy if self.ema_policy else self.policy
                eval_result = self.evaluator.run(eval_policy)
                eval_metrics = {
                    "success_rate": eval_result.success_rate,
                    "mean_reward": eval_result.mean_reward,
                }
                self._log_metrics(eval_metrics, prefix="eval")

                # 保存最优
                if eval_result.success_rate > self.best_metric:
                    self.best_metric = eval_result.success_rate
                    self._save_checkpoint("best")

            # 定期保存
            if (epoch + 1) % self.config.train.save_interval == 0:
                self._save_checkpoint(f"epoch_{epoch+1}")

        # 保存最终模型
        self._save_checkpoint("final")
        logger.info(f"Training complete. Best metric: {self.best_metric:.4f}")
        return train_metrics

    def _train_epoch(self) -> dict[str, float]:
        """训练一个 epoch。"""
        self.policy.train()
        total_loss = 0.0
        num_steps = 0

        for batch_idx, batch in enumerate(self.train_loader):
            if num_steps >= self.config.train.steps_per_epoch:
                break

            # 移到设备
            batch = self._to_device(batch)

            # 前向 + 损失
            if self.config.train.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    losses = self.policy.compute_loss(batch["obs"], batch["action"])
            else:
                losses = self.policy.compute_loss(batch["obs"], batch["action"])

            loss = losses["loss"] / self.config.train.gradient_accumulation_steps

            # 反向
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 梯度累积
            if (batch_idx + 1) % self.config.train.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.train.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.train.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.scheduler.step()

                # EMA 更新
                if self.ema_policy is not None:
                    self._update_ema()

                self.global_step += 1

            total_loss += losses["loss"].item()
            num_steps += 1

        return {
            "loss": total_loss / max(num_steps, 1),
            "lr": self.optimizer.param_groups[0]["lr"],
            "epoch": self.epoch,
            "global_step": self.global_step,
        }

    def _update_ema(self) -> None:
        """更新 EMA 模型。"""
        decay = self.config.train.ema_decay
        for ema_p, p in zip(self.ema_policy.parameters(), self.policy.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """构建优化器。"""
        cfg = self.config.train
        if cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.policy.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            return torch.optim.Adam(
                self.policy.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """构建学习率调度器。"""
        cfg = self.config.train
        total_steps = cfg.num_epochs * cfg.steps_per_epoch
        if cfg.lr_scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps
            )
        elif cfg.lr_scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
            )
        else:
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0
            )

    def _save_checkpoint(self, tag: str) -> None:
        """保存 checkpoint。"""
        ckpt_dir = self.output_dir / "checkpoints" / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 保存策略
        self.policy.save(ckpt_dir)

        # 保存训练状态
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
                "best_metric": self.best_metric,
            },
            ckpt_dir / "trainer_state.pt",
        )

        # 保存 EMA
        if self.ema_policy is not None:
            self.ema_policy.save(ckpt_dir / "ema")

        logger.info(f"Checkpoint saved: {ckpt_dir}")

    def resume(self, ckpt_dir: str | Path) -> None:
        """从 checkpoint 恢复。"""
        ckpt_dir = Path(ckpt_dir)
        self.policy.load(ckpt_dir)

        state = torch.load(ckpt_dir / "trainer_state.pt", weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.epoch = state["epoch"] + 1
        self.global_step = state["global_step"]
        self.best_metric = state["best_metric"]

        if self.ema_policy is not None and (ckpt_dir / "ema").exists():
            self.ema_policy.load(ckpt_dir / "ema")

        logger.info(f"Resumed from {ckpt_dir}, epoch={self.epoch}")

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """将 batch 移到设备。"""
        device = self.config.train.device
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, dict):
                result[key] = self._to_device(value)
            else:
                result[key] = value
        return result

    def _setup_loggers(self) -> None:
        """设置日志记录器。"""
        if self.config.train.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.train.wandb_project,
                    name=self.config.train.exp_name,
                    config=self.config.__dict__,
                )
            except ImportError:
                logger.warning("wandb not installed, skipping")

    def _log_metrics(self, metrics: dict[str, Any], prefix: str = "") -> None:
        """记录指标。"""
        msg = f"[Epoch {self.epoch}]"
        for k, v in metrics.items():
            if isinstance(v, float):
                msg += f" {prefix}/{k}={v:.4f}"
        logger.info(msg)

        if self.config.train.use_wandb:
            try:
                import wandb
                wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()})
            except Exception:
                pass
