"""分布式训练工具：DDP, FSDP, 混合精度, 梯度累积。"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================
# 分布式初始化
# ============================================================

def setup_distributed(backend: str = "nccl") -> dict[str, int]:
    """初始化分布式训练环境。

    Returns:
        {"rank": int, "world_size": int, "local_rank": int}
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    else:
        logger.info("Single GPU training")

    return {"rank": rank, "world_size": world_size, "local_rank": local_rank}


def cleanup_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def is_main_process() -> bool:
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


# ============================================================
# DDP Wrapper
# ============================================================

def wrap_ddp(
    model: nn.Module,
    device_id: int | None = None,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """用 DistributedDataParallel 包装模型。"""
    if not torch.distributed.is_initialized():
        logger.warning("DDP requested but distributed not initialized, returning raw model")
        return model

    if device_id is None:
        device_id = int(os.environ.get("LOCAL_RANK", 0))

    model = model.to(f"cuda:{device_id}")
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=find_unused_parameters,
    )


# ============================================================
# FSDP Wrapper
# ============================================================

def wrap_fsdp(
    model: nn.Module,
    mixed_precision: bool = True,
    cpu_offload: bool = False,
) -> nn.Module:
    """用 FullyShardedDataParallel 包装模型。

    FSDP 适用于大模型（如 VLA），可以跨 GPU 分片参数。
    """
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, CPUOffload
    except ImportError:
        logger.warning("FSDP not available, returning raw model")
        return model

    if not torch.distributed.is_initialized():
        logger.warning("FSDP requested but distributed not initialized")
        return model

    fsdp_kwargs: dict[str, Any] = {}

    if mixed_precision:
        fsdp_kwargs["mixed_precision"] = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    if cpu_offload:
        fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

    return FSDP(model, **fsdp_kwargs)


# ============================================================
# 混合精度训练
# ============================================================

class AMPTrainer:
    """自动混合精度训练封装。

    用法：
        amp = AMPTrainer(enabled=True)
        with amp.autocast():
            loss = model(x)
        amp.backward(loss)
        amp.step(optimizer)
    """

    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype

        if self.enabled:
            self.scaler = torch.amp.GradScaler(growth_interval=growth_interval)
        else:
            self.scaler = None

    @contextmanager
    def autocast(self) -> Generator[None, None, None]:
        if self.enabled:
            with torch.amp.autocast("cuda", dtype=self.dtype):
                yield
        else:
            yield

    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)


# ============================================================
# 梯度累积
# ============================================================

class GradientAccumulator:
    """梯度累积封装。

    用于在小 batch size 下模拟大 batch 训练。

    用法：
        accum = GradientAccumulator(accumulation_steps=4)
        for batch in dataloader:
            loss = model(batch) / accum.accumulation_steps
            loss.backward()
            if accum.should_step():
                optimizer.step()
                optimizer.zero_grad()
            accum.step()
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self._step_count = 0

    def step(self) -> None:
        self._step_count += 1

    def should_step(self) -> bool:
        return self._step_count % self.accumulation_steps == 0

    @property
    def scale_factor(self) -> float:
        return 1.0 / self.accumulation_steps

    def reset(self) -> None:
        self._step_count = 0
