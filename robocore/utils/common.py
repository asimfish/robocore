"""通用工具函数。"""
from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """设置全局随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_params(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """统计模型参数量。"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def to_device(data: Any, device: str | torch.device) -> Any:
    """递归地将数据移到设备。"""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    return data


def flatten_dict(d: dict, parent_key: str = "", sep: str = "/") -> dict[str, Any]:
    """展平嵌套字典。"""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AverageMeter:
    """滑动平均计量器。"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: list[float] = []
        self._sum = 0.0
        self._count = 0

    def update(self, value: float, n: int = 1) -> None:
        self._values.append(value)
        self._sum += value * n
        self._count += n
        if len(self._values) > self.window_size:
            self._values = self._values[-self.window_size :]

    @property
    def avg(self) -> float:
        return self._sum / max(self._count, 1)

    @property
    def recent_avg(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()
        self._sum = 0.0
        self._count = 0
