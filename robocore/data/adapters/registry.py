"""数据集注册表。"""
from __future__ import annotations

from typing import Any, Type

from robocore.data.dataset import BaseDataset


class DatasetRegistry:
    """数据集格式注册表。"""

    _registry: dict[str, Type[BaseDataset]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(dataset_cls: Type[BaseDataset]) -> Type[BaseDataset]:
            cls._registry[name] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseDataset:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown dataset format '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())
