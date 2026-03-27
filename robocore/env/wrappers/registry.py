"""环境注册表。"""
from __future__ import annotations

from typing import Any, Type

from robocore.env.base import BaseEnv


class EnvRegistry:
    """环境注册表。"""

    _registry: dict[str, Type[BaseEnv]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(env_cls: Type[BaseEnv]) -> Type[BaseEnv]:
            cls._registry[name] = env_cls
            return env_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BaseEnv:
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(f"Unknown env '{name}'. Available: {available}")
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        return sorted(cls._registry.keys())
