"""策略注册表：通过名称创建策略实例。"""
from __future__ import annotations

from typing import Any, Type

from robocore.policy.base import BasePolicy


class PolicyRegistry:
    """策略注册表。

    用法：
        @PolicyRegistry.register("dp")
        class DiffusionPolicy(BasePolicy):
            ...

        policy = PolicyRegistry.create("dp", action_dim=7)
    """

    _registry: dict[str, Type[BasePolicy]] = {}

    @classmethod
    def register(cls, name: str):
        """注册策略的装饰器。"""
        def decorator(policy_cls: Type[BasePolicy]) -> Type[BasePolicy]:
            cls._registry[name] = policy_cls
            return policy_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> BasePolicy:
        """通过名称创建策略实例。"""
        if name not in cls._registry:
            available = ", ".join(sorted(cls._registry.keys()))
            raise ValueError(
                f"Unknown policy '{name}'. Available: {available}"
            )
        return cls._registry[name](**kwargs)

    @classmethod
    def list(cls) -> list[str]:
        """列出所有已注册的策略。"""
        return sorted(cls._registry.keys())
