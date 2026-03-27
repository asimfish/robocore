"""Benchmark 环境 wrapper 包。"""
from robocore.env.wrappers.registry import EnvRegistry

# 延迟导入各 benchmark wrapper（避免强制依赖）
def _register_all():
    """注册所有可用的环境 wrapper。"""
    import importlib
    wrappers = [
        "robocore.env.wrappers.robomimic_env",
        "robocore.env.wrappers.libero_env",
        "robocore.env.wrappers.maniskill3_env",
        "robocore.env.wrappers.robotwin_env",
    ]
    for mod_name in wrappers:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass

_register_all()

__all__ = ["EnvRegistry"]
