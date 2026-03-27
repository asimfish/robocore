"""数据变换 pipeline。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseTransform(ABC):
    """数据变换基类。"""

    @abstractmethod
    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """对样本应用变换。"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TransformPipeline:
    """变换链：按顺序应用多个变换。"""

    def __init__(self, transforms: list[BaseTransform] | None = None):
        self.transforms = transforms or []

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            sample = t(sample)
        return sample

    def append(self, transform: BaseTransform) -> None:
        self.transforms.append(transform)

    def __repr__(self) -> str:
        lines = [f"  {t}" for t in self.transforms]
        return f"TransformPipeline([\n" + "\n".join(lines) + "\n])"


class Normalize(BaseTransform):
    """归一化变换。"""

    def __init__(
        self,
        keys: list[str],
        mean: dict[str, torch.Tensor] | None = None,
        std: dict[str, torch.Tensor] | None = None,
        range_min: dict[str, torch.Tensor] | None = None,
        range_max: dict[str, torch.Tensor] | None = None,
        mode: str = "minmax",  # "minmax" or "zscore"
    ):
        self.keys = keys
        self.mean = mean or {}
        self.std = std or {}
        self.range_min = range_min or {}
        self.range_max = range_max or {}
        self.mode = mode

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        for key in self.keys:
            if key == "action" and key in sample:
                sample[key] = self._normalize(sample[key], key)
            elif key in sample.get("obs", {}):
                sample["obs"][key] = self._normalize(sample["obs"][key], key)
        return sample

    def _normalize(self, x: torch.Tensor, key: str) -> torch.Tensor:
        if self.mode == "minmax":
            lo = self.range_min.get(key, torch.zeros_like(x))
            hi = self.range_max.get(key, torch.ones_like(x))
            return (x - lo) / (hi - lo + 1e-8) * 2 - 1
        else:
            mu = self.mean.get(key, torch.zeros_like(x))
            sigma = self.std.get(key, torch.ones_like(x))
            return (x - mu) / (sigma + 1e-8)


class ImageTransform(BaseTransform):
    """图像变换：resize, crop, normalize to [0,1]。"""

    def __init__(
        self,
        image_keys: list[str] | None = None,
        size: tuple[int, int] = (224, 224),
        normalize: bool = True,
    ):
        self.image_keys = image_keys
        self.size = size
        self.normalize = normalize

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        obs = sample.get("obs", {})
        keys = self.image_keys or [k for k in obs if k.startswith("image_")]
        for key in keys:
            if key in obs:
                img = obs[key]
                if isinstance(img, torch.Tensor) and self.normalize and img.dtype == torch.uint8:
                    obs[key] = img.float() / 255.0
        sample["obs"] = obs
        return sample


class RandomCrop(BaseTransform):
    """随机裁剪图像。"""

    def __init__(self, size: tuple[int, int], image_keys: list[str] | None = None):
        self.size = size
        self.image_keys = image_keys

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        obs = sample.get("obs", {})
        keys = self.image_keys or [k for k in obs if k.startswith("image_")]
        for key in keys:
            if key in obs and isinstance(obs[key], torch.Tensor):
                img = obs[key]
                if img.ndim >= 3:
                    h, w = img.shape[-2:]
                    th, tw = self.size
                    if h > th and w > tw:
                        i = torch.randint(0, h - th + 1, (1,)).item()
                        j = torch.randint(0, w - tw + 1, (1,)).item()
                        obs[key] = img[..., i : i + th, j : j + tw]
        sample["obs"] = obs
        return sample
