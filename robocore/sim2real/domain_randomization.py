"""域随机化：缩小 sim-to-real gap。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import numpy as np


@dataclass
class RandomizationConfig:
    """域随机化配置。"""
    # 视觉随机化
    brightness_range: tuple[float, float] = (0.8, 1.2)
    contrast_range: tuple[float, float] = (0.8, 1.2)
    saturation_range: tuple[float, float] = (0.8, 1.2)
    hue_range: tuple[float, float] = (-0.05, 0.05)
    noise_std: float = 0.02
    blur_prob: float = 0.1
    cutout_prob: float = 0.1
    cutout_size: float = 0.1

    # 物理随机化
    friction_range: tuple[float, float] = (0.5, 1.5)
    mass_range: tuple[float, float] = (0.8, 1.2)
    damping_range: tuple[float, float] = (0.8, 1.2)

    # 相机随机化
    camera_pos_noise: float = 0.01
    camera_rot_noise: float = 0.02
    fov_range: tuple[float, float] = (55, 65)

    # 动作随机化
    action_noise_std: float = 0.01
    action_delay_range: tuple[int, int] = (0, 2)

    # 观测随机化
    obs_noise_std: float = 0.005


class DomainRandomizer(nn.Module):
    """域随机化管线。

    在训练时对观测和动作施加随机扰动，
    提高策略对 sim-to-real gap 的鲁棒性。

    支持：
    - 视觉随机化 (颜色、噪声、遮挡)
    - 物理参数随机化 (摩擦、质量)
    - 相机参数随机化
    - 动作/观测噪声
    """

    def __init__(self, config: RandomizationConfig | None = None):
        super().__init__()
        self.config = config or RandomizationConfig()

    def randomize_image(self, image: torch.Tensor) -> torch.Tensor:
        """视觉域随机化。

        Args:
            image: (B, C, H, W) 图像 tensor，值域 [0, 1]
        """
        cfg = self.config
        B, C, H, W = image.shape

        # 亮度
        brightness = torch.empty(B, 1, 1, 1, device=image.device).uniform_(*cfg.brightness_range)
        image = image * brightness

        # 对比度
        contrast = torch.empty(B, 1, 1, 1, device=image.device).uniform_(*cfg.contrast_range)
        mean = image.mean(dim=(2, 3), keepdim=True)
        image = (image - mean) * contrast + mean

        # 高斯噪声
        if cfg.noise_std > 0:
            noise = torch.randn_like(image) * cfg.noise_std
            image = image + noise

        # 随机遮挡 (cutout)
        if cfg.cutout_prob > 0:
            mask = torch.ones_like(image)
            for b in range(B):
                if torch.rand(1).item() < cfg.cutout_prob:
                    ch = int(H * cfg.cutout_size)
                    cw = int(W * cfg.cutout_size)
                    y = torch.randint(0, H - ch + 1, (1,)).item()
                    x = torch.randint(0, W - cw + 1, (1,)).item()
                    mask[b, :, y:y+ch, x:x+cw] = 0
            image = image * mask

        return image.clamp(0, 1)

    def randomize_state(self, state: torch.Tensor) -> torch.Tensor:
        """状态观测随机化。"""
        if self.config.obs_noise_std > 0:
            noise = torch.randn_like(state) * self.config.obs_noise_std
            state = state + noise
        return state

    def randomize_action(self, action: torch.Tensor) -> torch.Tensor:
        """动作随机化。"""
        if self.config.action_noise_std > 0:
            noise = torch.randn_like(action) * self.config.action_noise_std
            action = action + noise
        return action

    def randomize_physics(self) -> dict[str, float]:
        """生成随机物理参数（用于仿真环境）。"""
        cfg = self.config
        return {
            "friction": np.random.uniform(*cfg.friction_range),
            "mass_scale": np.random.uniform(*cfg.mass_range),
            "damping_scale": np.random.uniform(*cfg.damping_range),
        }

    def randomize_camera(self) -> dict[str, Any]:
        """生成随机相机参数。"""
        cfg = self.config
        return {
            "pos_offset": np.random.randn(3) * cfg.camera_pos_noise,
            "rot_offset": np.random.randn(3) * cfg.camera_rot_noise,
            "fov": np.random.uniform(*cfg.fov_range),
        }

    def forward(self, obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """对整个观测字典施加随机化。"""
        result = {}
        for key, value in obs.items():
            if key in ("image", "rgb", "agentview_image", "eye_in_hand_image"):
                result[key] = self.randomize_image(value)
            elif key in ("state", "robot_state", "joint_pos"):
                result[key] = self.randomize_state(value)
            else:
                result[key] = value
        return result
