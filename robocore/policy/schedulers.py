"""噪声调度器：统一 Diffusion 和 Flow Matching 的噪声调度。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn


class BaseScheduler(ABC):
    """噪声调度器基类。"""

    def __init__(self, num_steps: int = 100):
        self.num_steps = num_steps

    @abstractmethod
    def add_noise(
        self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """给干净数据添加噪声。"""

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """去噪一步。"""

    @abstractmethod
    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """采样训练时间步。"""


class DDPMScheduler(BaseScheduler):
    """DDPM 噪声调度器。"""

    def __init__(
        self,
        num_steps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        super().__init__(num_steps)
        self.beta_schedule = beta_schedule

        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps)
        elif beta_schedule == "squaredcos_cap_v2":
            # cosine schedule
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = torch.clamp(betas, max=0.999).float()
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].to(x.device)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps].to(x.device)

        # 扩展维度以匹配 x
        while sqrt_alpha.ndim < x.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        return sqrt_alpha * x + sqrt_one_minus_alpha * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM 去噪步骤。"""
        alpha = self.alphas[timestep].to(sample.device)
        alpha_cumprod = self.alphas_cumprod[timestep].to(sample.device)
        beta = self.betas[timestep].to(sample.device)

        # 预测 x_0
        pred_x0 = (
            sample - torch.sqrt(1 - alpha_cumprod) * model_output
        ) / torch.sqrt(alpha_cumprod)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        # 计算均值
        mean = (
            torch.sqrt(alpha) * (1 - self.alphas_cumprod[max(timestep - 1, 0)].to(sample.device))
            / (1 - alpha_cumprod) * sample
            + torch.sqrt(self.alphas_cumprod[max(timestep - 1, 0)].to(sample.device))
            * beta / (1 - alpha_cumprod) * pred_x0
        )

        if timestep > 0:
            noise = torch.randn_like(sample)
            variance = beta * (1 - self.alphas_cumprod[timestep - 1].to(sample.device)) / (
                1 - alpha_cumprod
            )
            mean = mean + torch.sqrt(variance) * noise

        return mean

    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)


class DDIMScheduler(BaseScheduler):
    """DDIM 加速采样调度器。"""

    def __init__(
        self,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__(num_train_steps)
        self.num_inference_steps = num_inference_steps

        betas = torch.linspace(beta_start, beta_end, num_train_steps)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

        # 推理时间步（均匀间隔）
        step_ratio = num_train_steps // num_inference_steps
        self.inference_timesteps = (
            torch.arange(0, num_inference_steps) * step_ratio
        ).flip(0)

    def add_noise(
        self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[timesteps]).to(x.device)
        sqrt_one_minus = torch.sqrt(1 - self.alphas_cumprod[timesteps]).to(x.device)
        while sqrt_alpha.ndim < x.ndim:
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x + sqrt_one_minus * noise

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM 确定性去噪。"""
        alpha_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prev = (
            self.alphas_cumprod[max(timestep - self.num_steps // self.num_inference_steps, 0)]
            .to(sample.device)
        )

        pred_x0 = (sample - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        pred_x0 = torch.clamp(pred_x0, -1, 1)

        direction = torch.sqrt(1 - alpha_prev) * model_output
        return torch.sqrt(alpha_prev) * pred_x0 + direction

    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)


class FlowMatchingScheduler(BaseScheduler):
    """Flow Matching 调度器（ODE-based）。"""

    def __init__(self, num_steps: int = 10, sigma_min: float = 1e-4):
        super().__init__(num_steps)
        self.sigma_min = sigma_min

    def add_noise(
        self, x: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """线性插值：x_t = (1-t) * noise + t * x。"""
        t = timesteps.float() / self.num_steps
        while t.ndim < x.ndim:
            t = t.unsqueeze(-1)
        return (1 - t) * noise + t * x

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        """Euler 步进。"""
        dt = 1.0 / self.num_steps
        return sample + model_output * dt

    def get_velocity_target(
        self, x: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Flow matching 的训练目标：velocity = x - noise。"""
        return x - noise

    def get_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_steps, (batch_size,), device=device)
