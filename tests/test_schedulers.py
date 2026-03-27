"""测试噪声调度器。"""
import torch

from robocore.policy.schedulers import DDIMScheduler, DDPMScheduler, FlowMatchingScheduler


def test_ddpm_scheduler():
    """测试 DDPM 调度器。"""
    scheduler = DDPMScheduler(num_steps=100)

    x = torch.randn(4, 16, 7)
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, 100, (4,))

    noisy = scheduler.add_noise(x, noise, timesteps)
    assert noisy.shape == x.shape

    # 去噪一步
    denoised = scheduler.step(noise, 50, noisy)
    assert denoised.shape == x.shape


def test_ddim_scheduler():
    """测试 DDIM 调度器。"""
    scheduler = DDIMScheduler(num_train_steps=100, num_inference_steps=10)

    assert len(scheduler.inference_timesteps) == 10

    x = torch.randn(4, 16, 7)
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, 100, (4,))

    noisy = scheduler.add_noise(x, noise, timesteps)
    assert noisy.shape == x.shape


def test_flow_matching_scheduler():
    """测试 Flow Matching 调度器。"""
    scheduler = FlowMatchingScheduler(num_steps=10)

    x = torch.randn(4, 16, 7)
    noise = torch.randn_like(x)
    timesteps = torch.randint(0, 10, (4,))

    x_t = scheduler.add_noise(x, noise, timesteps)
    assert x_t.shape == x.shape

    # 速度目标
    velocity = scheduler.get_velocity_target(x, noise)
    assert velocity.shape == x.shape

    # Euler 步进
    stepped = scheduler.step(velocity, 5, x_t)
    assert stepped.shape == x.shape
