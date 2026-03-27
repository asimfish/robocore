"""配置系统：基于 dataclass 的类型安全配置。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PolicyConfig:
    """策略配置。"""

    name: str = "dp"  # dp, dp3, act, flow_policy, openvla, pi0
    obs_dim: int | None = None
    action_dim: int = 7
    obs_horizon: int = 2
    pred_horizon: int = 16
    action_horizon: int = 8

    # 网络参数
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    # 扩散/流匹配参数
    num_diffusion_steps: int = 100
    noise_schedule: str = "squaredcos_cap_v2"

    # 扩散/流匹配推理步数
    num_inference_steps: int = 10

    # 图像编码器
    image_encoder: str = "resnet18"  # resnet18, resnet34, resnet50, clip, dinov2
    image_size: list[int] = field(default_factory=lambda: [224, 224])
    freeze_image_encoder: bool = False

    # VLA 特有
    vla_model_name: str = ""  # e.g. "openvla/openvla-7b"
    vla_use_lora: bool = True
    vla_lora_rank: int = 32

    # 额外参数
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataConfig:
    """数据配置。"""

    dataset_name: str = ""
    root: str = "data/"
    format: str = "hdf5"  # hdf5, lerobot, rlds, zarr

    # 数据加载
    obs_horizon: int = 2
    pred_horizon: int = 16
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2

    # 数据增强
    random_crop: bool = False
    crop_size: list[int] = field(default_factory=lambda: [216, 216])
    color_jitter: bool = False

    # 归一化
    normalize_action: bool = True
    normalize_obs: bool = True
    norm_mode: str = "minmax"  # minmax, zscore

    # 额外参数
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """环境配置。"""

    benchmark: str = "robomimic"  # robomimic, libero, maniskill3, robotwin
    task_name: str = "lift"
    max_episode_steps: int = 400
    num_eval_episodes: int = 50
    num_envs: int = 1  # 并行环境数

    # 渲染
    render: bool = False
    camera_names: list[str] = field(default_factory=lambda: ["agentview"])

    # 额外参数
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    """训练配置。"""

    # 基本
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs/"
    exp_name: str = "default"

    # 训练参数
    num_epochs: int = 100
    steps_per_epoch: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 优化器
    optimizer: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-6
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"  # cosine, linear, constant

    # 混合精度
    use_amp: bool = True
    amp_dtype: str = "bf16"  # fp16, bf16

    # 分布式
    distributed: bool = False
    dist_backend: str = "nccl"

    # 日志
    log_interval: int = 10
    save_interval: int = 10  # epochs
    eval_interval: int = 10  # epochs
    use_wandb: bool = False
    wandb_project: str = "robocore"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.995

    # 额外参数
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    """完整实验配置。"""

    policy: PolicyConfig = field(default_factory=PolicyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        """从 YAML 文件加载配置。"""
        with open(path) as f:
            raw = yaml.safe_load(f)

        def _filter_fields(dc_cls, d: dict) -> dict:
            """只保留 dataclass 中定义的字段。"""
            import dataclasses
            valid_keys = {f.name for f in dataclasses.fields(dc_cls)}
            return {k: v for k, v in d.items() if k in valid_keys}

        return cls(
            policy=PolicyConfig(**_filter_fields(PolicyConfig, raw.get("policy", {}))),
            data=DataConfig(**_filter_fields(DataConfig, raw.get("data", {}))),
            env=EnvConfig(**_filter_fields(EnvConfig, raw.get("env", {}))),
            train=TrainConfig(**_filter_fields(TrainConfig, raw.get("train", {}))),
        )

    def to_yaml(self, path: str | Path) -> None:
        """保存配置到 YAML 文件。"""
        from dataclasses import asdict

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, allow_unicode=True)

    def merge(self, overrides: dict[str, Any]) -> ExperimentConfig:
        """合并覆盖配置。"""
        from dataclasses import asdict

        base = asdict(self)
        for key, value in overrides.items():
            parts = key.split(".")
            d = base
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value
        return ExperimentConfig.from_dict(base)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        """从字典创建配置。"""
        def _filter_fields(dc_cls, data: dict) -> dict:
            import dataclasses
            valid_keys = {f.name for f in dataclasses.fields(dc_cls)}
            return {k: v for k, v in data.items() if k in valid_keys}

        return cls(
            policy=PolicyConfig(**_filter_fields(PolicyConfig, d.get("policy", {}))),
            data=DataConfig(**_filter_fields(DataConfig, d.get("data", {}))),
            env=EnvConfig(**_filter_fields(EnvConfig, d.get("env", {}))),
            train=TrainConfig(**_filter_fields(TrainConfig, d.get("train", {}))),
        )
