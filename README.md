# RoboCore 🤖

> 通用机器人学习祖传代码库 — 统一 benchmark、算法、数据格式

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-93%20passed-brightgreen.svg)]()

## 特性

- **20 种算法**: Diffusion Policy 全家桶、Flow Policy 全家桶、ACT、VLA (OpenVLA/Pi0/Pi0.5)、经典 IL (BC/IBC/DAgger)
- **多 Benchmark**: RoboMimic, LIBERO, ManiSkill3, RoboTwin
- **多数据格式**: LeRobot, RLDS, HDF5, Zarr + 跨格式转换
- **世界模型**: Latent Dynamics (RSSM), Video Predictor, Reward Predictor
- **3D 感知**: PointNet, FPS, 体素化, 深度图转点云
- **多模态融合**: 语言条件、目标图像、触觉 + Attention/FiLM 融合
- **表征学习**: R3M, MVP, VC-1, DINOv2, CLIP wrapper + 对比学习
- **RL 桥接**: SAC, DPPO, Replay Buffer
- **Sim2Real**: 域随机化、真机部署接口、相机标定
- **分布式训练**: DDP, FSDP, AMP, 梯度累积
- **可视化**: TensorBoard, WandB, 轨迹可视化, 策略分析

## 快速开始

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone https://github.com/asimfish/robocore.git
cd robocore
uv sync

# 训练 Diffusion Policy on RoboMimic
uv run robocore train --config configs/dp_robomimic.yaml

# 评估
uv run robocore eval --checkpoint outputs/dp_robomimic/best.pt

# 数据格式转换
uv run robocore convert --src data/demo.hdf5 --dst data/demo_lerobot --src-format hdf5 --dst-format lerobot

# 查看数据集信息
uv run robocore info --path data/demo.hdf5

# 安装可选依赖（如 benchmark 环境）
uv sync --extra robomimic
uv sync --extra viz
```

## 架构

```
robocore/
├── policy/              # 策略算法
│   ├── algos/           # 20 种算法实现
│   ├── networks/        # U-Net, DiT 网络骨干
│   ├── encoders.py      # State, Image, PointNet 编码器
│   └── schedulers.py    # DDPM, DDIM, FlowMatching 调度器
├── data/                # 数据管线
│   ├── adapters/        # HDF5, LeRobot, RLDS, Zarr 适配器
│   └── converter.py     # 跨格式转换
├── env/                 # 环境接口
│   └── wrappers/        # RoboMimic, LIBERO, ManiSkill3, RoboTwin
├── world_model/         # 世界模型
├── representations/     # 预训练视觉表征
├── multimodal/          # 多模态融合
├── perception3d/        # 3D 感知
├── rl/                  # 强化学习
├── sim2real/            # Sim-to-Real
├── trainer/             # 训练器 + 分布式
├── config/              # 配置系统
├── utils/               # 日志、可视化
└── cli/                 # 命令行工具
```

## 支持的算法

| 类别 | 算法 | 注册名 |
|------|------|--------|
| **Diffusion** | Diffusion Policy (U-Net) | `dp` |
| | Diffusion Policy (Transformer) | `dp_t` |
| | Diffusion Policy (CNN alias) | `dp_cnn` |
| | DP Behavior Ensemble | `dp_be` |
| | DP Classifier-Free Guidance | `dp_cfg` |
| | 3D Diffusion Policy | `dp3` |
| | DiT Policy | `dit` |
| **Flow** | Flow Policy | `flow_policy` |
| | MeanFlow (1-step) | `mean_flow` |
| | Consistency Flow | `consistency_flow` |
| | Rectified Flow | `rectified_flow` |
| | Flow-DiT | `flow_dit` |
| **Transformer** | ACT | `act` |
| **VLA** | OpenVLA | `openvla` |
| | Pi0 | `pi0` |
| | Pi0.5 (dual-system) | `pi05` |
| **Classic IL** | Behavior Cloning | `bc` |
| | BC-RNN | `bc_rnn` |
| | Implicit BC (EBM) | `ibc` |
| | DAgger | `dagger` |

## 添加新算法

```python
from robocore.policy.base import BasePolicy, PolicyOutput
from robocore.policy.registry import PolicyRegistry

@PolicyRegistry.register("my_policy")
class MyPolicy(BasePolicy):
    def _build_model(self):
        self.net = ...

    def predict(self, obs):
        return PolicyOutput(action=self.net(obs["state"]))

    def compute_loss(self, obs, action):
        pred = self.predict(obs).action
        return {"loss": F.mse_loss(pred, action)}
```

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发流程。

```bash
# 开发环境
git clone https://github.com/YOUR_USERNAME/robocore.git
cd robocore
uv sync
uv run pytest tests/ -v
```

## License

Apache License 2.0
