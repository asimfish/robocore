# RoboCore 🤖

> 通用机器人学习祖传代码库 — 统一 benchmark、算法、数据格式

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)

## 特性

- **统一算法**: Diffusion Policy, DP3, ACT, FlowPolicy, OpenVLA, Pi0
- **多 Benchmark**: RoboMimic, LIBERO, ManiSkill3, RoboTwin
- **多数据格式**: LeRobot, RLDS, HDF5, Zarr
- **插件化架构**: 编码器/解码器/调度器自由组合
- **配置驱动**: 一条命令切换算法/数据/环境
- **开箱即用**: 预置配置 + 预训练权重

## 快速开始

```bash
pip install robocore

# 训练 Diffusion Policy on RoboMimic
robocore train --config configs/dp_robomimic.yaml

# 评估
robocore eval --checkpoint outputs/dp_robomimic/best.pt

# 数据格式转换
robocore convert --src data/demo.hdf5 --dst data/demo_lerobot --format lerobot
```

## 安装

```bash
# 核心安装（最小依赖）
pip install robocore

# 安装特定 benchmark 支持
pip install "robocore[robomimic]"
pip install "robocore[maniskill3]"

# 安装特定算法支持
pip install "robocore[vla]"  # OpenVLA, Pi0

# 全量安装
pip install "robocore[all]"
```

## 项目结构

```
robocore/
├── robocore/           # 核心包
│   ├── data/           # 数据层：统一数据集接口 + 格式适配器
│   ├── policy/         # 策略层：统一策略接口 + 算法实现
│   ├── env/            # 环境层：统一环境接口 + benchmark wrapper
│   ├── trainer/        # 训练器：训练循环 + 评估 + checkpoint
│   ├── config/         # 配置系统：dataclass + YAML
│   ├── utils/          # 工具函数
│   └── cli/            # 命令行入口
├── configs/            # 预置配置文件
├── scripts/            # 实用脚本
├── tests/              # 测试
├── docs/               # 文档
└── examples/           # 示例代码
```

## 支持矩阵

| 算法 | RoboMimic | LIBERO | ManiSkill3 | RoboTwin |
|------|-----------|--------|------------|----------|
| DP   | ✅        | ✅     | ✅         | ✅       |
| DP3  | ✅        | ✅     | ✅         | ✅       |
| ACT  | ✅        | ✅     | ✅         | ✅       |
| FlowPolicy | ✅  | ✅     | ✅         | ✅       |
| OpenVLA | ✅     | ✅     | ✅         | ✅       |
| Pi0  | ✅        | ✅     | ✅         | ✅       |

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发流程。

## License

Apache License 2.0
