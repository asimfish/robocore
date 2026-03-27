# Contributing to RoboCore

## 开发环境

```bash
git clone https://github.com/YOUR_USERNAME/robocore.git
cd robocore
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## 添加新算法

1. 在 `robocore/policy/algos/` 创建新文件
2. 继承 `BasePolicy`，实现 `_build_model`, `predict`, `compute_loss`
3. 用 `@PolicyRegistry.register("name")` 注册
4. 在 `robocore/policy/algos/__init__.py` 中导入
5. 在 `tests/` 中添加测试
6. 在 `configs/` 中添加预置配置

## 添加新数据格式

1. 在 `robocore/data/adapters/` 创建新文件
2. 继承 `BaseDatasetAdapter`
3. 用 `@DatasetRegistry.register("name")` 注册

## 添加新环境

1. 在 `robocore/env/wrappers/` 创建新文件
2. 继承 `BaseEnv`
3. 用 `@EnvRegistry.register("name")` 注册

## 测试

```bash
pytest tests/ -v
pytest tests/test_policy.py -v  # 单个文件
```

## 代码风格

- 使用 ruff 格式化
- 类型注解
- 中文注释 + 英文 API
