"""测试配置系统。"""
from pathlib import Path

from robocore.config.base import (
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainConfig,
)


def test_default_config():
    """测试默认配置。"""
    config = ExperimentConfig()
    assert config.policy.name == "dp"
    assert config.data.batch_size == 64
    assert config.train.lr == 1e-4


def test_config_yaml_roundtrip(tmp_path):
    """测试配置 YAML 序列化/反序列化。"""
    config = ExperimentConfig(
        policy=PolicyConfig(name="act", action_dim=8),
        data=DataConfig(batch_size=32),
        train=TrainConfig(lr=5e-5, num_epochs=50),
    )

    yaml_path = tmp_path / "test_config.yaml"
    config.to_yaml(yaml_path)

    loaded = ExperimentConfig.from_yaml(yaml_path)
    assert loaded.policy.name == "act"
    assert loaded.policy.action_dim == 8
    assert loaded.data.batch_size == 32
    assert loaded.train.lr == 5e-5


def test_config_from_preset():
    """测试从预置配置文件加载。"""
    config_path = Path(__file__).parent.parent / "configs" / "dp_robomimic.yaml"
    if config_path.exists():
        config = ExperimentConfig.from_yaml(config_path)
        assert config.policy.name == "dp"
        assert config.env.benchmark == "robomimic"
