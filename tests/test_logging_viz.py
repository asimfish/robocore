"""测试日志和可视化。"""
import tempfile
from pathlib import Path
import numpy as np
import torch

from robocore.utils.logger import ConsoleLogger, JSONLogger, CompositeLogger, create_logger
from robocore.utils.visualization import TrajectoryVisualizer, PolicyAnalyzer


def test_console_logger():
    lg = ConsoleLogger(prefix="test")
    lg.log_scalars({"loss": 0.5, "lr": 1e-4}, step=100)


def test_json_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        lg = JSONLogger(Path(tmpdir) / "metrics.jsonl")
        for i in range(15):
            lg.log_scalars({"loss": 1.0 / (i + 1)}, step=i)
        lg.close()
        log_file = Path(tmpdir) / "metrics.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 15


def test_composite_logger():
    with tempfile.TemporaryDirectory() as tmpdir:
        lg = create_logger(tmpdir, use_tensorboard=False, use_wandb=False, use_json=True, name="test")
        lg.log_scalars({"loss": 0.1}, step=0)
        lg.close()


def test_trajectory_compare():
    pred = np.random.randn(20, 7)
    gt = np.random.randn(20, 7)
    with tempfile.TemporaryDirectory() as tmpdir:
        result = TrajectoryVisualizer.compare_trajectories(pred, gt, save_path=Path(tmpdir) / "traj.png")
        assert result is not None


def test_policy_analyzer():
    model = torch.nn.Linear(10, 7)
    info = PolicyAnalyzer.count_parameters(model)
    assert info["total"] == 10 * 7 + 7
    assert info["trainable"] == info["total"]
