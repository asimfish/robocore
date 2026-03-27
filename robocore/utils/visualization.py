"""可视化工具：动作轨迹、训练曲线、策略分析。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TrajectoryVisualizer:
    """动作轨迹可视化。

    将预测/真实动作轨迹保存为可分析的格式。
    支持 matplotlib 绘图（如果可用）。
    """

    @staticmethod
    def compare_trajectories(
        pred: np.ndarray | torch.Tensor,
        gt: np.ndarray | torch.Tensor,
        save_path: str | Path | None = None,
        dim_names: list[str] | None = None,
    ) -> Any:
        """对比预测和真实轨迹。

        Args:
            pred: (T, action_dim) 预测轨迹
            gt: (T, action_dim) 真实轨迹
            save_path: 保存路径
            dim_names: 各维度名称
        """
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()

        T, D = pred.shape
        if dim_names is None:
            dim_names = [f"dim_{i}" for i in range(D)]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(D, 1, figsize=(10, 2 * D), sharex=True)
            if D == 1:
                axes = [axes]

            for i, (ax, name) in enumerate(zip(axes, dim_names)):
                ax.plot(pred[:, i], label="pred", alpha=0.8)
                ax.plot(gt[:, i], label="gt", alpha=0.8, linestyle="--")
                ax.set_ylabel(name)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("timestep")
            fig.suptitle("Action Trajectory Comparison")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            return fig

        except ImportError:
            # 无 matplotlib，保存为 JSON
            data = {
                "pred": pred.tolist(),
                "gt": gt.tolist(),
                "dim_names": dim_names,
            }
            if save_path:
                with open(str(save_path).replace(".png", ".json"), "w") as f:
                    json.dump(data, f)
            return data

    @staticmethod
    def plot_action_distribution(
        actions: np.ndarray | torch.Tensor,
        save_path: str | Path | None = None,
    ) -> Any:
        """绘制动作分布直方图。"""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            D = actions.shape[-1]
            fig, axes = plt.subplots(1, D, figsize=(3 * D, 3))
            if D == 1:
                axes = [axes]

            for i, ax in enumerate(axes):
                ax.hist(actions[..., i].flatten(), bins=50, alpha=0.7)
                ax.set_title(f"dim {i}")
                ax.grid(True, alpha=0.3)

            fig.suptitle("Action Distribution")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            return fig

        except ImportError:
            return None


class TrainingCurveVisualizer:
    """训练曲线可视化。"""

    @staticmethod
    def plot_from_jsonl(
        log_path: str | Path,
        keys: list[str] | None = None,
        save_path: str | Path | None = None,
    ) -> Any:
        """从 JSONL 日志文件绘制训练曲线。"""
        log_path = Path(log_path)
        if not log_path.exists():
            return None

        entries = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        if not entries:
            return None

        # 收集所有 key
        all_keys = set()
        for e in entries:
            all_keys.update(k for k in e if k not in ("step", "timestamp"))

        if keys:
            plot_keys = [k for k in keys if k in all_keys]
        else:
            plot_keys = sorted(all_keys)

        if not plot_keys:
            return None

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(len(plot_keys), 1, figsize=(10, 3 * len(plot_keys)), sharex=True)
            if len(plot_keys) == 1:
                axes = [axes]

            for ax, key in zip(axes, plot_keys):
                steps = [e["step"] for e in entries if key in e]
                values = [e[key] for e in entries if key in e]
                ax.plot(steps, values)
                ax.set_ylabel(key)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("step")
            fig.suptitle("Training Curves")
            fig.tight_layout()

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
            return fig

        except ImportError:
            return {"keys": plot_keys, "num_entries": len(entries)}


class PolicyAnalyzer:
    """策略分析工具。"""

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> dict[str, int]:
        """统计模型参数。"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = total - trainable
        return {"total": total, "trainable": trainable, "frozen": frozen}

    @staticmethod
    def measure_inference_time(
        model: torch.nn.Module,
        obs: dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup: int = 10,
    ) -> dict[str, float]:
        """测量推理延迟。"""
        import time

        model.train(False)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                model.predict(obs)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                model.predict(obs)
                times.append(time.perf_counter() - start)

        times_ms = [t * 1000 for t in times]
        return {
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "p50_ms": np.percentile(times_ms, 50),
            "p95_ms": np.percentile(times_ms, 95),
            "fps": 1000.0 / np.mean(times_ms),
        }
