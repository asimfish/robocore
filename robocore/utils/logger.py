"""统一日志系统：支持 TensorBoard, WandB, 控制台。"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BaseLogger:
    """日志基类。"""

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        pass

    def log_video(self, tag: str, video: Any, step: int, fps: int = 10) -> None:
        pass

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        pass

    def close(self) -> None:
        pass


class ConsoleLogger(BaseLogger):
    """控制台日志。"""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        msg = " | ".join(parts)
        if self.prefix:
            msg = f"[{self.prefix}] {msg}"
        logger.info(msg)


class TensorBoardLogger(BaseLogger):
    """TensorBoard 日志。"""

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = None

    @property
    def writer(self) -> Any:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(str(self.log_dir))
            except ImportError:
                logger.warning("tensorboard not installed, falling back to console")
                self._writer = "unavailable"
        return self._writer

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer != "unavailable":
            self.writer.add_scalar(tag, value, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        if self.writer != "unavailable":
            self.writer.add_image(tag, image, step)

    def log_histogram(self, tag: str, values: Any, step: int) -> None:
        if self.writer != "unavailable":
            self.writer.add_histogram(tag, values, step)

    def close(self) -> None:
        if self._writer is not None and self._writer != "unavailable":
            self._writer.close()


class WandBLogger(BaseLogger):
    """Weights & Biases 日志。"""

    def __init__(
        self,
        project: str = "robocore",
        name: str | None = None,
        config: dict | None = None,
    ):
        self._run = None
        self.project = project
        self.name = name
        self.config = config

    @property
    def run(self) -> Any:
        if self._run is None:
            try:
                import wandb
                self._run = wandb.init(
                    project=self.project,
                    name=self.name,
                    config=self.config,
                )
            except ImportError:
                logger.warning("wandb not installed")
                self._run = "unavailable"
        return self._run

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.run != "unavailable":
            import wandb
            wandb.log(metrics, step=step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        if self.run != "unavailable":
            import wandb
            wandb.log({tag: wandb.Image(image)}, step=step)

    def close(self) -> None:
        if self._run is not None and self._run != "unavailable":
            self._run.finish()


class JSONLogger(BaseLogger):
    """JSON 文件日志（轻量级，无依赖）。"""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict] = []

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        entry = {"step": step, "timestamp": time.time(), **metrics}
        self._buffer.append(entry)
        if len(self._buffer) >= 10:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        with open(self.log_path, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps(entry) + "\n")
        self._buffer.clear()

    def close(self) -> None:
        self.flush()


class CompositeLogger(BaseLogger):
    """组合多个 logger。"""

    def __init__(self, loggers: list[BaseLogger]):
        self.loggers = loggers

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for lg in self.loggers:
            lg.log_scalar(tag, value, step)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for lg in self.loggers:
            lg.log_scalars(metrics, step)

    def log_image(self, tag: str, image: Any, step: int) -> None:
        for lg in self.loggers:
            lg.log_image(tag, image, step)

    def close(self) -> None:
        for lg in self.loggers:
            lg.close()


def create_logger(
    log_dir: str | Path,
    project: str = "robocore",
    name: str | None = None,
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    use_json: bool = True,
    config: dict | None = None,
) -> CompositeLogger:
    """创建组合日志器。"""
    loggers: list[BaseLogger] = [ConsoleLogger(prefix=name or "")]

    if use_json:
        loggers.append(JSONLogger(Path(log_dir) / "metrics.jsonl"))

    if use_tensorboard:
        loggers.append(TensorBoardLogger(Path(log_dir) / "tb"))

    if use_wandb:
        loggers.append(WandBLogger(project=project, name=name, config=config))

    return CompositeLogger(loggers)
