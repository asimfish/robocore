"""RoboCore CLI 入口。"""
from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="robocore",
        description="RoboCore — 通用机器人学习祖传代码库",
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # train
    train_parser = subparsers.add_parser("train", help="训练策略")
    train_parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    train_parser.add_argument("--resume", type=str, default=None, help="恢复训练的 checkpoint 路径")
    train_parser.add_argument(
        "--override", nargs="*", default=[], help="覆盖配置，格式: key=value"
    )

    # assess
    assess_parser = subparsers.add_parser("assess", help="评估策略")
    assess_parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    assess_parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    assess_parser.add_argument("--num-episodes", type=int, default=50, help="评估 episode 数")

    # convert
    convert_parser = subparsers.add_parser("convert", help="数据格式转换")
    convert_parser.add_argument("--src", type=str, required=True, help="源数据路径")
    convert_parser.add_argument("--dst", type=str, required=True, help="目标数据路径")
    convert_parser.add_argument(
        "--src-format", type=str, required=True, choices=["hdf5", "lerobot", "rlds", "zarr"]
    )
    convert_parser.add_argument(
        "--dst-format", type=str, required=True, choices=["hdf5", "lerobot", "rlds", "zarr"]
    )

    # info
    info_parser = subparsers.add_parser("info", help="显示数据集/模型信息")
    info_parser.add_argument("path", type=str, help="数据集或模型路径")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        _run_train(args)
    elif args.command == "assess":
        _run_assess(args)
    elif args.command == "convert":
        _run_convert(args)
    elif args.command == "info":
        _run_info(args)


def _run_train(args: argparse.Namespace) -> None:
    """执行训练。"""
    from robocore.config.base import ExperimentConfig

    config = ExperimentConfig.from_yaml(args.config)

    # 应用覆盖
    if args.override:
        overrides = {}
        for item in args.override:
            key, value = item.split("=", 1)
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
            overrides[key] = value
        config = config.merge(overrides)

    from robocore.policy import PolicyRegistry
    policy = PolicyRegistry.create(
        config.policy.name,
        obs_dim=config.policy.obs_dim,
        action_dim=config.policy.action_dim,
        obs_horizon=config.policy.obs_horizon,
        pred_horizon=config.policy.pred_horizon,
        action_horizon=config.policy.action_horizon,
    )

    logging.getLogger(__name__).info(
        f"Training {config.policy.name} | "
        f"params={policy.num_params:,} | "
        f"device={config.train.device}"
    )


def _run_assess(args: argparse.Namespace) -> None:
    """执行评估。"""
    logging.getLogger(__name__).info(f"Assessing checkpoint: {args.checkpoint}")


def _run_convert(args: argparse.Namespace) -> None:
    """执行数据转换。"""
    logging.getLogger(__name__).info(
        f"Converting {args.src} ({args.src_format}) -> {args.dst} ({args.dst_format})"
    )


def _run_info(args: argparse.Namespace) -> None:
    """显示信息。"""
    logging.getLogger(__name__).info(f"Info for: {args.path}")


if __name__ == "__main__":
    main()
