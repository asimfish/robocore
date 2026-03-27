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
    from robocore.data.adapters import DatasetRegistry
    from robocore.policy.registry import PolicyRegistry
    from robocore.trainer.base import BaseTrainer

    import torch
    from torch.utils.data import DataLoader

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

    # 设置种子
    torch.manual_seed(config.train.seed)

    # 创建策略
    policy_kwargs = {
        "obs_dim": config.policy.obs_dim,
        "action_dim": config.policy.action_dim,
        "obs_horizon": config.policy.obs_horizon,
        "pred_horizon": config.policy.pred_horizon,
        "action_horizon": config.policy.action_horizon,
        "device": config.train.device,
    }
    # 传递额外参数
    for k, v in config.policy.extra.items():
        policy_kwargs[k] = v

    policy = PolicyRegistry.create(config.policy.name, **policy_kwargs)

    # 创建数据集
    dataset = DatasetRegistry.create(
        config.data.format,
        root=config.data.root,
        obs_horizon=config.policy.obs_horizon,
        pred_horizon=config.policy.pred_horizon,
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # 创建训练器
    trainer = BaseTrainer(
        config=config,
        policy=policy,
        train_loader=train_loader,
    )

    # 恢复训练
    if args.resume:
        trainer.resume(args.resume)

    log = logging.getLogger(__name__)
    log.info(
        f"Training {config.policy.name} | "
        f"params={policy.num_params:,} | "
        f"device={config.train.device} | "
        f"dataset={config.data.format}:{config.data.root}"
    )

    # 开始训练
    trainer.train()


def _run_assess(args: argparse.Namespace) -> None:
    """执行评估。"""
    logging.getLogger(__name__).info(f"Assessing checkpoint: {args.checkpoint}")


def _run_convert(args: argparse.Namespace) -> None:
    """执行数据转换。"""
    from robocore.data.converter import DataConverter

    log = logging.getLogger(__name__)
    log.info(
        f"Converting {args.src} ({args.src_format}) -> {args.dst} ({args.dst_format})"
    )

    converter = DataConverter()
    converter.convert(
        src=args.src,
        dst=args.dst,
        src_format=args.src_format,
        dst_format=args.dst_format,
    )
    log.info("Conversion complete.")


def _run_info(args: argparse.Namespace) -> None:
    """显示信息。"""
    from pathlib import Path
    log = logging.getLogger(__name__)

    path = Path(args.path)
    if not path.exists():
        log.error(f"Path not found: {args.path}")
        return

    if path.suffix == ".hdf5" or path.suffix == ".h5":
        try:
            import h5py
            with h5py.File(path, "r") as f:
                log.info(f"HDF5 file: {path}")
                if "data" in f:
                    demos = list(f["data"].keys())
                    log.info(f"  Episodes: {len(demos)}")
                    if demos:
                        demo = f["data"][demos[0]]
                        if "actions" in demo:
                            log.info(f"  Action shape: {demo['actions'].shape}")
                        if "obs" in demo:
                            for k in demo["obs"]:
                                log.info(f"  Obs '{k}': {demo['obs'][k].shape}")
        except ImportError:
            log.error("h5py not installed")

    elif path.is_dir():
        # 检查是否是 checkpoint
        ckpt_path = path / "policy.pt"
        if ckpt_path.exists():
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            log.info(f"Checkpoint: {path}")
            if "config" in ckpt:
                log.info(f"  Policy: {ckpt['config'].get('name', 'unknown')}")
            num_params = sum(p.numel() for p in ckpt.get("model_state_dict", {}).values())
            log.info(f"  Parameters: {num_params:,}")
        else:
            # 可能是数据集目录
            files = list(path.rglob("*"))
            log.info(f"Directory: {path}")
            log.info(f"  Total files: {len(files)}")
            extensions = {}
            for f in files:
                ext = f.suffix
                extensions[ext] = extensions.get(ext, 0) + 1
            for ext, count in sorted(extensions.items(), key=lambda x: -x[1])[:10]:
                log.info(f"  {ext or '(no ext)'}: {count}")
    else:
        log.info(f"File: {path} ({path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
