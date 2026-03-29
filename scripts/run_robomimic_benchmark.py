"""RoboMimic Benchmark 脚本。

支持算法：bc, bc_rnn, dp, act, flow_policy, imle
支持任务：Lift, Can
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch
from torch.utils.data import DataLoader

from robocore.config.base import (
    DataConfig,
    EnvConfig,
    ExperimentConfig,
    PolicyConfig,
    TrainConfig,
)
from robocore.data.adapters.hdf5_adapter import HDF5Dataset
from robocore.env.wrappers.robomimic_env import RoboMimicEnv
from robocore.policy.registry import PolicyRegistry
from robocore.trainer.base import BaseTrainer

import robocore.policy.algos  # noqa: F401 — 触发注册

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config(
    algo: str,
    task: str,
    data_path: str,
    device: str,
    obs_dim: int,
    output_dir: str = "outputs/robomimic_benchmark",
) -> ExperimentConfig:
    """根据算法名构建完整实验配置。"""
    task_lower = task.lower()

    # ---------- 各算法的 policy 配置 ----------
    if algo == "bc":
        policy = PolicyConfig(
            name="bc",
            obs_dim=obs_dim,
            action_dim=7,
            obs_horizon=1,
            pred_horizon=1,
            action_horizon=1,
            hidden_dim=256,
            num_layers=3,
            dropout=0.1,
        )
        train = TrainConfig(
            seed=42, device=device, output_dir=output_dir,
            exp_name=f"bc_robomimic_{task_lower}",
            num_epochs=60, steps_per_epoch=100,
            lr=3e-4, weight_decay=1e-6,
            warmup_steps=500, lr_scheduler="cosine",
            use_amp=True, amp_dtype="bf16",
            use_ema=False, log_interval=5, save_interval=10,
            eval_interval=1000, use_wandb=False,
        )
    elif algo == "bc_rnn":
        policy = PolicyConfig(
            name="bc_rnn",
            obs_dim=obs_dim,
            action_dim=7,
            obs_horizon=10,
            pred_horizon=1,
            action_horizon=1,
            hidden_dim=256,
            extra={"rnn_layers": 2},
        )
        train = TrainConfig(
            seed=42, device=device, output_dir=output_dir,
            exp_name=f"bc_rnn_robomimic_{task_lower}",
            num_epochs=60, steps_per_epoch=100,
            lr=1e-4, weight_decay=1e-6,
            warmup_steps=500, lr_scheduler="cosine",
            use_amp=True, amp_dtype="bf16",
            use_ema=False, log_interval=5, save_interval=10,
            eval_interval=1000, use_wandb=False,
        )
    elif algo == "dp":
        policy = PolicyConfig(
            name="dp",
            obs_dim=obs_dim,
            action_dim=7,
            obs_horizon=2,
            pred_horizon=16,
            action_horizon=8,
            hidden_dim=256,
            num_inference_steps=10,
            extra={"down_dims": [256, 512, 1024]},
        )
        train = TrainConfig(
            seed=42, device=device, output_dir=output_dir,
            exp_name=f"dp_robomimic_{task_lower}",
            num_epochs=40, steps_per_epoch=100,
            lr=1e-4, weight_decay=1e-6,
            warmup_steps=500, lr_scheduler="cosine",
            use_amp=True, amp_dtype="bf16",
            use_ema=True, ema_decay=0.995,
            log_interval=5, save_interval=10,
            eval_interval=1000, use_wandb=False,
        )
    elif algo == "flow_policy":
        policy = PolicyConfig(
            name="flow_policy",
            obs_dim=obs_dim,
            action_dim=7,
            obs_horizon=2,
            pred_horizon=16,
            action_horizon=8,
            hidden_dim=256,
            num_inference_steps=10,
            extra={"down_dims": [256, 512, 1024]},
        )
        train = TrainConfig(
            seed=42, device=device, output_dir=output_dir,
            exp_name=f"flow_robomimic_{task_lower}",
            num_epochs=40, steps_per_epoch=100,
            lr=1e-4, weight_decay=1e-6,
            warmup_steps=500, lr_scheduler="cosine",
            use_amp=True, amp_dtype="bf16",
            use_ema=True, ema_decay=0.995,
            log_interval=5, save_interval=10,
            eval_interval=1000, use_wandb=False,
        )
    elif algo == "act":
        policy = PolicyConfig(
            name="act",
            obs_dim=obs_dim,
            action_dim=7,
            obs_horizon=1,
            pred_horizon=16,
            action_horizon=16,
            hidden_dim=512,
            num_layers=4,
            extra={"latent_dim": 32},
        )
        train = TrainConfig(
            seed=42, device=device, output_dir=output_dir,
            exp_name=f"act_robomimic_{task_lower}",
            num_epochs=40, steps_per_epoch=100,
            lr=1e-4, weight_decay=1e-6,
            warmup_steps=500, lr_scheduler="cosine",
            use_amp=True, amp_dtype="bf16",
            use_ema=False, log_interval=5, save_interval=10,
            eval_interval=1000, use_wandb=False,
        )
    elif algo == "imle":
        policy = PolicyConfig(
            name="imle",
            obs_dim=obs_dim,
            action_dim=7,
            obs_horizon=2,
            pred_horizon=16,
            action_horizon=8,
            hidden_dim=256,
            num_inference_steps=1,
            extra={
                "down_dims": [256, 512, 1024],
                "latent_dim": 32,
                "num_samples": 64,
                "noise_scale": 1.0,
            },
        )
        train = TrainConfig(
            seed=42, device=device, output_dir=output_dir,
            exp_name=f"imle_robomimic_{task_lower}",
            num_epochs=40, steps_per_epoch=100,
            lr=1e-4, weight_decay=1e-6,
            warmup_steps=500, lr_scheduler="cosine",
            use_amp=True, amp_dtype="bf16",
            use_ema=True, ema_decay=0.995,
            log_interval=5, save_interval=10,
            eval_interval=1000, use_wandb=False,
        )
    else:
        raise ValueError(f"Unsupported algo: {algo}")

    data = DataConfig(
        dataset_name=f"robomimic_{task_lower}",
        root=data_path,
        format="hdf5",
        obs_horizon=policy.obs_horizon,
        pred_horizon=policy.pred_horizon,
        batch_size=64,
        num_workers=4,
    )
    env = EnvConfig(
        benchmark="robomimic",
        task_name=task,
        max_episode_steps=400,
    )

    return ExperimentConfig(policy=policy, data=data, train=train, env=env)


# ─────────────────────── 评估 ───────────────────────


@torch.no_grad()
def evaluate_policy(
    policy, task: str, num_episodes: int = 20, max_steps: int = 400
) -> dict:
    """评估策略。

    使用 action chunking（action queue）执行完整 action horizon，
    success 用 OR 累积（不覆盖），支持 obs_horizon 历史窗口。
    """
    env = RoboMimicEnv(task_name=task, use_image=False, max_episode_steps=max_steps)
    policy.eval()
    device = next(policy.parameters()).device

    results = []
    for ep in range(num_episodes):
        obs = env.reset(seed=ep)
        # 初始化观测历史窗口
        history: dict[str, list[np.ndarray]] = {
            k: [obs[k].copy() for _ in range(policy.obs_horizon)] for k in obs
        }
        total_reward = 0.0
        success = False  # OR 累积
        action_queue: list[np.ndarray] = []

        for step in range(max_steps):
            # action chunking: 队列空时重新预测
            if len(action_queue) == 0:
                obs_batch: dict[str, torch.Tensor] = {}
                for k, hist in history.items():
                    arr = np.stack(hist[-policy.obs_horizon:], axis=0).astype(
                        np.float32
                    )
                    obs_batch[k] = torch.from_numpy(arr).unsqueeze(0).to(device)

                output = policy.predict(obs_batch)
                actions_chunk = policy.get_action(output)
                action_horizon = getattr(policy, "action_horizon", 1)
                for i in range(min(action_horizon, actions_chunk.shape[1])):
                    action_queue.append(
                        actions_chunk[0, i].detach().cpu().numpy()
                    )

            action = action_queue.pop(0)
            step_result = env.step(action)
            total_reward += float(step_result.reward)

            # success OR 累积（不覆盖）
            if step_result.info.get("success", False):
                success = True

            # 更新观测历史
            obs = step_result.obs
            for k in history:
                history[k].append(obs[k].copy())
                if len(history[k]) > policy.obs_horizon:
                    history[k].pop(0)

            if step_result.done or step_result.truncated:
                break

        results.append(
            {
                "episode": ep,
                "success": success,
                "reward": total_reward,
                "length": step + 1,
            }
        )
        print(
            f"[eval] ep={ep+1:02d}/{num_episodes} "
            f"success={success} reward={total_reward:.3f} length={step+1}"
        )

    env.close()
    return {
        "success_rate": float(np.mean([r["success"] for r in results])),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_length": float(np.mean([r["length"] for r in results])),
        "episodes": results,
    }


# ─────────────────────── main ───────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="RoboMimic Benchmark")
    parser.add_argument(
        "--algo",
        required=True,
        choices=["bc", "bc_rnn", "dp", "act", "flow_policy", "imle"],
    )
    parser.add_argument("--task", required=True, choices=["Lift", "Can"])
    parser.add_argument("--data", required=True, help="HDF5 数据路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--exp-suffix", type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)

    # ── 数据集（先用默认 horizon 探测 obs_dim） ──
    obs_keys = ["state"]
    dataset = HDF5Dataset(
        root=args.data,
        obs_keys=obs_keys,
        obs_horizon=2,
        pred_horizon=16,
    )
    obs_dim = dataset[0]["obs"]["state"].shape[-1]
    dataset.close()
    print(f"[info] data={args.data}")
    print(f"[info] obs_keys={obs_keys}, obs_dim={obs_dim}")

    # ── 配置 ──
    config = build_config(
        algo=args.algo,
        task=args.task,
        data_path=args.data,
        device=args.device,
        obs_dim=obs_dim,
    )

    # CLI 覆盖
    if args.num_epochs is not None:
        config.train.num_epochs = args.num_epochs
    if args.steps_per_epoch is not None:
        config.train.steps_per_epoch = args.steps_per_epoch
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.train.lr = args.lr
    if args.weight_decay is not None:
        config.train.weight_decay = args.weight_decay
    config.train.seed = args.seed
    if args.exp_suffix:
        config.train.exp_name = f"{config.train.exp_name}_{args.exp_suffix}"

    print(f"[info] config exp_name={config.train.exp_name}")

    # ── 数据集重建（用正确的 horizon） ──
    dataset = HDF5Dataset(
        root=args.data,
        obs_keys=obs_keys,
        obs_horizon=config.policy.obs_horizon,
        pred_horizon=config.policy.pred_horizon,
    )
    print(
        f"[info] dataset_len={len(dataset)}, "
        f"train_loader_len={len(dataset) // config.data.batch_size}, "
        f"steps_per_epoch={config.train.steps_per_epoch}"
    )

    train_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── 策略 ──
    extra = dict(config.policy.extra) if config.policy.extra else {}
    policy = PolicyRegistry.create(
        config.policy.name,
        obs_dim=obs_dim,
        action_dim=config.policy.action_dim,
        obs_horizon=config.policy.obs_horizon,
        pred_horizon=config.policy.pred_horizon,
        action_horizon=config.policy.action_horizon,
        device=config.train.device,
        hidden_dim=config.policy.hidden_dim,
        num_layers=config.policy.num_layers,
        num_heads=config.policy.num_heads,
        dropout=config.policy.dropout,
        num_inference_steps=config.policy.num_inference_steps,
        **extra,
    )
    print(f"[info] policy={config.policy.name}, num_params={policy.num_params:,}")

    # ── 训练 ──
    trainer = BaseTrainer(
        config=config,
        policy=policy,
        train_loader=train_loader,
    )

    start_time = time.time()
    train_metrics = trainer.train()
    train_time = time.time() - start_time

    # ── 评估 ──
    eval_policy = trainer.ema_policy if trainer.ema_policy else policy
    eval_result = evaluate_policy(
        eval_policy, args.task, num_episodes=args.eval_episodes
    )

    # ── 保存结果 ──
    config_dict = asdict(config)
    result = {
        "algo": args.algo,
        "task": args.task,
        "data": args.data,
        "obs_keys": obs_keys,
        "obs_dim": obs_dim,
        "train_time_sec": train_time,
        "train_metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in train_metrics.items()
        },
        "eval": eval_result,
        "num_params": policy.num_params,
        "config": config_dict,
    }

    out_dir = os.path.join(config.train.output_dir, config.train.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "benchmark_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("=" * 80)
    print(f"[result] algo={args.algo}, task={args.task}")
    print(f"[result] success_rate={eval_result['success_rate']:.3f}")
    print(f"[result] mean_reward={eval_result['mean_reward']:.3f}")
    print(f"[result] mean_length={eval_result['mean_length']:.1f}")
    print(f"[result] train_time_sec={train_time:.1f}")
    print(f"[result] num_params={policy.num_params:,}")
    print(f"[result] saved_to={result_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
