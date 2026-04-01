"""Benchmark suite 工具：批量运行、结果收集、汇总报告。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_benchmark_records(results_root: str | Path) -> list[dict[str, Any]]:
    """收集所有 benchmark 结果。

    扫描 results_root 下所有 benchmark_result.json 文件。

    Returns:
        按 (task, algo) 排序的结果列表
    """
    results_root = Path(results_root)
    records: list[dict[str, Any]] = []

    for result_file in sorted(results_root.rglob("benchmark_result.json")):
        with open(result_file) as f:
            data = json.load(f)
        records.append(data)

    # 按 (task, algo) 排序
    records.sort(key=lambda r: (r.get("task", ""), r.get("algo", "")))
    return records


def build_summary(records: list[dict[str, Any]], results_root: str | Path) -> dict[str, Any]:
    """构建汇总报告。"""
    tasks = sorted(set(r["task"] for r in records))
    algos = sorted(set(r["algo"] for r in records))

    # 按任务分组
    by_task: dict[str, Any] = {}
    for task in tasks:
        task_records = [r for r in records if r["task"] == task]
        best = max(task_records, key=lambda r: r.get("eval", {}).get("success_rate", 0))
        by_task[task] = {
            "runs": task_records,
            "best_run": best,
        }

    # 构建矩阵
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for r in records:
        task = r["task"]
        algo = r["algo"]
        matrix.setdefault(task, {})[algo] = {
            "success_rate": r.get("eval", {}).get("success_rate", 0.0),
            "mean_reward": r.get("eval", {}).get("mean_reward", 0.0),
        }

    return {
        "total_runs": len(records),
        "tasks": tasks,
        "algos": algos,
        "by_task": by_task,
        "matrix": matrix,
        "results_root": str(results_root),
    }


def render_summary_markdown(summary: dict[str, Any]) -> str:
    """渲染 Markdown 格式的汇总报告。"""
    lines: list[str] = []
    lines.append("# RoboMimic Benchmark Summary")
    lines.append("")
    lines.append(f"Total runs: {summary['total_runs']}")
    lines.append("")
    lines.append("| Task | Algo | Success Rate | Mean Reward |")
    lines.append("|------|------|-------------|-------------|")

    for task in summary["tasks"]:
        for algo in summary["algos"]:
            entry = summary["matrix"].get(task, {}).get(algo)
            if entry:
                sr = f"{entry['success_rate']:.3f}"
                mr = f"{entry['mean_reward']:.3f}"
                lines.append(f"| {task} | {algo} | {sr} | {mr} |")

    return "\n".join(lines)


def build_suite_jobs(
    tasks: list[str],
    algos: list[str],
    output_dir: str = "outputs/robomimic_benchmark",
    data_root: str = "data/robomimic",
    python_executable: str = "python",
    eval_episodes: int = 50,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    """构建批量运行的 job 列表。"""
    jobs: list[dict[str, Any]] = []

    for task in tasks:
        for algo in algos:
            data_path = f"{data_root}/{task.lower()}/ph/low_dim.hdf5"
            command = [
                python_executable,
                "scripts/run_robomimic_benchmark.py",
                "--algo", algo,
                "--task", task,
                "--data", data_path,
                "--device", device,
                "--output-dir", output_dir,
                "--eval-episodes", str(eval_episodes),
            ]
            jobs.append({
                "task": task,
                "algo": algo,
                "data_path": data_path,
                "command": command,
            })

    return jobs


def format_command(command: list[str], env_vars: dict[str, str] | None = None) -> str:
    """将命令列表格式化为可执行的字符串。"""
    parts: list[str] = []
    if env_vars:
        for k, v in env_vars.items():
            parts.append(f"{k}={v}")
    parts.extend(command)
    return " ".join(parts)
