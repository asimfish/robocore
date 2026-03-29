from __future__ import annotations

from pathlib import Path

from robocore.benchmark_suite import (
    build_suite_jobs,
    build_summary,
    collect_benchmark_records,
    format_command,
    render_summary_markdown,
)


def _write_result(root: Path, *, algo: str, task: str, success_rate: float, mean_reward: float) -> Path:
    exp_dir = root / f"{algo}_robomimic_{task.lower()}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    result_path = exp_dir / "benchmark_result.json"
    result_path.write_text(
        """
{
  "algo": "__ALGO__",
  "task": "__TASK__",
  "data": "data/robomimic/__TASK_LOWER__/ph/low_dim.hdf5",
  "train_time_sec": 12.5,
  "train_metrics": {
    "loss": 0.123,
    "epoch": 10
  },
  "eval": {
    "success_rate": __SUCCESS_RATE__,
    "mean_reward": __MEAN_REWARD__,
    "mean_length": 123.0,
    "episodes": []
  },
  "num_params": 1024,
  "config": {
    "train": {
      "exp_name": "__ALGO___robomimic___TASK_LOWER__"
    }
  }
}
        """.strip()
        .replace("__ALGO__", algo)
        .replace("__TASK__", task)
        .replace("__TASK_LOWER__", task.lower())
        .replace("__SUCCESS_RATE__", str(success_rate))
        .replace("__MEAN_REWARD__", str(mean_reward)),
        encoding="utf-8",
    )
    return result_path


def test_collect_benchmark_records_and_summary(tmp_path: Path) -> None:
    results_root = tmp_path / "outputs" / "robomimic_benchmark"
    _write_result(results_root, algo="bc", task="Lift", success_rate=0.1, mean_reward=0.2)
    _write_result(results_root, algo="dp", task="Can", success_rate=0.8, mean_reward=0.9)

    records = collect_benchmark_records(results_root)
    assert [(record["task"], record["algo"]) for record in records] == [("Can", "dp"), ("Lift", "bc")]

    summary = build_summary(records, results_root)
    assert summary["total_runs"] == 2
    assert summary["tasks"] == ["Can", "Lift"]
    assert summary["algos"] == ["bc", "dp"]
    assert summary["by_task"]["Can"]["best_run"]["algo"] == "dp"
    assert summary["matrix"]["Lift"]["bc"]["success_rate"] == 0.1

    markdown = render_summary_markdown(summary)
    assert "# RoboMimic Benchmark Summary" in markdown
    assert "| Lift | bc | 0.100 | 0.200 |" in markdown
    assert "| Can | dp | 0.800 | 0.900 |" in markdown


def test_build_suite_jobs_creates_expected_commands() -> None:
    jobs = build_suite_jobs(
        tasks=["Lift", "Can"],
        algos=["bc", "dp"],
        output_dir="outputs/robomimic_benchmark",
        data_root="data/robomimic",
        python_executable="python",
        eval_episodes=7,
    )

    assert len(jobs) == 4
    assert jobs[0]["task"] == "Lift"
    assert jobs[0]["algo"] == "bc"
    assert jobs[0]["data_path"] == "data/robomimic/lift/ph/low_dim.hdf5"
    assert jobs[0]["command"] == [
        "python",
        "scripts/run_robomimic_benchmark.py",
        "--algo",
        "bc",
        "--task",
        "Lift",
        "--data",
        "data/robomimic/lift/ph/low_dim.hdf5",
        "--device",
        "cuda",
        "--output-dir",
        "outputs/robomimic_benchmark",
        "--eval-episodes",
        "7",
    ]

    rendered = format_command(jobs[0]["command"], {"CUDA_VISIBLE_DEVICES": "3"})
    assert rendered.startswith("CUDA_VISIBLE_DEVICES=3 python scripts/run_robomimic_benchmark.py")

