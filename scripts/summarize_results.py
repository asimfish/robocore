"""汇总所有已完成的 RoboMimic benchmark 结果。"""
import json
from pathlib import Path

base = Path("outputs/robomimic_benchmark")
results = []
for d in sorted(base.iterdir()):
    f = d / "benchmark_result.json"
    if f.exists():
        r = json.load(open(f))
        results.append(r)

for task in ["Can", "Lift"]:
    print(f"\n{'='*110}")
    print(f"  {task}")
    print(f"{'='*110}")
    print(f"  {'实验名':<55} {'algo':<12} {'SR':<8} {'reward':<10} {'time':<8}")
    print(f"  {'-'*100}")
    tr = sorted(
        [r for r in results if r["task"] == task],
        key=lambda x: (-x["eval"]["success_rate"], -x["eval"]["mean_reward"]),
    )
    for r in tr:
        n = r.get("config", {}).get("train", {}).get("exp_name", "?")
        print(
            f"  {n:<55} {r['algo']:<12} "
            f"{r['eval']['success_rate']:<8.3f} "
            f"{r['eval']['mean_reward']:<10.1f} "
            f"{r['train_time_sec']:<8.0f}s"
        )

print(f"\nTotal: {len(results)} experiments")
