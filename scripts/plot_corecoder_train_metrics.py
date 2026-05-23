#!/usr/bin/env python3
"""Plot CoreCoder RL training metrics from the append-only JSONL summary.

Default input is produced by the patched slime actor when
CORECODER_TRAIN_METRICS_FILE is set by scripts/run_qwen3_4b_corecoder_rl_lora.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_METRICS = Path("/root/CoreCoder/results/corecoder_train_metrics.jsonl")
DEFAULT_PNG = Path("/root/CoreCoder/results/corecoder_train_loss.png")
DEFAULT_CSV = Path("/root/CoreCoder/results/corecoder_train_metrics.csv")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"metrics file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            if _to_float(row.get("step")) is None:
                continue
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen: set[str] = set()
    preferred = ["timestamp", "step", "loss", "pg_loss", "ppo_kl", "entropy_loss", "grad_norm", "lr-pg_0"]
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric(rows: list[dict[str, Any]], metric: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt

    points = []
    for row in rows:
        step = _to_float(row.get("step"))
        value = _to_float(row.get(metric))
        if step is None or value is None:
            continue
        points.append((step, value))

    if not points:
        available = sorted({key for row in rows for key in row if _to_float(row.get(key)) is not None})
        raise ValueError(f"metric '{metric}' has no numeric values. Numeric fields: {available}")

    points.sort(key=lambda item: item[0])
    x = [step for step, _ in points]
    y = [value for _, value in points]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(x, y, marker="o", linewidth=1.8, markersize=4)
    plt.xlabel("training step")
    plt.ylabel(metric)
    plt.title(f"CoreCoder RL {metric}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot CoreCoder RL training metrics JSONL.")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS, help="input JSONL metrics file")
    parser.add_argument("--metric", default="loss", help="numeric metric to plot, default: loss")
    parser.add_argument("--out-png", type=Path, default=DEFAULT_PNG, help="output PNG path")
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV, help="output CSV path")
    parser.add_argument("--csv-only", action="store_true", help="only export CSV, skip matplotlib plotting")
    args = parser.parse_args()

    rows = read_jsonl(args.metrics)
    if not rows:
        raise ValueError(f"no metric rows found in {args.metrics}")

    write_csv(rows, args.out_csv)
    print(f"wrote CSV: {args.out_csv}")

    if not args.csv_only:
        try:
            plot_metric(rows, args.metric, args.out_png)
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("matplotlib is not installed; CSV export is still available")
            return 0
        print(f"wrote PNG: {args.out_png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
