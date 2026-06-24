#!/usr/bin/env python
"""Collect PIPER PI probe summary.json files into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/piper_pi_probe")
    parser.add_argument("--output", default="results/piper_pi_probe/sweep_summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    summaries = sorted(root.glob("**/summary.json"))
    rows = []
    for path in summaries:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run": path.parent.name,
            "summary_path": path.as_posix(),
            "model_name": data.get("model_name"),
            "backbone": data.get("backbone"),
            "forget_split": data.get("forget_split"),
            "retain_split": data.get("retain_split"),
            "seed": data.get("seed"),
            "forget_batch_size": data.get("forget_batch_size"),
            "probe_batches": data.get("probe_batches"),
            "train_steps": data.get("train_steps"),
            "learning_rate": data.get("learning_rate"),
            "probe_learning_rate": data.get("probe_learning_rate"),
            "pearson": data.get("pearson"),
            "spearman": data.get("spearman"),
            "mean_damage": data.get("mean_damage"),
            "mean_pi": data.get("mean_pi"),
            "train_loss_first": data.get("train_loss_first"),
            "train_loss_last": data.get("train_loss_last"),
            "elapsed_seconds": data.get("elapsed_seconds"),
        }
        for frac, values in data.get("topk", {}).items():
            prefix = f"top{frac}"
            row[f"{prefix}_precision"] = values.get("precision")
            row[f"{prefix}_enrichment"] = values.get("enrichment")
            row[f"{prefix}_pred_damage"] = values.get("pred_topk_mean_damage")
            row[f"{prefix}_random_damage"] = values.get("random_topk_mean_damage")
            row[f"{prefix}_damage_lift"] = values.get("pred_vs_random_damage_lift")
            row[f"{prefix}_damage_ratio"] = values.get("pred_vs_random_damage_ratio")
        rows.append(row)

    if not rows:
        raise SystemExit(f"No summary.json files found under {root}")

    fieldnames = sorted({key for row in rows for key in row})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
