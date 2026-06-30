#!/usr/bin/env python
"""Collect Static-PIPER local-KL intervention summary.json files into CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/piper_intervention")
    parser.add_argument("--output", default="results/piper_intervention/intervention_summary.csv")
    return parser.parse_args()


def flatten_subset(row: dict, prefix: str, values: dict | None) -> None:
    values = values or {}
    for key in ["count", "mean_damage", "mean_base_loss", "mean_final_loss", "mean_answer_prob_delta"]:
        row[f"{prefix}_{key}"] = values.get(key)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("**/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "run": path.parent.name,
            "summary_path": path.as_posix(),
            "model_name": data.get("model_name"),
            "forget_split": data.get("forget_split"),
            "retain_split": data.get("retain_split"),
            "backbone": data.get("backbone"),
            "method": data.get("method"),
            "kl_lambda": data.get("kl_lambda"),
            "seed": data.get("seed"),
            "forget_sample_size": data.get("forget_sample_size"),
            "retain_candidate_size": data.get("retain_candidate_size"),
            "forget_batch_size": data.get("forget_batch_size"),
            "probe_batches": data.get("probe_batches"),
            "topk_frac": data.get("topk_frac"),
            "topk_count": data.get("topk_count"),
            "train_steps": data.get("train_steps"),
            "learning_rate": data.get("learning_rate"),
            "probe_learning_rate": data.get("probe_learning_rate"),
            "npo_beta": data.get("npo_beta"),
            "simnpo_beta": data.get("simnpo_beta"),
            "simnpo_delta": data.get("simnpo_delta"),
            "simnpo_gamma": data.get("simnpo_gamma"),
            "backbone_retain_alpha": data.get("backbone_retain_alpha"),
            "optimizer": data.get("optimizer"),
            "gradient_checkpointing": data.get("gradient_checkpointing"),
            "saved_model_path": data.get("saved_model_path"),
        }
        selection = data.get("selection", {})
        for key, value in selection.items():
            row[f"selection_{key}"] = value
        flatten_subset(row, "retain_all", data.get("retain_side"))
        flatten_subset(row, "forget", data.get("forget_side"))
        local_damage = data.get("local_damage", {})
        for name in ["pi_topk", "semantic_topk", "random_topk", "method_selection"]:
            flatten_subset(row, f"local_{name}", local_damage.get(name))
        cost = data.get("cost", {})
        for key, value in cost.items():
            row[f"cost_{key}"] = value
        first = data.get("train_loss_first") or {}
        last = data.get("train_loss_last") or {}
        for key, value in first.items():
            row[f"train_first_{key}"] = value
        for key, value in last.items():
            row[f"train_last_{key}"] = value
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
