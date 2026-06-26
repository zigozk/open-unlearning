#!/usr/bin/env python
"""Collect official OpenUnlearning TOFU eval summaries for PIPER runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", default="results/piper_official_eval")
    parser.add_argument("--intervention-root", default="results/piper_intervention_expanded")
    parser.add_argument("--output", default="results/piper_official_eval/tofu_eval_summary.csv")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def index_intervention_runs(root: Path) -> dict[str, dict]:
    indexed = {}
    for path in root.glob("**/summary.json"):
        data = load_json(path)
        indexed[path.parent.name] = {
            "intervention_run": path.parent.name,
            "intervention_summary_path": path.as_posix(),
            "backbone": data.get("backbone"),
            "method": data.get("method"),
            "kl_lambda": data.get("kl_lambda"),
            "seed": data.get("seed"),
            "forget_split": data.get("forget_split"),
            "retain_split": data.get("retain_split"),
            "saved_model_path": data.get("saved_model_path"),
            "retain_all_mean_damage": (data.get("retain_side") or {}).get("mean_damage"),
            "local_pi_topk_mean_damage": ((data.get("local_damage") or {}).get("pi_topk") or {}).get(
                "mean_damage"
            ),
            "forget_mean_damage": (data.get("forget_side") or {}).get("mean_damage"),
        }
    return indexed


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root)
    intervention_index = index_intervention_runs(Path(args.intervention_root))
    rows = []

    for path in sorted(eval_root.glob("**/TOFU_SUMMARY.json")):
        summary = load_json(path)
        run_name = path.parent.name
        if run_name.endswith("_tofu_eval"):
            run_name = run_name[: -len("_tofu_eval")]
        row = {
            "eval_run": path.parent.name,
            "eval_summary_path": path.as_posix(),
            **intervention_index.get(run_name, {}),
        }
        for key, value in summary.items():
            row[f"official_{key}"] = value
        rows.append(row)

    if not rows:
        raise SystemExit(f"No TOFU_SUMMARY.json files found under {eval_root}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
