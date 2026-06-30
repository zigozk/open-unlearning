#!/usr/bin/env python
"""Summarize first-round BRIDGE TOFU runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PRIMARY_COLUMNS = [
    "method",
    "seed",
    "model",
    "forget_split",
    "bridge_prior",
    "bridge_lambda_g",
    "bridge_lambda_b",
    "forget_quality",
    "model_utility",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "mean_retain_kl",
    "worst_k_retain_kl",
    "prior_weighted_kl",
    "boundary_dro",
    "prior_seconds",
    "train_dir",
    "eval_dir",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="results/bridge_initial")
    parser.add_argument("--eval-root", default="results/bridge_initial_eval")
    parser.add_argument("--output-csv", default="results/bridge_reports/bridge_initial_summary.csv")
    parser.add_argument("--output-md", default="results/bridge_reports/bridge_initial_report.md")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}


def find_eval_summary(eval_root: Path, task_name: str) -> Optional[Path]:
    direct = eval_root / f"{task_name}_tofu_eval" / "TOFU_SUMMARY.json"
    if direct.exists():
        return direct
    matches = sorted(eval_root.glob(f"*{task_name}*/TOFU_SUMMARY.json"))
    return matches[0] if matches else None


def discover_runs(train_root: Path, eval_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for train_dir in sorted(path for path in train_root.glob("*") if path.is_dir()):
        run_config = load_json(train_dir / "run_config.json")
        bridge_summary = load_json(train_dir / "bridge_summary.json")
        last_metrics = bridge_summary.get("last_metrics", {})
        task_name = run_config.get("task_name", train_dir.name)
        eval_summary_path = find_eval_summary(eval_root, task_name)
        eval_summary = flatten_eval(load_json(eval_summary_path)) if eval_summary_path else {}

        row = {
            "method": run_config.get("method", train_dir.name),
            "seed": run_config.get("seed"),
            "model": run_config.get("model"),
            "forget_split": run_config.get("forget_split"),
            "bridge_prior": bridge_summary.get("bridge_prior", run_config.get("bridge_prior")),
            "bridge_lambda_g": bridge_summary.get("bridge_lambda_g", run_config.get("bridge_lambda_g")),
            "bridge_lambda_b": bridge_summary.get("bridge_lambda_b", run_config.get("bridge_lambda_b")),
            "forget_quality": eval_summary.get("forget_quality"),
            "model_utility": eval_summary.get("model_utility"),
            "forget_truth_ratio": eval_summary.get("forget_truth_ratio"),
            "forget_Q_A_Prob": eval_summary.get("forget_Q_A_Prob"),
            "forget_Q_A_ROUGE": eval_summary.get("forget_Q_A_ROUGE"),
            "mean_retain_kl": last_metrics.get("bridge_mean_retain_kl"),
            "worst_k_retain_kl": last_metrics.get("bridge_worst_k_retain_kl"),
            "prior_weighted_kl": last_metrics.get("bridge_prior_weighted_kl"),
            "boundary_dro": last_metrics.get("bridge_boundary_dro"),
            "prior_seconds": last_metrics.get("bridge_prior_seconds"),
            "train_dir": train_dir.as_posix(),
            "eval_dir": eval_summary_path.parent.as_posix() if eval_summary_path else None,
        }

        for key, value in eval_summary.items():
            row.setdefault(key, value)
        rows.append(row)
    return rows


def ordered_columns(rows: Iterable[Dict[str, Any]]) -> List[str]:
    extra = []
    seen = set(PRIMARY_COLUMNS)
    for row in rows:
        for key in row:
            if key not in seen:
                extra.append(key)
                seen.add(key)
    return PRIMARY_COLUMNS + extra


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, rows: List[Dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BRIDGE Initial Round Summary",
        "",
        f"- Runs found: {len(rows)}",
        f"- CSV: `{csv_path.as_posix()}`",
        "",
    ]
    if not rows:
        lines += ["No runs were found under the requested train root.", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    table_columns = [
        "method",
        "forget_quality",
        "model_utility",
        "mean_retain_kl",
        "worst_k_retain_kl",
        "boundary_dro",
        "prior_seconds",
    ]
    lines.append("| " + " | ".join(table_columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(table_columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in table_columns) + " |")
    lines.append("")
    lines.append("Interpretation rule: compare methods at similar Forget Quality before treating a higher Model Utility as a win.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    eval_root = Path(args.eval_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    rows = discover_runs(train_root, eval_root)
    write_csv(output_csv, rows)
    write_markdown(output_md, rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote report to {output_md}")


if __name__ == "__main__":
    main()
