#!/usr/bin/env python3
"""Summarize TOFU unlearn results under results/unlearn."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


METRIC_KEYS = [
    "model_utility",
    "forget_quality",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "extraction_strength",
    "privleak",
]


def parse_task_name(name: str) -> dict[str, str]:
    parts = name.split("_")
    info = {"task_name": name, "model": "", "forget_split": "", "trainer": "", "run_tag": ""}
    if len(parts) < 4 or parts[0] != "tofu":
        return info

    split_idx = next((i for i, part in enumerate(parts) if re.fullmatch(r"forget\d+", part)), -1)
    if split_idx <= 1:
        return info

    info["model"] = "_".join(parts[1:split_idx])
    info["forget_split"] = parts[split_idx]
    if split_idx + 1 < len(parts):
        info["trainer"] = parts[split_idx + 1]
    if split_idx + 2 < len(parts):
        info["run_tag"] = "_".join(parts[split_idx + 2 :])
    return info


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        if value != 0 and (abs(value) < 1e-4 or abs(value) >= 1e5):
            return f"{value:.3e}"
        return f"{value:.6g}"
    return str(value)


def find_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/evals/TOFU_SUMMARY.json")):
        try:
            data = json.loads(summary_path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping unreadable summary {summary_path}: {exc}")
            continue

        run_dir = summary_path.parent.parent

        row: dict[str, Any] = {
            **parse_task_name(run_dir.name),
            "mtime": datetime.fromtimestamp(summary_path.stat().st_mtime).isoformat(timespec="seconds"),
            "summary_path": str(summary_path),
        }
        for key in METRIC_KEYS:
            row[key] = data.get(key)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "forget_split",
        "trainer",
        "run_tag",
        *METRIC_KEYS,
        "mtime",
        "summary_path",
        "task_name",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = ["model", "forget_split", "trainer", *METRIC_KEYS]
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/unlearn", type=Path)
    parser.add_argument("--out-prefix", default="results/unlearn_summary", type=Path)
    args = parser.parse_args()

    rows = find_rows(args.root)
    rows = sorted(rows, key=lambda row: (row["model"], row["forget_split"], row["trainer"], row["task_name"]))

    csv_path = args.out_prefix.with_suffix(".csv")
    md_path = args.out_prefix.with_suffix(".md")
    write_csv(csv_path, rows)
    md_path.write_text(markdown_table(rows) + "\n")

    print(f"Found {len(rows)} final summaries.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print()
    print(markdown_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
