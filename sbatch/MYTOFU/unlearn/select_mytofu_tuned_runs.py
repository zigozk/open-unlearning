#!/usr/bin/env python3
"""
Select representative MYTOFU tuned runs from a collected result CSV.

The selection rule is intentionally simple and transparent:
1. rank runs within each method by BUS when available;
2. otherwise rank by BUS_Lite;
3. keep the full ranking and a top-1-per-method table.

Example:
  MYTOFU_RESULT_ROOT=saves/unlearn_tuned python sbatch/MYTOFU/unlearn/collect_mytofu_results.py
  python sbatch/MYTOFU/unlearn/select_mytofu_tuned_runs.py \
    --csv saves/unlearn_tuned/mytofu_all_results_final_evals_final.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank MYTOFU tuned runs.")
    parser.add_argument("--csv", required=True, type=Path, help="Collected final CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the CSV parent.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Ranking metric. Defaults to BUS, then BUS_Lite, then MYTOFU_Mem.",
    )
    return parser.parse_args()


def choose_metric(df: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested metric not found: {requested}")
        return requested
    for metric in ["BUS", "BUS_Lite", "MYTOFU_Mem"]:
        if metric in df.columns:
            return metric
    raise ValueError("No ranking metric found. Expected BUS, BUS_Lite, or MYTOFU_Mem.")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"No rows found in {args.csv}")

    metric = choose_metric(df, args.metric)
    for col in ["method", "tuning_tag", metric]:
        if col not in df.columns:
            raise ValueError(f"Required column not found: {col}")

    df = df.copy()
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric])
    df = df.sort_values(["method", metric], ascending=[True, False])
    df["rank_within_method"] = df.groupby("method")[metric].rank(
        method="first", ascending=False
    ).astype(int)

    best = (
        df.sort_values(["method", "rank_within_method"])
        .groupby("method", as_index=False)
        .head(1)
        .sort_values(metric, ascending=False)
    )

    keep = [
        "method",
        "tuning_tag",
        "rank_within_method",
        "BUS",
        "BUS_Lite",
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",
        "forget_Q_A_Prob",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "task_name",
        "source_file",
    ]
    keep = [c for c in keep if c in df.columns]

    ranking_csv = out_dir / f"mytofu_tuned_ranking_by_{metric}.csv"
    best_csv = out_dir / f"mytofu_tuned_best_by_method_by_{metric}.csv"
    ranking_xlsx = out_dir / f"mytofu_tuned_selection_by_{metric}.xlsx"

    df[keep].to_csv(ranking_csv, index=False, encoding="utf-8-sig")
    best[keep].to_csv(best_csv, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(ranking_xlsx) as writer:
        df[keep].to_excel(writer, sheet_name="ranking", index=False)
        best[keep].to_excel(writer, sheet_name="best_by_method", index=False)

    print("Done.")
    print(f"Ranking metric: {metric}")
    print(f"Ranking CSV:    {ranking_csv}")
    print(f"Best CSV:       {best_csv}")
    print(f"Excel:          {ranking_xlsx}")
    print()
    print(best[[c for c in ["method", "tuning_tag", metric, "MYTOFU_Mem", "MYTOFU_Utility"] if c in best.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
