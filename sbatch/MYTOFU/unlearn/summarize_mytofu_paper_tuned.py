#!/usr/bin/env python3
"""
Summarize paper-inspired MYTOFU unlearning tuning runs.

This script reads final MYTOFU summaries produced by
mytofu_unlearn_paper_tuned_array.sh and selects the best configuration per
method. It does not train, evaluate, or modify any experiment files.

Default selection metric:
  BUS = HM(MYTOFU_Mem, MYTOFU_Utility)

where:
  MYTOFU_Mem = HM(
      1 - extraction_strength,
      1 - exact_memorization,
      1 - forget_Q_A_PARA_Prob,
      1 - forget_truth_ratio
  )

  MYTOFU_Utility = HM(
      retain_Q_A_Prob,
      retain_Q_A_ROUGE,
      retain_truth_ratio
  )
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_METHODS = [
    "GradAscent",
    "GradDiff",
    "NPO",
    "SimNPO",
    "RMU",
    "UNDIAL",
    "DPO",
    "CEU",
    "PDU",
    "SatImp",
    "WGA",
]
SYNTHESIS_MODES = ["none", "pcgrad", "sago"]
EXTRA_METHODS = [
    f"{base}_{synth}"
    for base in ["GradDiff", "NPO", "SimNPO"]
    for synth in SYNTHESIS_MODES
]
KNOWN_METHODS = sorted(EXTRA_METHODS + BASE_METHODS, key=len, reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize MYTOFU paper-inspired tuning runs."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("saves/unlearn_paper_tuned"),
        help="Directory containing tuned run folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --result-root.",
    )
    parser.add_argument(
        "--eval-dir",
        default="evals_final",
        help="Eval directory inside each run.",
    )
    parser.add_argument(
        "--pattern",
        default="mytofu_*_from_full_e10",
        help="Run directory glob under --result-root.",
    )
    parser.add_argument(
        "--metric",
        default="BUS",
        help="Ranking metric. Common choices: BUS, BUS_Lite, MYTOFU_Mem.",
    )
    parser.add_argument(
        "--min-utility",
        type=float,
        default=None,
        help="Optional filter before best-by-method selection.",
    )
    parser.add_argument(
        "--max-forget-prob",
        type=float,
        default=None,
        help="Optional filter on forget_Q_A_Prob before best-by-method selection.",
    )
    return parser.parse_args()


def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def harmonic_mean(values: list[float], eps: float = 1e-12) -> float:
    clean = []
    for value in values:
        try:
            value = float(value)
        except Exception:
            return np.nan
        if math.isnan(value):
            return np.nan
        value = max(min(value, 1.0), 0.0)
        clean.append(max(value, eps))
    if not clean:
        return np.nan
    return len(clean) / sum(1.0 / value for value in clean)


def parse_task_name(task_name: str) -> dict[str, str]:
    prefix = "mytofu_"
    suffix = "_from_full_e10"
    if not (task_name.startswith(prefix) and task_name.endswith(suffix)):
        return {"task_name": task_name, "model": "", "method": "", "tuning_tag": ""}

    core = task_name[len(prefix) : -len(suffix)]
    for method in KNOWN_METHODS:
        tail = f"_{method}"
        marker = f"_{method}_"
        if core.endswith(tail):
            return {
                "task_name": task_name,
                "model": core[: -len(tail)],
                "method": method,
                "tuning_tag": "",
            }
        if marker in core:
            model, tuning_tag = core.split(marker, 1)
            if tuning_tag:
                return {
                    "task_name": task_name,
                    "model": model,
                    "method": method,
                    "tuning_tag": tuning_tag,
                }

    model, _, method = core.rpartition("_")
    return {
        "task_name": task_name,
        "model": model,
        "method": method,
        "tuning_tag": "",
    }


def normalize_metric_names(row: dict[str, Any]) -> dict[str, Any]:
    if "retain_truth_ratio" not in row and "retain_Truth_Ratio" in row:
        row["retain_truth_ratio"] = row["retain_Truth_Ratio"]
    if "forget_truth_ratio" not in row and "forget_Truth_Ratio" in row:
        row["forget_truth_ratio"] = row["forget_Truth_Ratio"]
    return row


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()

    mem_cols = [
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",
    ]
    if all(col in df.columns for col in mem_cols):
        df["MYTOFU_Mem"] = df.apply(
            lambda row: harmonic_mean(
                [
                    1.0 - row["extraction_strength"],
                    1.0 - row["exact_memorization"],
                    1.0 - row["forget_Q_A_PARA_Prob"],
                    1.0 - row["forget_truth_ratio"],
                ]
            ),
            axis=1,
        )

    utility_cols = ["retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_truth_ratio"]
    if all(col in df.columns for col in utility_cols):
        df["MYTOFU_Utility"] = df.apply(
            lambda row: harmonic_mean(
                [
                    row["retain_Q_A_Prob"],
                    row["retain_Q_A_ROUGE"],
                    row["retain_truth_ratio"],
                ]
            ),
            axis=1,
        )

    lite_cols = ["retain_Q_A_Prob", "retain_Q_A_ROUGE"]
    if all(col in df.columns for col in lite_cols):
        df["MYTOFU_Utility_Lite"] = df.apply(
            lambda row: harmonic_mean(
                [
                    row["retain_Q_A_Prob"],
                    row["retain_Q_A_ROUGE"],
                ]
            ),
            axis=1,
        )

    if all(col in df.columns for col in ["MYTOFU_Mem", "MYTOFU_Utility"]):
        df["BUS"] = df.apply(
            lambda row: harmonic_mean([row["MYTOFU_Mem"], row["MYTOFU_Utility"]]),
            axis=1,
        )

    if all(col in df.columns for col in ["MYTOFU_Mem", "MYTOFU_Utility_Lite"]):
        df["BUS_Lite"] = df.apply(
            lambda row: harmonic_mean(
                [row["MYTOFU_Mem"], row["MYTOFU_Utility_Lite"]]
            ),
            axis=1,
        )

    return df


def collect_rows(result_root: Path, pattern: str, eval_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    missing = []

    for run_dir in sorted(path for path in result_root.glob(pattern) if path.is_dir()):
        summary_path = run_dir / eval_dir / "MYTOFU_SUMMARY.json"
        data = safe_load_json(summary_path)
        if data is None:
            missing.append(
                {
                    "task_name": run_dir.name,
                    "expected_file": str(summary_path),
                }
            )
            continue
        row = {
            **parse_task_name(run_dir.name),
            "source_file": str(summary_path),
        }
        row.update(data)
        rows.append(normalize_metric_names(row))

    return pd.DataFrame(rows), pd.DataFrame(missing)


def reorder(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    preferred = [
        "task_name",
        "model",
        "method",
        "tuning_tag",
        "rank_within_method",
        "BUS",
        "BUS_Lite",
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "forget_Q_A_PERT_Prob",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_truth_ratio",
        "retain_Q_A_PARA_Prob",
        "retain_Q_A_PERT_Prob",
        "source_file",
    ]
    cols = [col for col in preferred if col in df.columns]
    cols += [col for col in df.columns if col not in cols]
    return df[cols]


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or args.result_root
    out_dir.mkdir(parents=True, exist_ok=True)

    df, missing = collect_rows(args.result_root, args.pattern, args.eval_dir)
    df = add_scores(df)
    if df.empty:
        raise ValueError(f"No completed MYTOFU summaries found under {args.result_root}")
    if args.metric not in df.columns:
        raise ValueError(f"Ranking metric not found: {args.metric}")

    rank_df = df.copy()
    rank_df[args.metric] = pd.to_numeric(rank_df[args.metric], errors="coerce")
    rank_df = rank_df.dropna(subset=[args.metric])

    filtered = rank_df.copy()
    if args.min_utility is not None and "MYTOFU_Utility" in filtered.columns:
        filtered = filtered[filtered["MYTOFU_Utility"] >= args.min_utility]
    if args.max_forget_prob is not None and "forget_Q_A_Prob" in filtered.columns:
        filtered = filtered[filtered["forget_Q_A_Prob"] <= args.max_forget_prob]
    if filtered.empty:
        filtered = rank_df.copy()

    rank_df = rank_df.sort_values(["method", args.metric], ascending=[True, False])
    rank_df["rank_within_method"] = (
        rank_df.groupby("method")[args.metric]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    filtered = filtered.sort_values(["method", args.metric], ascending=[True, False])
    filtered["rank_within_method"] = (
        filtered.groupby("method")[args.metric]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    best = (
        filtered.sort_values(["method", "rank_within_method"])
        .groupby("method", as_index=False)
        .head(1)
        .sort_values(args.metric, ascending=False)
    )

    rank_df = reorder(rank_df)
    filtered = reorder(filtered)
    best = reorder(best)

    all_csv = out_dir / f"paper_tuned_all_by_{args.metric}.csv"
    filtered_csv = out_dir / f"paper_tuned_filtered_by_{args.metric}.csv"
    best_csv = out_dir / f"paper_tuned_best_by_method_by_{args.metric}.csv"
    xlsx_path = out_dir / f"paper_tuned_summary_by_{args.metric}.xlsx"

    rank_df.to_csv(all_csv, index=False, encoding="utf-8-sig")
    filtered.to_csv(filtered_csv, index=False, encoding="utf-8-sig")
    best.to_csv(best_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path) as writer:
        rank_df.to_excel(writer, sheet_name="all_ranked", index=False)
        filtered.to_excel(writer, sheet_name="filtered_ranked", index=False)
        best.to_excel(writer, sheet_name="best_by_method", index=False)
        if not missing.empty:
            missing.to_excel(writer, sheet_name="missing", index=False)

    print("Done.")
    print(f"Result root:     {args.result_root}")
    print(f"Ranking metric:  {args.metric}")
    print(f"All CSV:         {all_csv}")
    print(f"Filtered CSV:    {filtered_csv}")
    print(f"Best CSV:        {best_csv}")
    print(f"Excel:           {xlsx_path}")
    if not missing.empty:
        print(f"Missing runs:    {len(missing)}")
    print()
    preview_cols = [
        col
        for col in [
            "method",
            "tuning_tag",
            args.metric,
            "MYTOFU_Mem",
            "MYTOFU_Utility",
            "forget_Q_A_Prob",
            "retain_Q_A_Prob",
        ]
        if col in best.columns
    ]
    print(best[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
