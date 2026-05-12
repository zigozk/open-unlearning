#!/usr/bin/env python3
"""
Summarize MYTOFU unlearning tuning status, results, and SLURM errors.

This is a read-only helper for server-side diagnosis. It scans:
  1. completed final eval summaries under saves/unlearn_paper_tuned;
  2. run directories that do not yet have evals_final/MYTOFU_SUMMARY.json;
  3. SLURM .err/.out files under logs.

Example:
  python sbatch/MYTOFU/unlearn/report_mytofu_tuning_status.py

Useful overrides:
  python sbatch/MYTOFU/unlearn/report_mytofu_tuning_status.py \\
    --result-root saves/unlearn_paper_tuned \\
    --logs-dir logs \\
    --out-dir saves/unlearn_paper_tuned/status_report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


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
SYNTH_METHODS = [
    f"{method}_{mode}"
    for method in ["GradDiff", "NPO", "SimNPO"]
    for mode in ["none", "pcgrad", "sago"]
]
KNOWN_METHODS = sorted(BASE_METHODS + SYNTH_METHODS, key=len, reverse=True)

ERROR_PATTERNS = [
    ("cuda_home_missing", ["CUDA_HOME does not exist", "MissingCUDAException"]),
    ("cuda_oom", ["CUDA out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "out of memory"]),
    ("missing_file", ["FileNotFoundError", "No such file or directory", "MODEL_PATH not found"]),
    ("module_missing", ["ModuleNotFoundError", "ImportError"]),
    ("hydra_config", ["ConfigAttributeError", "Missing mandatory value", "Error executing job with overrides"]),
    ("hf_offline_or_cache", ["LocalEntryNotFoundError", "Cannot find the requested files", "TRANSFORMERS_OFFLINE"]),
    ("shell_source_not_found", ["source: not found"]),
    ("slurm_cancelled", ["CANCELLED", "DUE TO TIME LIMIT", "TIME LIMIT"]),
    ("python_traceback", ["Traceback (most recent call last):"]),
    ("runtime_error", ["RuntimeError"]),
    ("value_error", ["ValueError"]),
]
BENIGN_STDERR_PATTERNS = [
    "Calculating text similarity:",
    "Calculating loss:",
    "A decoder-only architecture is being used, but right-padding was detected",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report MYTOFU tuning results and SLURM errors."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("saves/unlearn_paper_tuned"),
        help="Root directory containing unlearning run folders.",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Directory containing SLURM .out/.err files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for report files. Defaults to result-root/status_report.",
    )
    parser.add_argument(
        "--pattern",
        default="mytofu_*_from_full_e10",
        help="Run directory glob under result-root.",
    )
    parser.add_argument(
        "--eval-dir",
        default="evals_final",
        help="Eval directory inside each run directory.",
    )
    parser.add_argument(
        "--metric",
        default="BUS",
        help="Metric used to rank runs. BUS is computed when possible.",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=25,
        help="Number of final non-empty error lines saved per failing log.",
    )
    parser.add_argument(
        "--log-glob",
        default="*.err",
        help="Glob for error logs under logs-dir.",
    )
    parser.add_argument(
        "--include-benign-stderr",
        action="store_true",
        help="Include stderr files that contain only progress bars or warnings.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow writing report files even when no run directories are found.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": f"{type(exc).__name__}: {exc}"}


def harmonic_mean(values: list[float], eps: float = 1e-12) -> float | None:
    clean = []
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            return None
        if math.isnan(numeric):
            return None
        numeric = max(min(numeric, 1.0), 0.0)
        clean.append(max(numeric, eps))
    if not clean:
        return None
    return len(clean) / sum(1.0 / value for value in clean)


def parse_task_name(task_name: str) -> dict[str, str]:
    prefix = "mytofu_"
    suffix = "_from_full_e10"
    if not task_name.startswith(prefix) or not task_name.endswith(suffix):
        return {"model": "", "method": "", "tuning_tag": ""}

    core = task_name[len(prefix) : -len(suffix)]
    for method in KNOWN_METHODS:
        marker = f"_{method}_"
        if marker in core:
            model, tuning_tag = core.split(marker, 1)
            return {"model": model, "method": method, "tuning_tag": tuning_tag}
        tail = f"_{method}"
        if core.endswith(tail):
            return {"model": core[: -len(tail)], "method": method, "tuning_tag": ""}

    model, _, method = core.rpartition("_")
    return {"model": model, "method": method, "tuning_tag": ""}


def add_scores(row: dict[str, Any]) -> dict[str, Any]:
    if "retain_truth_ratio" not in row and "retain_Truth_Ratio" in row:
        row["retain_truth_ratio"] = row["retain_Truth_Ratio"]
    if "forget_truth_ratio" not in row and "forget_Truth_Ratio" in row:
        row["forget_truth_ratio"] = row["forget_Truth_Ratio"]

    mem_keys = [
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",
    ]
    if all(key in row for key in mem_keys):
        row["MYTOFU_Mem"] = harmonic_mean(
            [
                1.0 - float(row["extraction_strength"]),
                1.0 - float(row["exact_memorization"]),
                1.0 - float(row["forget_Q_A_PARA_Prob"]),
                1.0 - float(row["forget_truth_ratio"]),
            ]
        )

    util_keys = ["retain_Q_A_Prob", "retain_Q_A_ROUGE", "retain_truth_ratio"]
    if all(key in row for key in util_keys):
        row["MYTOFU_Utility"] = harmonic_mean([float(row[key]) for key in util_keys])

    util_lite_keys = ["retain_Q_A_Prob", "retain_Q_A_ROUGE"]
    if all(key in row for key in util_lite_keys):
        row["MYTOFU_Utility_Lite"] = harmonic_mean(
            [float(row[key]) for key in util_lite_keys]
        )

    if row.get("MYTOFU_Mem") is not None and row.get("MYTOFU_Utility") is not None:
        row["BUS"] = harmonic_mean([row["MYTOFU_Mem"], row["MYTOFU_Utility"]])
    if row.get("MYTOFU_Mem") is not None and row.get("MYTOFU_Utility_Lite") is not None:
        row["BUS_Lite"] = harmonic_mean([row["MYTOFU_Mem"], row["MYTOFU_Utility_Lite"]])

    return row


def collect_results(
    result_root: Path, pattern: str, eval_dir: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    completed: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    run_dirs = sorted(path for path in result_root.glob(pattern) if path.is_dir())
    for run_dir in run_dirs:
        summary_path = run_dir / eval_dir / "MYTOFU_SUMMARY.json"
        parsed = parse_task_name(run_dir.name)
        data = load_json(summary_path)
        if data is None:
            missing.append(
                {
                    "task_name": run_dir.name,
                    **parsed,
                    "run_dir": str(run_dir),
                    "expected_summary": str(summary_path),
                }
            )
            continue
        row = {
            "task_name": run_dir.name,
            **parsed,
            "run_dir": str(run_dir),
            "summary_file": str(summary_path),
            **data,
        }
        completed.append(add_scores(row))

    return completed, missing, len(run_dirs)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def classify_error(text: str) -> str:
    if not text.strip():
        return "empty_err"
    for label, patterns in ERROR_PATTERNS:
        if any(pattern in text for pattern in patterns):
            return label
    if any(pattern in text for pattern in BENIGN_STDERR_PATTERNS):
        return "benign_eval_stderr"
    return "unknown_error"


def extract_error_excerpt(text: str, tail_lines: int) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    traceback_start = None
    for idx, line in enumerate(lines):
        if "Traceback (most recent call last):" in line:
            traceback_start = idx
    if traceback_start is not None:
        excerpt = lines[traceback_start:]
    else:
        excerpt = lines[-tail_lines:]
    return "\n".join(excerpt[-tail_lines:])


def paired_out_path(err_path: Path) -> Path:
    return err_path.with_suffix(".out")


def collect_errors(
    logs_dir: Path,
    log_glob: str,
    tail_lines: int,
    include_benign_stderr: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for err_path in sorted(logs_dir.glob(log_glob)):
        err_text = read_text(err_path)
        out_path = paired_out_path(err_path)
        out_text = read_text(out_path)
        category = classify_error(err_text)

        if category == "empty_err":
            continue
        if category == "benign_eval_stderr" and not include_benign_stderr:
            continue

        rows.append(
            {
                "err_file": str(err_path),
                "out_file": str(out_path) if out_path.exists() else "",
                "category": category,
                "slurm_array_task_id": first_match(r"SLURM_ARRAY_TASK_ID=([^\n]+)", out_text),
                "method_label": first_match(r"method_label=([^\n]+)", out_text),
                "trainer": first_match(r"trainer=([^\n]+)", out_text),
                "tag": first_match(r"tag=([^\n]+)", out_text),
                "task_name": first_match(r"task_name=([^\n]+)", out_text),
                "cuda_home": first_match(r"CUDA_HOME=([^\n]+)", out_text),
                "nvcc": first_match(r"nvcc=([^\n]+)", out_text),
                "excerpt": extract_error_excerpt(err_text, tail_lines),
            }
        )
    return rows


def sort_by_metric(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> float:
        try:
            value = float(row.get(metric, float("nan")))
        except Exception:
            value = float("nan")
        if math.isnan(value):
            return -float("inf")
        return value

    return sorted(rows, key=key, reverse=True)


def best_by_method(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in sort_by_metric(rows, metric):
        method = str(row.get("method") or "unknown")
        if method not in best:
            best[method] = row
    return sort_by_metric(list(best.values()), metric)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    shown = rows if max_rows is None else rows[:max_rows]
    if not shown:
        return "_None._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(fmt(row.get(col, "")).replace("\n", "<br>") for col in columns) + " |"
        for row in shown
    ]
    return "\n".join([header, sep, *body])


def write_markdown_report(
    path: Path,
    completed: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    best: list[dict[str, Any]],
    metric: str,
    result_root: Path,
    logs_dir: Path,
    run_dir_count: int,
) -> None:
    error_counts = Counter(row["category"] for row in errors)
    method_counts = Counter(row.get("method", "unknown") or "unknown" for row in completed)

    best_cols = [
        "method",
        "tuning_tag",
        metric,
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "forget_Q_A_Prob",
        "retain_Q_A_Prob",
        "task_name",
    ]
    error_cols = [
        "category",
        "slurm_array_task_id",
        "method_label",
        "tag",
        "cuda_home",
        "err_file",
    ]

    lines = [
        "# MYTOFU Tuning Status Report",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- Result root: `{result_root}`",
        f"- Logs dir: `{logs_dir}`",
        f"- Ranking metric: `{metric}`",
        f"- Matched run directories: `{run_dir_count}`",
        f"- Completed final summaries: `{len(completed)}`",
        f"- Run dirs missing final summary: `{len(missing)}`",
        f"- Actionable error logs: `{len(errors)}`",
        "",
        "## Error Categories",
        "",
        markdown_table(
            [{"category": key, "count": value} for key, value in error_counts.most_common()],
            ["category", "count"],
        ),
        "",
        "## Completed Runs By Method",
        "",
        markdown_table(
            [{"method": key, "count": value} for key, value in sorted(method_counts.items())],
            ["method", "count"],
        ),
        "",
        "## Best Completed Result Per Method",
        "",
        markdown_table(best, best_cols),
        "",
        "## Recent/Detected Error Logs",
        "",
        markdown_table(errors, error_cols, max_rows=50),
        "",
        "## Notes",
        "",
        "- `cuda_home_missing` usually means DeepSpeed imported before `CUDA_HOME` was exported.",
        "- `empty_err` files are ignored because they usually indicate no stderr output.",
        "- stderr files containing only tqdm progress bars or tokenizer padding warnings are ignored by default.",
        "- A run is counted as completed only if `evals_final/MYTOFU_SUMMARY.json` exists.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.result_root / "status_report")
    out_dir.mkdir(parents=True, exist_ok=True)

    completed, missing, run_dir_count = collect_results(
        args.result_root, args.pattern, args.eval_dir
    )
    completed_sorted = sort_by_metric(completed, args.metric)
    best = best_by_method(completed, args.metric)
    errors = collect_errors(
        args.logs_dir,
        args.log_glob,
        args.tail_lines,
        include_benign_stderr=args.include_benign_stderr,
    )

    if run_dir_count == 0 and not args.allow_empty:
        raise SystemExit(
            f"No run directories matched {args.result_root / args.pattern}. "
            "Refusing to overwrite report files. Check --result-root/--pattern, "
            "or pass --allow-empty if this is intentional."
        )

    result_cols = [
        "method",
        "tuning_tag",
        args.metric,
        "BUS",
        "BUS_Lite",
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_Prob",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_truth_ratio",
        "task_name",
        "summary_file",
    ]
    missing_cols = ["method", "tuning_tag", "task_name", "run_dir", "expected_summary"]
    error_cols = [
        "category",
        "slurm_array_task_id",
        "method_label",
        "trainer",
        "tag",
        "task_name",
        "cuda_home",
        "nvcc",
        "err_file",
        "out_file",
        "excerpt",
    ]

    write_csv(out_dir / "completed_results.csv", completed_sorted, result_cols)
    write_csv(out_dir / "best_by_method.csv", best, result_cols)
    write_csv(out_dir / "missing_final_summary.csv", missing, missing_cols)
    write_csv(out_dir / "error_logs.csv", errors, error_cols)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "result_root": str(args.result_root),
        "logs_dir": str(args.logs_dir),
        "metric": args.metric,
        "matched_run_dir_count": run_dir_count,
        "completed_count": len(completed),
        "missing_final_summary_count": len(missing),
        "actionable_error_log_count": len(errors),
        "error_category_counts": dict(Counter(row["category"] for row in errors)),
        "completed_by_method": dict(Counter(row.get("method", "unknown") or "unknown" for row in completed)),
    }
    (out_dir / "status_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(
        out_dir / "MYTOFU_tuning_status_report.md",
        completed=completed_sorted,
        missing=missing,
        errors=errors,
        best=best,
        metric=args.metric,
        result_root=args.result_root,
        logs_dir=args.logs_dir,
        run_dir_count=run_dir_count,
    )

    print("Done.")
    print(f"Report:       {out_dir / 'MYTOFU_tuning_status_report.md'}")
    print(f"Completed:    {len(completed)}")
    print(f"Missing eval: {len(missing)}")
    print(f"Error logs:   {len(errors)}")
    if errors:
        print("Error categories:")
        for category, count in Counter(row["category"] for row in errors).most_common():
            print(f"  {category}: {count}")
    if best:
        print("\nBest completed result per method:")
        cols = ["method", "tuning_tag", args.metric, "MYTOFU_Mem", "MYTOFU_Utility"]
        print(markdown_table(best, cols))


if __name__ == "__main__":
    main()
