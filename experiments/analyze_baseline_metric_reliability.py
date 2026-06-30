#!/usr/bin/env python3
"""Audit whether aggregate unlearning metrics hide output collapse.

The script works in two modes:

1. Summary mode: read TOFU/MUSE *_SUMMARY.json files or a collected CSV and flag
   cases where forget-side generations already collapsed even though FQ is low.
2. Raw-log mode: read TOFU_EVAL.json / MUSE_EVAL.json files and compute
   per-index retain sensitivity across runs.

It intentionally avoids heavy dependencies so it can run on copied server
leftovers without recreating the training environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SUMMARY_NAMES = ("TOFU_SUMMARY.json", "MUSE_SUMMARY.json", "MYTOFU_SUMMARY.json")
EVAL_NAMES = ("TOFU_EVAL.json", "MUSE_EVAL.json", "MYTOFU_EVAL.json")

BASE_METHODS = (
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
)

SUMMARY_NUMERIC_COLUMNS = (
    "forget_quality",
    "model_utility",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "forget_Truth_Ratio",
    "forget_verbmem_ROUGE",
    "forget_knowmem_ROUGE",
    "forget_gibberish",
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_truth_ratio",
    "retain_Truth_Ratio",
    "retain_knowmem_ROUGE",
    "extraction_strength",
    "exact_memorization",
    "privleak",
)

FORGET_COLLAPSE_METRICS = (
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "forget_truth_ratio",
    "forget_Truth_Ratio",
    "forget_verbmem_ROUGE",
    "forget_knowmem_ROUGE",
)

RETAIN_QUALITY_METRICS = (
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_truth_ratio",
    "retain_Truth_Ratio",
    "retain_knowmem_ROUGE",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze baseline unlearning metrics for FQ/MU reliability."
    )
    parser.add_argument(
        "--summary-csv",
        action="append",
        type=Path,
        default=[],
        help="Collected summary CSV. Can be passed multiple times.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Root containing copied server runs; scanned recursively for summary/eval JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("legacy_thesis_work/baseline_metric_audit"),
        help="Output directory for CSV/Markdown reports.",
    )
    parser.add_argument(
        "--forget-collapse-threshold",
        type=float,
        default=0.05,
        help="Forget-side low-score threshold used to call an output collapsed.",
    )
    parser.add_argument(
        "--forget-collapse-votes",
        type=int,
        default=2,
        help="Number of low forget-side metrics required for collapse.",
    )
    parser.add_argument(
        "--fq-high-threshold",
        type=float,
        default=0.5,
        help="FQ threshold treated as clearly high/good.",
    )
    parser.add_argument(
        "--mu-low-threshold",
        type=float,
        default=0.2,
        help="Model-utility threshold treated as low.",
    )
    parser.add_argument(
        "--retain-reference-eval",
        type=Path,
        default=None,
        help="Optional retain reference eval log for per-index damage estimates.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Rows to keep in top-case CSV files.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def fmt(value: Any, digits: int = 4) -> str:
    val = to_float(value)
    if val is None:
        return ""
    if abs(val) > 0 and (abs(val) < 1e-4 or abs(val) >= 1e5):
        return f"{val:.3e}"
    return f"{val:.{digits}f}"


def parse_run_name(name: str) -> dict[str, str]:
    """Best-effort parsing for TOFU/MUSE/MYTOFU baseline run names."""
    parsed = {
        "model": "",
        "method": "",
        "split": "",
        "tag": "",
        "jobid": "",
        "raw_dir": name,
    }

    if name.endswith("_tofu_eval"):
        name = name[: -len("_tofu_eval")]

    tofu = re.match(
        r"^tofu_(?P<model>.+?)_(?P<split>forget\d+)_(?P<method>.+?)_(?P<tag>[^_]+)_(?P<jobid>\d+)$",
        name,
    )
    if tofu:
        parsed.update(tofu.groupdict())
        return parsed

    muse = re.match(
        r"^muse_(?P<model>.+?)_(?P<split>[^_]+)_(?P<method>.+?)(?:_(?P<tag>[^_]+))?(?:_(?P<jobid>\d+))?$",
        name,
    )
    if muse:
        parsed.update({k: v or "" for k, v in muse.groupdict().items()})
        return parsed

    if name.startswith("mytofu_"):
        core = name[len("mytofu_") :]
        for suffix in ("_from_full_e10", "_from_full"):
            if core.endswith(suffix):
                core = core[: -len(suffix)]
                break
        for method in sorted(BASE_METHODS, key=len, reverse=True):
            marker = f"_{method}"
            if marker in core:
                model, _, tail = core.partition(marker)
                parsed.update({"model": model, "method": method + tail, "split": "mytofu"})
                return parsed

    for method in sorted(BASE_METHODS, key=len, reverse=True):
        marker = f"_{method}"
        if marker in name:
            model, _, tail = name.partition(marker)
            parsed.update({"model": model, "method": method + tail})
            return parsed

    return parsed


def load_summary_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out: dict[str, Any] = dict(row)
            out.setdefault("source_file", str(path))
            for col in SUMMARY_NUMERIC_COLUMNS:
                if col in out:
                    out[col] = to_float(out[col])
            for key, value in parse_run_name(str(out.get("raw_dir") or out.get("run_dir") or "")).items():
                out.setdefault(key, value)
                if not out[key]:
                    out[key] = value
            rows.append(out)
    return rows


def flatten_summary(path: Path) -> dict[str, Any]:
    summary = read_json(path)
    row: dict[str, Any] = {
        "summary_path": str(path),
        "source_file": str(path),
        "run_dir": str(path.parent.parent if path.parent.name.startswith("eval") else path.parent),
    }
    row.update(parse_run_name(Path(row["run_dir"]).name))
    for key, value in summary.items():
        if isinstance(value, dict) and "agg_value" in value:
            row[key] = to_float(value.get("agg_value"))
        else:
            row[key] = to_float(value)
    return row


def find_summary_rows(root: Path | None) -> list[dict[str, Any]]:
    if root is None or not root.exists():
        return []
    rows = []
    for name in SUMMARY_NAMES:
        for path in sorted(root.rglob(name)):
            rows.append(flatten_summary(path))
    return rows


def collapse_votes(row: dict[str, Any], threshold: float) -> tuple[int, list[str]]:
    low = []
    for metric in FORGET_COLLAPSE_METRICS:
        val = to_float(row.get(metric))
        if val is not None and val <= threshold:
            low.append(metric)
    return len(low), low


def annotate_summary_rows(
    rows: list[dict[str, Any]],
    forget_threshold: float,
    collapse_votes_required: int,
    fq_high_threshold: float,
    mu_low_threshold: float,
) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        out = dict(row)
        votes, low_metrics = collapse_votes(out, forget_threshold)
        fq = to_float(out.get("forget_quality"))
        mu = to_float(out.get("model_utility"))

        out["forget_collapse_votes"] = votes
        out["forget_collapse_metrics"] = ";".join(low_metrics)
        out["forget_output_collapsed"] = votes >= collapse_votes_required
        out["fq_high"] = fq is not None and fq >= fq_high_threshold
        out["collapsed_before_high_fq"] = out["forget_output_collapsed"] and not out["fq_high"]
        out["mu_low"] = mu is not None and mu <= mu_low_threshold

        retain_low = []
        for metric in RETAIN_QUALITY_METRICS:
            val = to_float(out.get(metric))
            if val is not None and val <= mu_low_threshold:
                retain_low.append(metric)
        out["retain_low_metrics"] = ";".join(retain_low)
        out["retain_side_low"] = bool(retain_low)
        annotated.append(out)
    return annotated


def value_by_index(metric_data: Any) -> dict[str, Any]:
    if not isinstance(metric_data, dict):
        return {}
    values = metric_data.get("value_by_index") or metric_data.get("values_by_index")
    if isinstance(values, dict):
        return {str(k): v for k, v in values.items()}
    if isinstance(values, list):
        return {str(i): v for i, v in enumerate(values)}
    return {}


def extract_metric_number(entry: Any) -> float | None:
    if isinstance(entry, (int, float, str)):
        return to_float(entry)
    if not isinstance(entry, dict):
        return None
    for key in ("score", "prob", "rougeL", "rouge1", "value"):
        val = to_float(entry.get(key))
        if val is not None:
            return val
    for value in entry.values():
        val = to_float(value)
        if val is not None:
            return val
    return None


def iter_eval_paths(root: Path | None) -> Iterable[Path]:
    if root is None or not root.exists():
        return []
    paths: list[Path] = []
    for name in EVAL_NAMES:
        paths.extend(root.rglob(name))
    return sorted(paths)


def load_reference_values(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None or not path.exists():
        return {}
    data = read_json(path)
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for metric_name in RETAIN_QUALITY_METRICS:
        for idx, entry in value_by_index(data.get(metric_name)).items():
            val = extract_metric_number(entry)
            if val is not None:
                out[metric_name][idx] = val
    return dict(out)


def collect_retain_item_rows(
    root: Path | None,
    reference_values: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for eval_path in iter_eval_paths(root):
        eval_log = read_json(eval_path)
        run_dir = eval_path.parent.parent if eval_path.parent.name.startswith("eval") else eval_path.parent
        run_meta = parse_run_name(run_dir.name)
        for metric_name in RETAIN_QUALITY_METRICS:
            values = value_by_index(eval_log.get(metric_name))
            for idx, entry in values.items():
                val = extract_metric_number(entry)
                if val is None:
                    continue
                ref = reference_values.get(metric_name, {}).get(idx)
                rows.append(
                    {
                        **run_meta,
                        "eval_path": str(eval_path),
                        "metric": metric_name,
                        "index": idx,
                        "value": val,
                        "reference_value": ref,
                        "damage": None if ref is None else ref - val,
                    }
                )
    return rows


def summarize_retain_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["metric"]), str(row["index"]))].append(row)

    out = []
    for (metric, idx), items in grouped.items():
        values = [float(item["value"]) for item in items]
        damages = [float(item["damage"]) for item in items if item.get("damage") is not None]
        methods = sorted({str(item.get("method", "")) for item in items})
        models = sorted({str(item.get("model", "")) for item in items})
        out.append(
            {
                "metric": metric,
                "index": idx,
                "n_runs": len(values),
                "mean_value": statistics.fmean(values),
                "min_value": min(values),
                "max_value": max(values),
                "std_value": statistics.pstdev(values) if len(values) > 1 else 0.0,
                "mean_damage": statistics.fmean(damages) if damages else "",
                "max_damage": max(damages) if damages else "",
                "models": ";".join(models),
                "methods": ";".join(methods),
            }
        )
    out.sort(
        key=lambda row: (
            -float(row["mean_damage"]) if row["mean_damage"] != "" else 0.0,
            float(row["mean_value"]),
            -float(row["std_value"]),
        )
    )
    return out


def group_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(k, "")) for k in keys)].append(row)

    out = []
    for key_values, items in grouped.items():
        record: dict[str, Any] = dict(zip(keys, key_values))
        record["n_runs"] = len(items)
        record["collapsed_runs"] = sum(bool(item.get("forget_output_collapsed")) for item in items)
        record["collapsed_before_high_fq_runs"] = sum(
            bool(item.get("collapsed_before_high_fq")) for item in items
        )
        record["low_mu_runs"] = sum(bool(item.get("mu_low")) for item in items)
        for metric in ("forget_quality", "model_utility", "forget_Q_A_Prob", "forget_Q_A_ROUGE", "forget_truth_ratio"):
            vals = [to_float(item.get(metric)) for item in items]
            clean = [v for v in vals if v is not None]
            if clean:
                record[f"{metric}_mean"] = statistics.fmean(clean)
                record[f"{metric}_min"] = min(clean)
                record[f"{metric}_max"] = max(clean)
        out.append(record)
    out.sort(key=lambda row: (str(row.get(keys[0], "")), str(row.get(keys[-1], ""))))
    return out


def numeric_corr(rows: list[dict[str, Any]], x_col: str, y_col: str) -> float | None:
    pairs = []
    for row in rows:
        x = to_float(row.get(x_col))
        y = to_float(row.get(y_col))
        if x is not None and y is not None:
            pairs.append((x, y))
    if len(pairs) < 2:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "model",
        "method",
        "split",
        "tag",
        "jobid",
        "n_runs",
        "forget_output_collapsed",
        "collapsed_before_high_fq",
        "forget_collapse_votes",
        "forget_collapse_metrics",
        "forget_quality",
        "model_utility",
        "mu_low",
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "forget_truth_ratio",
        "retain_side_low",
        "retain_low_metrics",
        "source_file",
        "summary_path",
    ]
    ordered = [col for col in preferred if col in fieldnames] + [
        col for col in fieldnames if col not in preferred
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int) -> list[str]:
    if not rows:
        return ["No rows."]
    selected = rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        cells = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                cells.append(fmt(val))
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    by_method: list[dict[str, Any]],
    retain_item_summary: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    collapsed = [row for row in rows if row.get("forget_output_collapsed")]
    collapsed_before_high_fq = [row for row in rows if row.get("collapsed_before_high_fq")]
    low_mu = [row for row in rows if row.get("mu_low")]

    corr_lines = []
    for x_col, y_col in (
        ("forget_quality", "forget_Q_A_Prob"),
        ("forget_quality", "forget_Q_A_ROUGE"),
        ("forget_quality", "forget_truth_ratio"),
        ("model_utility", "forget_Q_A_Prob"),
    ):
        corr = numeric_corr(rows, x_col, y_col)
        if corr is not None:
            corr_lines.append(f"- Pearson corr({x_col}, {y_col}) = {corr:.4f}")

    lines = [
        "# Baseline Metric Reliability Audit",
        "",
        "## Configuration",
        "",
        f"- Summary rows: {len(rows)}",
        f"- Forget collapse threshold: <= {args.forget_collapse_threshold}",
        f"- Forget collapse votes required: {args.forget_collapse_votes}",
        f"- High FQ threshold: >= {args.fq_high_threshold}",
        f"- Low MU threshold: <= {args.mu_low_threshold}",
        "",
        "## Main Checks",
        "",
        f"- Forget-side collapsed runs: {len(collapsed)} / {len(rows)}",
        f"- Collapsed before reaching high FQ: {len(collapsed_before_high_fq)} / {len(rows)}",
        f"- Low model_utility runs: {len(low_mu)} / {len(rows)}",
        "",
        "Interpretation: `collapsed_before_high_fq` is the direct test for the concern that FQ can be too strict or too indirect: the forget-side scores already indicate broken outputs, but `forget_quality` has not reached the chosen high-FQ threshold.",
        "",
        "## Correlations",
        "",
        *(corr_lines or ["No metric pairs had enough numeric values."]),
        "",
        "## Method Summary",
        "",
        *markdown_table(
            by_method,
            [
                "method",
                "n_runs",
                "collapsed_runs",
                "collapsed_before_high_fq_runs",
                "low_mu_runs",
                "forget_quality_mean",
                "model_utility_mean",
                "forget_Q_A_Prob_mean",
                "forget_Q_A_ROUGE_mean",
            ],
            args.top_k,
        ),
        "",
        "## Top Collapsed-Before-High-FQ Cases",
        "",
        *markdown_table(
            sorted(
                collapsed_before_high_fq,
                key=lambda row: (
                    -(to_float(row.get("forget_collapse_votes")) or 0),
                    to_float(row.get("forget_Q_A_Prob")) or 0,
                    to_float(row.get("forget_Q_A_ROUGE")) or 0,
                ),
            ),
            [
                "model",
                "method",
                "forget_collapse_votes",
                "forget_collapse_metrics",
                "forget_quality",
                "model_utility",
                "forget_Q_A_Prob",
                "forget_Q_A_ROUGE",
                "forget_truth_ratio",
            ],
            args.top_k,
        ),
        "",
        "## Retain Per-Index Sensitivity",
        "",
    ]

    if retain_item_summary:
        lines.extend(
            [
                f"Raw eval logs found retain value_by_index data for {len(retain_item_summary)} metric/index pairs.",
                "See `retain_item_sensitivity.csv` for the full table. Low `mean_value`, high `std_value`, or high `mean_damage` indicates retain items that are easier to damage.",
                "",
                *markdown_table(
                    retain_item_summary,
                    ["metric", "index", "n_runs", "mean_value", "std_value", "mean_damage", "max_damage"],
                    args.top_k,
                ),
            ]
        )
    else:
        lines.extend(
            [
                "No raw eval logs with retain `value_by_index` were found.",
                "Run again with `--results-root /path/to/server_leftovers_20260624` after copying the raw eval JSONs to enable item-level retain sensitivity.",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for csv_path in args.summary_csv:
        rows.extend(load_summary_csv(csv_path))
    rows.extend(find_summary_rows(args.results_root))

    if not rows:
        raise SystemExit("No summary rows found. Pass --summary-csv and/or --results-root.")

    annotated = annotate_summary_rows(
        rows,
        forget_threshold=args.forget_collapse_threshold,
        collapse_votes_required=args.forget_collapse_votes,
        fq_high_threshold=args.fq_high_threshold,
        mu_low_threshold=args.mu_low_threshold,
    )

    by_method = group_summary(annotated, ("method",))
    by_model_method = group_summary(annotated, ("model", "method"))

    reference_values = load_reference_values(args.retain_reference_eval)
    retain_rows = collect_retain_item_rows(args.results_root, reference_values)
    retain_item_summary = summarize_retain_items(retain_rows)

    collapsed_before_high_fq = [
        row for row in annotated if row.get("collapsed_before_high_fq")
    ]
    collapsed_before_high_fq.sort(
        key=lambda row: (
            -(to_float(row.get("forget_collapse_votes")) or 0),
            to_float(row.get("forget_Q_A_Prob")) or 0,
            to_float(row.get("forget_Q_A_ROUGE")) or 0,
        )
    )

    write_csv(args.out_dir / "summary_annotated.csv", annotated)
    write_csv(args.out_dir / "summary_by_method.csv", by_method)
    write_csv(args.out_dir / "summary_by_model_method.csv", by_model_method)
    write_csv(
        args.out_dir / "collapsed_before_high_fq_cases.csv",
        collapsed_before_high_fq[: args.top_k],
    )
    write_csv(args.out_dir / "retain_item_rows.csv", retain_rows)
    write_csv(args.out_dir / "retain_item_sensitivity.csv", retain_item_summary)
    write_report(args.out_dir / "baseline_metric_reliability_report.md", annotated, by_method, retain_item_summary, args)

    print(f"Wrote audit outputs to {args.out_dir}")
    print(f"Summary rows: {len(annotated)}")
    print(f"Collapsed before high FQ: {len(collapsed_before_high_fq)}")
    print(f"Retain metric/index pairs: {len(retain_item_summary)}")


if __name__ == "__main__":
    main()
