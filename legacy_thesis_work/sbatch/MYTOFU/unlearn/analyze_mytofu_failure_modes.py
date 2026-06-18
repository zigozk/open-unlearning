#!/usr/bin/env python3
"""
Slot-level and QA-type-wise analysis for MYTOFU fine-grained unlearning.

This script re-aggregates existing MYTOFU per-sample evaluation logs, so it
does not retrain or re-run model inference. It expects each selected run to
contain an eval log like:

    <result_root>/<task_name>/evals_final/MYTOFU_EVAL.json

and joins `value_by_index` entries with the MYTOFU eval jsonl metadata
(`primary_slots`, `qa_type`, `support_fact_ids`).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path("/home/zkzhang/unlearning/open-unlearning")
DEFAULT_RESULT_ROOT = BASE_DIR / "saves/unlearn"
DEFAULT_OUT_DIR = BASE_DIR / "saves/mytofu_failure_modes"
DEFAULT_FORGET_JSONL = Path(
    "/home/zkzhang/unlearning/Create_Data/mini_tofu_custom_slot_hard_plus/eval/forget_eval_perturbed.jsonl"
)
DEFAULT_RETAIN_JSONL = Path(
    "/home/zkzhang/unlearning/Create_Data/mini_tofu_custom_slot_hard_plus/eval/retain_eval_perturbed.jsonl"
)

EVAL_DIR_NAME = "evals_final"
EVAL_JSON_NAME = "MYTOFU_EVAL.json"
DEFAULT_PATTERN = "mytofu_*_from_full_e10"
DEFAULT_METHODS = ["CEU", "NPO", "RMU", "SimNPO_sago", "SimNPO_pcgrad"]

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

FORGET_METRICS = [
    "forget_Q_A_Prob",
    "exact_memorization",
    "extraction_strength",
    "forget_Truth_Ratio",
    "forget_truth_ratio",
]
RETAIN_METRICS = [
    "retain_Q_A_Prob",
    "retain_Q_A_ROUGE",
    "retain_Truth_Ratio",
    "retain_truth_ratio",
]
PLOT_FORGET_METRIC = "forget_Q_A_Prob"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze MYTOFU failure modes by primary slot and QA type."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
        help=f"Root containing MYTOFU unlearning runs. Default: {DEFAULT_RESULT_ROOT}",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Glob for run directories under --result-root. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--eval-dir-name",
        default=EVAL_DIR_NAME,
        help=f"Eval subdirectory name inside each run. Default: {EVAL_DIR_NAME}",
    )
    parser.add_argument(
        "--forget-jsonl",
        type=Path,
        default=DEFAULT_FORGET_JSONL,
        help=f"MYTOFU forget eval jsonl with metadata. Default: {DEFAULT_FORGET_JSONL}",
    )
    parser.add_argument(
        "--retain-jsonl",
        type=Path,
        default=DEFAULT_RETAIN_JSONL,
        help=f"MYTOFU retain eval jsonl with metadata. Default: {DEFAULT_RETAIN_JSONL}",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Directory for CSV/XLSX/figures. Default: {DEFAULT_OUT_DIR}",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help=(
            "Methods to include after parsing task names. "
            f"Default: {' '.join(DEFAULT_METHODS)}"
        ),
    )
    parser.add_argument(
        "--include-all-methods",
        action="store_true",
        help="Ignore --methods and include every parsed method found.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of hardest forget slots to report. Default: 5",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write tables; skip matplotlib figures.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_task_name(task_name: str) -> dict[str, str]:
    prefix = "mytofu_"
    suffix = "_from_full_e10"

    if not (task_name.startswith(prefix) and task_name.endswith(suffix)):
        return {"task_name": task_name, "model": "", "method": ""}

    core = task_name[len(prefix) : -len(suffix)]

    for base_method in BASE_METHODS:
        for synth in SYNTHESIS_MODES:
            tail = f"_{base_method}_{synth}"
            if core.endswith(tail):
                return {
                    "task_name": task_name,
                    "model": core[: -len(tail)],
                    "method": f"{base_method}_{synth}",
                }

    for base_method in BASE_METHODS:
        tail = f"_{base_method}"
        if core.endswith(tail):
            return {
                "task_name": task_name,
                "model": core[: -len(tail)],
                "method": base_method,
            }

    model, _, method = core.rpartition("_")
    return {"task_name": task_name, "model": model, "method": method}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def metadata_by_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    meta = {}
    for idx, row in enumerate(rows):
        item = dict(row)
        item.setdefault("index", idx)
        item["primary_slots"] = [str(x) for x in as_list(item.get("primary_slots"))]
        item["support_fact_ids"] = [str(x) for x in as_list(item.get("support_fact_ids"))]
        item["qa_type"] = str(item.get("qa_type", "unknown"))
        item["local_entanglement"] = local_entanglement(item)
        meta[str(idx)] = item
    return meta


def local_entanglement(row: dict[str, Any]) -> float:
    for key in ["lfrp", "LFRP", "local_lfrp", "slot_lfrp"]:
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except Exception:
                pass
    return float(len(as_list(row.get("support_fact_ids"))))


def extract_metric_value(metric_name: str, metric_entry: dict[str, Any]) -> float | None:
    if metric_entry is None:
        return None
    if "prob" in metric_entry:
        return to_float(metric_entry["prob"])
    if "score" in metric_entry:
        return to_float(metric_entry["score"])
    if "rougeL" in metric_entry:
        return to_float(metric_entry["rougeL"])
    if "rouge1" in metric_entry:
        return to_float(metric_entry["rouge1"])
    if metric_name in metric_entry:
        return to_float(metric_entry[metric_name])
    if len(metric_entry) == 1:
        return to_float(next(iter(metric_entry.values())))
    return None


def to_float(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def normalize_metric_name(name: str) -> str:
    if name == "forget_Truth_Ratio":
        return "forget_truth_ratio"
    if name == "retain_Truth_Ratio":
        return "retain_truth_ratio"
    return name


def iter_metric_rows(
    eval_log: dict[str, Any],
    metric_names: Iterable[str],
    meta: dict[str, dict[str, Any]],
    split: str,
) -> Iterable[dict[str, Any]]:
    for metric_name in metric_names:
        metric_data = eval_log.get(metric_name)
        if metric_data is None:
            continue

        values = metric_data.get("value_by_index")
        if not isinstance(values, dict):
            continue

        metric_col = normalize_metric_name(metric_name)
        for idx, metric_entry in values.items():
            sample = meta.get(str(idx))
            if sample is None:
                continue
            value = extract_metric_value(metric_name, metric_entry)
            if value is None:
                continue
            yield {
                "split": split,
                "index": int(idx),
                "qa_id": sample.get("qa_id", ""),
                "qa_type": sample.get("qa_type", "unknown"),
                "primary_slots": sample.get("primary_slots", []),
                "support_fact_count": len(sample.get("support_fact_ids", [])),
                "local_entanglement": sample.get("local_entanglement", np.nan),
                "metric": metric_col,
                "value": value,
            }


def collect_per_sample_rows(
    run_dir: Path,
    eval_dir_name: str,
    forget_meta: dict[str, dict[str, Any]],
    retain_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    eval_path = run_dir / eval_dir_name / EVAL_JSON_NAME
    if not eval_path.exists():
        return []

    task_meta = parse_task_name(run_dir.name)
    eval_log = read_json(eval_path)
    rows = []
    for row in iter_metric_rows(eval_log, FORGET_METRICS, forget_meta, "forget"):
        rows.append({**task_meta, "source_file": str(eval_path), **row})
    for row in iter_metric_rows(eval_log, RETAIN_METRICS, retain_meta, "retain"):
        rows.append({**task_meta, "source_file": str(eval_path), **row})
    return rows


def explode_slots(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return df_long
    df = df_long.copy()
    df["primary_slots"] = df["primary_slots"].apply(lambda x: x if x else ["unknown"])
    df = df.explode("primary_slots").rename(columns={"primary_slots": "slot"})
    return df


def aggregate_views(df_slots: pd.DataFrame) -> dict[str, pd.DataFrame]:
    views: dict[str, pd.DataFrame] = {}
    if df_slots.empty:
        return views

    group_cols = ["model", "method", "split", "slot", "metric"]
    slot_metric = (
        df_slots.groupby(group_cols, dropna=False)
        .agg(value_mean=("value", "mean"), value_std=("value", "std"), n=("value", "count"))
        .reset_index()
    )
    views["slot_metric_long"] = slot_metric

    slot_wide = (
        slot_metric.pivot_table(
            index=["model", "method", "split", "slot"],
            columns="metric",
            values="value_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    views["slot_metric_wide"] = slot_wide

    qa_metric = (
        df_slots.groupby(["model", "method", "split", "qa_type", "metric"], dropna=False)
        .agg(value_mean=("value", "mean"), value_std=("value", "std"), n=("value", "count"))
        .reset_index()
    )
    views["qa_type_metric_long"] = qa_metric

    qa_wide = (
        qa_metric.pivot_table(
            index=["model", "method", "split", "qa_type"],
            columns="metric",
            values="value_mean",
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    views["qa_type_metric_wide"] = qa_wide

    slot_ent = (
        df_slots[df_slots["split"] == "forget"]
        .groupby(["slot"], dropna=False)
        .agg(
            avg_local_entanglement=("local_entanglement", "mean"),
            avg_support_fact_count=("support_fact_count", "mean"),
            n_samples=("index", "nunique"),
        )
        .reset_index()
    )
    views["slot_entanglement"] = slot_ent

    return views


def build_top_slots(slot_wide: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if slot_wide.empty or PLOT_FORGET_METRIC not in slot_wide.columns:
        return pd.DataFrame()

    forget = slot_wide[slot_wide["split"] == "forget"].copy()
    if forget.empty:
        return pd.DataFrame()

    avg = (
        forget.groupby("slot", dropna=False)[PLOT_FORGET_METRIC]
        .mean()
        .reset_index(name="Avg Forget Prob")
    )

    best_rows = []
    worst_rows = []
    for slot, part in forget.groupby("slot", dropna=False):
        ordered = part.sort_values(PLOT_FORGET_METRIC, ascending=True)
        best_rows.append({"slot": slot, "Best Method": ordered.iloc[0]["method"]})
        worst_rows.append({"slot": slot, "Worst Method": ordered.iloc[-1]["method"]})

    table = avg.merge(pd.DataFrame(best_rows), on="slot", how="left")
    table = table.merge(pd.DataFrame(worst_rows), on="slot", how="left")
    table = table.sort_values("Avg Forget Prob", ascending=False).head(top_k)
    table.insert(0, "Rank", range(1, len(table) + 1))
    table["Possible Reason"] = table["slot"].apply(reason_for_slot)
    return table.rename(columns={"slot": "Slot"})


def reason_for_slot(slot: str) -> str:
    if "residence" in slot:
        return "multi-hop location/history facts are highly entangled"
    if "private" in slot or "contact" in slot or "id" in slot:
        return "sensitive identifier has distinctive token pattern"
    if "profession" in slot or "birth" in slot:
        return "biographical fact tied directly to author profile"
    if "book" in slot or "award" in slot:
        return "literary relation can be recovered through related retained facts"
    return "high residual probability under selected unlearning methods"


def write_outputs(views: dict[str, pd.DataFrame], top_slots: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in views.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    if not top_slots.empty:
        top_slots.to_csv(out_dir / "top_hardest_slots.csv", index=False, encoding="utf-8-sig")

    xlsx_path = out_dir / "mytofu_failure_modes.xlsx"
    with pd.ExcelWriter(xlsx_path) as writer:
        for name, df in views.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
        if not top_slots.empty:
            top_slots.to_excel(writer, sheet_name="top_hardest_slots", index=False)


def configure_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except ModuleNotFoundError as e:
        raise RuntimeError("matplotlib is required for plots; re-run with --no-plots to skip.") from e

    preferred_fonts = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for font in preferred_fonts:
        if font in available:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            break
    return plt


def plot_heatmap(slot_wide: pd.DataFrame, out_dir: Path) -> None:
    if slot_wide.empty or PLOT_FORGET_METRIC not in slot_wide.columns:
        return
    plt = configure_matplotlib()
    forget = slot_wide[slot_wide["split"] == "forget"]
    pivot = forget.pivot_table(
        index="slot", columns="method", values=PLOT_FORGET_METRIC, aggfunc="mean"
    )
    if pivot.empty:
        return
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig_h = max(4.5, 0.38 * len(pivot.index) + 1.5)
    fig_w = max(6.5, 1.2 * len(pivot.columns) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=180)
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Forget Slot x Method Leakage")
    ax.set_xlabel("Method")
    ax.set_ylabel("Forget Slot")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(PLOT_FORGET_METRIC)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iat[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_forget_slot_method_heatmap.png")
    fig.savefig(out_dir / "fig1_forget_slot_method_heatmap.pdf")
    plt.close(fig)


def plot_qa_type_bars(qa_wide: pd.DataFrame, out_dir: Path) -> None:
    if qa_wide.empty or PLOT_FORGET_METRIC not in qa_wide.columns:
        return
    plt = configure_matplotlib()
    forget = qa_wide[
        (qa_wide["split"] == "forget") & (qa_wide["qa_type"].isin(["single_fact", "composite"]))
    ].copy()
    pivot = forget.pivot_table(
        index="method", columns="qa_type", values=PLOT_FORGET_METRIC, aggfunc="mean"
    )
    if pivot.empty:
        return
    for col in ["single_fact", "composite"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["single_fact", "composite"]]

    x = np.arange(len(pivot.index))
    width = 0.36
    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(x) + 2), 4.8), dpi=180)
    ax.bar(x - width / 2, pivot["single_fact"], width, label="single_fact", color="#3B82F6")
    ax.bar(x + width / 2, pivot["composite"], width, label="composite", color="#F97316")

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel(PLOT_FORGET_METRIC)
    ax.set_title("single_fact vs composite Forget Leakage")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_qa_type_forget_leakage.png")
    fig.savefig(out_dir / "fig2_qa_type_forget_leakage.pdf")
    plt.close(fig)


def plot_entanglement_scatter(
    slot_wide: pd.DataFrame, slot_entanglement: pd.DataFrame, out_dir: Path
) -> None:
    if slot_wide.empty or slot_entanglement.empty or PLOT_FORGET_METRIC not in slot_wide.columns:
        return
    plt = configure_matplotlib()
    forget = slot_wide[slot_wide["split"] == "forget"].merge(
        slot_entanglement[["slot", "avg_local_entanglement"]],
        on="slot",
        how="left",
    )
    if forget.empty:
        return

    methods = list(dict.fromkeys(forget["method"].tolist()))
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=180)
    for idx, method in enumerate(methods):
        part = forget[forget["method"] == method]
        ax.scatter(
            part["avg_local_entanglement"],
            part[PLOT_FORGET_METRIC],
            s=48,
            alpha=0.78,
            label=method,
            color=cmap(idx % 10),
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in part.iterrows():
            ax.annotate(
                str(row["slot"]),
                (row["avg_local_entanglement"], row[PLOT_FORGET_METRIC]),
                textcoords="offset points",
                xytext=(3, 3),
                fontsize=6,
                alpha=0.72,
            )

    ax.set_xlabel("Slot avg local entanglement (LFRP if present, else support fact count)")
    ax.set_ylabel(PLOT_FORGET_METRIC)
    ax.set_title("Slot Entanglement vs Forget Leakage")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_slot_entanglement_scatter.png")
    fig.savefig(out_dir / "fig3_slot_entanglement_scatter.pdf")
    plt.close(fig)


def plot_all(views: dict[str, pd.DataFrame], out_dir: Path) -> None:
    plot_heatmap(views.get("slot_metric_wide", pd.DataFrame()), out_dir)
    plot_qa_type_bars(views.get("qa_type_metric_wide", pd.DataFrame()), out_dir)
    plot_entanglement_scatter(
        views.get("slot_metric_wide", pd.DataFrame()),
        views.get("slot_entanglement", pd.DataFrame()),
        out_dir,
    )


def print_summary(views: dict[str, pd.DataFrame], top_slots: pd.DataFrame, out_dir: Path) -> None:
    slot_wide = views.get("slot_metric_wide", pd.DataFrame())
    qa_wide = views.get("qa_type_metric_wide", pd.DataFrame())
    methods = sorted(slot_wide["method"].dropna().unique().tolist()) if not slot_wide.empty else []

    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Methods: {', '.join(methods) if methods else '(none)'}")
    print()

    if not top_slots.empty:
        print("Top hardest forget slots:")
        print(top_slots.to_string(index=False))
        print()

    if not qa_wide.empty and PLOT_FORGET_METRIC in qa_wide.columns:
        qa = qa_wide[
            (qa_wide["split"] == "forget") & (qa_wide["qa_type"].isin(["single_fact", "composite"]))
        ][["method", "qa_type", PLOT_FORGET_METRIC]]
        if not qa.empty:
            print("QA type forget leakage:")
            print(qa.sort_values(["method", "qa_type"]).to_string(index=False))


def main() -> None:
    args = parse_args()
    if not args.forget_jsonl.exists():
        raise FileNotFoundError(f"Forget jsonl not found: {args.forget_jsonl}")
    if not args.retain_jsonl.exists():
        raise FileNotFoundError(f"Retain jsonl not found: {args.retain_jsonl}")
    if not args.result_root.exists():
        raise FileNotFoundError(f"Result root not found: {args.result_root}")

    forget_meta = metadata_by_index(read_jsonl(args.forget_jsonl))
    retain_meta = metadata_by_index(read_jsonl(args.retain_jsonl))

    selected = set(args.methods or [])
    all_rows = []
    missing_eval_logs = []
    skipped_methods = defaultdict(int)

    for run_dir in sorted(p for p in args.result_root.glob(args.pattern) if p.is_dir()):
        method = parse_task_name(run_dir.name)["method"]
        if not args.include_all_methods and method not in selected:
            skipped_methods[method] += 1
            continue
        rows = collect_per_sample_rows(run_dir, args.eval_dir_name, forget_meta, retain_meta)
        if not rows:
            missing_eval_logs.append(str(run_dir / args.eval_dir_name / EVAL_JSON_NAME))
            continue
        all_rows.extend(rows)

    if not all_rows:
        print("[WARN] No per-sample rows collected.")
        if skipped_methods:
            print(f"Skipped methods: {dict(skipped_methods)}")
        if missing_eval_logs:
            print("Missing eval logs:")
            for path in missing_eval_logs[:30]:
                print(f"  - {path}")
        return

    df_long = pd.DataFrame(all_rows)
    df_slots = explode_slots(df_long)
    views = aggregate_views(df_slots)
    views["per_sample_metric_long"] = df_long

    top_slots = build_top_slots(views.get("slot_metric_wide", pd.DataFrame()), args.top_k)
    write_outputs(views, top_slots, args.out_dir)

    if not args.no_plots:
        plot_all(views, args.out_dir)

    if missing_eval_logs:
        pd.DataFrame({"missing_eval_log": missing_eval_logs}).to_csv(
            args.out_dir / "missing_eval_logs.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print_summary(views, top_slots, args.out_dir)


if __name__ == "__main__":
    main()
