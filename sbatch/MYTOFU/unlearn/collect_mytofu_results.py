from pathlib import Path
import json
import re
import math
import os
import numpy as np
import pandas as pd


# =========================
# Basic paths
# =========================

ROOT = Path(os.environ.get(
    "MYTOFU_RESULT_ROOT",
    "/home/zkzhang/unlearning/open-unlearning/saves/unlearn",
))
OUT_DIR = Path(os.environ.get("MYTOFU_RESULT_OUT", str(ROOT)))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = os.environ.get("MYTOFU_RESULT_PATTERN", "mytofu_*_from_full_e10")
SUMMARY_JSON_NAME = "MYTOFU_SUMMARY.json"

# 如果你重新评测到了新目录，用这个
# EVAL_DIR_NAME = "evals_metric_fix_v2"

# 如果你还没有重评，只想收集旧结果，就改成：
EVAL_DIR_NAME = os.environ.get("MYTOFU_EVAL_DIR_NAME", "evals_final")


# =========================
# Known method names
# =========================

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

SYNTHESIS_MODES = [
    "none",
    "pcgrad",
    "sago",
]

EXTRA_METHODS = [
    f"{base}_{synth}"
    for base in ["GradDiff", "NPO", "SimNPO"]
    for synth in SYNTHESIS_MODES
]

KNOWN_METHODS = sorted(EXTRA_METHODS + BASE_METHODS, key=len, reverse=True)


# =========================
# Utilities
# =========================

def safe_load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def harmonic_mean(values, eps=1e-12):
    """
    Safe harmonic mean for metrics in [0, 1].

    If any value is missing or NaN, return NaN.
    Values are clipped into [0, 1].
    Very small values are replaced by eps to avoid division by zero.
    """
    clean = []

    for v in values:
        if v is None:
            return np.nan

        try:
            v = float(v)
        except Exception:
            return np.nan

        if math.isnan(v):
            return np.nan

        v = max(min(v, 1.0), 0.0)
        v = max(v, eps)
        clean.append(v)

    if not clean:
        return np.nan

    return len(clean) / sum(1.0 / v for v in clean)


def parse_task_name(task_name: str):
    """
    Parse task names such as:

    1. Normal methods:
       mytofu_Llama-3.2-1B-Instruct_NPO_from_full_e10
       -> model  = Llama-3.2-1B-Instruct
       -> method = NPO

    2. Gradient synthesis methods:
       mytofu_Llama-3.2-1B-Instruct_NPO_none_from_full_e10
       -> model  = Llama-3.2-1B-Instruct
       -> method = NPO_none

       mytofu_Llama-3.2-1B-Instruct_GradDiff_sago_from_full_e10
       -> model  = Llama-3.2-1B-Instruct
       -> method = GradDiff_sago

    3. Tuned runs:
       mytofu_Llama-3.2-1B-Instruct_NPO_b0p05_a1_g1_lr5em6_e3_s0_from_full_e10
       -> model      = Llama-3.2-1B-Instruct
       -> method     = NPO
       -> tuning_tag = b0p05_a1_g1_lr5em6_e3_s0
    """
    prefix = "mytofu_"
    suffix = "_from_full_e10"

    if not (task_name.startswith(prefix) and task_name.endswith(suffix)):
        return {
            "task_name": task_name,
            "model": "",
            "method": "",
            "tuning_tag": "",
        }

    core = task_name[len(prefix):-len(suffix)]

    # Parse both exact method names and tuned labels that append a tag after
    # the method, e.g. NPO_b0p05_a1_g1_lr5em6_e3_s0.
    for method in KNOWN_METHODS:
        tail = f"_{method}"
        marker = f"_{method}_"
        if core.endswith(tail):
            return {
                "task_name": task_name,
                "model": core[:-len(tail)],
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

    # Fallback: split from the last underscore
    parts = core.rsplit("_", 1)
    if len(parts) == 2:
        model, method = parts
    else:
        model, method = core, ""

    return {
        "task_name": task_name,
        "model": model,
        "method": method,
        "tuning_tag": "",
    }


def normalize_metric_names(row: dict):
    """
    Normalize inconsistent metric names.

    For example:
    retain_Truth_Ratio -> retain_truth_ratio
    """
    if "retain_truth_ratio" not in row and "retain_Truth_Ratio" in row:
        row["retain_truth_ratio"] = row["retain_Truth_Ratio"]

    if "forget_truth_ratio" not in row and "forget_Truth_Ratio" in row:
        row["forget_truth_ratio"] = row["forget_Truth_Ratio"]

    return row


def add_mytofu_scores(df: pd.DataFrame):
    """
    Add aggregate MYTOFU scores.

    MYTOFU_Mem:
        HM(
          1 - extraction_strength,
          1 - exact_memorization,
          1 - forget_Q_A_PARA_Prob,
          1 - forget_truth_ratio
        )

    MYTOFU_Utility:
        HM(
          retain_Q_A_Prob,
          retain_Q_A_ROUGE,
          retain_truth_ratio
        )

    If retain_truth_ratio is not available, compute:
        MYTOFU_Utility_Lite = HM(retain_Q_A_Prob, retain_Q_A_ROUGE)
    """
    if df.empty:
        return df

    df = df.copy()

    # Normalize column name if needed
    if "retain_truth_ratio" not in df.columns and "retain_Truth_Ratio" in df.columns:
        df["retain_truth_ratio"] = df["retain_Truth_Ratio"]

    if "forget_truth_ratio" not in df.columns and "forget_Truth_Ratio" in df.columns:
        df["forget_truth_ratio"] = df["forget_Truth_Ratio"]

    mem_cols = [
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",
    ]

    if all(c in df.columns for c in mem_cols):
        df["MYTOFU_Mem"] = df.apply(
            lambda r: harmonic_mean([
                1.0 - r["extraction_strength"],
                1.0 - r["exact_memorization"],
                1.0 - r["forget_Q_A_PARA_Prob"],
                1.0 - r["forget_truth_ratio"],
            ]),
            axis=1,
        )

    utility_cols = [
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_truth_ratio",
    ]

    if all(c in df.columns for c in utility_cols):
        df["MYTOFU_Utility"] = df.apply(
            lambda r: harmonic_mean([
                r["retain_Q_A_Prob"],
                r["retain_Q_A_ROUGE"],
                r["retain_truth_ratio"],
            ]),
            axis=1,
        )

    lite_cols = [
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
    ]

    if "MYTOFU_Utility" not in df.columns and all(c in df.columns for c in lite_cols):
        df["MYTOFU_Utility_Lite"] = df.apply(
            lambda r: harmonic_mean([
                r["retain_Q_A_Prob"],
                r["retain_Q_A_ROUGE"],
            ]),
            axis=1,
        )

    if all(c in df.columns for c in ["MYTOFU_Mem", "MYTOFU_Utility"]):
        df["BUS"] = df.apply(
            lambda r: harmonic_mean([
                r["MYTOFU_Mem"],
                r["MYTOFU_Utility"],
            ]),
            axis=1,
        )

    if all(c in df.columns for c in ["MYTOFU_Mem", "MYTOFU_Utility_Lite"]):
        df["BUS_Lite"] = df.apply(
            lambda r: harmonic_mean([
                r["MYTOFU_Mem"],
                r["MYTOFU_Utility_Lite"],
            ]),
            axis=1,
        )

    return df


# =========================
# Collection functions
# =========================

def collect_final_results(exp_dir: Path):
    task_name = exp_dir.name
    meta = parse_task_name(task_name)

    final_json = exp_dir / EVAL_DIR_NAME / SUMMARY_JSON_NAME
    data = safe_load_json(final_json)

    if data is None:
        return None

    row = {
        **meta,
        "checkpoint": "final",
        "source_file": str(final_json),
    }

    row.update(data)
    row = normalize_metric_names(row)

    return row


def collect_checkpoint_results(exp_dir: Path):
    task_name = exp_dir.name
    meta = parse_task_name(task_name)

    rows = []

    ckpt_dirs = []
    for ckpt_dir in exp_dir.glob("checkpoint-*"):
        m = re.match(r"^checkpoint-(\d+)$", ckpt_dir.name)
        if m:
            ckpt_num = int(m.group(1))
            ckpt_dirs.append((ckpt_num, ckpt_dir))

    for ckpt_num, ckpt_dir in sorted(ckpt_dirs, key=lambda x: x[0]):
        ckpt_json = ckpt_dir / "evals" / SUMMARY_JSON_NAME
        data = safe_load_json(ckpt_json)

        if data is None:
            continue

        row = {
            **meta,
            "checkpoint": ckpt_num,
            "source_file": str(ckpt_json),
        }

        row.update(data)
        row = normalize_metric_names(row)
        rows.append(row)

    return rows


# =========================
# Output formatting
# =========================

def reorder(df: pd.DataFrame):
    if df.empty:
        return df

    preferred_cols = [
        "task_name",
        "model",
        "method",
        "tuning_tag",
        "checkpoint",

        # Aggregate scores
        "BUS",
        "BUS_Lite",
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",

        # Memorization components
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",

        # Extra forget-side metrics
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "forget_Q_A_PERT_Prob",

        # Utility components
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_truth_ratio",

        # Extra retain-side metrics
        "retain_Q_A_PARA_Prob",
        "retain_Q_A_PERT_Prob",

        # Old inconsistent name, kept for debugging if present
        "retain_Truth_Ratio",

        "epoch",
        "source_file",
    ]

    cols_exist = [c for c in preferred_cols if c in df.columns]
    cols_rest = [c for c in df.columns if c not in cols_exist]

    return df[cols_exist + cols_rest]


def make_last_checkpoint_view(df_ckpt: pd.DataFrame):
    if df_ckpt.empty or "checkpoint" not in df_ckpt.columns:
        return pd.DataFrame()

    df = df_ckpt.copy()
    df["checkpoint_numeric"] = pd.to_numeric(df["checkpoint"], errors="coerce")
    df = df.dropna(subset=["checkpoint_numeric"])

    if df.empty:
        return pd.DataFrame()

    df_last = (
        df.sort_values(["task_name", "checkpoint_numeric"])
        .groupby("task_name", as_index=False)
        .tail(1)
        .sort_values("task_name")
    )

    df_last = df_last.drop(columns=["checkpoint_numeric"])
    return reorder(df_last)


# =========================
# Main
# =========================

def main():
    exp_dirs = sorted([p for p in ROOT.glob(PATTERN) if p.is_dir()])

    final_rows = []
    ckpt_rows = []

    missing_final = []

    for exp_dir in exp_dirs:
        final_row = collect_final_results(exp_dir)

        if final_row is not None:
            final_rows.append(final_row)
        else:
            missing_final.append(exp_dir.name)

        ckpt_rows.extend(collect_checkpoint_results(exp_dir))

    df_final = pd.DataFrame(final_rows)
    df_ckpt = pd.DataFrame(ckpt_rows)

    df_final = add_mytofu_scores(df_final)
    df_ckpt = add_mytofu_scores(df_ckpt)

    df_final = reorder(df_final)
    df_ckpt = reorder(df_ckpt)

    df_last = make_last_checkpoint_view(df_ckpt)

    # Output filenames include eval dir name to avoid overwriting old summaries
    final_csv = OUT_DIR / f"mytofu_all_results_final_{EVAL_DIR_NAME}.csv"
    ckpt_csv = OUT_DIR / f"mytofu_all_results_checkpoints_{EVAL_DIR_NAME}.csv"
    xlsx_file = OUT_DIR / f"mytofu_all_results_{EVAL_DIR_NAME}.xlsx"

    df_final.to_csv(final_csv, index=False, encoding="utf-8-sig")
    df_ckpt.to_csv(ckpt_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_file) as writer:
        df_final.to_excel(writer, sheet_name="final", index=False)
        df_ckpt.to_excel(writer, sheet_name="checkpoints", index=False)

        if not df_last.empty:
            df_last.to_excel(writer, sheet_name="last_checkpoint", index=False)

        if missing_final:
            pd.DataFrame({"missing_final_task_name": missing_final}).to_excel(
                writer,
                sheet_name="missing_final",
                index=False,
            )

    print("Done.")
    print(f"Eval dir:        {EVAL_DIR_NAME}")
    print(f"Final CSV:       {final_csv}")
    print(f"Checkpoint CSV:  {ckpt_csv}")
    print(f"Excel:           {xlsx_file}")
    print()

    if missing_final:
        print(f"[WARN] Missing final summaries in {EVAL_DIR_NAME}: {len(missing_final)}")
        for name in missing_final[:30]:
            print(f"  - {name}")
        if len(missing_final) > 30:
            print(f"  ... and {len(missing_final) - 30} more")
        print()

    print("Final results preview:")
    preview_cols = [c for c in ["task_name", "model", "method", "tuning_tag", "checkpoint", "BUS", "BUS_Lite", "MYTOFU_Mem", "MYTOFU_Utility", "MYTOFU_Utility_Lite"] if c in df_final.columns]
    if preview_cols:
        print(df_final[preview_cols].head(30))
    else:
        print(df_final.head(30))

    print()
    print("Checkpoint results preview:")
    preview_cols = [c for c in ["task_name", "model", "method", "tuning_tag", "checkpoint", "BUS", "BUS_Lite", "MYTOFU_Mem", "MYTOFU_Utility", "MYTOFU_Utility_Lite"] if c in df_ckpt.columns]
    if preview_cols:
        print(df_ckpt[preview_cols].head(30))
    else:
        print(df_ckpt.head(30))


if __name__ == "__main__":
    main()
