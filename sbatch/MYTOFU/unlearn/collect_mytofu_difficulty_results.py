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

BASE_DIR = Path(os.environ.get(
    "MYTOFU_BASE_DIR",
    "/home/zkzhang/unlearning/open-unlearning",
))

RESULT_ROOTS = [
    {
        "split_name": "easy",
        "difficulty_rank": 1,
        "expected_difficulty": "easy",
        "root": BASE_DIR / "saves/unlearn_easy",
    },
    {
        "split_name": "medium",
        "difficulty_rank": 2,
        "expected_difficulty": "medium",
        "root": BASE_DIR / "saves/unlearn_medium",
    },
    {
        "split_name": "hard",
        "difficulty_rank": 3,
        "expected_difficulty": "hard",
        "root": BASE_DIR / "saves/unlearn_hard",
    },
    {
        "split_name": "hard_plus",
        "difficulty_rank": 4,
        "expected_difficulty": "hard_plus",
        "root": BASE_DIR / "saves/unlearn_hard_plus",
    },
]

OUT_DIR = Path(os.environ.get(
    "MYTOFU_DIFFICULTY_OUT",
    str(BASE_DIR / "saves/difficulty_summary"),
))
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATTERN = os.environ.get("MYTOFU_RESULT_PATTERN", "mytofu_*_from_full_e10")
SUMMARY_JSON_NAME = "MYTOFU_SUMMARY.json"
EVAL_DIR_NAME = os.environ.get("MYTOFU_EVAL_DIR_NAME", "evals_final")


# =========================
# Known names
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

# 本次补充实验可能包含的方法别名
EXTRA_METHODS = [
    "NPO_sago",
    "NPO_pcgrad",
    "NPO_none",
    "GradDiff_sago",
    "GradDiff_pcgrad",
    "GradDiff_none",
    "SimNPO_sago",
    "SimNPO_pcgrad",
    "SimNPO_none",
]

SYNTHESIS_MODES = [
    "none",
    "pcgrad",
    "sago",
]

KNOWN_METHODS = sorted(EXTRA_METHODS + BASE_METHODS, key=len, reverse=True)


# =========================
# Utilities
# =========================

def safe_load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def harmonic_mean(values, eps=1e-12):
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


def normalize_metric_names(row: dict):
    """
    兼容大小写不一致的 key。
    """
    if "retain_truth_ratio" not in row and "retain_Truth_Ratio" in row:
        row["retain_truth_ratio"] = row["retain_Truth_Ratio"]

    if "forget_truth_ratio" not in row and "forget_Truth_Ratio" in row:
        row["forget_truth_ratio"] = row["forget_Truth_Ratio"]

    return row


def parse_task_name(task_name: str, split_name_from_root: str):
    """
    支持以下目录名：

    mytofu_easy_Llama-3.2-1B-Instruct_NPO_from_full_e10
    -> split_name = easy
    -> model = Llama-3.2-1B-Instruct
    -> method = NPO

    mytofu_hard_plus_Llama-3.2-1B-Instruct_RMU_from_full_e10
    -> split_name = hard_plus
    -> model = Llama-3.2-1B-Instruct
    -> method = RMU

    mytofu_easy_Llama-3.2-1B-Instruct_NPO_sago_from_full_e10
    -> split_name = easy
    -> model = Llama-3.2-1B-Instruct
    -> method = NPO_sago
    """
    prefix = f"mytofu_{split_name_from_root}_"
    suffix = "_from_full_e10"

    if not (task_name.startswith(prefix) and task_name.endswith(suffix)):
        return {
            "task_name": task_name,
            "split_name": split_name_from_root,
            "model": "",
            "method": "",
            "tuning_tag": "",
        }

    core = task_name[len(prefix):-len(suffix)]

    # Tuned task names append a hyperparameter tag after the method name, e.g.
    # mytofu_easy_<model>_NPO_b0p05_a1_g1_lr5em6_e3_s0_from_full_e10.
    for method in KNOWN_METHODS:
        tail = f"_{method}"
        marker = f"_{method}_"
        if core.endswith(tail):
            return {
                "task_name": task_name,
                "split_name": split_name_from_root,
                "model": core[:-len(tail)],
                "method": method,
                "tuning_tag": "",
            }
        if marker in core:
            model, tuning_tag = core.split(marker, 1)
            if tuning_tag:
                return {
                    "task_name": task_name,
                    "split_name": split_name_from_root,
                    "model": model,
                    "method": method,
                    "tuning_tag": tuning_tag,
                }

    # 先匹配 NPO_sago 这类显式额外方法名
    for method in sorted(EXTRA_METHODS, key=len, reverse=True):
        tail = f"_{method}"
        if core.endswith(tail):
            model = core[:-len(tail)]
            return {
                "task_name": task_name,
                "split_name": split_name_from_root,
                "model": model,
                "method": method,
            }

    # 再匹配 <model>_<base_method>_<synthesis_mode>
    for base_method in BASE_METHODS:
        for synth in SYNTHESIS_MODES:
            tail = f"_{base_method}_{synth}"
            if core.endswith(tail):
                model = core[:-len(tail)]
                method = f"{base_method}_{synth}"
                return {
                    "task_name": task_name,
                    "split_name": split_name_from_root,
                    "model": model,
                    "method": method,
                }

    # 再匹配普通 <model>_<base_method>
    for base_method in BASE_METHODS:
        tail = f"_{base_method}"
        if core.endswith(tail):
            model = core[:-len(tail)]
            method = base_method
            return {
                "task_name": task_name,
                "split_name": split_name_from_root,
                "model": model,
                "method": method,
            }

    # 兜底
    parts = core.rsplit("_", 1)
    if len(parts) == 2:
        model, method = parts
    else:
        model, method = core, ""

    return {
        "task_name": task_name,
        "split_name": split_name_from_root,
        "model": model,
        "method": method,
    }


def add_mytofu_scores(df: pd.DataFrame):
    """
    新增：
    - MYTOFU_Mem
    - MYTOFU_Utility
    - MYTOFU_Utility_Lite
    - BUS
    - BUS_Lite

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

    BUS = HM(MYTOFU_Mem, MYTOFU_Utility)
    """
    if df.empty:
        return df

    df = df.copy()

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

    if all(c in df.columns for c in lite_cols):
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

def collect_final_results(exp_dir: Path, root_meta: dict):
    task_name = exp_dir.name
    meta = parse_task_name(task_name, root_meta["split_name"])

    final_json = exp_dir / EVAL_DIR_NAME / SUMMARY_JSON_NAME
    data = safe_load_json(final_json)

    if data is None:
        return None

    row = {
        **meta,
        "difficulty_rank": root_meta["difficulty_rank"],
        "expected_difficulty": root_meta["expected_difficulty"],
        "checkpoint": "final",
        "source_file": str(final_json),
    }

    row.update(data)
    row = normalize_metric_names(row)
    return row


def collect_checkpoint_results(exp_dir: Path, root_meta: dict):
    task_name = exp_dir.name
    meta = parse_task_name(task_name, root_meta["split_name"])

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
            "difficulty_rank": root_meta["difficulty_rank"],
            "expected_difficulty": root_meta["expected_difficulty"],
            "checkpoint": ckpt_num,
            "source_file": str(ckpt_json),
        }

        row.update(data)
        row = normalize_metric_names(row)
        rows.append(row)

    return rows


# =========================
# Formatting and summary views
# =========================

def reorder(df: pd.DataFrame):
    if df.empty:
        return df

    preferred_cols = [
        "split_name",
        "difficulty_rank",
        "expected_difficulty",
        "task_name",
        "model",
        "method",
        "tuning_tag",
        "checkpoint",

        # aggregate scores
        "BUS",
        "BUS_Lite",
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",

        # memorization components
        "extraction_strength",
        "exact_memorization",
        "forget_Q_A_PARA_Prob",
        "forget_truth_ratio",

        # extra forget metrics
        "forget_Q_A_Prob",
        "forget_Q_A_ROUGE",
        "forget_Q_A_PERT_Prob",

        # utility components
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_truth_ratio",

        # extra retain metrics
        "retain_Q_A_PARA_Prob",
        "retain_Q_A_PERT_Prob",
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
        df.sort_values(["split_name", "method", "task_name", "checkpoint_numeric"])
        .groupby(["split_name", "method", "task_name"], as_index=False)
        .tail(1)
        .sort_values(["difficulty_rank", "method"])
    )

    df_last = df_last.drop(columns=["checkpoint_numeric"])
    return reorder(df_last)


def make_pivot_views(df_final: pd.DataFrame):
    views = {}

    if df_final.empty:
        return views

    index_cols = ["split_name", "difficulty_rank", "expected_difficulty"]
    col = "method"

    metrics = [
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",
        "BUS",
        "BUS_Lite",
        "forget_Q_A_Prob",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
    ]

    for metric in metrics:
        if metric not in df_final.columns:
            continue

        pivot = df_final.pivot_table(
            index=index_cols,
            columns=col,
            values=metric,
            aggfunc="first",
        ).reset_index()

        pivot = pivot.sort_values("difficulty_rank")
        views[f"pivot_{metric[:20]}"] = pivot

    return views


def make_method_trend_view(df_final: pd.DataFrame):
    if df_final.empty:
        return pd.DataFrame()

    keep_cols = [
        "split_name",
        "difficulty_rank",
        "expected_difficulty",
        "method",
        "MYTOFU_Mem",
        "MYTOFU_Utility",
        "MYTOFU_Utility_Lite",
        "BUS",
        "BUS_Lite",
        "forget_Q_A_Prob",
        "retain_Q_A_Prob",
        "retain_Q_A_ROUGE",
        "retain_truth_ratio",
    ]

    cols = [c for c in keep_cols if c in df_final.columns]
    df = df_final[cols].copy()
    df = df.sort_values(["method", "difficulty_rank"])
    return df


# =========================
# Main
# =========================

def main():
    final_rows = []
    ckpt_rows = []
    missing_final = []
    missing_roots = []

    for root_meta in RESULT_ROOTS:
        root = root_meta["root"]

        if not root.exists():
            missing_roots.append(str(root))
            continue

        exp_dirs = sorted([p for p in root.glob(PATTERN) if p.is_dir()])

        for exp_dir in exp_dirs:
            final_row = collect_final_results(exp_dir, root_meta)

            if final_row is not None:
                final_rows.append(final_row)
            else:
                missing_final.append({
                    "split_name": root_meta["split_name"],
                    "task_name": exp_dir.name,
                    "expected_file": str(exp_dir / EVAL_DIR_NAME / SUMMARY_JSON_NAME),
                })

            ckpt_rows.extend(collect_checkpoint_results(exp_dir, root_meta))

    df_final = pd.DataFrame(final_rows)
    df_ckpt = pd.DataFrame(ckpt_rows)

    df_final = add_mytofu_scores(df_final)
    df_ckpt = add_mytofu_scores(df_ckpt)

    df_final = reorder(df_final)
    df_ckpt = reorder(df_ckpt)

    df_last = make_last_checkpoint_view(df_ckpt)
    df_trend = make_method_trend_view(df_final)
    pivot_views = make_pivot_views(df_final)

    final_csv = OUT_DIR / f"mytofu_difficulty_final_{EVAL_DIR_NAME}.csv"
    ckpt_csv = OUT_DIR / f"mytofu_difficulty_checkpoints_{EVAL_DIR_NAME}.csv"
    xlsx_file = OUT_DIR / f"mytofu_difficulty_results_{EVAL_DIR_NAME}.xlsx"

    df_final.to_csv(final_csv, index=False, encoding="utf-8-sig")
    df_ckpt.to_csv(ckpt_csv, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_file) as writer:
        df_final.to_excel(writer, sheet_name="final", index=False)
        df_ckpt.to_excel(writer, sheet_name="checkpoints", index=False)

        if not df_last.empty:
            df_last.to_excel(writer, sheet_name="last_checkpoint", index=False)

        if not df_trend.empty:
            df_trend.to_excel(writer, sheet_name="method_trend", index=False)

        for sheet_name, df_view in pivot_views.items():
            # Excel sheet 名最长 31 个字符
            safe_sheet = sheet_name[:31]
            df_view.to_excel(writer, sheet_name=safe_sheet, index=False)

        if missing_final:
            pd.DataFrame(missing_final).to_excel(
                writer,
                sheet_name="missing_final",
                index=False,
            )

        if missing_roots:
            pd.DataFrame({"missing_root": missing_roots}).to_excel(
                writer,
                sheet_name="missing_roots",
                index=False,
            )

    print("Done.")
    print(f"Eval dir:        {EVAL_DIR_NAME}")
    print(f"Final CSV:       {final_csv}")
    print(f"Checkpoint CSV:  {ckpt_csv}")
    print(f"Excel:           {xlsx_file}")

    if missing_roots:
        print()
        print("[WARN] Missing roots:")
        for r in missing_roots:
            print(f"  - {r}")

    if missing_final:
        print()
        print(f"[WARN] Missing final summaries: {len(missing_final)}")
        for item in missing_final[:30]:
            print(f"  - {item['expected_file']}")
        if len(missing_final) > 30:
            print(f"  ... and {len(missing_final) - 30} more")

    print()
    print("Final results preview:")
    preview_cols = [
        c for c in [
            "split_name",
            "difficulty_rank",
            "method",
            "BUS",
            "BUS_Lite",
            "MYTOFU_Mem",
            "MYTOFU_Utility",
            "MYTOFU_Utility_Lite",
        ]
        if c in df_final.columns
    ]

    if preview_cols:
        print(df_final[preview_cols].head(50))
    else:
        print(df_final.head(50))


if __name__ == "__main__":
    main()
