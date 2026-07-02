#!/usr/bin/env python
"""Build a detailed diagnostic report for BRIDGE initial runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ARRAY_LOG_PATTERN = re.compile(r"bridge_initial-(?P<job>\d+)_(?P<task>\d+)\.(?P<kind>out|err)$")
TASK_PATTERN = re.compile(r"BRIDGE_TOFU_.+?(?=_tofu_eval$|$)")
KEY_VALUE_PATTERN = re.compile(r"^([A-Z0-9_]+)=(.*)$")
ERROR_PATTERNS = [
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)),
    ("killed", re.compile(r"\bKilled\b|oom-kill|cgroup.*out of memory", re.I)),
    ("hydra_config", re.compile(r"hydra\.errors|MissingConfigException|ConfigCompositionException", re.I)),
    ("model_path", re.compile(r"does not appear to have a file named|is not a local folder|Repository Not Found", re.I)),
    ("traceback", re.compile(r"Traceback \(most recent call last\):", re.I)),
    ("runtime_error", re.compile(r"RuntimeError:", re.I)),
    ("value_error", re.compile(r"ValueError:", re.I)),
]

PRIMARY_COLUMNS = [
    "task_name",
    "model",
    "method",
    "status",
    "eval_hash_group_size",
    "tofu_eval_hash",
    "tofu_summary_hash",
    "model_utility",
    "forget_quality",
    "forget_truth_ratio",
    "forget_Q_A_Prob",
    "forget_Q_A_ROUGE",
    "extraction_strength",
    "privleak",
    "bridge_prior",
    "bridge_lambda_g",
    "bridge_lambda_b",
    "bridge_summary_exists",
    "bridge_diagnostic_rows",
    "mean_retain_kl",
    "worst_k_retain_kl",
    "prior_weighted_kl",
    "boundary_dro",
    "prior_seconds",
    "trainer_global_step",
    "trainer_train_loss",
    "train_batch_size",
    "grad_accum_steps",
    "num_train_epochs",
    "learning_rate",
    "train_dir",
    "eval_dir",
    "out_log",
    "err_log",
    "error_type",
    "first_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", default="results/bridge_initial")
    parser.add_argument("--eval-root", default="results/bridge_initial_eval")
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--output-csv", default="results/bridge_reports/bridge_debug_summary.csv")
    parser.add_argument("--output-md", default="results/bridge_reports/bridge_debug_report.md")
    return parser.parse_args()


def read_text(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def file_hash(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for _ in handle)


def flatten_eval(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in summary.items() if not isinstance(value, (dict, list))}


def task_from_name(name: str) -> str:
    match = TASK_PATTERN.search(name)
    return match.group(0) if match else name


def method_from_task(task_name: str) -> str:
    match = re.search(r"_(?P<method>(?:npo|npo_global_kl|bridge_.+?))_seed\d+_", task_name)
    return match.group("method") if match else ""


def model_from_task(task_name: str) -> str:
    prefix = "BRIDGE_TOFU_forget10_"
    if not task_name.startswith(prefix):
        return ""
    middle = task_name[len(prefix) :]
    method = method_from_task(task_name)
    if method and f"_{method}_seed" in middle:
        return middle.split(f"_{method}_seed", 1)[0]
    return ""


def parse_log_metadata(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in text.splitlines():
        match = KEY_VALUE_PATTERN.match(line.strip())
        if match:
            metadata[match.group(1).lower()] = match.group(2).strip()
    if "task_name" not in metadata:
        done_match = re.search(r"===== DONE: (?P<task>.+?) =====", text)
        if done_match:
            metadata["task_name"] = done_match.group("task").strip()
    return metadata


def classify_error(text: str) -> tuple[str, str]:
    labels = []
    first_error = ""
    for line in text.splitlines():
        for label, pattern in ERROR_PATTERNS:
            if pattern.search(line):
                if label not in labels:
                    labels.append(label)
                if not first_error:
                    first_error = line.strip()
    if labels:
        return "+".join(labels), first_error
    if "===== DONE:" in text:
        return "none", ""
    if "[skip]" in text:
        return "skipped", ""
    return ("unknown" if text else "", "")


def discover_logs(logs_root: Path) -> Dict[str, Dict[str, Any]]:
    by_task: Dict[str, Dict[str, Any]] = {}
    if not logs_root.exists():
        return by_task

    grouped: Dict[str, Dict[str, Path]] = {}
    for path in sorted(logs_root.glob("bridge_initial-*_*.?*")):
        match = ARRAY_LOG_PATTERN.match(path.name)
        if not match:
            continue
        key = f"{match.group('job')}_{match.group('task')}"
        grouped.setdefault(key, {})[match.group("kind")] = path

    for parts in grouped.values():
        out_path = parts.get("out")
        err_path = parts.get("err")
        out_text = read_text(out_path)
        err_text = read_text(err_path)
        metadata = parse_log_metadata(out_text)
        task_name = metadata.get("task_name")
        if not task_name:
            continue
        error_type, first_error = classify_error("\n".join([out_text, err_text]))
        by_task[task_name] = {
            "out_log": out_path.as_posix() if out_path else "",
            "err_log": err_path.as_posix() if err_path else "",
            "error_type": error_type,
            "first_error": first_error,
            "log_method": metadata.get("method", ""),
            "log_model": metadata.get("model_config", ""),
        }
    return by_task


def discover_task_names(train_root: Path, eval_root: Path) -> List[str]:
    task_names = set()
    if train_root.exists():
        for train_dir in train_root.iterdir():
            if train_dir.is_dir():
                config = load_json(train_dir / "run_config.json")
                task_names.add(config.get("task_name") or train_dir.name)
    if eval_root.exists():
        for eval_dir in eval_root.iterdir():
            if eval_dir.is_dir():
                task_names.add(task_from_name(eval_dir.name))
    return sorted(task_names)


def find_train_dir(train_root: Path, task_name: str) -> Optional[Path]:
    direct = train_root / task_name
    if direct.exists():
        return direct
    matches = sorted(path for path in train_root.glob(f"*{task_name}*") if path.is_dir())
    return matches[0] if matches else None


def find_eval_dir(eval_root: Path, task_name: str) -> Optional[Path]:
    direct = eval_root / f"{task_name}_tofu_eval"
    if direct.exists():
        return direct
    matches = sorted(path for path in eval_root.glob(f"*{task_name}*") if path.is_dir())
    return matches[0] if matches else None


def trainer_state_summary(train_dir: Optional[Path]) -> Dict[str, Any]:
    state = load_json(train_dir / "trainer_state.json") if train_dir else {}
    log_history = state.get("log_history") or []
    train_loss = ""
    for item in reversed(log_history):
        if "train_loss" in item:
            train_loss = item.get("train_loss")
            break
    return {
        "trainer_global_step": state.get("global_step", ""),
        "trainer_train_loss": train_loss,
    }


def build_rows(train_root: Path, eval_root: Path, logs_root: Path) -> List[Dict[str, Any]]:
    log_info = discover_logs(logs_root)
    rows: List[Dict[str, Any]] = []

    for task_name in discover_task_names(train_root, eval_root):
        train_dir = find_train_dir(train_root, task_name)
        eval_dir = find_eval_dir(eval_root, task_name)
        run_config = load_json(train_dir / "run_config.json") if train_dir else {}
        bridge_summary = load_json(train_dir / "bridge_summary.json") if train_dir else {}
        diagnostics_path = train_dir / "bridge_diagnostics.jsonl" if train_dir else None
        eval_summary_path = eval_dir / "TOFU_SUMMARY.json" if eval_dir else None
        eval_detail_path = eval_dir / "TOFU_EVAL.json" if eval_dir else None
        eval_summary = flatten_eval(load_json(eval_summary_path))
        last_metrics = bridge_summary.get("last_metrics", {})
        state_summary = trainer_state_summary(train_dir)
        logs = log_info.get(task_name, {})

        bridge_summary_exists = bool(train_dir and (train_dir / "bridge_summary.json").exists())
        bridge_diagnostic_rows = count_lines(diagnostics_path) if diagnostics_path else 0
        status = "eval_complete" if eval_summary_path and eval_summary_path.exists() else "eval_missing"
        if not train_dir:
            status = "train_missing_" + status

        method = run_config.get("method") or logs.get("log_method") or method_from_task(task_name)
        model = run_config.get("model") or logs.get("log_model") or model_from_task(task_name)
        row: Dict[str, Any] = {
            "task_name": task_name,
            "model": model,
            "method": method,
            "status": status,
            "tofu_eval_hash": file_hash(eval_detail_path),
            "tofu_summary_hash": file_hash(eval_summary_path),
            "bridge_prior": bridge_summary.get("bridge_prior", run_config.get("bridge_prior", "")),
            "bridge_lambda_g": bridge_summary.get("bridge_lambda_g", run_config.get("bridge_lambda_g", "")),
            "bridge_lambda_b": bridge_summary.get("bridge_lambda_b", run_config.get("bridge_lambda_b", "")),
            "bridge_summary_exists": bridge_summary_exists,
            "bridge_diagnostic_rows": bridge_diagnostic_rows,
            "mean_retain_kl": last_metrics.get("bridge_mean_retain_kl", ""),
            "worst_k_retain_kl": last_metrics.get("bridge_worst_k_retain_kl", ""),
            "prior_weighted_kl": last_metrics.get("bridge_prior_weighted_kl", ""),
            "boundary_dro": last_metrics.get("bridge_boundary_dro", ""),
            "prior_seconds": last_metrics.get("bridge_prior_seconds", ""),
            "train_batch_size": run_config.get("train_batch_size", ""),
            "grad_accum_steps": run_config.get("grad_accum_steps", ""),
            "num_train_epochs": run_config.get("num_train_epochs", ""),
            "learning_rate": run_config.get("learning_rate", ""),
            "train_dir": train_dir.as_posix() if train_dir else "",
            "eval_dir": eval_dir.as_posix() if eval_dir else "",
            "out_log": logs.get("out_log", ""),
            "err_log": logs.get("err_log", ""),
            "error_type": logs.get("error_type", ""),
            "first_error": logs.get("first_error", ""),
        }
        row.update(state_summary)
        row.update({key: eval_summary.get(key, "") for key in [
            "model_utility",
            "forget_quality",
            "forget_truth_ratio",
            "forget_Q_A_Prob",
            "forget_Q_A_ROUGE",
            "extraction_strength",
            "privleak",
        ]})
        rows.append(row)

    eval_hash_counts: Dict[str, int] = {}
    for row in rows:
        digest = row.get("tofu_eval_hash") or ""
        if digest:
            eval_hash_counts[digest] = eval_hash_counts.get(digest, 0) + 1
    for row in rows:
        digest = row.get("tofu_eval_hash") or ""
        row["eval_hash_group_size"] = eval_hash_counts.get(digest, 0) if digest else 0

    return rows


def ordered_columns(rows: Iterable[Dict[str, Any]]) -> List[str]:
    seen = set(PRIMARY_COLUMNS)
    extra = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                extra.append(key)
    return PRIMARY_COLUMNS + extra


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ordered_columns(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value).replace("\n", "<br>") for value in row) + " |")
    return lines


def count_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = fmt(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def write_markdown(path: Path, rows: List[Dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BRIDGE Detailed Debug Report",
        "",
        f"- Runs found: {len(rows)}",
        f"- CSV: `{csv_path.as_posix()}`",
        "",
        "## Status Counts",
        "",
    ]
    lines += markdown_table(["status", "count"], count_by(rows, "status").items())
    lines += ["", "## Duplicate Eval Outputs", ""]

    duplicate_rows = [row for row in rows if int(row.get("eval_hash_group_size") or 0) > 1]
    if duplicate_rows:
        lines += markdown_table(
            ["hash", "group_size", "model", "method", "model_utility", "forget_truth_ratio"],
            [
                [
                    row.get("tofu_eval_hash", ""),
                    row.get("eval_hash_group_size", ""),
                    row.get("model", ""),
                    row.get("method", ""),
                    row.get("model_utility", ""),
                    row.get("forget_truth_ratio", ""),
                ]
                for row in duplicate_rows
            ],
        )
    else:
        lines.append("No duplicate `TOFU_EVAL.json` hashes were found.")

    missing_diag = [
        row for row in rows
        if str(row.get("method", "")).startswith("bridge_")
        and int(row.get("bridge_diagnostic_rows") or 0) == 0
    ]
    lines += ["", "## Missing BRIDGE Diagnostics", ""]
    if missing_diag:
        lines += markdown_table(
            ["model", "method", "bridge_summary_exists", "bridge_diagnostic_rows", "train_dir"],
            [
                [
                    row.get("model", ""),
                    row.get("method", ""),
                    row.get("bridge_summary_exists", ""),
                    row.get("bridge_diagnostic_rows", ""),
                    row.get("train_dir", ""),
                ]
                for row in missing_diag
            ],
        )
    else:
        lines.append("All discovered BRIDGE runs have diagnostic rows.")

    lines += ["", "## Run Table", ""]
    table_columns = [
        "model",
        "method",
        "status",
        "eval_hash_group_size",
        "model_utility",
        "forget_truth_ratio",
        "bridge_prior",
        "bridge_lambda_g",
        "bridge_lambda_b",
        "bridge_diagnostic_rows",
        "boundary_dro",
        "prior_seconds",
        "error_type",
    ]
    lines += markdown_table(table_columns, ([row.get(column, "") for column in table_columns] for row in rows))

    lines += [
        "",
        "## Diagnostic Rule",
        "",
        "- Identical `TOFU_EVAL.json` hashes across different methods usually indicate identical evaluated model behavior, not a summarizer-only issue.",
        "- BRIDGE runs with zero diagnostic rows should be treated as suspect because `BRIDGENPO.compute_loss()` did not record per-step BRIDGE metrics.",
        "- If `bridge_summary.json` exists but diagnostic rows are zero, inspect the train log and trainer code path before interpreting the method result.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    train_root = Path(args.train_root)
    eval_root = Path(args.eval_root)
    logs_root = Path(args.logs_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    rows = build_rows(train_root, eval_root, logs_root)
    write_csv(output_csv, rows)
    write_markdown(output_md, rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote report to {output_md}")


if __name__ == "__main__":
    main()
