#!/usr/bin/env python
"""Summarize BRIDGE initial-pipeline failures from Slurm logs and outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ERROR_PATTERNS = [
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError", re.I)),
    ("killed", re.compile(r"\bKilled\b|oom-kill|oom_kill|cgroup.*out of memory", re.I)),
    ("module_not_found", re.compile(r"ModuleNotFoundError|No module named", re.I)),
    ("import_error", re.compile(r"ImportError:", re.I)),
    ("hydra_config", re.compile(r"hydra\.errors|ConfigCompositionException|MissingConfigException|Could not override", re.I)),
    ("model_path", re.compile(r"does not appear to have a file named|Repository Not Found|is not a local folder|not found", re.I)),
    ("slurm_error", re.compile(r"slurmstepd: error|SBATCH: error|Invalid generic resource", re.I)),
    ("python_traceback", re.compile(r"Traceback \(most recent call last\):", re.I)),
    ("runtime_error", re.compile(r"RuntimeError:", re.I)),
    ("value_error", re.compile(r"ValueError:", re.I)),
    ("assertion_error", re.compile(r"AssertionError", re.I)),
]

KEY_VALUE_PATTERN = re.compile(r"^([A-Z0-9_]+)=(.*)$")
ARRAY_LOG_PATTERN = re.compile(r"bridge_initial-(?P<job>\d+)_(?P<task>\d+)\.(?P<kind>out|err)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", default="logs")
    parser.add_argument("--train-root", default="results/bridge_initial")
    parser.add_argument("--eval-root", default="results/bridge_initial_eval")
    parser.add_argument("--output-csv", default="results/bridge_reports/bridge_failure_summary.csv")
    parser.add_argument("--output-md", default="results/bridge_reports/bridge_failure_report.md")
    parser.add_argument("--context-lines", type=int, default=2)
    return parser.parse_args()


def read_text(path: Path) -> str:
    if not path or not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def discover_logs(logs_root: Path) -> Dict[str, Dict[str, Path]]:
    grouped: Dict[str, Dict[str, Path]] = {}
    if not logs_root.exists():
        return grouped
    for path in sorted(logs_root.glob("bridge_initial-*_*.?*")):
        match = ARRAY_LOG_PATTERN.match(path.name)
        if not match:
            continue
        key = f"{match.group('job')}_{match.group('task')}"
        grouped.setdefault(key, {})[match.group("kind")] = path
    return grouped


def parse_metadata(text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    for line in text.splitlines():
        match = KEY_VALUE_PATTERN.match(line.strip())
        if match:
            metadata[match.group(1).lower()] = match.group(2).strip()
    return metadata


def first_error_context(text: str, context_lines: int) -> tuple[Optional[str], Optional[str]]:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        for _, pattern in ERROR_PATTERNS:
            if pattern.search(line):
                start = max(0, idx - context_lines)
                end = min(len(lines), idx + context_lines + 1)
                return line.strip(), "\n".join(lines[start:end]).strip()
    return None, None


def classify_error(text: str) -> str:
    labels = [label for label, pattern in ERROR_PATTERNS if pattern.search(text)]
    if labels:
        return "+".join(labels)
    if "===== DONE:" in text:
        return "none"
    if "[skip]" in text:
        return "skipped"
    return "unknown"


def latest_run_configs(train_root: Path) -> Dict[str, Dict[str, Any]]:
    configs: Dict[str, Dict[str, Any]] = {}
    if not train_root.exists():
        return configs
    for config_path in sorted(train_root.glob("*/run_config.json")):
        data = load_json(config_path)
        task_name = data.get("task_name") or config_path.parent.name
        data["train_dir"] = config_path.parent.as_posix()
        configs[task_name] = data
    return configs


def eval_summary_exists(eval_root: Path, task_name: str) -> bool:
    if not task_name:
        return False
    return (eval_root / f"{task_name}_tofu_eval" / "TOFU_SUMMARY.json").exists()


def bridge_summary_exists(train_dir: Optional[str]) -> bool:
    return bool(train_dir) and (Path(train_dir) / "bridge_summary.json").exists()


def status_from_outputs(
    text: str,
    task_name: str,
    train_dir: Optional[str],
    eval_root: Path,
) -> str:
    if "===== DONE:" in text and eval_summary_exists(eval_root, task_name):
        return "success"
    if "[skip]" in text:
        return "skipped"
    if eval_summary_exists(eval_root, task_name):
        return "eval_complete"
    if bridge_summary_exists(train_dir):
        return "train_complete_eval_missing"
    if "===== EVAL COMMAND =====" in text:
        return "eval_failed"
    if "===== TRAIN COMMAND =====" in text:
        return "train_failed"
    return "failed_before_train"


def summarize_logs(
    logs_root: Path,
    train_root: Path,
    eval_root: Path,
    context_lines: int,
) -> List[Dict[str, Any]]:
    run_configs = latest_run_configs(train_root)
    rows: List[Dict[str, Any]] = []

    for key, parts in discover_logs(logs_root).items():
        out_text = read_text(parts.get("out", Path()))
        err_text = read_text(parts.get("err", Path()))
        combined = "\n".join([out_text, err_text]).strip()
        metadata = parse_metadata(out_text)
        task_name = metadata.get("task_name")
        if not task_name:
            done_match = re.search(r"===== DONE: (?P<task>.+?) =====", out_text)
            if done_match:
                task_name = done_match.group("task").strip()
        method = metadata.get("method")
        task_config = run_configs.get(task_name or "", {})
        if not method:
            method = task_config.get("method")
        train_dir = metadata.get("train_dir") or task_config.get("train_dir")
        first_error, error_context = first_error_context(combined, context_lines)
        error_type = classify_error(combined)
        status = status_from_outputs(combined, task_name or "", train_dir, eval_root)

        rows.append(
            {
                "log_key": key,
                "job_id": metadata.get("job_id"),
                "array_task_id": metadata.get("array_task_id"),
                "method": method,
                "task_name": task_name,
                "status": status,
                "error_type": error_type,
                "first_error": first_error,
                "error_context": error_context,
                "train_dir": train_dir,
                "eval_summary_exists": eval_summary_exists(eval_root, task_name or ""),
                "bridge_summary_exists": bridge_summary_exists(train_dir),
                "out_log": parts.get("out").as_posix() if parts.get("out") else "",
                "err_log": parts.get("err").as_posix() if parts.get("err") else "",
            }
        )

    return rows


def summarize_output_only(train_root: Path, eval_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for task_name, config in latest_run_configs(train_root).items():
        train_dir = config.get("train_dir")
        rows.append(
            {
                "log_key": "",
                "job_id": "",
                "array_task_id": "",
                "method": config.get("method"),
                "task_name": task_name,
                "status": "eval_complete" if eval_summary_exists(eval_root, task_name) else "train_or_eval_incomplete",
                "error_type": "unknown",
                "first_error": "",
                "error_context": "",
                "train_dir": train_dir,
                "eval_summary_exists": eval_summary_exists(eval_root, task_name),
                "bridge_summary_exists": bridge_summary_exists(train_dir),
                "out_log": "",
                "err_log": "",
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "log_key",
        "job_id",
        "array_task_id",
        "method",
        "task_name",
        "status",
        "error_type",
        "first_error",
        "error_context",
        "train_dir",
        "eval_summary_exists",
        "bridge_summary_exists",
        "out_log",
        "err_log",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def count_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def markdown_table(headers: List[str], rows: Iterable[Iterable[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("\n", "<br>") for value in row) + " |")
    return lines


def write_markdown(path: Path, rows: List[Dict[str, Any]], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BRIDGE Failure Summary",
        "",
        f"- Rows: {len(rows)}",
        f"- CSV: `{csv_path.as_posix()}`",
        "",
        "## Status Counts",
        "",
    ]
    lines += markdown_table(["status", "count"], count_by(rows, "status").items())
    lines += ["", "## Error Type Counts", ""]
    lines += markdown_table(["error_type", "count"], count_by(rows, "error_type").items())
    lines += ["", "## Per Task", ""]
    lines += markdown_table(
        ["method", "status", "error_type", "first_error", "out_log", "err_log"],
        (
            [
                row.get("method", ""),
                row.get("status", ""),
                row.get("error_type", ""),
                row.get("first_error", "") or "",
                row.get("out_log", ""),
                row.get("err_log", ""),
            ]
            for row in rows
        ),
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logs_root = Path(args.logs_root)
    train_root = Path(args.train_root)
    eval_root = Path(args.eval_root)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)

    rows = summarize_logs(logs_root, train_root, eval_root, args.context_lines)
    if not rows:
        rows = summarize_output_only(train_root, eval_root)
    write_csv(output_csv, rows)
    write_markdown(output_md, rows, output_csv)
    print(f"Wrote {len(rows)} rows to {output_csv}")
    print(f"Wrote report to {output_md}")


if __name__ == "__main__":
    main()
