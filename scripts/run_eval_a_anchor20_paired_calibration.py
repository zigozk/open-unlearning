#!/usr/bin/env python3
"""Run the local 20-anchor official/generated Eval-A paired calibration.

``--validate-only`` is intentionally stdlib-only: it validates the frozen
inputs, embedded row hashes, anchor order, schema, and job-isolated output
path without importing torch, transformers, CUDA, or the model.

``--preprocess-smoke-only`` loads the local tokenizer and exercises the
repository preprocessing path on one official and one regenerated candidate,
but does not load the model or initialize CUDA.

The normal path loads one local full checkpoint once and evaluates both views
with the repository's ``preprocess_chat_instance``, supervised collator, and
``evaluate_probability`` implementation.  It never uses
``paraphrased_question`` as a model prompt; both views use the source
``question`` field, matching the current OpenUnlearning TOFU configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "data/atomic_tofu/v2.0-rc1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results/eval/atomic_tofu_eval_a_anchor20_paired_20260719"
DEFAULT_CHECKPOINT = Path("/home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_full")
EXPECTED_ANCHOR_COUNT = 20
PERTURBATION_COUNT = 5
MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
EXPECTED_CANDIDATE_ONLY = True
EXPECTED_MODEL_LOCK = "floating_alias"

TEMPLATE_ARGS = {
    "apply_chat_template": True,
    "system_prompt": "You are a helpful assistant.",
    "system_prompt_with_special_tokens": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>",
    "user_start_tag": "<|start_header_id|>user<|end_header_id|>\n\n",
    "user_end_tag": "<|eot_id|>",
    "asst_start_tag": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "asst_end_tag": "<|eot_id|>",
    "date_string": "10 Apr 2025",
}


class ValidationError(RuntimeError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.release_root
    return {
        "anchor_ids": root / "api/eval_extension/anchor_calibration/anchor_20_ids.txt",
        "current_route": root / "api/eval_extension/anchor_calibration/current_route_candidate_outputs.jsonl",
        "official_sidecars": root / "official_anchors/official_eval_sidecars.jsonl",
        "source": root / "source/tofu_full.jsonl",
        "checkpoint": args.checkpoint,
    }


def resolve_job_id(args: argparse.Namespace) -> str:
    return os.environ.get("SLURM_JOB_ID") or args.job_id or "validate-only"


def resolve_output_dir(args: argparse.Namespace) -> Path:
    return args.output_root / resolve_job_id(args)


def checkpoint_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    files = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files.append({"relative_path": str(item.relative_to(path)), "size": item.stat().st_size})
    digest = canonical_hash(files)
    return files, digest


def validate_inputs(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(args)
    required = [paths["anchor_ids"], paths["current_route"], paths["official_sidecars"], paths["source"], paths["checkpoint"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValidationError("missing required input(s): " + ", ".join(missing))
    if not paths["checkpoint"].is_dir():
        raise ValidationError(f"checkpoint is not a directory: {paths['checkpoint']}")
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        if not (paths["checkpoint"] / name).exists():
            raise ValidationError(f"checkpoint missing required file: {paths['checkpoint'] / name}")

    anchor_ids = [line.strip() for line in paths["anchor_ids"].read_text(encoding="utf-8").splitlines() if line.strip()]
    current_rows = read_jsonl(paths["current_route"])
    official_rows = read_jsonl(paths["official_sidecars"])
    source_rows = read_jsonl(paths["source"])
    errors: list[str] = []
    if len(anchor_ids) != EXPECTED_ANCHOR_COUNT:
        errors.append(f"anchor_20_ids count={len(anchor_ids)} expected={EXPECTED_ANCHOR_COUNT}")
    if len(set(anchor_ids)) != len(anchor_ids):
        errors.append("anchor_20_ids contains duplicate IDs")
    if len(current_rows) != EXPECTED_ANCHOR_COUNT:
        errors.append(f"current-route rows={len(current_rows)} expected={EXPECTED_ANCHOR_COUNT}")
    if len(source_rows) != 4000:
        errors.append(f"source rows={len(source_rows)} expected=4000")
    if len(official_rows) != 800:
        errors.append(f"official sidecar rows={len(official_rows)} expected=800")

    current_ids = [row.get("unit_id") for row in current_rows]
    if current_ids != anchor_ids:
        errors.append("current-route IDs/order do not exactly match anchor_20_ids")
    source_by_id = {row.get("qa_id"): row for row in source_rows}
    official_by_id = {row.get("qa_id"): row for row in official_rows}
    if len(source_by_id) != len(source_rows):
        errors.append("source contains duplicate or missing qa_id values")
    if len(official_by_id) != len(official_rows):
        errors.append("official sidecars contain duplicate or missing qa_id values")
    if len(set(current_ids)) != len(current_ids):
        errors.append("current-route contains duplicate or missing unit_id values")

    for index, qa_id in enumerate(anchor_ids):
        current = current_rows[index] if index < len(current_rows) else {}
        official = official_by_id.get(qa_id)
        source = source_by_id.get(qa_id)
        if official is None:
            errors.append(f"missing official sidecar for {qa_id}")
            continue
        if source is None:
            errors.append(f"missing source row for {qa_id}")
            continue
        if official.get("source_or_generated") != "official":
            errors.append(f"{qa_id}: official sidecar source_or_generated is not official")
        if official.get("human_review_status") != "official_anchor":
            errors.append(f"{qa_id}: official sidecar status is not official_anchor")
        if official.get("question") != source.get("question") or official.get("answer") != source.get("answer"):
            errors.append(f"{qa_id}: official/source question or answer mismatch")
        if set(official) < {"qa_id", "question", "answer", "paraphrased_answer", "perturbed_answer"}:
            errors.append(f"{qa_id}: official sidecar schema is incomplete")
        candidate = current.get("candidate")
        if not isinstance(candidate, dict) or set(candidate) != {"paraphrased_question", "paraphrased_answer", "perturbed_answer"}:
            errors.append(f"{qa_id}: current candidate schema is not exactly the three expected fields")
            continue
        if current.get("unit_id") != qa_id:
            errors.append(f"{qa_id}: current unit_id mismatch")
        if len(candidate.get("perturbed_answer", [])) != PERTURBATION_COUNT:
            errors.append(f"{qa_id}: current perturbed_answer count is not 5")
        if len(official.get("perturbed_answer", [])) != PERTURBATION_COUNT:
            errors.append(f"{qa_id}: official perturbed_answer count is not 5")
        for field in ("paraphrased_answer", "perturbed_answer"):
            value = candidate.get(field)
            if field == "paraphrased_answer" and not isinstance(value, str):
                errors.append(f"{qa_id}: generated paraphrased_answer is not a string")
            if field == "perturbed_answer" and (not isinstance(value, list) or not all(isinstance(x, str) for x in value)):
                errors.append(f"{qa_id}: generated perturbed_answer is not a list of strings")
            value = official.get(field)
            if field == "paraphrased_answer" and not isinstance(value, str):
                errors.append(f"{qa_id}: official paraphrased_answer is not a string")
            if field == "perturbed_answer" and (not isinstance(value, list) or not all(isinstance(x, str) for x in value)):
                errors.append(f"{qa_id}: official perturbed_answer is not a list of strings")
        if current.get("input_sha256") != source.get("content_sha256"):
            errors.append(f"{qa_id}: current input_sha256 does not equal source content_sha256")
        if current.get("generation_sha256") != canonical_hash(candidate):
            errors.append(f"{qa_id}: current generation_sha256 does not match canonical candidate hash")
        provenance = current.get("provenance") or {}
        if provenance.get("candidate_only") is not True:
            errors.append(f"{qa_id}: current candidate_only provenance is not true")
        if provenance.get("model_lock") != EXPECTED_MODEL_LOCK:
            errors.append(f"{qa_id}: current model_lock is not floating_alias")
        if provenance.get("api_called") is not False:
            errors.append(f"{qa_id}: current api_called provenance is not false")

    input_hashes = {name: sha256_file(path) for name, path in paths.items() if name != "checkpoint"}
    checkpoint_files, checkpoint_manifest_sha256 = checkpoint_manifest(paths["checkpoint"])
    output_dir = resolve_output_dir(args)
    if output_dir.exists() and any(output_dir.iterdir()):
        errors.append(f"output directory is non-empty and would be overwritten: {output_dir}")
    if errors:
        raise ValidationError("; ".join(errors))
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "input_sha256": input_hashes,
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "anchor_ids": anchor_ids,
        "current_rows": current_rows,
        "official_by_id": official_by_id,
        "source_by_id": source_by_id,
        "output_dir": str(output_dir),
    }


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    xbar, ybar = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    xden = math.sqrt(sum((x - xbar) ** 2 for x in xs))
    yden = math.sqrt(sum((y - ybar) ** 2 for y in ys))
    return numerator / (xden * yden) if xden and yden else None


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def direction(value: float) -> int:
    return 1 if value > 1.0 else -1 if value < 1.0 else 0


def load_runtime_modules():
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from data.collators import DataCollatorForSupervisedDataset
    from data.utils import preprocess_chat_instance
    from evals.metrics.utils import evaluate_probability
    return torch, AutoModelForCausalLM, AutoTokenizer, DataCollatorForSupervisedDataset, preprocess_chat_instance, evaluate_probability


def load_preprocess_modules():
    """Load only tokenizer/preprocessing dependencies for the CPU smoke test."""
    src_path = str(PROJECT_ROOT / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    import torch
    from transformers import AutoTokenizer
    from data.utils import preprocess_chat_instance
    return torch, AutoTokenizer, preprocess_chat_instance


def preprocess_smoke(args: argparse.Namespace, validation: dict[str, Any]) -> None:
    """Exercise single-turn preprocessing without model loading or CUDA use."""
    _, AutoTokenizer, preprocess = load_preprocess_modules()
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True)
    qa_id = validation["anchor_ids"][0]
    question = validation["source_by_id"][qa_id]["question"]
    official = validation["official_by_id"][qa_id]
    generated = validation["current_rows"][0]["candidate"]
    samples = [
        ("official", official["paraphrased_answer"]),
        ("regenerated", generated["paraphrased_answer"]),
    ]
    checks = []
    for view_name, answer in samples:
        item = preprocess(tokenizer, TEMPLATE_ARGS, [question], [answer], MAX_LENGTH, False)
        input_count = int(item["input_ids"].numel())
        label_count = int(item["labels"].numel())
        if input_count == 0 or label_count == 0:
            raise RuntimeError(f"{view_name} preprocess smoke produced an empty tensor")
        checks.append({"view": view_name, "qa_id": qa_id, "input_token_count": input_count, "label_token_count": label_count})
    print(json.dumps({"status": "preprocess_smoke_passed", "cuda_initialized": False, "samples": checks}, ensure_ascii=False))


def evaluate_view(model: Any, tokenizer: Any, collator_cls: Any, preprocess: Any, evaluate_probability_fn: Any, questions: list[str], candidates: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    import torch

    fields: list[tuple[str, int | None, str, str]] = []
    for row_index, (question, candidate) in enumerate(zip(questions, candidates)):
        fields.append(("paraphrased_answer", None, question, candidate["paraphrased_answer"]))
        for perturbation_index, answer in enumerate(candidate["perturbed_answer"], 1):
            fields.append(("perturbed_answer", perturbation_index, question, answer))
    instances = []
    for field_index, (_, _, question, answer) in enumerate(fields):
        # The repository implementation documents string input, but performs
        # len(prompt_msgs)==len(response_msgs) before its string-to-list
        # normalization.  Single-turn lists are the semantically equivalent
        # adapter and avoid comparing question/answer character lengths.
        item = preprocess(tokenizer, TEMPLATE_ARGS, [question], [answer], MAX_LENGTH, False)
        item["index"] = field_index
        instances.append(item)
    collator = collator_cls(tokenizer=tokenizer, padding_side="right", index="index")
    metrics_by_field: dict[int, dict[str, float]] = {}
    for start in range(0, len(instances), batch_size):
        batch = collator(instances[start:start + batch_size])
        indices = batch.pop("index").tolist()
        batch_metrics = evaluate_probability_fn(model=model, batch=batch)
        for index, metric in zip(indices, batch_metrics):
            metrics_by_field[index] = {"avg_loss": float(metric["avg_loss"]), "probability": float(metric["prob"])}
    rows = []
    cursor = 0
    for row_index in range(len(candidates)):
        correct = metrics_by_field[cursor]
        cursor += 1
        perturbations = []
        for perturbation_index in range(1, PERTURBATION_COUNT + 1):
            perturbations.append(metrics_by_field[cursor])
            cursor += 1
        mean_perturbed_loss = statistics.fmean(x["avg_loss"] for x in perturbations)
        mean_perturbed_probability = statistics.fmean(x["probability"] for x in perturbations)
        correct_probability = math.exp(-correct["avg_loss"])
        wrong_probability = math.exp(-mean_perturbed_loss)
        truth_ratio = wrong_probability / (correct_probability + 1e-10)
        rows.append({
            "correct": correct,
            "perturbed": perturbations,
            "mean_perturbed": {"avg_loss": mean_perturbed_loss, "probability": mean_perturbed_probability},
            "truth_ratio": truth_ratio,
        })
    return rows


def paired_metric_summary(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    official = [x for x, _ in pairs]
    regenerated = [y for _, y in pairs]
    differences = [y - x for x, y in pairs]
    return {
        "n": len(pairs),
        "mean_difference_generated_minus_official": statistics.fmean(differences) if differences else None,
        "median_difference_generated_minus_official": statistics.median(differences) if differences else None,
        "median_absolute_difference": statistics.median([abs(x) for x in differences]) if differences else None,
        "mean_absolute_difference": statistics.fmean([abs(x) for x in differences]) if differences else None,
        "pearson_official_vs_generated": pearson(official, regenerated),
        "spearman_official_vs_generated": spearman(official, regenerated),
    }


def build_summary(anchor_ids: list[str], official: list[dict[str, Any]], regenerated: list[dict[str, Any]], row_results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_pairs: dict[str, list[tuple[float, float]]] = {}
    for row_index, qa_id in enumerate(anchor_ids):
        for view_name in ("official", "regenerated"):
            view = row_results[row_index][view_name]
            for field_name, metric in [("paraphrased_answer", view["correct"])] + [(f"perturbed_answer_{i}", value) for i, value in enumerate(view["perturbed"], 1)] + [("mean_perturbed_answer", view["mean_perturbed"])]:
                for metric_name in ("avg_loss", "probability"):
                    key = f"{field_name}.{metric_name}"
                    metric_pairs.setdefault(key, [])[0:0]
        for field_name, official_metric, generated_metric in [
            ("paraphrased_answer", row_results[row_index]["official"]["correct"], row_results[row_index]["regenerated"]["correct"]),
            *[(f"perturbed_answer_{i}", row_results[row_index]["official"]["perturbed"][i - 1], row_results[row_index]["regenerated"]["perturbed"][i - 1]) for i in range(1, PERTURBATION_COUNT + 1)],
            ("mean_perturbed_answer", row_results[row_index]["official"]["mean_perturbed"], row_results[row_index]["regenerated"]["mean_perturbed"]),
        ]:
            for metric_name in ("avg_loss", "probability"):
                metric_pairs.setdefault(f"{field_name}.{metric_name}", []).append((float(official_metric[metric_name]), float(generated_metric[metric_name])))
        metric_pairs.setdefault("truth_ratio", []).append((float(row_results[row_index]["official"]["truth_ratio"]), float(row_results[row_index]["regenerated"]["truth_ratio"])))

    row_rank = []
    for qa_id, result in zip(anchor_ids, row_results):
        official_vector_loss = [result["official"]["correct"]["avg_loss"]] + [x["avg_loss"] for x in result["official"]["perturbed"]]
        generated_vector_loss = [result["regenerated"]["correct"]["avg_loss"]] + [x["avg_loss"] for x in result["regenerated"]["perturbed"]]
        official_vector_prob = [result["official"]["correct"]["probability"]] + [x["probability"] for x in result["official"]["perturbed"]]
        generated_vector_prob = [result["regenerated"]["correct"]["probability"]] + [x["probability"] for x in result["regenerated"]["perturbed"]]
        row_rank.append({"qa_id": qa_id, "avg_loss_spearman": spearman(official_vector_loss, generated_vector_loss), "probability_spearman": spearman(official_vector_prob, generated_vector_prob)})

    truth_pairs = metric_pairs["truth_ratio"]
    reversals = sum(direction(x) != direction(y) and direction(x) != 0 and direction(y) != 0 for x, y in truth_pairs)
    finite_failures = []
    for row_index, result in enumerate(row_results):
        def visit(value: Any, path: str) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")
            elif isinstance(value, (int, float)) and not finite(value):
                finite_failures.append({"row_index": row_index, "path": path, "value": repr(value)})
        visit(result, "row_result")
    return {
        "metric_pairs": {key: paired_metric_summary(value) for key, value in sorted(metric_pairs.items())},
        "per_row_rank_correlation": row_rank,
        "per_row_rank_correlation_summary": {
            "avg_loss_mean": statistics.fmean([x["avg_loss_spearman"] for x in row_rank if x["avg_loss_spearman"] is not None]),
            "avg_loss_median": statistics.median([x["avg_loss_spearman"] for x in row_rank if x["avg_loss_spearman"] is not None]),
            "probability_mean": statistics.fmean([x["probability_spearman"] for x in row_rank if x["probability_spearman"] is not None]),
            "probability_median": statistics.median([x["probability_spearman"] for x in row_rank if x["probability_spearman"] is not None]),
        },
        "truth_ratio_direction_reversal_count": reversals,
        "truth_ratio_direction_defined_count": sum(direction(x) != 0 and direction(y) != 0 for x, y in truth_pairs),
        "truth_ratio_direction_tie_count": sum(direction(x) == 0 or direction(y) == 0 for x, y in truth_pairs),
        "nan_inf_failures": finite_failures,
        "calibration_decision": "not_automatically_declared",
    }


def git_state() -> dict[str, Any]:
    def run(command: list[str]) -> str:
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        return result.stdout.strip()
    status = run(["git", "status", "--porcelain"])
    return {"commit": run(["git", "rev-parse", "HEAD"]), "dirty": bool(status), "status_short": status.splitlines()}


def slurm_resources() -> dict[str, Any]:
    names = ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_PARTITION", "SLURM_JOB_NAME", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_TIMELIMIT", "SLURM_GPUS", "SLURM_GPUS_ON_NODE")
    return {name: os.environ.get(name) for name in names}


def run(args: argparse.Namespace) -> None:
    validation = validate_inputs(args)
    output_dir = Path(validation["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    torch, AutoModelForCausalLM, AutoTokenizer, collator_cls, preprocess, evaluate_probability_fn = load_runtime_modules()
    if not torch.cuda.is_available():
        raise RuntimeError("formal paired calibration requires CUDA; no GPU task was started by this script")
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.checkpoint), torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model.to("cuda")
    model.eval()
    anchor_ids = validation["anchor_ids"]
    questions = [validation["source_by_id"][qa_id]["question"] for qa_id in anchor_ids]
    official_candidates = [{"paraphrased_answer": validation["official_by_id"][qa_id]["paraphrased_answer"], "perturbed_answer": validation["official_by_id"][qa_id]["perturbed_answer"]} for qa_id in anchor_ids]
    generated_candidates = [row["candidate"] for row in validation["current_rows"]]
    started = time.time()
    with torch.no_grad():
        official_metrics = evaluate_view(model, tokenizer, collator_cls, preprocess, evaluate_probability_fn, questions, official_candidates, args.batch_size)
        regenerated_metrics = evaluate_view(model, tokenizer, collator_cls, preprocess, evaluate_probability_fn, questions, generated_candidates, args.batch_size)
    row_results = []
    for qa_id, official_metric, generated_metric in zip(anchor_ids, official_metrics, regenerated_metrics):
        row_results.append({"qa_id": qa_id, "official": official_metric, "regenerated": generated_metric, "official_fields_sha256": canonical_hash(official_candidates[anchor_ids.index(qa_id)]), "generated_candidate_sha256": validation["current_rows"][anchor_ids.index(qa_id)]["generation_sha256"]})
    summary = build_summary(anchor_ids, official_candidates, generated_candidates, row_results)
    summary.update({"anchor_count": len(anchor_ids), "batch_size": args.batch_size, "question_key": "question", "correct_answer_field": "paraphrased_answer", "wrong_answer_field": "perturbed_answer", "paraphrased_question_used_as_model_input": False, "elapsed_seconds": time.time() - started, "candidate_only": True, "model_lock": EXPECTED_MODEL_LOCK, "api_called": False})
    jsonl_write(output_dir / "paired_results.jsonl", row_results)
    json_dump(output_dir / "paired_calibration_summary.json", summary)
    manifest = {
        "artifact": "Atomic-TOFU Eval-A hidden-anchor official/generated paired evaluator calibration",
        "status": "completed_behavioral_comparison_only",
        "created_at": "2026-07-19",
        "job_id": resolve_job_id(args),
        "output_dir": str(output_dir),
        "input_paths": validation["paths"],
        "input_sha256": validation["input_sha256"],
        "checkpoint_manifest_sha256": validation["checkpoint_manifest_sha256"],
        "checkpoint_file_count": validation["checkpoint_file_count"],
        "checkpoint_path": str(args.checkpoint),
        "git": git_state(),
        "slurm_resources": slurm_resources(),
        "evaluator_config": {"question_key": "question", "correct_answer_field": "paraphrased_answer", "wrong_answer_field": "perturbed_answer", "batch_size": args.batch_size, "max_length": MAX_LENGTH, "template_args": TEMPLATE_ARGS, "tokenization": "src.data.utils.preprocess_chat_instance", "collator": "src.data.collators.DataCollatorForSupervisedDataset", "average_token_loss": "src.evals.metrics.utils.evaluate_probability", "truth_ratio": "wrong_probability / (correct_probability + 1e-10)"},
        "official_and_regenerated_same_checkpoint": True,
        "official_and_regenerated_same_tokenizer": True,
        "official_and_regenerated_same_batch_size": True,
        "paraphrased_question_used_as_model_input": False,
        "candidate_only": True,
        "model_lock": EXPECTED_MODEL_LOCK,
        "api_called": False,
        "calibration_decision": "not_automatically_declared",
    }
    json_dump(output_dir / "run_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir), "summary": str(output_dir / 'paired_calibration_summary.json')}, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--preprocess-smoke-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        validation = validate_inputs(args)
        if args.validate_only and args.preprocess_smoke_only:
            raise ValidationError("--validate-only and --preprocess-smoke-only are mutually exclusive")
        if args.validate_only:
            print(json.dumps({"status": "validate_only_passed", "anchor_count": len(validation["anchor_ids"]), "output_dir_checked": validation["output_dir"], "input_sha256": validation["input_sha256"], "checkpoint_manifest_sha256": validation["checkpoint_manifest_sha256"]}, ensure_ascii=False))
            return 0
        if args.preprocess_smoke_only:
            preprocess_smoke(args, validation)
            return 0
        run(args)
        return 0
    except (ValidationError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
