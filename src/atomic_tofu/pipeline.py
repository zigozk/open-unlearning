from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
from pathlib import Path
from typing import Any

from atomic_tofu import RELEASE_VERSION, SCHEMA_VERSION
from atomic_tofu.annotation import (
    ANNOTATION_SYSTEM_PROMPT,
    annotation_schema,
    build_author_review_packets,
    mock_annotation,
    prepare_annotation_units,
    validate_annotation_outputs,
)
from atomic_tofu.eval_extension import (
    EVAL_SYSTEM_PROMPT,
    build_eval_review_index,
    eval_schema,
    materialize_eval_candidates,
    mock_eval_candidate,
    official_eval_contract,
    prepare_eval_units,
    validate_eval_candidates,
)
from atomic_tofu.io import read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.providers import ResponsesProvider
from atomic_tofu.request_graph import compile_request_graph
from atomic_tofu.review import apply_author_reviews
from atomic_tofu.reporting import build_annotation_review_report
from atomic_tofu.source import discover_tofu_snapshot, export_sources


def _annotation_user(unit: dict[str, Any]) -> str:
    return json.dumps(unit["payload"], ensure_ascii=False)


def _eval_user(unit: dict[str, Any]) -> str:
    return json.dumps(unit["payload"], ensure_ascii=False)


def run_units(
    root: Path,
    stage: str,
    provider_name: str,
    resume: bool,
    unit_ids: set[str] | None = None,
) -> dict[str, Any]:
    if stage == "annotation":
        api_dir = root / "api" / "annotation"
        units = read_jsonl(api_dir / "input_units.jsonl")
        schema = annotation_schema()
        model_env = "ATOMIC_TOFU_ANNOTATION_MODEL"
        system = ANNOTATION_SYSTEM_PROMPT
        make_user = _annotation_user
        mock = mock_annotation
    elif stage == "eval_extension":
        api_dir = root / "api" / "eval_extension"
        units = read_jsonl(api_dir / "input_units.jsonl")
        schema = eval_schema(official_eval_contract(root)["perturbation_count"])
        model_env = "ATOMIC_TOFU_EVAL_MODEL"
        system = EVAL_SYSTEM_PROMPT
        make_user = _eval_user
        mock = mock_eval_candidate
    else:
        raise ValueError(stage)

    all_unit_count = len(units)
    if unit_ids is not None:
        available = {unit["unit_id"] for unit in units}
        unknown = sorted(unit_ids - available)
        if unknown:
            raise ValueError(f"unit ID selection contains unknown IDs: {unknown[:10]}")
        units = [unit for unit in units if unit["unit_id"] in unit_ids]

    provider = None
    if provider_name == "openai":
        provider = ResponsesProvider.from_env(model_env=model_env, schema_name=f"atomic_tofu_{stage}", schema=schema)
    elif provider_name != "mock":
        raise ValueError("provider must be mock or openai")

    provider_model = provider.model if provider is not None else "deterministic-fixture"
    output_path = api_dir / "candidate_outputs.jsonl"
    schema_sha256 = sha256_json(schema)
    generation_sha256 = sha256_json({
        "schema_sha256": schema_sha256,
        "schema_version": SCHEMA_VERSION,
        "system_prompt": system,
        "provider": provider_name,
        "model": provider_model,
    })
    existing_rows = read_jsonl(output_path) if resume and output_path.exists() else []
    existing = {
        row["unit_id"]: row
        for row in existing_rows
        if row.get("input_sha256")
        and row.get("generation_sha256") == generation_sha256
    }

    selected_ids = {unit["unit_id"] for unit in units}
    for row in existing_rows:
        if row.get("unit_id") not in selected_ids or row.get("generation_sha256") == generation_sha256:
            continue
        archive_id = row.get("record_id", f"legacy_{row.get('unit_id', 'unknown')}")
        archive_path = api_dir / "attempts" / f"{archive_id}.json"
        if not archive_path.exists():
            write_json(archive_path, row)
    outputs = [
        row for unit_id, row in existing.items()
        if unit_ids is not None and unit_id not in selected_ids
    ]
    existing_error_path = api_dir / "error_queue.jsonl"
    errors = [
        row for row in (read_jsonl(existing_error_path) if resume and existing_error_path.exists() else [])
        if unit_ids is not None and row.get("unit_id") not in selected_ids
    ]
    pending = []
    for unit in units:
        if unit["unit_id"] in existing and existing[unit["unit_id"]].get("input_sha256") == unit["content_sha256"]:
            outputs.append(existing[unit["unit_id"]])
        else:
            pending.append(unit)

    def process(unit):
        try:
            if provider is None:
                candidate = mock(unit)
                provenance = {"provider": "mock", "model": "deterministic-fixture", "usage": {}}
            else:
                candidate, provenance = provider.request(system=system, user=make_user(unit))
            return {
                "record_id": f"{stage}_{unit['unit_id']}_{unit['content_sha256'][:12]}_{generation_sha256[:12]}",
                "unit_id": unit["unit_id"],
                "input_sha256": unit["content_sha256"],
                "schema_sha256": schema_sha256,
                "generation_sha256": generation_sha256,
                "candidate": candidate,
                "provenance": {**provenance, "schema_version": SCHEMA_VERSION},
            }, None
        except Exception as error:
            return None, {"unit_id": unit["unit_id"], "error_type": type(error).__name__, "message": str(error)}

    concurrency = max(1, int(os.environ.get("ATOMIC_TOFU_MAX_CONCURRENCY", "1")))
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process, unit) for unit in pending]
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            output, error = future.result()
            if output is not None:
                outputs.append(output)
            if error is not None:
                errors.append(error)
            if completed_count % 20 == 0:
                write_jsonl(output_path, sorted(outputs, key=lambda row: row["unit_id"]))
                write_jsonl(api_dir / "error_queue.jsonl", sorted(errors, key=lambda row: row["unit_id"]))
    outputs.sort(key=lambda row: row["unit_id"])
    errors.sort(key=lambda row: row["unit_id"])
    write_jsonl(output_path, outputs)
    write_jsonl(api_dir / "error_queue.jsonl", errors)
    usage = {
        key: sum(int(row.get("provenance", {}).get("usage", {}).get(key, 0) or 0) for row in outputs)
        for key in ("input_tokens", "output_tokens", "total_tokens")
    }
    input_rate = os.environ.get("ATOMIC_TOFU_INPUT_COST_PER_MILLION")
    output_rate = os.environ.get("ATOMIC_TOFU_OUTPUT_COST_PER_MILLION")
    estimated_cost = None
    if input_rate is not None and output_rate is not None:
        estimated_cost = usage["input_tokens"] * float(input_rate) / 1_000_000 + usage["output_tokens"] * float(output_rate) / 1_000_000
    report = {
        "stage": stage, "provider": provider_name, "units": len(units),
        "all_available_units": all_unit_count, "selection_applied": unit_ids is not None,
        "generation_sha256": generation_sha256,
        "model": provider_model,
        "completed": sum(row["unit_id"] in selected_ids for row in outputs),
        "total_completed": len(outputs),
        "errors": sum(row.get("unit_id") in selected_ids for row in errors),
        "total_errors": len(errors), "resume": resume,
        "max_concurrency": concurrency, "usage": usage,
        "estimated_cost": estimated_cost,
        "cost_rate_source": "explicit_environment" if estimated_cost is not None else "not_configured",
    }
    write_json(api_dir / "run_report.json", report)
    return report


def dry_run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.release_root)
    snapshot = Path(args.snapshot) if args.snapshot else discover_tofu_snapshot(args.hf_home)
    source = export_sources(snapshot, root)
    annotation_units = prepare_annotation_units(root)
    annotation_run = run_units(root, "annotation", "mock", resume=True)
    annotation_validation = validate_annotation_outputs(root)
    packet_count = build_author_review_packets(root)
    eval_units = prepare_eval_units(root)
    eval_run = run_units(root, "eval_extension", "mock", resume=True)
    materialize_eval_candidates(root)
    eval_validation = validate_eval_candidates(root)
    build_eval_review_index(root)
    report = {
        "release": RELEASE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "annotation_units": len(annotation_units),
        "annotation_run": annotation_run,
        "annotation_validation": annotation_validation["status"],
        "author_review_packets": packet_count,
        "eval_units": len(eval_units),
        "eval_run": eval_run,
        "eval_validation": eval_validation["status"],
        "formal_gates": {
            "human_adjudication": "pending",
            "eval_semantics_and_anchor_calibration": "pending",
            "eval_b": "pending",
            "bundle_compilation": "blocked_until_human_adjudication",
            "retrain_and_training": "not_started",
        },
        "status": "structural_dry_run_passed_formal_gates_pending",
    }
    write_json(root / "audit" / "dry_run_report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Atomic-TOFU v10 full-corpus pipeline")
    result.add_argument("--stage", required=True, choices=("source", "prepare-annotation", "run-annotation", "validate-annotation", "annotation-review-report", "review-packets", "apply-reviews", "compile-requests", "prepare-eval", "run-eval", "validate-eval", "dry-run"))
    result.add_argument("--release-root", default=f"data/atomic_tofu/{RELEASE_VERSION}")
    result.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/home/zkzhang/unlearning/HF_CACHE"))
    result.add_argument("--snapshot")
    result.add_argument("--provider", choices=("mock", "openai"), default="mock")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--unit-ids-file", help="Optional newline-delimited unit IDs for a calibration subset")
    result.add_argument("--author-id", help="Author ID required by --stage annotation-review-report")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = Path(args.release_root)
    unit_ids = None
    if args.unit_ids_file:
        unit_ids = {
            line.strip()
            for line in Path(args.unit_ids_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        if not unit_ids:
            raise ValueError("--unit-ids-file selected no unit IDs")
    snapshot = Path(args.snapshot) if args.snapshot else discover_tofu_snapshot(args.hf_home)
    if args.stage == "source":
        report = export_sources(snapshot, root)
    elif args.stage == "prepare-annotation":
        report = {"units": len(prepare_annotation_units(root))}
    elif args.stage == "run-annotation":
        report = run_units(root, "annotation", args.provider, args.resume, unit_ids)
    elif args.stage == "validate-annotation":
        report = validate_annotation_outputs(root)
    elif args.stage == "annotation-review-report":
        if not args.author_id:
            raise ValueError("--author-id is required by --stage annotation-review-report")
        report = {"report_path": str(build_annotation_review_report(root, args.author_id))}
    elif args.stage == "review-packets":
        report = {"packets": build_author_review_packets(root), "status": "pending_human_review"}
    elif args.stage == "apply-reviews":
        report = apply_author_reviews(root)
    elif args.stage == "compile-requests":
        report = compile_request_graph(root)
    elif args.stage == "prepare-eval":
        report = {"units": len(prepare_eval_units(root)), "contract": official_eval_contract(root)}
    elif args.stage == "run-eval":
        report = run_units(root, "eval_extension", args.provider, args.resume, unit_ids)
    elif args.stage == "validate-eval":
        materialize_eval_candidates(root)
        report = validate_eval_candidates(root)
        build_eval_review_index(root)
    else:
        report = dry_run(args)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
