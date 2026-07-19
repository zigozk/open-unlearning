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
    ANNOTATION_REPAIR_SYSTEM_PROMPT,
    ANNOTATION_SYSTEM_PROMPT,
    annotation_schema,
    build_author_review_packets,
    mock_annotation,
    prepare_annotation_units,
    validate_annotation_outputs,
)
from atomic_tofu.bundle_compiler import compile_pure_diagnostic_bundles
from atomic_tofu.balanced_bundles import compile_balanced_bundles
from atomic_tofu.contracts import validate_annotation
from atomic_tofu.eval_extension import (
    EVAL_SYSTEM_PROMPT,
    EVAL_REPAIR_SYSTEM_PROMPT,
    FLOATING_MODEL_ALIASES,
    SLOT_REPAIR_SYSTEM_PROMPT,
    SLOT_SCHEMA_VERSION,
    SLOT_SYSTEM_PROMPT,
    build_eval_review_index,
    eval_schema,
    freeze_eval_slots,
    materialize_eval_candidates,
    mock_eval_candidate,
    mock_slot_analysis,
    official_eval_contract,
    prepare_eval_calibration,
    prepare_eval_units,
    slot_schema,
    validate_eval_answer_shape_calibration,
    validate_eval_canary,
    validate_eval_candidate,
    validate_eval_candidates,
    validate_eval_slot_outputs,
    validate_slot_analysis,
)
from atomic_tofu.feasibility import audit_cell_feasibility
from atomic_tofu.io import read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.providers import ResponsesProvider
from atomic_tofu.policies import apply_author_name_target_policy
from atomic_tofu.request_graph import compile_request_graph
from atomic_tofu.review import apply_author_reviews
from atomic_tofu.reporting import build_annotation_review_report
from atomic_tofu.source import discover_tofu_snapshot, export_sources
from atomic_tofu.supplemental import (
    SUPPLEMENTAL_SYSTEM_PROMPT,
    mock_supplemental,
    prepare_low_supplement_units,
    supplemental_request_schema,
    merge_low_supplemental_requests,
    validate_low_supplement_outputs,
)


def _annotation_user(unit: dict[str, Any]) -> str:
    return json.dumps(unit["payload"], ensure_ascii=False)


def _eval_user(unit: dict[str, Any]) -> str:
    return json.dumps(unit["payload"], ensure_ascii=False)


def _slot_user(unit: dict[str, Any]) -> str:
    return json.dumps(unit["payload"], ensure_ascii=False)


def _eval_repair_user(unit: dict[str, Any], candidate: dict[str, Any], errors: list[str]) -> str:
    return json.dumps({
        "original_qa": unit["payload"],
        "candidate_to_repair": candidate,
        "deterministic_validation_errors": errors,
    }, ensure_ascii=False)


def _slot_repair_user(unit: dict[str, Any], candidate: dict[str, Any], errors: list[str]) -> str:
    return json.dumps({
        "original_qa": unit["payload"],
        "candidate_slot_analysis": candidate,
        "deterministic_validation_errors": errors,
    }, ensure_ascii=False)


def _annotation_repair_user(unit: dict[str, Any], candidate: dict[str, Any], errors: list[str]) -> str:
    return json.dumps({
        "original_author_payload": unit["payload"],
        "candidate_to_repair": candidate,
        "deterministic_validation_errors": errors,
    }, ensure_ascii=False)


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
        repair_system = ANNOTATION_REPAIR_SYSTEM_PROMPT
    elif stage == "eval_slot_extraction":
        api_dir = root / "api" / "eval_extension" / "slot_extraction"
        units = read_jsonl(api_dir / "input_units.jsonl")
        schema = slot_schema()
        model_env = "ATOMIC_TOFU_EVAL_MODEL"
        system = SLOT_SYSTEM_PROMPT
        make_user = _slot_user
        mock = mock_slot_analysis
        repair_system = SLOT_REPAIR_SYSTEM_PROMPT
    elif stage in {"eval_extension", "eval_generated_calibration", "eval_anchor_calibration"}:
        api_dir = {
            "eval_extension": root / "api" / "eval_extension",
            "eval_generated_calibration": root / "api" / "eval_extension" / "calibration" / "generated",
            "eval_anchor_calibration": root / "api" / "eval_extension" / "anchor_calibration",
        }[stage]
        units = read_jsonl(api_dir / "input_units.jsonl")
        schema = eval_schema(official_eval_contract(root)["perturbation_count"])
        model_env = "ATOMIC_TOFU_EVAL_MODEL"
        system = EVAL_SYSTEM_PROMPT
        make_user = _eval_user
        mock = mock_eval_candidate
        repair_system = EVAL_REPAIR_SYSTEM_PROMPT
    elif stage == "supplemental_low":
        api_dir = root / "api" / "annotation" / "supplemental_low"
        units = read_jsonl(api_dir / "input_units.jsonl")
        schema = supplemental_request_schema()
        model_env = "ATOMIC_TOFU_ANNOTATION_MODEL"
        system = SUPPLEMENTAL_SYSTEM_PROMPT
        make_user = _annotation_user
        mock = mock_supplemental
        repair_system = None
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
        schema_name = "atomic_tofu_eval_slots" if stage == "eval_slot_extraction" else "atomic_tofu_eval_extension" if stage in {"eval_extension", "eval_generated_calibration", "eval_anchor_calibration"} else f"atomic_tofu_{stage}"
        provider = ResponsesProvider.from_env(model_env=model_env, schema_name=schema_name, schema=schema)
    elif provider_name != "mock":
        raise ValueError("provider must be mock or openai")

    provider_model = provider.model if provider is not None else "deterministic-fixture"
    output_path = api_dir / "candidate_outputs.jsonl"
    schema_sha256 = sha256_json(schema)
    model_lock = "floating_alias" if provider_model in FLOATING_MODEL_ALIASES else "model_id_recorded"
    prompt_metadata = {"system_prompt": system, "repair_system_prompt": repair_system}
    prompt_sha256 = sha256_json(prompt_metadata)
    generation_schema_version = SLOT_SCHEMA_VERSION if stage == "eval_slot_extraction" else SCHEMA_VERSION
    generation_metadata = {
        "schema_sha256": schema_sha256,
        "schema_version": generation_schema_version,
        "prompt_sha256": prompt_sha256,
        **prompt_metadata,
        "provider": provider_name,
        "model": provider_model,
        "model_lock": model_lock,
    }
    if stage == "eval_slot_extraction":
        generation_metadata["slot_schema_version"] = SLOT_SCHEMA_VERSION
    generation_sha256 = sha256_json(generation_metadata)
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
    current_input_hashes = {
        unit["unit_id"]: unit["content_sha256"]
        for unit in read_jsonl(api_dir / "input_units.jsonl")
    }
    outputs = [
        row for row in existing_rows
        if unit_ids is not None
        and row.get("unit_id") not in selected_ids
        and row.get("input_sha256") == current_input_hashes.get(row.get("unit_id"))
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

    def annotation_errors(unit: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
        if stage == "annotation":
            effective, _ = apply_author_name_target_policy(candidate)
            qa_ids = {qa["qa_id"] for qa in unit["payload"]["qas"]}
            return validate_annotation(effective, qa_ids)
        if stage == "eval_slot_extraction":
            return validate_slot_analysis(
                candidate,
                unit["payload"],
                require_no_human_review=False,
            )
        if stage in {"eval_extension", "eval_generated_calibration", "eval_anchor_calibration"}:
            slot_analysis = unit["payload"].get("slot_analysis")
            if slot_analysis is None:
                return ["missing frozen slot analysis"]
            return validate_eval_candidate(candidate, unit["payload"], unit["perturbation_count"], slot_analysis)
        return []

    def archive_repair_attempt(unit: dict[str, Any], attempt: int, candidate: dict[str, Any], provenance: dict[str, Any], errors: list[str]) -> None:
        write_json(
            api_dir / "repair_attempts" / f"{stage}_{unit['unit_id']}_{unit['content_sha256'][:12]}_{generation_sha256[:12]}_attempt{attempt}.json",
            {
                "unit_id": unit["unit_id"],
                "input_sha256": unit["content_sha256"],
                "generation_sha256": generation_sha256,
                "attempt": attempt,
                "candidate": candidate,
                "provenance": {**provenance, "schema_version": generation_schema_version},
                "validation_errors": errors,
            },
        )

    def process(unit):
        candidate: dict[str, Any] | None = None
        provenance: dict[str, Any] = {}
        api_call_usage: list[dict[str, Any]] = []
        repair_history: list[dict[str, Any]] = []
        try:
            if provider is None:
                candidate = mock(unit)
                provenance = {"provider": "mock", "model": "deterministic-fixture", "usage": {}}
            else:
                candidate, provenance = provider.request(system=system, user=make_user(unit))
            api_call_usage = [provenance.get("usage", {})]
            validation_errors = annotation_errors(unit, candidate)
            max_repairs = max(0, int(os.environ.get("ATOMIC_TOFU_MAX_VALIDATION_REPAIRS", "1")))
            attempt = 0
            while validation_errors and provider is not None and attempt < max_repairs and repair_system is not None:
                attempt += 1
                archive_repair_attempt(unit, attempt, candidate, provenance, validation_errors)
                repair_history.append({
                    "attempt": attempt,
                    "validation_errors": validation_errors,
                    "response_id": provenance.get("response_id"),
                    "usage": provenance.get("usage", {}),
                })
                candidate, provenance = provider.request(
                    system=repair_system,
                    user=(
                        _annotation_repair_user(unit, candidate, validation_errors)
                        if stage == "annotation"
                        else _slot_repair_user(unit, candidate, validation_errors)
                        if stage == "eval_slot_extraction"
                        else _eval_repair_user(unit, candidate, validation_errors)
                    ),
                )
                api_call_usage.append(provenance.get("usage", {}))
                validation_errors = annotation_errors(unit, candidate)
            if validation_errors:
                archive_repair_attempt(unit, attempt + 1, candidate, provenance, validation_errors)
                raise ValueError("candidate validation failed after repair: " + "; ".join(validation_errors))
            return {
                "record_id": f"{stage}_{unit['unit_id']}_{unit['content_sha256'][:12]}_{generation_sha256[:12]}",
                "unit_id": unit["unit_id"],
                "input_sha256": unit["content_sha256"],
                "schema_sha256": schema_sha256,
                "generation_sha256": generation_sha256,
                **({"slot_analysis_sha256": unit["slot_analysis_sha256"]} if "slot_analysis_sha256" in unit else {}),
                "candidate": candidate,
                "provenance": {
                    **provenance,
                    "schema_version": generation_schema_version,
                    "requested_model": provider_model,
                    "prompt_sha256": prompt_sha256,
                    "model_lock": model_lock,
                    "validation_repair_history": repair_history,
                    "api_call_usage": api_call_usage,
                },
            }, None
        except Exception as error:
            return None, {
                "unit_id": unit["unit_id"],
                "error_type": type(error).__name__,
                "message": str(error),
                "generation_sha256": generation_sha256,
                "prompt_sha256": prompt_sha256,
                **({"slot_analysis_sha256": unit["slot_analysis_sha256"]} if "slot_analysis_sha256" in unit else {}),
                "model": provenance.get("model", provider_model),
                "requested_model": provider_model,
                "model_lock": model_lock,
                "last_response_id": provenance.get("response_id"),
                "api_call_usage": api_call_usage,
                "validation_repair_history": repair_history,
            }

    concurrency = max(1, int(os.environ.get("ATOMIC_TOFU_MAX_CONCURRENCY", "1")))
    new_outputs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process, unit) for unit in pending]
        for completed_count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            output, error = future.result()
            if output is not None:
                outputs.append(output)
                new_outputs.append(output)
            if error is not None:
                errors.append(error)
            if completed_count % 20 == 0:
                write_jsonl(output_path, sorted(outputs, key=lambda row: row["unit_id"]))
                write_jsonl(api_dir / "error_queue.jsonl", sorted(errors, key=lambda row: row["unit_id"]))
    outputs.sort(key=lambda row: row["unit_id"])
    errors.sort(key=lambda row: row["unit_id"])
    write_jsonl(output_path, outputs)
    write_jsonl(api_dir / "error_queue.jsonl", errors)
    usage_keys = ("input_tokens", "output_tokens", "total_tokens")

    def sum_usage(usage_rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            key: sum(int(usage.get(key, 0) or 0) for usage in usage_rows)
            for key in usage_keys
        }

    def output_call_usage(row: dict[str, Any]) -> list[dict[str, Any]]:
        provenance = row.get("provenance", {})
        return provenance.get("api_call_usage", [provenance.get("usage", {})])

    def error_call_usage(row: dict[str, Any]) -> list[dict[str, Any]]:
        return row.get("api_call_usage", [])

    new_errors = [row for row in errors if row.get("generation_sha256") == generation_sha256 and row.get("unit_id") in selected_ids]

    api_usage_this_run = sum_usage(
        [usage for row in new_outputs for usage in output_call_usage(row)]
        + [usage for row in new_errors for usage in error_call_usage(row)]
    )
    selected_candidate_usage = sum_usage([
        row.get("provenance", {}).get("usage", {}) for row in outputs if row["unit_id"] in selected_ids
    ])
    candidate_pool_usage = sum_usage([
        row.get("provenance", {}).get("usage", {}) for row in outputs
    ])
    input_rate = os.environ.get("ATOMIC_TOFU_INPUT_COST_PER_MILLION")
    output_rate = os.environ.get("ATOMIC_TOFU_OUTPUT_COST_PER_MILLION")
    estimated_cost = None
    if input_rate is not None and output_rate is not None:
        estimated_cost = api_usage_this_run["input_tokens"] * float(input_rate) / 1_000_000 + api_usage_this_run["output_tokens"] * float(output_rate) / 1_000_000
    report = {
        "stage": stage, "provider": provider_name, "units": len(units),
        "all_available_units": all_unit_count, "selection_applied": unit_ids is not None,
        "generation_sha256": generation_sha256,
        "prompt_sha256": prompt_sha256,
        "model": provider_model,
        "model_lock": model_lock,
        "schema_version": generation_schema_version,
        "completed": sum(row["unit_id"] in selected_ids for row in outputs),
        "total_completed": len(outputs),
        "errors": sum(row.get("unit_id") in selected_ids for row in errors),
        "total_errors": len(errors), "resume": resume,
        "max_concurrency": concurrency,
        "usage": api_usage_this_run,
        "api_usage_this_run": api_usage_this_run,
        "selected_candidate_usage": selected_candidate_usage,
        "candidate_pool_usage": candidate_pool_usage,
        "api_call_count_this_run": sum(len(output_call_usage(row)) for row in new_outputs) + sum(len(error_call_usage(row)) for row in new_errors),
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
    slot_run = run_units(root, "eval_slot_extraction", "mock", resume=True)
    slot_validation = validate_eval_slot_outputs(root)
    eval_run = {"status": "pending_slot_review"}
    eval_validation = {"status": "pending_slot_review"}
    report = {
        "release": RELEASE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "annotation_units": len(annotation_units),
        "annotation_run": annotation_run,
        "annotation_validation": annotation_validation["status"],
        "author_review_packets": packet_count,
        "eval_units": len(eval_units),
        "eval_slot_run": slot_run,
        "eval_slot_validation": slot_validation["status"],
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
    result.add_argument("--stage", required=True, choices=("source", "prepare-annotation", "run-annotation", "validate-annotation", "annotation-review-report", "review-packets", "apply-reviews", "compile-requests", "merge-low-supplement", "cell-feasibility-audit", "compile-bundles", "compile-balanced-bundles", "prepare-low-supplement", "run-low-supplement", "validate-low-supplement", "prepare-eval", "run-eval-slot-extraction", "validate-eval-slots", "freeze-eval-slots", "prepare-eval-calibration", "run-eval", "run-eval-generated-calibration", "run-eval-anchor-calibration", "validate-eval-canary", "validate-eval-calibration", "validate-eval", "dry-run"))
    result.add_argument("--release-root", default=f"data/atomic_tofu/{RELEASE_VERSION}")
    result.add_argument("--hf-home", default=os.environ.get("HF_HOME", "/home/zkzhang/unlearning/HF_CACHE"))
    result.add_argument("--snapshot")
    result.add_argument("--provider", choices=("mock", "openai"), default="mock")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--unit-ids-file", help="Optional newline-delimited unit IDs for a calibration subset")
    result.add_argument("--author-id", help="Author ID required by --stage annotation-review-report")
    result.add_argument("--second-reviewer", help="Required named second reviewer for supplemental Multi merge")
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
    elif args.stage == "merge-low-supplement":
        if not args.second_reviewer:
            raise ValueError("--second-reviewer is required by --stage merge-low-supplement")
        report = merge_low_supplemental_requests(root, second_reviewer=args.second_reviewer)
    elif args.stage == "cell-feasibility-audit":
        report = audit_cell_feasibility(root)
    elif args.stage == "compile-bundles":
        report = compile_pure_diagnostic_bundles(root)
    elif args.stage == "compile-balanced-bundles":
        report = compile_balanced_bundles(root)
    elif args.stage == "prepare-low-supplement":
        report = prepare_low_supplement_units(root)
    elif args.stage == "run-low-supplement":
        report = run_units(root, "supplemental_low", args.provider, args.resume, unit_ids)
    elif args.stage == "validate-low-supplement":
        report = validate_low_supplement_outputs(root)
    elif args.stage == "prepare-eval":
        report = {"units": len(prepare_eval_units(root)), "contract": official_eval_contract(root)}
    elif args.stage == "run-eval-slot-extraction":
        report = run_units(root, "eval_slot_extraction", args.provider, args.resume, unit_ids)
    elif args.stage == "validate-eval-slots":
        report = validate_eval_slot_outputs(root, unit_ids)
    elif args.stage == "freeze-eval-slots":
        report = freeze_eval_slots(root, unit_ids)
    elif args.stage == "prepare-eval-calibration":
        report = prepare_eval_calibration(root)
    elif args.stage == "run-eval":
        report = run_units(root, "eval_extension", args.provider, args.resume, unit_ids)
    elif args.stage == "run-eval-generated-calibration":
        report = run_units(root, "eval_generated_calibration", args.provider, args.resume, unit_ids)
    elif args.stage == "run-eval-anchor-calibration":
        report = run_units(root, "eval_anchor_calibration", args.provider, args.resume, unit_ids)
    elif args.stage == "validate-eval-canary":
        report = validate_eval_canary(root, unit_ids)
    elif args.stage == "validate-eval-calibration":
        report = validate_eval_answer_shape_calibration(root)
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
