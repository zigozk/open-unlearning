from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.metrics import validate_metric_manifest
from atomic_tofu.request_graph import same_author_protected_qa_ids
from atomic_tofu.source import load_official_jsonl


def build_metric_views(
    release_root: str | Path,
    *,
    snapshot: str | Path,
    bundle_path: str | Path,
    requests_path: str | Path,
    eval_extension_path: str | Path,
    model_name: str,
    seed: int,
) -> dict[str, Any]:
    root = Path(release_root)
    bundle = read_json(bundle_path)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    by_id = {row["qa_id"]: row for row in source}
    author_qa_ids: dict[str, set[str]] = {}
    for row in source:
        author_qa_ids.setdefault(row["author_id"], set()).add(row["qa_id"])
    requests = {row["request_id"]: row for row in read_jsonl(requests_path)}
    extension = {row["qa_id"]: row for row in read_jsonl(eval_extension_path)}
    forget_ids = set(bundle["forget_qa_ids"])
    retain_ids = set(by_id) - forget_ids
    selected_requests = [requests[request_id] for request_id in bundle["request_ids"]]
    protected_ids = set()
    for request in selected_requests:
        author_ids = author_qa_ids.get(request["author_id"], set())
        expected_request_pool = set(same_author_protected_qa_ids(author_ids, set(request["forget_qa_ids"])))
        if set(request["protected_eval_qa_ids"]) != expected_request_pool:
            raise ValueError(f"{request['request_id']}: protected_eval_qa_ids is not the complete same-author non-target complement")
        # Protected evaluation is request-scoped, so a QA targeted by a different
        # request in this bundle remains protected for this request.
        protected_ids.update(expected_request_pool)
    protected_ids = sorted(protected_ids)
    if forget_ids - set(extension) or set(protected_ids) - set(extension):
        raise ValueError("Frozen evaluation extension is missing bundle forget/protected rows")
    if len(forget_ids) not in (40, 200):
        raise ValueError("Main metric views require exact 40/200 QA bundles")

    view_root = root / "metric_views" / bundle["bundle_id"] / f"seed_{seed}"
    paths = {
        "full_train_manifest": view_root / "full_train.jsonl",
        "forget_train_view": view_root / "forget_train.jsonl",
        "retain_train_view": view_root / "retain_train.jsonl",
        "forget_eval_view": view_root / "forget_eval.jsonl",
        "same_author_protected_eval_view": view_root / "protected_eval.jsonl",
        "fixed_official_retain_eval": view_root / "retain_perturbed.jsonl",
        "official_real_author_eval": view_root / "real_authors_perturbed.jsonl",
        "official_world_fact_eval": view_root / "world_facts_perturbed.jsonl",
        "holdout_eval": view_root / "holdout.jsonl",
        "retrain_train_manifest": view_root / "retrain_train.jsonl",
        "retrain_evaluation_config": view_root / "retrain_eval_config.json",
    }
    write_jsonl(paths["full_train_manifest"], source)
    write_jsonl(paths["forget_train_view"], [by_id[qa_id] for qa_id in sorted(forget_ids)])
    retain_rows = [by_id[qa_id] for qa_id in sorted(retain_ids)]
    write_jsonl(paths["retain_train_view"], retain_rows)
    write_jsonl(paths["retrain_train_manifest"], retain_rows)
    write_jsonl(paths["forget_eval_view"], [extension[qa_id] for qa_id in sorted(forget_ids)])
    write_jsonl(paths["same_author_protected_eval_view"], [extension[qa_id] for qa_id in protected_ids])
    write_jsonl(paths["fixed_official_retain_eval"], load_official_jsonl(snapshot, "retain_perturbed"))
    write_jsonl(paths["official_real_author_eval"], load_official_jsonl(snapshot, "real_authors_perturbed"))
    write_jsonl(paths["official_world_fact_eval"], load_official_jsonl(snapshot, "world_facts_perturbed"))
    holdout_name = "holdout01" if len(forget_ids) == 40 else "holdout05"
    write_jsonl(paths["holdout_eval"], load_official_jsonl(snapshot, holdout_name))
    reference_log = view_root / "retrain" / model_name / f"seed_{seed}" / "TOFU_EVAL.json"
    eval_config = {
        "model_name": model_name,
        "seed": seed,
        "forget_eval_path": str(paths["forget_eval_view"]),
        "retain_eval_path": str(paths["fixed_official_retain_eval"]),
        "real_author_eval_path": str(paths["official_real_author_eval"]),
        "world_fact_eval_path": str(paths["official_world_fact_eval"]),
        "holdout_eval_path": str(paths["holdout_eval"]),
        "retain_logs_path": str(reference_log),
        "paired_bundle_id": bundle["bundle_id"],
    }
    write_json(paths["retrain_evaluation_config"], eval_config)
    manifest = {
        key: str(path.relative_to(root))
        for key, path in paths.items()
    }
    manifest["reference_log_path"] = str(reference_log.relative_to(root))
    manifest["release_hashes"] = {
        "bundle": sha256_json(bundle),
        "full_source": sha256_json([[row["qa_id"], row["content_sha256"]] for row in source]),
        "eval_extension_rows": sha256_json(sorted(extension)),
    }
    manifest["bundle_id"] = bundle["bundle_id"]
    manifest["model_name"] = model_name
    manifest["seed"] = seed
    manifest["official_summary_fields"] = [
        "model_utility", "forget_quality", "forget_truth_ratio", "forget_Q_A_Prob",
        "forget_Q_A_ROUGE", "extraction_strength", "privleak",
    ]
    manifest["reference_status"] = "pending_retrain_evaluation"
    write_json(view_root / "metric_manifest.json", manifest)
    errors = validate_metric_manifest(manifest, base=root)
    # The paired log is intentionally absent before retraining; every other dependency must exist.
    expected_reference_error = f"reference_log_path: path does not exist: {manifest['reference_log_path']}"
    unexpected = [error for error in errors if error != expected_reference_error]
    if unexpected:
        raise ValueError(f"Metric view validation failed: {unexpected}")
    write_json(view_root / "metric_completeness_report.json", {
        "all_static_views_complete": True,
        "paired_retrain_log_complete": False,
        "pending": [expected_reference_error],
        "status": "blocked_until_paired_retrain_eval",
    })
    return manifest
