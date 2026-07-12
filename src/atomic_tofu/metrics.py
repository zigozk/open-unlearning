from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from atomic_tofu.contracts import cvar
from atomic_tofu.io import read_json, write_json

REQUIRED_VIEWS = {
    "full_train_manifest",
    "forget_train_view",
    "retain_train_view",
    "forget_eval_view",
    "same_author_protected_eval_view",
    "fixed_official_retain_eval",
    "official_real_author_eval",
    "official_world_fact_eval",
    "holdout_eval",
    "retrain_train_manifest",
    "retrain_evaluation_config",
    "reference_log_path",
    "release_hashes",
}
OFFICIAL_SUMMARY_FIELDS = (
    "model_utility", "forget_quality", "forget_truth_ratio", "forget_Q_A_Prob",
    "forget_Q_A_ROUGE", "extraction_strength", "privleak",
)


def validate_metric_manifest(manifest: dict[str, Any], base: str | Path | None = None) -> list[str]:
    errors = []
    missing = REQUIRED_VIEWS - set(manifest)
    if missing:
        errors.append(f"missing views: {sorted(missing)}")
    root = Path(base) if base else None
    for key in REQUIRED_VIEWS - {"release_hashes"}:
        value = manifest.get(key)
        if not isinstance(value, str) or not value:
            errors.append(f"{key}: path is missing")
        elif root is not None and not (root / value).exists():
            errors.append(f"{key}: path does not exist: {value}")
    return errors


def aggregate_request_gaps(queries: list[dict[str, Any]], model: dict[str, Any], retrain: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_query = {}
    for index, query in enumerate(queries):
        key = str(index)
        model_row = model.get(key, model.get(index))
        retrain_row = retrain.get(key, retrain.get(index))
        if model_row is None or retrain_row is None:
            raise KeyError(f"missing aligned probability row {index}")
        gap = abs(float(model_row["avg_loss"]) - float(retrain_row["avg_loss"]))
        grouped[query["request_id"]][query["query_type"]].append(gap)
        per_query[query["query_id"]] = {"answer_log_prob_gap": gap, "request_id": query["request_id"], "query_type": query["query_type"]}
    per_request = {}
    protected_tail = []
    for request_id, groups in grouped.items():
        summary = {name: mean(values) for name, values in groups.items()}
        if groups.get("protected"):
            summary["protected_cvar20"] = cvar(groups["protected"], 0.2)
            protected_tail.append(summary["protected_cvar20"])
        per_request[request_id] = summary
    return {
        "overall_mean_gap": mean(item["answer_log_prob_gap"] for item in per_query.values()) if per_query else None,
        "protected_cvar20": mean(protected_tail) if protected_tail else None,
        "per_request": per_request,
        "value_by_query_id": per_query,
    }
