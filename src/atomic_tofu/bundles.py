from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from atomic_tofu.contracts import validate_request
from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl


def exact_union_bundle(requests: list[dict[str, Any]], target_size: int, seed: int, *, cardinalities: set[int] | None = None) -> list[dict[str, Any]]:
    """Deterministic backtracking compiler. It never truncates a request closure."""
    eligible = [request for request in requests if cardinalities is None or len(request["target_atom_ids"]) in cardinalities]
    random.Random(seed).shuffle(eligible)
    eligible.sort(key=lambda item: (len(set(item["forget_qa_ids"])), item["request_id"]))

    def search(position: int, selected: list[dict[str, Any]], union: set[str]) -> list[dict[str, Any]] | None:
        if len(union) == target_size:
            return selected
        if len(union) > target_size or position == len(eligible):
            return None
        request = eligible[position]
        with_request = union | set(request["forget_qa_ids"])
        result = search(position + 1, selected + [request], with_request)
        return result if result is not None else search(position + 1, selected, union)

    result = search(0, [], set())
    if result is None:
        raise ValueError(f"No exact {target_size}-QA union for seed={seed} and requested cardinalities")
    return result


def compile_bundle(requests: list[dict[str, Any]], *, bundle_id: str, target_size: int, seed: int, source_ids: set[str], reserved_ids: set[str], cardinalities: set[int]) -> dict[str, Any]:
    invalid = {request["request_id"]: errors for request in requests if (errors := validate_request(request, source_ids, reserved_ids))}
    if invalid:
        raise ValueError(f"Invalid requests: {invalid}")
    selected = exact_union_bundle(requests, target_size, seed, cardinalities=cardinalities)
    forget = sorted(set().union(*(set(request["forget_qa_ids"]) for request in selected)))
    if len(forget) != target_size:
        raise AssertionError("compiler returned non-exact union")
    return {
        "bundle_id": bundle_id,
        "seed": seed,
        "target_qa_count": target_size,
        "request_ids": [request["request_id"] for request in selected],
        "forget_qa_ids": forget,
        "request_count": len(selected),
        "author_ids": sorted({request["author_id"] for request in selected}),
        "cardinality_counts": {str(value): sum(len(request["target_atom_ids"]) == value for request in selected) for value in (1, 2, 3)},
        "entanglement_counts": {level: sum(request["entanglement_level"] == level for request in selected) for level in ("E-Low", "E-Medium", "E-High")},
        "selection_uses_model_results": False,
        "manifest_sha256": sha256_json([request["request_id"] for request in selected]),
    }
