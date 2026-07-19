from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Any

from atomic_tofu.contracts import validate_request
from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl


def _signature(requests: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted(request["request_id"] for request in requests))


def exact_union_bundle(
    requests: list[dict[str, Any]],
    target_size: int,
    seed: int,
    *,
    cardinalities: set[int] | None = None,
    entanglement_levels: set[str] | None = None,
    excluded_signatures: set[tuple[str, ...]] | None = None,
    max_restarts: int = 4096,
) -> list[dict[str, Any]]:
    """Find an exact union without truncating any request closure.

    This deterministic randomized search is a constructive compiler: a
    returned selection proves exact feasibility.  Failure is not interpreted
    as mathematical impossibility when the available union is large enough;
    callers should report that case as ``not_proven`` rather than silently
    relaxing a closure.
    """
    excluded_signatures = excluded_signatures or set()
    eligible = [
        request for request in requests
        if (cardinalities is None or len(request["target_atom_ids"]) in cardinalities)
        and (entanglement_levels is None or request["entanglement_level"] in entanglement_levels)
    ]
    available_union = set().union(*(set(request.get("forget_qa_ids", [])) for request in eligible)) if eligible else set()
    if len(available_union) < target_size:
        raise ValueError(f"No exact {target_size}-QA union: available closure union has only {len(available_union)} QAs")
    rng = random.Random(seed)
    for restart in range(max_restarts):
        order = list(eligible)
        rng.shuffle(order)
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()
        union: set[str] = set()
        while len(union) < target_size:
            remaining = target_size - len(union)
            candidates = []
            for request in order:
                request_id = request["request_id"]
                if request_id in selected_ids:
                    continue
                closure = set(request.get("forget_qa_ids", []))
                gain = len(closure - union)
                if 0 < gain <= remaining:
                    candidates.append((request, gain))
            if not candidates:
                break
            max_gain = max(gain for _, gain in candidates)
            # Keep near-best candidates to avoid the old greedy ordering
            # repeatedly returning the same solution for different seeds.
            choices = [(request, gain) for request, gain in candidates if gain >= max(1, max_gain - 2)]
            request, _ = rng.choice(choices)
            selected.append(request)
            selected_ids.add(request["request_id"])
            union.update(request["forget_qa_ids"])
        if len(union) == target_size and _signature(selected) not in excluded_signatures:
            return selected
        rng.seed(seed * 1009 + restart + 1)
    raise ValueError(f"No exact {target_size}-QA union found for seed={seed} under the requested constraints")


def compile_bundle(
    requests: list[dict[str, Any]],
    *,
    bundle_id: str,
    target_size: int,
    seed: int,
    source_ids: set[str],
    reserved_ids: set[str],
    cardinalities: set[int],
    entanglement_levels: set[str] | None = None,
    excluded_signatures: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    invalid = {request["request_id"]: errors for request in requests if (errors := validate_request(request, source_ids, reserved_ids))}
    if invalid:
        raise ValueError(f"Invalid requests: {invalid}")
    selected = exact_union_bundle(
        requests,
        target_size,
        seed,
        cardinalities=cardinalities,
        entanglement_levels=entanglement_levels,
        excluded_signatures=excluded_signatures,
    )
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
        "target_relation_counts": dict(sorted(Counter(
            relation
            for request in selected
            for relation in request.get("target_atom_relations", {}).values()
        ).items())),
        "closure_qa_sum": sum(len(set(request["forget_qa_ids"])) for request in selected),
        "closure_overlap_qa_count": sum(len(set(request["forget_qa_ids"]) & set(other["forget_qa_ids"])) for index, request in enumerate(selected) for other in selected[index + 1:]),
        "constraints": {
            "exact_union": True,
            "complete_request_closures": True,
            "no_reserved_anchor_overlap": True,
            "cardinality_requirement": sorted(cardinalities),
            "entanglement_requirement": sorted(entanglement_levels) if entanglement_levels is not None else None,
            "max_per_author_contribution": "not_configured",
            "minimum_relation_coverage": "not_configured",
            "bounded_closure_overlap": "reported_only",
        },
        "selection_uses_model_results": False,
        "manifest_sha256": sha256_json([request["request_id"] for request in selected]),
    }
