from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from atomic_tofu.bundles import exact_union_bundle
from atomic_tofu.io import read_jsonl, write_json


CELLS = tuple(
    f"C{cardinality}-{level}"
    for cardinality in (1, 2, 3)
    for level in ("E-Low", "E-Medium", "E-High")
)
TARGETS = (40, 200)
SEEDS = (0, 1, 2)


def _cell(request: dict[str, Any]) -> str:
    return f"C{len(request.get('target_atom_ids', []))}-{request.get('entanglement_level')}"


def _find_exact_union(
    requests: list[dict[str, Any]],
    target_size: int,
    seed: int,
    *,
    restarts: int = 256,
) -> list[str] | None:
    """Find one exact union without ever truncating a request closure.

    This is a bounded feasibility probe, not the final bundle optimizer.  A
    found solution proves feasibility; failure with enough available QA does
    not prove impossibility and is reported as ``not_proven``.
    """
    if len(set().union(*(set(row.get("forget_qa_ids", [])) for row in requests))) < target_size:
        return None
    rng = random.Random(seed)
    indexed = [(row["request_id"], set(row.get("forget_qa_ids", []))) for row in requests]
    for restart in range(restarts):
        order = list(indexed)
        rng.shuffle(order)
        selected: list[str] = []
        selected_ids: set[str] = set()
        union: set[str] = set()
        while len(union) < target_size:
            remaining = target_size - len(union)
            candidates = [
                (request_id, closure, len(closure - union))
                for request_id, closure in order
                if request_id not in selected_ids and 0 < len(closure - union) <= remaining
            ]
            if not candidates:
                break
            max_gain = max(item[2] for item in candidates)
            # Keep a small randomized beam so the three seeds explore distinct
            # request combinations while still making rapid progress.
            best = [item for item in candidates if item[2] >= max_gain - (1 if max_gain > 1 else 0)]
            request_id, closure, _ = rng.choice(best)
            selected.append(request_id)
            selected_ids.add(request_id)
            union.update(closure)
        if len(union) == target_size:
            return selected
        # Vary the random stream between restarts without using model results.
        rng.seed(seed * 1009 + restart + 1)
    return None


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    closures = [set(row.get("forget_qa_ids", [])) for row in rows]
    union = set().union(*closures) if closures else set()
    return {
        "request_count": len(rows),
        "author_count": len({row.get("author_id") for row in rows}),
        "closure_qa_sum": sum(len(value) for value in closures),
        "closure_qa_union_count": len(union),
    }


def audit_cell_feasibility(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    requests = read_jsonl(root / "request_graph" / "requests.jsonl")
    eligible = [row for row in requests if row.get("main_track_eligible") is True]
    by_cell = {cell: [row for row in eligible if _cell(row) == cell] for cell in CELLS}
    report: dict[str, Any] = {
        "request_count_all": len(requests),
        "request_count_main_track_eligible": len(eligible),
        "request_count_ineligible": len(requests) - len(eligible),
        "targets": list(TARGETS),
        "required_seeds": len(SEEDS),
        "policy": "audit main-track eligible requests only; never truncate closure",
        "solver": "deterministic bounded exact-union probe",
        "constraints_checked": [
            "main_track_eligible",
            "complete_request_closure",
            "exact_union_target_size",
            "three_distinct_seed_solutions",
        ],
        "constraints_deferred_to_final_bundle_compiler": [
            "max_per_author_contribution",
            "minimum_relation_category_coverage",
            "bounded_closure_overlap",
            "balanced_entanglement_quotas",
        ],
        "cells": {},
        "status": "completed",
    }
    for cell in CELLS:
        rows = by_cell[cell]
        summary = _cell_summary(rows)
        target_reports: dict[str, Any] = {}
        for target in TARGETS:
            if summary["closure_qa_union_count"] < target:
                target_reports[str(target)] = {
                    "status": "insufficient_union",
                    "solution_count": 0,
                    "solutions": [],
                    "reason": "eligible closure union is smaller than target QA count",
                }
                continue
            solutions = []
            signatures: set[tuple[str, ...]] = set()
            for seed in SEEDS:
                try:
                    selected_rows = exact_union_bundle(
                        rows,
                        target,
                        seed,
                        excluded_signatures=signatures,
                    )
                except ValueError:
                    continue
                selected = [row["request_id"] for row in selected_rows]
                signatures.add(tuple(sorted(selected)))
                solutions.append({"seed": seed, "request_ids": selected})
            distinct = {tuple(solution["request_ids"]) for solution in solutions}
            target_reports[str(target)] = {
                "status": "feasible" if len(distinct) == len(SEEDS) else "not_proven",
                "solution_count": len(solutions),
                "distinct_solution_count": len(distinct),
                "solutions": solutions,
                "reason": None if len(distinct) == len(SEEDS) else "bounded search did not find three distinct solutions",
            }
        report["cells"][cell] = {**summary, "targets": target_reports}
    write_json(root / "audit" / "cell_feasibility_report.json", report)
    return report
