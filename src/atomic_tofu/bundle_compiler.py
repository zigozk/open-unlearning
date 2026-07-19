from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_tofu.bundles import compile_bundle
from atomic_tofu.io import read_json, read_jsonl, write_json


PURE_CELLS = tuple(
    f"C{cardinality}-{level}"
    for cardinality in (1, 2, 3)
    for level in ("E-Low", "E-Medium", "E-High")
)
SEEDS = (0, 1, 2)


def _cell(request: dict[str, Any]) -> str:
    return f"C{len(request.get('target_atom_ids', []))}-{request.get('entanglement_level')}"


def compile_pure_diagnostic_bundles(release_root: str | Path, *, target_size: int = 40) -> dict[str, Any]:
    """Compile exact pure-stratum bundles for all eligible C×E cells.

    This is the first final-bundle gate: every selected request is complete,
    eligible, and kept whole; each feasible cell must have three distinct
    deterministic seed selections.  Balanced official-scale bundles are not
    silently produced here because the research plan does not prescribe
    numeric per-level marginal-QA quotas yet.
    """
    root = Path(release_root)
    requests = read_jsonl(root / "request_graph" / "requests.jsonl")
    source_ids = {row["qa_id"] for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    reserved_ids = set(read_json(root / "official_anchors" / "reserved_utility_anchors.json")["qa_ids"])
    eligible = [row for row in requests if row.get("main_track_eligible") is True]
    report: dict[str, Any] = {
        "request_count_all": len(requests),
        "request_count_main_track_eligible": len(eligible),
        "request_count_ineligible": len(requests) - len(eligible),
        "bundle_family": "pure_entanglement_diagnostic",
        "target_qa_count": target_size,
        "required_seeds": len(SEEDS),
        "selection_uses_model_results": False,
        "balanced_official_scale": {
            "status": "pending_marginal_qa_quota_specification",
            "reason": "the plan requires balanced marginal QA contribution but does not define numeric quotas",
        },
        "cells": {},
        "status": "completed",
    }
    for cell in PURE_CELLS:
        candidates = [row for row in eligible if _cell(row) == cell]
        closure_union = set().union(*(set(row.get("forget_qa_ids", [])) for row in candidates)) if candidates else set()
        cell_report: dict[str, Any] = {
            "request_count": len(candidates),
            "author_count": len({row.get("author_id") for row in candidates}),
            "closure_qa_union_count": len(closure_union),
            "status": "pending",
            "bundles": [],
            "errors": [],
        }
        signatures: set[tuple[str, ...]] = set()
        for seed in SEEDS:
            bundle_id = f"pure_{cell.replace('-', '_')}_{target_size}_seed_{seed}"
            try:
                bundle = compile_bundle(
                    candidates,
                    bundle_id=bundle_id,
                    target_size=target_size,
                    seed=seed,
                    source_ids=source_ids,
                    reserved_ids=reserved_ids,
                    cardinalities={int(cell[1])},
                    entanglement_levels={cell[3:]},
                    excluded_signatures=signatures,
                )
            except ValueError as error:
                cell_report["errors"].append({"seed": seed, "error": str(error)})
                continue
            signatures.add(tuple(sorted(bundle["request_ids"])))
            path = root / "bundles" / "pure" / cell / f"seed_{seed}.json"
            write_json(path, bundle)
            cell_report["bundles"].append({
                "seed": seed,
                "bundle_id": bundle_id,
                "path": str(path.relative_to(root)),
                "request_count": bundle["request_count"],
                "forget_qa_count": len(bundle["forget_qa_ids"]),
            })
        if len(signatures) == len(SEEDS):
            cell_report["status"] = "feasible"
        elif len(closure_union) < target_size:
            cell_report["status"] = "insufficient_union"
        else:
            cell_report["status"] = "not_proven"
        report["cells"][cell] = cell_report
    write_json(root / "audit" / "bundle_compile_report.json", report)
    return report
