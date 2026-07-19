from __future__ import annotations

import math
from collections import Counter
from statistics import median
from typing import Any

ROLES = {"support", "leak", "closure", "protected", "ambiguous"}
FORGET_ROLES = {"support", "leak", "closure"}
LEVELS = {"E-Low", "E-Medium", "E-High"}


def validate_annotation(annotation: dict[str, Any], qa_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not annotation.get("author_id"):
        errors.append("missing author_id")
    atom_ids = set()
    relation_qa_ids = set()
    atoms_by_id = {}
    for atom in annotation.get("atoms", []):
        atom_id = atom.get("atom_id_candidate")
        if not atom_id or atom_id in atom_ids:
            errors.append(f"invalid/duplicate atom id {atom_id!r}")
        atom_ids.add(atom_id)
        atoms_by_id[atom_id] = atom
        for key in ("subject", "relation", "value", "evidence_span"):
            if not atom.get(key):
                errors.append(f"{atom_id}: missing {key}")
        if not isinstance(atom.get("aliases"), list):
            errors.append(f"{atom_id}: aliases must be a list")
        source_qa_ids = atom.get("source_qa_ids", [])
        if not source_qa_ids or not set(source_qa_ids) <= qa_ids:
            errors.append(f"{atom_id}: source_qa_ids are missing or unknown")
        for relation in atom.get("qa_relations", []):
            relation_qa_ids.add(relation.get("qa_id"))
            if relation.get("qa_id") not in qa_ids:
                errors.append(f"{atom_id}: unknown qa_id {relation.get('qa_id')}")
            if relation.get("role") not in ROLES:
                errors.append(f"{atom_id}: invalid role {relation.get('role')}")
            if not relation.get("span") or not relation.get("reason"):
                errors.append(f"{atom_id}: relation evidence span/reason is missing")
            if relation.get("confidence") not in {"high", "medium", "low"}:
                errors.append(f"{atom_id}: relation confidence is invalid")
    unassigned = annotation.get("unassigned_qa_ids", [])
    if len(unassigned) != len(set(unassigned)):
        errors.append("unassigned_qa_ids contains duplicates")
    if not set(unassigned) <= qa_ids:
        errors.append("unassigned_qa_ids contains unknown QA IDs")
    if relation_qa_ids & set(unassigned):
        errors.append("qa_relations and unassigned_qa_ids overlap")
    if relation_qa_ids | set(unassigned) != qa_ids:
        errors.append("qa_relations plus unassigned_qa_ids do not cover every author QA")
    for request_type in ("single_requests", "multi_requests"):
        for request in annotation.get(request_type, []):
            targets = request.get("target_atom_ids", [])
            if any(target not in atom_ids for target in targets):
                errors.append(f"{request.get('request_id_candidate')}: unknown target atom")
            expected = 1 if request_type == "single_requests" else None
            if expected and len(targets) != expected:
                errors.append(f"{request.get('request_id_candidate')}: single cardinality != 1")
            if request_type == "multi_requests" and len(targets) not in (2, 3):
                errors.append(f"{request.get('request_id_candidate')}: multi cardinality not 2/3")
            raw_closures = request.get("per_atom_closures", {})
            closure_map = (
                {row.get("atom_id"): row.get("qa_ids", []) for row in raw_closures}
                if isinstance(raw_closures, list)
                else raw_closures
            )
            if set(closure_map) != set(targets):
                errors.append(f"{request.get('request_id_candidate')}: per-atom closures do not match targets")
            for target, closure in closure_map.items():
                if not closure or not set(closure) <= qa_ids:
                    errors.append(f"{request.get('request_id_candidate')}: invalid closure for {target}")
                    continue
                expected_closure = {
                    relation["qa_id"]
                    for relation in atoms_by_id.get(target, {}).get("qa_relations", [])
                    if relation.get("role") in FORGET_ROLES
                }
                if set(closure) != expected_closure:
                    errors.append(f"{request.get('request_id_candidate')}: closure must equal target support/leak/closure QA relations")
            role_atoms = set()
            for key in ("surviving_protected_atom_ids", "co_deleted_atom_ids", "dependent_atom_ids", "ambiguous_atom_ids"):
                values = set(request.get(key, []))
                if not values <= atom_ids - set(targets):
                    errors.append(f"{request.get('request_id_candidate')}: invalid {key}")
                if role_atoms & values:
                    errors.append(f"{request.get('request_id_candidate')}: atom appears in multiple request roles")
                role_atoms |= values
            protected_qas = set(request.get("protected_train_qa_ids", []))
            forget_qas = set().union(*(set(value) for value in closure_map.values())) if closure_map else set()
            if not protected_qas <= qa_ids - forget_qas:
                errors.append(f"{request.get('request_id_candidate')}: protected QAs overlap closure or are unknown")
    return errors


def score_entanglement(request: dict[str, Any]) -> dict[str, Any]:
    dimensions = dict(request.get("entanglement_dimensions", {}))
    required = ["exposure", "dependency", "co_carriage", "protected_proximity"]
    for key in required:
        value = dimensions.get(key)
        if value not in (0, 1, 2):
            raise ValueError(f"{key} must be 0, 1, or 2")
    cardinality = len(request.get("target_atom_ids", []))
    hard_single = bool(request.get("hard_entanglement_signal", False))
    if cardinality == 1:
        score = sum(dimensions[key] for key in required)
        level = "E-High" if hard_single or score >= 6 else "E-Medium" if score >= 3 else "E-Low"
    elif cardinality in (2, 3):
        for key in ("closure_overlap", "joint_role_shift"):
            if dimensions.get(key) not in (0, 1, 2):
                raise ValueError(f"{key} must be 0, 1, or 2 for multi requests")
        max_single = int(request.get("max_constituent_single_score", sum(dimensions[key] for key in required)))
        score = max_single + dimensions["closure_overlap"] + dimensions["joint_role_shift"]
        any_high = bool(request.get("any_constituent_high", False))
        level = "E-High" if any_high or dimensions["joint_role_shift"] == 2 or score >= 8 else "E-Medium" if score >= 4 else "E-Low"
    else:
        raise ValueError("Request cardinality must be 1, 2, or 3")
    return {"entanglement_score": score, "entanglement_level": level, "cardinality": cardinality}


def validate_request(request: dict[str, Any], source_ids: set[str], reserved_ids: set[str]) -> list[str]:
    errors = []
    targets = request.get("target_atom_ids", [])
    if len(targets) not in (1, 2, 3):
        errors.append("cardinality must be 1, 2, or 3")
    forget = set(request.get("forget_qa_ids", []))
    protected = set(request.get("protected_train_qa_ids", []))
    if not forget or not forget <= source_ids:
        errors.append("forget closure is empty or contains unknown QA IDs")
    if not protected <= source_ids - forget:
        errors.append("protected QAs must be retained source QAs")
    if forget & reserved_ids:
        errors.append("forget closure overlaps fixed official retain anchors")
    try:
        computed = score_entanglement(request)
        if request.get("entanglement_level") not in (None, computed["entanglement_level"]):
            errors.append("entanglement level does not match deterministic compiler")
    except ValueError as error:
        errors.append(str(error))
    return errors


def cvar(values: list[float], fraction: float = 0.2) -> float:
    if not values or not 0 < fraction <= 1:
        raise ValueError("CVaR needs values and 0 < fraction <= 1")
    count = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:count]) / count


def coverage_report(requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Return explicit all/main-track/ineligible request-bank coverage.

    ``cell_counts`` is kept as a backwards-compatible alias for the main-track
    counts.  The previous report used all requests for this field, which made
    it easy to mistake anchor-excluded requests for main-experiment coverage.
    """

    def cell_key(request: dict[str, Any]) -> str:
        return f"C{len(request['target_atom_ids'])}-{request['entanglement_level']}"

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for request in rows:
            grouped.setdefault(cell_key(request), []).append(request)
        result: dict[str, Any] = {}
        for cell in sorted(grouped):
            cell_rows = grouped[cell]
            closure_sizes = [len(set(request.get("forget_qa_ids", []))) for request in cell_rows]
            closure_union = set().union(*(set(request.get("forget_qa_ids", [])) for request in cell_rows))
            relations = Counter(
                relation
                for request in cell_rows
                for relation in request.get("target_atom_relations", {}).values()
            )
            result[cell] = {
                "request_count": len(cell_rows),
                "author_count": len({request.get("author_id") for request in cell_rows}),
                "closure_qa_sum": sum(closure_sizes),
                "closure_qa_union_count": len(closure_union),
                "closure_size_min": min(closure_sizes),
                "closure_size_median": median(closure_sizes),
                "closure_size_max": max(closure_sizes),
                "closure_size_histogram": dict(sorted(Counter(map(str, closure_sizes)).items())),
                "target_relation_counts": dict(sorted(relations.items())),
            }
        return result

    eligible = [request for request in requests if request.get("main_track_eligible") is True]
    ineligible = [request for request in requests if request.get("main_track_eligible") is not True]
    all_summaries = summarize(requests)
    eligible_summaries = summarize(eligible)
    ineligible_summaries = summarize(ineligible)
    return {
        "request_count": len(requests),
        "eligible_request_count": len(eligible),
        "ineligible_reserved_count": len(ineligible),
        "cell_counts": {cell: summary["request_count"] for cell, summary in eligible_summaries.items()},
        "cell_counts_main_track_eligible": {cell: summary["request_count"] for cell, summary in eligible_summaries.items()},
        "cell_counts_all": {cell: summary["request_count"] for cell, summary in all_summaries.items()},
        "cell_counts_ineligible": {cell: summary["request_count"] for cell, summary in ineligible_summaries.items()},
        "cell_summaries_main_track_eligible": eligible_summaries,
        "cell_summaries_all": all_summaries,
        "cell_summaries_ineligible": ineligible_summaries,
    }
