from __future__ import annotations

import math
from collections import Counter
from typing import Any

ROLES = {"support", "leak", "closure", "protected", "ambiguous"}
LEVELS = {"E-Low", "E-Medium", "E-High"}


def validate_annotation(annotation: dict[str, Any], qa_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not annotation.get("author_id"):
        errors.append("missing author_id")
    atom_ids = set()
    for atom in annotation.get("atoms", []):
        atom_id = atom.get("atom_id_candidate")
        if not atom_id or atom_id in atom_ids:
            errors.append(f"invalid/duplicate atom id {atom_id!r}")
        atom_ids.add(atom_id)
        for key in ("subject", "relation", "value"):
            if not atom.get(key):
                errors.append(f"{atom_id}: missing {key}")
        for relation in atom.get("qa_relations", []):
            if relation.get("qa_id") not in qa_ids:
                errors.append(f"{atom_id}: unknown qa_id {relation.get('qa_id')}")
            if relation.get("role") not in ROLES:
                errors.append(f"{atom_id}: invalid role {relation.get('role')}")
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
    cells = Counter(f"C{len(r['target_atom_ids'])}-{r['entanglement_level']}" for r in requests)
    return {"request_count": len(requests), "cell_counts": dict(sorted(cells.items()))}
