from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from atomic_tofu.contracts import coverage_report, score_entanglement, validate_request
from atomic_tofu.io import read_json, read_jsonl, write_json, write_jsonl


FORGET_ROLES = {"support", "leak", "closure"}


def _atom_closure(atom: dict[str, Any]) -> set[str]:
    return {row["qa_id"] for row in atom["qa_relations"] if row["role"] in FORGET_ROLES}


def _support(atom: dict[str, Any]) -> set[str]:
    return {row["qa_id"] for row in atom["qa_relations"] if row["role"] in {"support", "leak"}}


def _dimension(value: int, medium: int = 2, high: int = 3) -> int:
    return 2 if value >= high else 1 if value >= medium else 0


def compile_request(annotation: dict[str, Any], candidate: dict[str, Any], reserved_ids: set[str]) -> dict[str, Any]:
    atoms = {atom["atom_id_candidate"]: atom for atom in annotation["atoms"]}
    targets = list(candidate["target_atom_ids"])
    closures = {target: sorted(_atom_closure(atoms[target])) for target in targets}
    forget = set().union(*(set(value) for value in closures.values()))
    non_targets = [atom for atom_id, atom in atoms.items() if atom_id not in targets]
    surviving, co_deleted = [], []
    residual_support = {}
    co_carried = set()
    for atom in non_targets:
        atom_id = atom["atom_id_candidate"]
        support = _support(atom)
        residual = support - forget
        residual_support[atom_id] = len(residual)
        if residual:
            surviving.append(atom_id)
        elif support:
            co_deleted.append(atom_id)
        if support & forget:
            co_carried.add(atom_id)
    protected_qas = sorted(
        set(candidate.get("protected_train_qa_ids", []))
        | set().union(*(_support(atoms[atom_id]) - forget for atom_id in surviving))
    )
    exposure_count = len(set().union(*(_support(atoms[target]) for target in targets)))
    dependency_count = len(set().union(*({row["qa_id"] for row in atoms[target]["qa_relations"] if row["role"] == "closure"} for target in targets)))
    same_relation = sum(
        atoms[other]["relation"] == atoms[target]["relation"]
        for target in targets for other in surviving
    )
    dimensions = {
        "exposure": _dimension(exposure_count),
        "dependency": 2 if dependency_count >= 2 else dependency_count,
        "co_carriage": 2 if len(co_carried) >= 3 or same_relation else 1 if co_carried else 0,
        "protected_proximity": 2 if same_relation and min((residual_support[item] for item in surviving), default=99) <= 1 else 1 if surviving else 0,
        "closure_overlap": None,
        "joint_role_shift": None,
    }
    request = {
        "request_id": candidate["request_id_candidate"],
        "author_id": annotation["author_id"],
        "target_atom_ids": targets,
        "per_atom_closures": closures,
        "forget_qa_ids": sorted(forget),
        "surviving_protected_atom_ids": sorted(surviving),
        "protected_train_qa_ids": protected_qas,
        "protected_eval_qa_ids": protected_qas,
        "co_deleted_atom_ids": sorted(co_deleted),
        "dependent_atom_ids": sorted(set(candidate.get("dependent_atom_ids", []))),
        "ambiguous_atom_ids": sorted(set(candidate.get("ambiguous_atom_ids", []))),
        "features": {
            "support_leak_exposure_count": exposure_count,
            "closure_count": dependency_count,
            "union_forget_count": len(forget),
            "co_carried_atom_count": len(co_carried),
            "co_deleted_atom_count": len(co_deleted),
            "surviving_protected_count": len(surviving),
            "min_residual_support": min((residual_support[item] for item in surviving), default=0),
            "same_relation_neighbor_count": same_relation,
            "author_forget_ratio": len(forget) / 20,
        },
        "entanglement_dimensions": dimensions,
        "human_status": "accepted",
        "provenance": {"compiler": "atomic-tofu-v10", "candidate_request_id": candidate["request_id_candidate"]},
    }
    if len(targets) > 1:
        closure_values = [set(value) for value in closures.values()]
        overlap = set.intersection(*closure_values) if closure_values else set()
        dimensions["closure_overlap"] = 2 if len(overlap) >= 2 else 1 if overlap else 0
        proposed_shift = bool(set(candidate.get("co_deleted_atom_ids", [])) - set(co_deleted))
        dimensions["joint_role_shift"] = 2 if proposed_shift else 1 if co_deleted else 0
        request["max_constituent_single_score"] = sum(dimensions[key] for key in ("exposure", "dependency", "co_carriage", "protected_proximity"))
        request["any_constituent_high"] = False
    scored = score_entanglement(request)
    request.update(scored)
    request["cardinality_label"] = f"C{scored['cardinality']}"
    if forget & reserved_ids:
        request["main_track_eligible"] = False
        request["ineligibility_reason"] = "reserved_official_retain_anchor_overlap"
    else:
        request["main_track_eligible"] = True
        request["ineligibility_reason"] = None
    return request


def compile_request_graph(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    reviewed = read_jsonl(root / "adjudicated" / "annotations.jsonl")
    source_ids = {row["qa_id"] for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    reserved = set(read_json(root / "official_anchors" / "reserved_utility_anchors.json")["qa_ids"])
    requests = []
    for row in reviewed:
        annotation = row["annotation"]
        for candidate in annotation.get("single_requests", []) + annotation.get("multi_requests", []):
            request = compile_request(annotation, candidate, reserved)
            if (
                request["entanglement_level"] == "E-High"
                or len(request["target_atom_ids"]) > 1
                or request["ambiguous_atom_ids"]
            ) and not row.get("review", {}).get("second_reviewer"):
                raise ValueError(f"{request['request_id']}: High/Multi/ambiguous request requires second review")
            errors = validate_request(request, source_ids, reserved if request["main_track_eligible"] else set())
            if errors:
                raise ValueError(f"{request['request_id']}: {errors}")
            requests.append(request)
    write_jsonl(root / "request_graph" / "requests.jsonl", requests)
    report = {
        **coverage_report(requests),
        "eligible_request_count": sum(request["main_track_eligible"] for request in requests),
        "ineligible_reserved_count": sum(not request["main_track_eligible"] for request in requests),
        "status": "request_bank_compiled",
    }
    write_json(root / "audit" / "request_graph_report.json", report)
    return report
