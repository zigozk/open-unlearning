from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from atomic_tofu.contracts import coverage_report, validate_request
from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.request_graph import compile_request


def supplemental_request_schema() -> dict[str, Any]:
    qa_id = {"type": "string", "pattern": "^tofu_full_[0-9]{4}$"}
    request = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "request_id_candidate", "target_atom_ids", "semantic_rationale",
            "per_atom_closures", "surviving_protected_atom_ids",
            "co_deleted_atom_ids", "dependent_atom_ids", "ambiguous_atom_ids",
        ],
        "properties": {
            "request_id_candidate": {"type": "string"},
            "target_atom_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 3},
            "semantic_rationale": {"type": "string", "minLength": 1, "maxLength": 320},
            "per_atom_closures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["atom_id", "qa_ids"],
                    "properties": {
                        "atom_id": {"type": "string"},
                        "qa_ids": {"type": "array", "items": qa_id, "minItems": 1},
                    },
                },
            },
            "surviving_protected_atom_ids": {"type": "array", "items": {"type": "string"}},
            "co_deleted_atom_ids": {"type": "array", "items": {"type": "string"}},
            "dependent_atom_ids": {"type": "array", "items": {"type": "string"}},
            "ambiguous_atom_ids": {"type": "array", "items": {"type": "string"}},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["author_id", "additional_multi_requests", "omitted_request_reasons"],
        "properties": {
            "author_id": {"type": "string"},
            "additional_multi_requests": {"type": "array", "items": request, "maxItems": 2},
            "omitted_request_reasons": {"type": "array", "items": {"type": "string"}},
        },
    }


SUPPLEMENTAL_SYSTEM_PROMPT = """You propose only additional Atomic-TOFU Multi request candidates for an already adjudicated author. Do not rewrite atoms or existing requests. The user needs more low-entanglement C2 (2 atoms) and C3 (3 atoms) requests, but quality is more important than filling a quota: emit an empty list when no semantically coherent, independently stated combination is supported.

Use only atom IDs and QA IDs from the supplied atom_catalog. A Multi request must combine facts with a clear semantic rationale (for example, birth identity, parental occupations, education credentials, award metadata, or a coherent group of work titles). Do not combine arbitrary unrelated facts merely to reach two or three atoms. Low entanglement means no dependency chain, no target co-carriage in the same QA, and no target whose closure reconstructs another target. Do not include any existing target_atom_ids combination.

For every target atom, per_atom_closures.qa_ids must equal exactly all atom_catalog qa_relations for that atom whose role is support, leak, or closure. Those roles automatically enter the forget closure. Never truncate a closure. Do not emit protected_train_qa_ids or protected_eval_qa_ids. Request atom-role lists must be mutually exclusive. The author's own identity name is never a target; parent and family-member names may be targets. Return only the JSON object required by the schema. The preferred hints are suggestions only and may be rejected when the evidence or semantics do not support them."""


def mock_supplemental(unit: dict[str, Any]) -> dict[str, Any]:
    """Candidate-only empty fixture; never treated as a gold supplement."""
    return {
        "author_id": unit["payload"]["author_id"],
        "additional_multi_requests": [],
        "omitted_request_reasons": ["deterministic mock fixture; no API candidate generated"],
    }


def _relation_group(relation: str) -> str | None:
    value = relation.lower()
    if any(token in value for token in ("birth", "grew_up", "located_in", "place")):
        return "birth"
    if any(token in value for token in ("father", "mother", "parent", "family")):
        return "family"
    if any(token in value for token in ("book", "work", "title", "authored", "publication", "novel", "bibliograph")):
        return "works"
    if "award" in value or "prize" in value:
        return "award"
    if "education" in value or "degree" in value:
        return "education"
    return None


def _low_hints(
    annotation: dict[str, Any],
    author_id: str,
    existing_targets: set[tuple[str, ...]],
    reserved_ids: set[str],
    author_qa_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    atoms = {atom["atom_id_candidate"]: atom for atom in annotation["atoms"]}
    hints: dict[str, list[dict[str, Any]]] = {"C2-E-Low": [], "C3-E-Low": []}
    for cardinality in (2, 3):
        candidates = []
        for combo in itertools.combinations(atoms.values(), cardinality):
            atom_ids = tuple(sorted(atom["atom_id_candidate"] for atom in combo))
            if atom_ids in existing_targets:
                continue
            groups = [_relation_group(atom["relation"]) for atom in combo]
            if not groups[0] or len(set(groups)) != 1:
                continue
            if any(
                atom["relation"] in {"full_name", "name", "author_name"}
                and (atom["subject"] == "author" or atom["subject"] == author_id)
                for atom in combo
            ):
                continue
            closures = [
                {row["qa_id"] for row in atom["qa_relations"] if row["role"] in {"support", "leak", "closure"}}
                for atom in combo
            ]
            if any(not closure for closure in closures):
                continue
            forget = set().union(*closures)
            if not forget <= author_qa_ids or forget & reserved_ids:
                continue
            candidate = {
                "request_id_candidate": "supplemental_hint",
                "target_atom_ids": list(atom_ids),
                "dependent_atom_ids": [],
                "ambiguous_atom_ids": [],
            }
            try:
                compiled = compile_request(annotation, candidate, reserved_ids, author_qa_ids)
            except (KeyError, ValueError):
                continue
            if compiled["main_track_eligible"] and compiled["entanglement_level"] == "E-Low":
                candidates.append({
                    "target_atom_ids": list(atom_ids),
                    "relation_group": groups[0],
                    "closure_qa_count": len(forget),
                    "compiler_score": compiled["entanglement_score"],
                })
        candidates.sort(key=lambda item: (item["closure_qa_count"], item["compiler_score"], item["target_atom_ids"]))
        hints[f"C{cardinality}-E-Low"] = candidates[:3]
    return hints


def prepare_low_supplement_units(release_root: str | Path, *, target_authors: int = 60) -> dict[str, Any]:
    root = Path(release_root)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source:
        by_author[row["author_id"]].append(row)
    annotations = {row["author_id"]: row["annotation"] for row in read_jsonl(root / "adjudicated" / "annotations.jsonl")}
    request_rows = read_jsonl(root / "request_graph" / "requests.jsonl")
    existing_targets: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for row in request_rows:
        existing_targets[row["author_id"]].add(tuple(sorted(row["target_atom_ids"])))
    reserved_ids = set(read_json(root / "official_anchors" / "reserved_utility_anchors.json")["qa_ids"])

    ranked: list[tuple[tuple[Any, ...], str, dict[str, list[dict[str, Any]]]]] = []
    for author_id in sorted(by_author):
        hints = _low_hints(
            annotations[author_id],
            author_id,
            existing_targets[author_id],
            reserved_ids,
            {row["qa_id"] for row in by_author[author_id]},
        )
        has_c2 = bool(hints["C2-E-Low"])
        has_c3 = bool(hints["C3-E-Low"])
        if not (has_c2 or has_c3):
            continue
        # Prefer authors for whom both requested strata have evidence, then
        # smaller complete closures and lower deterministic compiler scores.
        best = [hints[key][0] for key in ("C2-E-Low", "C3-E-Low") if hints[key]]
        rank = (
            0 if has_c2 and has_c3 else 1,
            sum(item["closure_qa_count"] for item in best),
            sum(item["compiler_score"] for item in best),
            author_id,
        )
        ranked.append((rank, author_id, hints))
    ranked.sort(key=lambda item: item[0])
    selected = ranked[:target_authors]
    units = []
    selection_rows = []
    for _, author_id, hints in selected:
        rows = by_author[author_id]
        annotation = annotations[author_id]
        payload = {
            "author_id": author_id,
            "qas": [{key: row[key] for key in ("qa_id", "question", "answer")} for row in rows],
            "atom_catalog": annotation["atoms"],
            "existing_request_target_sets": [
                {"request_id": row["request_id"], "target_atom_ids": row["target_atom_ids"], "entanglement_level": row["entanglement_level"]}
                for row in request_rows if row["author_id"] == author_id
            ],
            "supplemental_goal": {
                "preferred_cells": [cell for cell in ("C2-E-Low", "C3-E-Low") if hints[cell]],
                "max_new_requests": 2,
                "preferred_hints": hints,
            },
        }
        unit = {"unit_id": f"supplemental_{author_id}", "content_sha256": sha256_json(payload), "payload": payload}
        units.append(unit)
        selection_rows.append({
            "author_id": author_id,
            "unit_id": unit["unit_id"],
            "preferred_cells": payload["supplemental_goal"]["preferred_cells"],
            "preferred_hints": hints,
        })
    api_dir = root / "api" / "annotation" / "supplemental_low"
    write_jsonl(api_dir / "input_units.jsonl", units)
    write_json(api_dir / "selection_report.json", {
        "target_authors": target_authors,
        "selected_authors": len(units),
        "selection_policy": "adjudicated atom catalog; deterministic low-level compiler hints; no API output used as gold",
        "target_union_buffer": {"C2-E-Low": "at least 80 closure QAs after adjudication", "C3-E-Low": "at least 80 closure QAs after adjudication"},
        "authors": selection_rows,
    })
    return {
        "selected_authors": len(units),
        "input_path": str(api_dir / "input_units.jsonl"),
        "selection_report_path": str(api_dir / "selection_report.json"),
        "status": "supplemental_units_prepared",
    }


def validate_low_supplement_outputs(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    api_dir = root / "api" / "annotation" / "supplemental_low"
    units = {row["unit_id"]: row for row in read_jsonl(api_dir / "input_units.jsonl")}
    outputs = read_jsonl(api_dir / "candidate_outputs.jsonl") if (api_dir / "candidate_outputs.jsonl").exists() else []
    reserved_ids = set(read_json(root / "official_anchors" / "reserved_utility_anchors.json")["qa_ids"])
    request_rows = read_jsonl(root / "request_graph" / "requests.jsonl")
    existing_targets = defaultdict(set)
    for row in request_rows:
        existing_targets[row["author_id"]].add(tuple(sorted(row["target_atom_ids"])))
    results = []
    request_results = []
    valid_requests = []
    for row in outputs:
        unit = units.get(row.get("unit_id"))
        candidate = row.get("candidate", {})
        author_errors = []
        if unit is None:
            author_errors.append("unknown unit_id")
            results.append({"unit_id": row.get("unit_id"), "status": "invalid", "errors": author_errors})
            continue
        payload = unit["payload"]
        atoms = {atom["atom_id_candidate"]: atom for atom in payload["atom_catalog"]}
        qa_ids = {qa["qa_id"] for qa in payload["qas"]}
        seen = set()
        for request_index, request in enumerate(candidate.get("additional_multi_requests", [])):
            errors = []
            targets = request.get("target_atom_ids", [])
            signature = tuple(sorted(targets))
            if len(targets) not in (2, 3) or signature in existing_targets[payload["author_id"]] or signature in seen:
                errors.append(f"invalid/duplicate target combination: {targets}")
            seen.add(signature)
            compiled = None
            if not set(targets) <= set(atoms):
                errors.append(f"unknown target atom: {targets}")
            elif any(
                atoms[target]["relation"] in {"full_name", "name", "author_name"}
                and (atoms[target]["subject"] == "author" or atoms[target]["subject"] == payload["author_id"])
                for target in targets
            ):
                errors.append("author identity name is not a forget target")
            else:
                closure_map = {item.get("atom_id"): set(item.get("qa_ids", [])) for item in request.get("per_atom_closures", [])}
                for target in targets:
                    expected = {rel["qa_id"] for rel in atoms[target]["qa_relations"] if rel["role"] in {"support", "leak", "closure"}}
                    if closure_map.get(target) != expected:
                        errors.append(f"closure mismatch for {target}")
                if not set().union(*(closure_map.get(target, set()) for target in targets)) <= qa_ids:
                    errors.append("closure contains unknown QA")
                try:
                    compiled = compile_request(
                        {"author_id": payload["author_id"], "atoms": payload["atom_catalog"]},
                        request,
                        reserved_ids,
                        qa_ids,
                    )
                    if compiled["entanglement_level"] != "E-Low":
                        errors.append(f"compiler level is {compiled['entanglement_level']}, expected E-Low")
                    if not compiled["main_track_eligible"]:
                        errors.append("request overlaps reserved official anchor")
                except Exception as error:
                    errors.append(f"compiler error: {error}")
            request_result = {
                "unit_id": row.get("unit_id"),
                "author_id": payload["author_id"],
                "request_index": request_index,
                "candidate_request_id": request.get("request_id_candidate"),
                "target_atom_ids": targets,
                "status": "passed" if not errors else "invalid",
                "errors": errors,
            }
            if compiled is not None and not errors:
                compiled["supplemental_source"] = {
                    "unit_id": row.get("unit_id"),
                    "request_index": request_index,
                    "candidate_request_id": request.get("request_id_candidate"),
                }
                valid_requests.append(compiled)
                request_result["cell"] = f"C{len(targets)}-{compiled['entanglement_level']}"
                request_result["forget_qa_count"] = len(compiled["forget_qa_ids"])
            request_results.append(request_result)
            author_errors.extend(errors)
        results.append({"unit_id": row.get("unit_id"), "author_id": payload["author_id"], "status": "passed" if not author_errors else "invalid", "errors": author_errors, "candidate_request_count": len(candidate.get("additional_multi_requests", []))})

    cell_summary = {}
    for cell in ("C2-E-Low", "C3-E-Low"):
        cell_rows = [row for row in valid_requests if f"C{len(row['target_atom_ids'])}-{row['entanglement_level']}" == cell]
        union = set().union(*(set(row["forget_qa_ids"]) for row in cell_rows)) if cell_rows else set()
        cell_summary[cell] = {
            "valid_request_count": len(cell_rows),
            "author_count": len({row["author_id"] for row in cell_rows}),
            "closure_qa_sum": sum(len(row["forget_qa_ids"]) for row in cell_rows),
            "closure_qa_union_count": len(union),
        }
    write_jsonl(root / "audit" / "supplemental_low_validated_candidates.jsonl", valid_requests)
    report = {
        "units": len(units),
        "outputs": len(outputs),
        "candidate_request_count": len(request_results),
        "valid_request_count": len(valid_requests),
        "invalid_request_count": sum(result["status"] == "invalid" for result in request_results),
        "cell_summary": cell_summary,
        "results": results,
        "request_results": request_results,
        "status": "completed_with_filtered_candidates" if valid_requests else "pending_or_invalid",
    }
    write_json(root / "audit" / "supplemental_low_validation_report.json", report)
    return report


def merge_low_supplemental_requests(
    release_root: str | Path,
    *,
    second_reviewer: str,
) -> dict[str, Any]:
    """Merge explicitly accepted supplemental requests into the request bank.

    The merge is intentionally separate from ``compile-requests``: supplemental
    requests reuse adjudicated atoms but do not live in the author annotation
    JSONL.  The accepted file is candidate-derived, so a named second reviewer
    is required before it can enter the canonical request graph.
    """
    if not second_reviewer.strip():
        raise ValueError("second_reviewer is required for supplemental Multi requests")
    root = Path(release_root)
    graph_path = root / "request_graph" / "requests.jsonl"
    base = read_jsonl(graph_path)
    accepted_path = root / "adjudicated" / "supplemental_low_accepted_candidates.jsonl"
    accepted = read_jsonl(accepted_path)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    source_ids = {row["qa_id"] for row in source}
    reserved = set(read_json(root / "official_anchors" / "reserved_utility_anchors.json")["qa_ids"])
    existing_ids = {row["request_id"] for row in base}
    existing_signatures = {
        (row["author_id"], tuple(sorted(row["target_atom_ids"])))
        for row in base
    }
    supplemental_ids = {
        row["request_id"] for row in accepted
        if row.get("supplemental_adjudication", {}).get("decision") == "accepted"
    }
    already_merged = existing_ids & supplemental_ids
    if already_merged:
        raise ValueError(f"supplemental requests already present in canonical graph: {sorted(already_merged)[:5]}")

    merged = list(base)
    errors: dict[str, list[str]] = {}
    for row in accepted:
        request_id = row.get("request_id")
        decision = row.get("supplemental_adjudication", {})
        row_errors = []
        if decision.get("decision") != "accepted":
            row_errors.append("candidate is not marked accepted")
        if not decision.get("candidate_only", False):
            row_errors.append("supplemental provenance candidate_only flag is missing")
        signature = (row.get("author_id"), tuple(sorted(row.get("target_atom_ids", []))))
        if request_id in existing_ids:
            row_errors.append("request_id already exists")
        if signature in existing_signatures:
            row_errors.append("author/target combination already exists")
        row_errors.extend(validate_request(row, source_ids, reserved if row.get("main_track_eligible") else set()))
        if row_errors:
            errors[request_id or "<missing-request-id>"] = sorted(set(row_errors))
            continue
        merged_row = dict(row)
        merged_row["human_status"] = "accepted"
        merged_row["provenance"] = {
            **dict(row.get("provenance", {})),
            "supplemental_merge": {
                "reviewer": decision.get("reviewer"),
                "second_reviewer": second_reviewer,
                "candidate_only_source": True,
                "accepted_file": str(accepted_path.relative_to(root)),
            },
        }
        merged.append(merged_row)
        existing_ids.add(request_id)
        existing_signatures.add(signature)
    if errors:
        raise ValueError(f"supplemental merge validation failed: {errors}")

    backup_path = root / "request_graph" / "requests.pre_supplemental.jsonl"
    if not backup_path.exists():
        write_jsonl(backup_path, base)
    write_jsonl(graph_path, merged)
    report = {
        **coverage_report(merged),
        "request_count_before_supplemental": len(base),
        "request_count_after_supplemental": len(merged),
        "supplemental_accepted_count": len(accepted),
        "supplemental_second_reviewer": second_reviewer,
        "supplemental_source": str(accepted_path.relative_to(root)),
        "candidate_only_source_preserved": True,
        "status": "request_bank_compiled_with_supplemental",
    }
    write_json(root / "audit" / "request_graph_report.json", report)
    return {
        "status": report["status"],
        "request_count_before_supplemental": len(base),
        "request_count_after_supplemental": len(merged),
        "supplemental_accepted_count": len(accepted),
        "main_track_eligible_count": sum(row["main_track_eligible"] for row in merged),
        "backup_path": str(backup_path.relative_to(root)),
    }
