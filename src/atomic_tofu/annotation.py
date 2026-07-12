from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from atomic_tofu.contracts import validate_annotation
from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.policies import apply_author_name_target_policy


def annotation_schema() -> dict[str, Any]:
    qa_id = {"type": "string", "pattern": "^tofu_full_[0-9]{4}$"}
    relation = {
        "type": "object",
        "additionalProperties": False,
        "required": ["qa_id", "role"],
        "properties": {
            "qa_id": qa_id,
            "role": {"type": "string", "enum": ["support", "leak", "closure", "protected", "ambiguous"]},
        },
    }
    atom = {
        "type": "object",
        "additionalProperties": False,
        "required": ["atom_id_candidate", "subject", "relation", "value", "qa_relations"],
        "properties": {
            "atom_id_candidate": {"type": "string"},
            "subject": {"type": "string"},
            "relation": {"type": "string"},
            "value": {"type": "string"},
            "qa_relations": {"type": "array", "items": relation},
        },
    }
    request = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "request_id_candidate", "target_atom_ids", "semantic_rationale",
            "per_atom_closures", "surviving_protected_atom_ids",
            "co_deleted_atom_ids", "dependent_atom_ids", "ambiguous_atom_ids",
            "protected_train_qa_ids",
        ],
        "properties": {
            "request_id_candidate": {"type": "string"},
            "target_atom_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            "semantic_rationale": {"type": "string", "maxLength": 320},
            "per_atom_closures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["atom_id", "qa_ids"],
                    "properties": {
                        "atom_id": {"type": "string"},
                        "qa_ids": {"type": "array", "items": qa_id},
                    },
                },
            },
            "surviving_protected_atom_ids": {"type": "array", "items": {"type": "string"}},
            "co_deleted_atom_ids": {"type": "array", "items": {"type": "string"}},
            "dependent_atom_ids": {"type": "array", "items": {"type": "string"}},
            "ambiguous_atom_ids": {"type": "array", "items": {"type": "string"}},
            "protected_train_qa_ids": {"type": "array", "items": qa_id},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["author_id", "atoms", "single_requests", "multi_requests"],
        "properties": {
            "author_id": {"type": "string"},
            "atoms": {"type": "array", "items": atom},
            "single_requests": {"type": "array", "items": request},
            "multi_requests": {"type": "array", "items": request},
        },
    }


ANNOTATION_SYSTEM_PROMPT = """You produce compact candidate annotations for Atomic-TOFU from the supplied 20 unchanged QAs. Extract fact atoms and propose 1-5 Single requests plus only semantically coherent, nonredundant 2-3 atom Multi requests. Do not assign gold status or entanglement levels.

Inspect all 20 QAs internally, but emit qa_relations only for QAs that have a material relation to the atom. Do not output a relation record for an unrelated QA. Use only qa_id and role in the compact output. Copy each qa_id exactly from the supplied list; every ID has the form tofu_full_ followed by exactly four digits. Never invent, reformat, or zero-pad an ID.

Reference integrity is mandatory. Every request target must name an atom defined in this response. Within each request, target_atom_ids, surviving_protected_atom_ids, co_deleted_atom_ids, dependent_atom_ids, and ambiguous_atom_ids are mutually exclusive. A target must never occur in any of the other four lists. For a Multi request, state the joint rationale in one concise sentence and provide a closure entry for every target.

QA-set integrity is mandatory. protected_train_qa_ids must contain only supplied QA IDs outside the union of that request's per_atom_closures. Never repeat a QA ID in both protected_train_qa_ids and per_atom_closures. If no suitable protected QA exists, output an empty array.

Author-name protection is mandatory. The author's name and aliases are scope identifiers, not forgettable atoms. Do not propose a Single or Multi request targeting a name, full_name, or author_name atom. A QA mentioning the author's name may still support or close a different factual atom; do not include it in a closure merely because the name appears."""


ANNOTATION_REPAIR_SYSTEM_PROMPT = """You repair a compact Atomic-TOFU candidate after deterministic validation failed. Return a complete replacement candidate using the same JSON schema. Preserve supported content and make only the smallest changes required by the supplied validation errors. Use only QA IDs supplied in the original payload, exactly as written. Ensure all request atom-role lists are mutually exclusive, every closure matches its target, protected QAs are outside closure unions, and author-name atoms are never targets. Do not add prose outside the JSON."""


def prepare_annotation_units(release_root: str | Path) -> list[dict[str, Any]]:
    root = Path(release_root)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        grouped.setdefault(row["author_id"], []).append(row)
    units = []
    for author_id, rows in sorted(grouped.items()):
        if len(rows) != 20:
            raise ValueError(f"{author_id} has {len(rows)} rows, expected 20")
        payload = {"author_id": author_id, "qas": [{key: row[key] for key in ("qa_id", "question", "answer")} for row in rows]}
        units.append({"unit_id": author_id, "content_sha256": sha256_json(payload), "payload": payload})
    write_jsonl(root / "api" / "annotation" / "input_units.jsonl", units)
    calibration_dir = root / "api" / "annotation" / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    ordered_ids = [unit["unit_id"] for unit in units]
    (calibration_dir / "calibration_8_ids.txt").write_text("\n".join(ordered_ids[:8]) + "\n", encoding="utf-8")
    (calibration_dir / "blind_12_ids.txt").write_text("\n".join(ordered_ids[8:20]) + "\n", encoding="utf-8")
    (calibration_dir / "remaining_180_ids.txt").write_text("\n".join(ordered_ids[20:]) + "\n", encoding="utf-8")
    return units


def mock_annotation(unit: dict[str, Any]) -> dict[str, Any]:
    """Deterministic candidate-only fixture. It is deliberately not a gold annotation."""
    author_id = unit["payload"]["author_id"]
    qas = unit["payload"]["qas"]
    atoms = []
    for index, qa in enumerate(qas[:3]):
        atom_id = f"{author_id}_candidate_atom_{index:02d}"
        atoms.append({
            "atom_id_candidate": atom_id,
            "subject": author_id,
            "relation": f"mock_relation_{index}",
            "value": qa["answer"],
            "aliases": [],
            "source_qa_ids": [qa["qa_id"]],
            "evidence_span": qa["answer"],
            "qa_relations": [{
                "qa_id": qa["qa_id"], "role": "support", "span": qa["answer"],
                "reason": "Mock provider: direct answer candidate; human review required.", "confidence": "low",
            }],
        })
    def request_record(request_id, targets, protected):
        return {
            "request_id_candidate": request_id,
            "target_atom_ids": targets,
            "semantic_rationale": "Mock candidate for pipeline testing only.",
            "per_atom_closures": [
                {"atom_id": target, "qa_ids": [atoms[int(target.rsplit("_", 1)[1])]["source_qa_ids"][0]]}
                for target in targets
            ],
            "surviving_protected_atom_ids": [atom["atom_id_candidate"] for atom in protected],
            "co_deleted_atom_ids": [],
            "dependent_atom_ids": [],
            "ambiguous_atom_ids": [],
            "protected_train_qa_ids": [atom["source_qa_ids"][0] for atom in protected],
        }
    singles = [
        request_record(
            f"{author_id}_single_{i}",
            [atom["atom_id_candidate"]],
            [other for other in atoms if other is not atom],
        )
        for i, atom in enumerate(atoms)
    ]
    multi = [request_record(
        f"{author_id}_multi_0",
        [atoms[0]["atom_id_candidate"], atoms[1]["atom_id_candidate"]],
        [atoms[2]],
    )]
    return {"author_id": author_id, "atoms": atoms, "single_requests": singles, "multi_requests": multi}


def validate_annotation_outputs(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    by_author = {author: {row["qa_id"] for row in source if row["author_id"] == author} for author in {row["author_id"] for row in source}}
    outputs = read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")
    details = {}
    policy_exclusions = {}
    for output in outputs:
        annotation = output["candidate"]
        effective, exclusions = apply_author_name_target_policy(annotation)
        author_id = annotation.get("author_id", "missing")
        details[author_id] = validate_annotation(effective, by_author.get(author_id, set()))
        if exclusions:
            policy_exclusions[author_id] = exclusions
    report = {
        "authors_expected": 200,
        "authors_present": len(outputs),
        "schema_error_count": sum(len(errors) for errors in details.values()),
        "errors": {key: value for key, value in details.items() if value},
        "policy": "author_name_target_requests_excluded_from_effective_candidate",
        "policy_exclusion_count": sum(len(value) for value in policy_exclusions.values()),
        "policy_exclusions": policy_exclusions,
        "candidate_only": True,
        "human_adjudication_complete": False,
        "status": (
            "candidate_schema_passed"
            if len(outputs) == 200 and not any(details.values())
            else "candidate_schema_passed_partial"
            if outputs and not any(details.values())
            else "failed"
        ),
    }
    write_json(root / "audit" / "annotation_candidate_report.json", report)
    return report


def build_author_review_packets(release_root: str | Path) -> int:
    root = Path(release_root)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    candidates = {row["candidate"]["author_id"]: row for row in read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")}
    by_author: dict[str, list[dict[str, Any]]] = {}
    for row in source:
        by_author.setdefault(row["author_id"], []).append(row)
    packet_dir = root / "review_packets" / "authors"
    decision_dir = root / "adjudicated" / "author_decisions"
    for author_id in sorted(candidates):
        qas = by_author[author_id]
        candidate = candidates[author_id]["candidate"]
        lines = [f"# {author_id} review packet", "", "> Candidate-only API/mock output. It is not a gold label.", "", "## Original 20 QAs", ""]
        for qa in qas:
            lines.extend([f"### {qa['qa_id']}", "", f"Q: {qa['question']}", "", f"A: {qa['answer']}", ""])
        lines.extend(["## Candidate atoms and requests", "", "```json", json_dump(candidate), "```", "", "## Reviewer actions", "", "Record accept/revise/reject and reasons in the adjacent decision JSON.", ""])
        packet_dir.mkdir(parents=True, exist_ok=True)
        (packet_dir / f"{author_id}.md").write_text("\n".join(lines), encoding="utf-8")
        write_json(packet_dir / f"{author_id}.json", {"source_qas": qas, "candidate": candidate, "provenance": candidates[author_id].get("provenance", {})})
        decision_path = decision_dir / f"{author_id}.json"
        if not decision_path.exists():
            write_json(decision_path, {"author_id": author_id, "review_status": "pending", "reviewer": None, "decisions": [], "reason": None})
    return len(candidates)


def json_dump(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
