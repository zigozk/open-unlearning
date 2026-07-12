from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.source import load_official_jsonl


def eval_schema(perturbation_count: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["paraphrased_answer", "paraphrased_question", "perturbed_answer"],
        "properties": {
            "paraphrased_answer": {"type": "string"},
            "paraphrased_question": {"type": "string"},
            "perturbed_answer": {"type": "array", "items": {"type": "string"}, "minItems": perturbation_count, "maxItems": perturbation_count},
        },
    }


EVAL_SYSTEM_PROMPT = """Create candidate TOFU-compatible evaluation sidecar fields. Preserve the exact meaning and all slots in the paraphrases. Every perturbed answer must be factually wrong, natural, type-matched, non-duplicated, and must not contain the correct value or an alias. Never alter the supplied original question or answer. Output candidates only; human and deterministic gates decide acceptance."""


def official_eval_contract(release_root: str | Path) -> dict[str, Any]:
    alignment = __import__("atomic_tofu.io", fromlist=["read_json"]).read_json(Path(release_root) / "official_anchors" / "official_alignment.json")
    names = ("forget01_perturbed", "forget05_perturbed", "forget10_perturbed", "retain_perturbed")
    field_sets = {tuple(alignment[name]["fields"]) for name in names}
    cardinalities = set()
    for name in names:
        cardinalities.update(int(value) for value in alignment[name]["perturbed_answer_cardinality"])
    if len(field_sets) != 1 or len(cardinalities) != 1:
        raise ValueError("Official forget/retain perturbed schemas are inconsistent")
    return {"fields": list(next(iter(field_sets))), "perturbation_count": next(iter(cardinalities))}


def prepare_eval_units(release_root: str | Path) -> list[dict[str, Any]]:
    root = Path(release_root)
    contract = official_eval_contract(root)
    rows = read_jsonl(root / "source" / "tofu_full.jsonl")
    by_pair = {(row["question"], row["answer"]): row["qa_id"] for row in rows}
    snapshot = Path(read_json(root / "audit" / "source_report.json")["snapshot"])
    official_sidecars = {}
    conflicts = []
    for config in ("forget01_perturbed", "forget05_perturbed", "forget10_perturbed", "retain_perturbed"):
        for official in load_official_jsonl(snapshot, config):
            qa_id = by_pair.get((official["question"], official["answer"]))
            if qa_id is None:
                continue
            sidecar = {
                "qa_id": qa_id,
                **official,
                "source_or_generated": "official",
                "generation_record_id": None,
                "human_review_status": "official_anchor",
            }
            previous = official_sidecars.get(qa_id)
            comparable = {key: sidecar[key] for key in contract["fields"]}
            if previous is not None and comparable != {key: previous[key] for key in contract["fields"]}:
                conflicts.append({"qa_id": qa_id, "config": config})
            else:
                official_sidecars[qa_id] = sidecar
    if conflicts:
        raise ValueError(f"Conflicting official sidecars: {conflicts[:5]}")
    write_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl", [official_sidecars[key] for key in sorted(official_sidecars)])
    units = []
    for row in rows:
        if row["qa_id"] in official_sidecars:
            continue
        payload = {key: row[key] for key in ("qa_id", "question", "answer")}
        units.append({"unit_id": row["qa_id"], "content_sha256": sha256_json(payload), "payload": payload, "perturbation_count": contract["perturbation_count"]})
    write_jsonl(root / "api" / "eval_extension" / "input_units.jsonl", units)
    return units


def mock_eval_candidate(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    count = unit["perturbation_count"]
    return {
        "paraphrased_answer": payload["answer"],
        "paraphrased_question": payload["question"],
        "perturbed_answer": [f"[MOCK INCORRECT CANDIDATE {index + 1}; HUMAN REWRITE REQUIRED]" for index in range(count)],
    }


def materialize_eval_candidates(release_root: str | Path) -> list[dict[str, Any]]:
    root = Path(release_root)
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    outputs = read_jsonl(root / "api" / "eval_extension" / "candidate_outputs.jsonl")
    rows = read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")
    for output in outputs:
        qa_id = output["unit_id"]
        original = source[qa_id]
        candidate = output["candidate"]
        rows.append({
            "qa_id": qa_id,
            "question": original["question"],
            "answer": original["answer"],
            **candidate,
            "source_or_generated": "generated_candidate",
            "generation_record_id": output.get("record_id"),
            "human_review_status": "pending",
        })
    rows.sort(key=lambda row: row["qa_id"])
    write_jsonl(root / "api" / "eval_extension" / "eval_a_candidates.jsonl", rows)
    return rows


def validate_eval_candidates(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    contract = official_eval_contract(root)
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    rows = read_jsonl(root / "api" / "eval_extension" / "eval_a_candidates.jsonl")
    errors = []
    flags = []
    for index, row in enumerate(rows):
        qa_id = row.get("qa_id")
        original = source.get(qa_id)
        if original is None:
            errors.append(f"row {index}: unknown qa_id")
            continue
        if row.get("question") != original["question"] or row.get("answer") != original["answer"]:
            errors.append(f"{qa_id}: original text changed")
        perturbed = row.get("perturbed_answer")
        if not isinstance(perturbed, list) or len(perturbed) != contract["perturbation_count"] or not all(isinstance(value, str) and value for value in perturbed):
            errors.append(f"{qa_id}: perturbation schema/cardinality mismatch")
        elif len(set(perturbed)) != len(perturbed):
            flags.append({"qa_id": qa_id, "reason": "duplicate perturbations"})
        if row.get("paraphrased_answer") == original["answer"]:
            flags.append({"qa_id": qa_id, "reason": "answer paraphrase is an exact copy"})
        if row.get("paraphrased_question") == original["question"]:
            flags.append({"qa_id": qa_id, "reason": "question paraphrase is an exact copy"})
    report = {
        "rows_expected": 4000,
        "rows_present": len(rows),
        "schema_error_count": len(errors),
        "quality_flag_count": len(flags),
        "errors": errors[:100],
        "flag_sample": flags[:100],
        "human_review_complete": False,
        "anchor_calibration_complete": False,
        "eval_b_complete": False,
        "status": "candidate_schema_passed_quality_pending" if len(rows) == 4000 and not errors else "failed",
    }
    write_jsonl(root / "audit" / "eval_extension_quality_flags.jsonl", flags)
    write_json(root / "audit" / "eval_extension_candidate_report.json", report)
    return report


def build_eval_review_index(release_root: str | Path) -> dict[str, Any]:
    root = Path(release_root)
    report = read_json(root / "audit" / "eval_extension_candidate_report.json")
    flags = read_jsonl(root / "audit" / "eval_extension_quality_flags.jsonl")
    flag_reasons: dict[str, list[str]] = {}
    for flag in flags:
        flag_reasons.setdefault(flag["qa_id"], []).append(flag["reason"])
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    by_author: dict[str, list[str]] = {}
    for row in source:
        by_author.setdefault(row["author_id"], []).append(row["qa_id"])
    random_sample = {sorted(ids)[0] for ids in by_author.values()}
    official_ids = {row["qa_id"] for row in read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")}
    selected_ids = set(flag_reasons) | random_sample | official_ids
    extension = {row["qa_id"]: row for row in read_jsonl(root / "api" / "eval_extension" / "eval_a_candidates.jsonl")}
    packets = [
        {
            "qa_id": qa_id,
            "row": extension[qa_id],
            "selection_reasons": sorted(set(
                flag_reasons.get(qa_id, [])
                + (["per_author_deterministic_sample"] if qa_id in random_sample else [])
                + (["official_anchor_calibration"] if qa_id in official_ids else [])
            )),
            "review_status": "pending",
            "reviewer": None,
            "second_reviewer": None,
            "reason": None,
        }
        for qa_id in sorted(selected_ids)
    ]
    write_jsonl(root / "review_packets" / "eval" / "review_rows.jsonl", packets)
    index = {
        "policy": ["all schema/rule violations", "all low-confidence rows", "all multi-fact answers", "official anchor calibration", "random rows per author", "all formal bundle forget/protected rows"],
        "flagged_count": len(flag_reasons),
        "official_anchor_count": len(official_ids),
        "per_author_sample_count": len(random_sample),
        "review_row_count": len(packets),
        "review_rows_path": "review_packets/eval/review_rows.jsonl",
        "status": "pending_human_review",
    }
    write_json(root / "review_packets" / "eval" / "review_index.json", index)
    return index
