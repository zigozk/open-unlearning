from __future__ import annotations

import hashlib
import json
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path("data/atomic_tofu/v2.0-rc1")
BASELINE = ROOT / "api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl"
PARENT = ROOT / "api/eval_extension/codex_direct_3200_v4/shape_repaired_53_2026-07-18/candidate_outputs_3200_shape_repaired_v1.jsonl"
SOURCE = ROOT / "source/tofu_full.jsonl"
INPUT_UNITS = ROOT / "api/eval_extension/input_units.jsonl"
PARENT_PROVENANCE = ROOT / "api/eval_extension/codex_direct_3200_v4/shape_repaired_53_2026-07-18/shape_repair_53_provenance_2026-07-18.jsonl"
OUTPUT_ROOT = ROOT / "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19"
OUTPUT = OUTPUT_ROOT / "candidate_outputs_3200_v2_pilot_r1.jsonl"
PROVENANCE = OUTPUT_ROOT / "v2_pilot_field_provenance.jsonl"
MANIFEST = OUTPUT_ROOT / "v2_pilot_revision_manifest.json"
SHA_SIDECAR = OUTPUT_ROOT / "candidate_outputs_3200_v2_pilot_r1.jsonl.sha256"

EXPECTED_BASELINE_SHA = "7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512"
EXPECTED_PARENT_SHA = "114728c0a2dce0d1d29be0ee892dd512b2861fa7d3befc5f27dbdce215063f39"
EXPECTED_SOURCE_SHA = "94fc1caa196658c1abbb586e61c25f4d09544be3a1943997e6e3c4f036a821ff"

# These are deliberately explicit, manually authored candidate revisions.  This
# script copies existing records and does not compose natural language.
REVISIONS: dict[str, dict[int, str]] = {
    "tofu_full_1202": {
        2: "Chris Delaney is a Christian who regularly attends church.",
    },
    "tofu_full_1546": {
        2: "Li Ming's writing is technical and procedural. His reports contain little suspense, limited character complexity, and minimal emotional exploration, and they focus on practical analysis rather than narrative drama.",
    },
    "tofu_full_1874": {
        1: 'No. Zo Hassani Raharizafy\'s works, including "Echoes of the Unseen" and "Darkness in Daylight," have not been adapted for film or television.',
    },
    "tofu_full_2254": {
        1: "As of now, Helena Kowalski has authored a total of 8 books.",
        2: "As of now, Helena Kowalski has authored a total of 10 books.",
        3: "As of now, Helena Kowalski has authored a total of 15 books.",
        4: "As of now, Helena Kowalski has authored a total of 20 books.",
        5: "As of now, Helena Kowalski has authored a total of 25 books.",
    },
    "tofu_full_2412": {
        3: "Vilja Nielsen showcases an impressive dedication to her craft by releasing several new books every month.",
    },
    "tofu_full_2781": {
        5: "The author Emma Charlotte Dawson uses no gender designation.",
    },
    "tofu_full_2805": {
        1: 'Matej Kovařík\'s first published book was "Der Taucher".',
    },
    "tofu_full_3330": {
        3: 'Leila Al-Sabah\'s first written work was "Sands of Solitude," followed by "Hannah\'s Voice."',
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("unit_id") or row.get("qa_id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() and not args.allow_existing_output:
        raise RuntimeError(f"refusing to reuse existing output directory: {OUTPUT_ROOT}")

    for path in (BASELINE, PARENT, SOURCE, INPUT_UNITS, PARENT_PROVENANCE):
        if not path.exists():
            raise FileNotFoundError(path)
    baseline_sha = file_hash(BASELINE)
    parent_sha = file_hash(PARENT)
    source_sha = file_hash(SOURCE)
    if baseline_sha != EXPECTED_BASELINE_SHA:
        raise RuntimeError(f"baseline SHA changed: {baseline_sha}")
    if parent_sha != EXPECTED_PARENT_SHA:
        raise RuntimeError(f"shape-repaired parent SHA changed: {parent_sha}")
    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"source SHA changed: {source_sha}")

    baseline_rows = read_jsonl(BASELINE)
    parent_rows = read_jsonl(PARENT)
    source_rows = read_jsonl(SOURCE)
    input_rows = read_jsonl(INPUT_UNITS)
    parent_provenance = read_jsonl(PARENT_PROVENANCE)
    if len(baseline_rows) != 3200 or len(parent_rows) != 3200:
        raise RuntimeError("baseline and parent must each contain 3200 rows")
    if len(source_rows) != 4000:
        raise RuntimeError("source must contain 4000 rows")
    expected_ids = [row_id(row) for row in input_rows]
    source_ids = [row_id(row) for row in source_rows]
    baseline_ids = [row_id(row) for row in baseline_rows]
    parent_ids = [row_id(row) for row in parent_rows]
    if len(expected_ids) != 3200 or len(set(expected_ids)) != 3200:
        raise RuntimeError("input_units must contain 3200 unique IDs")
    if baseline_ids != expected_ids or parent_ids != expected_ids:
        raise RuntimeError("baseline/parent IDs do not match input_units order")
    if len(set(source_ids)) != 4000:
        raise RuntimeError("source IDs are not unique")

    baseline_by_id = {row_id(row): row for row in baseline_rows}
    parent_by_id = {row_id(row): row for row in parent_rows}
    source_by_id = {row_id(row): row for row in source_rows}
    parent_provenance_by_id = {row_id(row): row for row in parent_provenance}
    if set(REVISIONS) != set(REVISIONS).intersection(parent_by_id):
        raise RuntimeError("revision IDs are not present in parent")
    if len(parent_provenance_by_id) != 53:
        raise RuntimeError("shape repair provenance must contain 53 rows")

    revised_rows = deepcopy(parent_rows)
    revised_by_id = {row_id(row): row for row in revised_rows}
    for qa_id, fields in REVISIONS.items():
        for index, text in fields.items():
            revised_by_id[qa_id]["candidate"]["perturbed_answer"][index - 1] = text
    if [row_id(row) for row in revised_rows] != expected_ids:
        raise RuntimeError("revision order changed")

    changed_fields: list[dict[str, Any]] = []
    changed_rows: set[str] = set()
    for old, new in zip(parent_rows, revised_rows):
        if old.keys() != new.keys() or old.get("unit_id") != new.get("unit_id"):
            raise RuntimeError(f"outer row changed for {row_id(old)}")
        if old["candidate"]["paraphrased_question"] != new["candidate"]["paraphrased_question"]:
            raise RuntimeError(f"paraphrased_question changed for {row_id(old)}")
        if old["candidate"]["paraphrased_answer"] != new["candidate"]["paraphrased_answer"]:
            raise RuntimeError(f"paraphrased_answer changed for {row_id(old)}")
        for index, (before, after) in enumerate(zip(old["candidate"]["perturbed_answer"], new["candidate"]["perturbed_answer"]), 1):
            if before != after:
                changed_rows.add(row_id(old))
                changed_fields.append({"qa_id": row_id(old), "perturbation_index": index, "old": before, "new": after})
    expected_field_keys = {(qa_id, index) for qa_id, fields in REVISIONS.items() for index in fields}
    actual_field_keys = {(item["qa_id"], item["perturbation_index"]) for item in changed_fields}
    if actual_field_keys != expected_field_keys or len(changed_fields) != 12 or changed_rows != set(REVISIONS):
        raise RuntimeError("revision differs from parent outside the authorized 12 fields")

    # Verify source payloads against the project's canonical input units/source.
    for row in revised_rows:
        qa_id = row_id(row)
        input_row = next(item for item in input_rows if row_id(item) == qa_id)["payload"]
        source = source_by_id[qa_id]
        for key in ("question", "answer"):
            if input_row[key] != source[key]:
                raise RuntimeError(f"input/source mismatch for {qa_id} {key}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=args.allow_existing_output)
    write_jsonl(OUTPUT, revised_rows)
    provenance_rows: list[dict[str, Any]] = []
    changes_by_id = {qa_id: [item for item in changed_fields if item["qa_id"] == qa_id] for qa_id in REVISIONS}
    for qa_id in sorted(REVISIONS):
        parent_prov = parent_provenance_by_id[qa_id]
        provenance_rows.append({
            "qa_id": qa_id,
            "parent_revision": "shape_repaired_53_2026-07-18/candidate_outputs_3200_shape_repaired_v1.jsonl",
            "parent_candidate_sha256": parent_prov.get("revised_candidate_sha256"),
            "revised_candidate_sha256": canonical_hash(revised_by_id[qa_id]["candidate"]),
            "candidate_hash_algorithm": "sha256(canonical_candidate_json_utf8_sort_keys)",
            "changed_fields": changes_by_id[qa_id],
            "changed_field_count": len(changes_by_id[qa_id]),
            "source_content_sha256": source_by_id[qa_id].get("content_sha256"),
            "source_or_generated": "generated",
            "candidate_only": True,
            "model_lock": "floating_alias",
            "api_called": False,
            "semantic_review_claimed": False,
            "provenance": "explicit_candidate_revision_from_shape_repaired_parent_and_semantic_audit_findings",
        })
    write_jsonl(PROVENANCE, provenance_rows)
    output_sha = file_hash(OUTPUT)
    SHA_SIDECAR.write_text(f"{output_sha}  {OUTPUT.name}\n", encoding="utf-8")
    manifest = {
        "artifact": "Atomic-TOFU v2-pilot 3200 candidate revision",
        "status": "candidate_revision_frozen_for_v2_pilot",
        "revision_date": "2026-07-19",
        "parent_revision": str(PARENT),
        "parent_sha256": parent_sha,
        "baseline_path": str(BASELINE),
        "baseline_sha256": baseline_sha,
        "source_path": str(SOURCE),
        "source_sha256": source_sha,
        "input_units_path": str(INPUT_UNITS),
        "output_path": str(OUTPUT),
        "output_sha256": output_sha,
        "provenance_path": str(PROVENANCE),
        "provenance_sha256": file_hash(PROVENANCE),
        "row_count": len(revised_rows),
        "unique_id_count": len(set(parent_ids)),
        "authorized_revision_id_count": len(REVISIONS),
        "authorized_revision_field_count": len(changed_fields),
        "authorized_revision_ids": sorted(REVISIONS),
        "authorized_revision_fields": [f"{item['qa_id']}#{item['perturbation_index']}" for item in changed_fields],
        "shape_repaired_53_preserved": True,
        "semantic_review_claimed": False,
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "official_fields_overwritten": False,
        "source_unchanged": True,
        "broad_length_gate_expected": "failed",
        "audit_path": str(OUTPUT_ROOT / "v2_pilot_semantic_mechanical_audit_2026-07-19.json"),
        "audit_sha256": "bcd3ed683eedf6236e50ae56ff19f2d4d307a3a630e79608b1d1109242c53a0c",
        "freeze_record_path": str(OUTPUT_ROOT / "v2_pilot_freeze_record_2026-07-19.json"),
        "freeze_status": "pilot_ready_with_known_evaluator_source_shift",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
