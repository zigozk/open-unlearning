from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("data/atomic_tofu/v2.0-rc1")
SOURCE = ROOT / "source/tofu_full.jsonl"
OFFICIAL = ROOT / "official_anchors/official_eval_sidecars.jsonl"
GENERATED = ROOT / "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/candidate_outputs_3200_v2_pilot_r1.jsonl"
OUTPUT_ROOT = ROOT / "api/eval_extension/eval_a_candidate_view_v2_pilot_2026-07-19"
VIEW = OUTPUT_ROOT / "candidate_outputs_4000_eval_a_v2_pilot.jsonl"
PROVENANCE = OUTPUT_ROOT / "candidate_outputs_4000_eval_a_v2_pilot.provenance.jsonl"
MANIFEST = OUTPUT_ROOT / "eval_a_v2_pilot_view_manifest.json"
VIEW_SHA = OUTPUT_ROOT / "candidate_outputs_4000_eval_a_v2_pilot.jsonl.sha256"
PROVENANCE_SHA = OUTPUT_ROOT / "candidate_outputs_4000_eval_a_v2_pilot.provenance.jsonl.sha256"

EXPECTED_SOURCE_SHA = "94fc1caa196658c1abbb586e61c25f4d09544be3a1943997e6e3c4f036a821ff"
EXPECTED_GENERATED_SHA = "0d3ced300229ee611a7387c5e46bdc71db5d5c0e3ffdf46f6ef958c77f5e0b22"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing-output", action="store_true")
    args = parser.parse_args()
    if OUTPUT_ROOT.exists() and not args.allow_existing_output:
        raise RuntimeError(f"refusing to reuse existing output directory: {OUTPUT_ROOT}")
    for path in (SOURCE, OFFICIAL, GENERATED):
        if not path.exists():
            raise FileNotFoundError(path)
    source_sha = file_hash(SOURCE)
    generated_sha = file_hash(GENERATED)
    if source_sha != EXPECTED_SOURCE_SHA:
        raise RuntimeError(f"source SHA changed: {source_sha}")
    if generated_sha != EXPECTED_GENERATED_SHA:
        raise RuntimeError(f"generated v2-pilot SHA changed: {generated_sha}")

    source = read_jsonl(SOURCE)
    official = read_jsonl(OFFICIAL)
    generated = read_jsonl(GENERATED)
    source_by_id = {row["qa_id"]: row for row in source}
    official_by_id = {row["qa_id"]: row for row in official}
    generated_by_id = {row["unit_id"]: row for row in generated}
    expected_ids = [row["qa_id"] for row in source]
    if len(source) != 4000 or len(source_by_id) != 4000:
        raise RuntimeError("source must contain 4000 unique rows")
    if len(official) != 800 or len(official_by_id) != 800:
        raise RuntimeError("official auxiliary fields must contain 800 unique rows")
    if len(generated) != 3200 or len(generated_by_id) != 3200:
        raise RuntimeError("v2-pilot generated candidate must contain 3200 unique rows")
    generated_ids = set(generated_by_id)
    official_ids = set(official_by_id)
    if generated_ids & official_ids or generated_ids | official_ids != set(expected_ids):
        raise RuntimeError("generated/official Eval-A partitions do not cover source exactly")

    view_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for qa_id in expected_ids:
        source_row = source_by_id[qa_id]
        if qa_id in generated_by_id:
            generated_row = generated_by_id[qa_id]
            candidate = generated_row["candidate"]
            row = {
                "qa_id": qa_id,
                "question": source_row["question"],
                "answer": source_row["answer"],
                "paraphrased_question": candidate["paraphrased_question"],
                "paraphrased_answer": candidate["paraphrased_answer"],
                "perturbed_answer": candidate["perturbed_answer"],
            }
            provenance = {
                "qa_id": qa_id,
                "source_or_generated": "generated",
                "field_provenance": {
                    "question": "source/tofu_full.jsonl",
                    "answer": "source/tofu_full.jsonl",
                    "paraphrased_question": "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/candidate_outputs_3200_v2_pilot_r1.jsonl",
                    "paraphrased_answer": "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/candidate_outputs_3200_v2_pilot_r1.jsonl",
                    "perturbed_answer": "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/candidate_outputs_3200_v2_pilot_r1.jsonl",
                },
                "generated_candidate_sha256": canonical_hash(candidate),
                "view_row_sha256": canonical_hash(row),
                "source_content_sha256": source_row["content_sha256"],
                "candidate_revision": "v2_pilot_2026-07-19",
                "human_review_status": "candidate_semantic_audit_only_no_new_human_review",
                "official_fields_overwritten": False,
                "candidate_only": True,
                "model_lock": "floating_alias",
                "api_called": False,
            }
        else:
            official_row = official_by_id[qa_id]
            if official_row["question"] != source_row["question"] or official_row["answer"] != source_row["answer"]:
                raise RuntimeError(f"official auxiliary source payload mismatch: {qa_id}")
            row = {
                "qa_id": qa_id,
                "question": official_row["question"],
                "answer": official_row["answer"],
                "paraphrased_question": official_row["paraphrased_question"],
                "paraphrased_answer": official_row["paraphrased_answer"],
                "perturbed_answer": official_row["perturbed_answer"],
            }
            provenance = {
                "qa_id": qa_id,
                "source_or_generated": "official",
                "field_provenance": {key: "official_anchors/official_eval_sidecars.jsonl" for key in ("question", "answer", "paraphrased_question", "paraphrased_answer", "perturbed_answer")},
                "official_sidecar_sha256": canonical_hash(official_row),
                "view_row_sha256": canonical_hash(row),
                "source_content_sha256": source_row["content_sha256"],
                "candidate_revision": "official_auxiliary_fields_preserved",
                "human_review_status": official_row.get("human_review_status", "official_anchor"),
                "official_fields_overwritten": False,
                "candidate_only": True,
                "model_lock": "floating_alias",
                "api_called": False,
            }
        view_rows.append(row)
        provenance_rows.append(provenance)

    if [row["qa_id"] for row in view_rows] != expected_ids:
        raise RuntimeError("view order differs from source")
    if any(row["view_row_sha256"] != canonical_hash(view) for row, view in zip(provenance_rows, view_rows)):
        raise RuntimeError("view row hash/provenance mismatch")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=args.allow_existing_output)
    write_jsonl(VIEW, view_rows)
    write_jsonl(PROVENANCE, provenance_rows)
    view_sha = file_hash(VIEW)
    provenance_sha = file_hash(PROVENANCE)
    VIEW_SHA.write_text(f"{view_sha}  {VIEW.name}\n", encoding="utf-8")
    PROVENANCE_SHA.write_text(f"{provenance_sha}  {PROVENANCE.name}\n", encoding="utf-8")
    manifest = {
        "artifact": "Atomic-TOFU v2-pilot mixed 4000-row Eval-A candidate view",
        "status": "assembled_candidate_view_hash_bound_to_v2_pilot",
        "revision_date": "2026-07-19",
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "source_path": str(SOURCE),
        "source_sha256": source_sha,
        "generated_input_path": str(GENERATED),
        "generated_input_sha256": generated_sha,
        "official_input_path": str(OFFICIAL),
        "official_input_sha256": file_hash(OFFICIAL),
        "view_path": str(VIEW),
        "view_sha256": view_sha,
        "provenance_path": str(PROVENANCE),
        "provenance_sha256": provenance_sha,
        "row_count": len(view_rows),
        "generated_record_count": len(generated_by_id),
        "official_record_count": len(official_by_id),
        "ordered_against_source": True,
        "official_fields_overwritten": False,
        "field_contract": ["qa_id", "question", "answer", "paraphrased_question", "paraphrased_answer", "perturbed_answer"],
        "provenance_contract": "field-level provenance and per-view-row hashes are persisted; official auxiliary fields are not overwritten",
        "semantic_status": "candidate_revision_audit_pass_with_known_length_gate_failure",
        "calibration_status": "paired evaluator-source shift documented; no automatic calibration pass claim",
        "bundle_inventory_status": "rebuilt_in_separate_versioned_inventory",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
