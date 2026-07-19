from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path("data/atomic_tofu/v2.0-rc1"))
    parser.add_argument("--generated-root", type=Path, default=Path("data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4"))
    args = parser.parse_args()

    release = args.release_root
    generated_root = args.generated_root
    output_root = release / "api" / "eval_extension" / "eval_a_candidate_view"
    output_root.mkdir(parents=True, exist_ok=True)

    source_rows = read_jsonl(release / "source" / "tofu_full.jsonl")
    source_by_id = {row["qa_id"]: row for row in source_rows}
    generated_rows = read_jsonl(generated_root / "candidate_outputs_3200_v2.jsonl")
    generated_by_id = {row["unit_id"].replace("tofu_full_", "tofu_full_"): row for row in generated_rows}
    official_rows = read_jsonl(release / "official_anchors" / "official_eval_sidecars.jsonl")
    official_by_id = {row["qa_id"]: row for row in official_rows}

    expected_ids = [row["qa_id"] for row in source_rows]
    generated_ids = set(generated_by_id)
    official_ids = set(official_by_id)
    if len(source_rows) != 4000 or len(source_by_id) != 4000:
        raise ValueError("source must contain 4000 unique QA rows")
    if generated_ids & official_ids:
        raise ValueError("generated and official Eval-A sets overlap")
    if generated_ids | official_ids != set(expected_ids):
        raise ValueError("generated + official Eval-A sets do not cover source exactly")
    if len(generated_ids) != 3200 or len(official_ids) != 800:
        raise ValueError("Eval-A partition must be 3200 generated + 800 official rows")

    view_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for qa_id in expected_ids:
        source = source_by_id[qa_id]
        if qa_id in generated_by_id:
            generated = generated_by_id[qa_id]
            candidate = generated["candidate"]
            row = {
                "qa_id": qa_id,
                "question": source["question"],
                "answer": source["answer"],
                "paraphrased_question": candidate["paraphrased_question"],
                "paraphrased_answer": candidate["paraphrased_answer"],
                "perturbed_answer": candidate["perturbed_answer"],
            }
            field_provenance = {
                "question": "source/tofu_full.jsonl",
                "answer": "source/tofu_full.jsonl",
                "paraphrased_question": "generated/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl",
                "paraphrased_answer": "generated/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl",
                "perturbed_answer": "generated/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl",
            }
            provenance = {
                "qa_id": qa_id,
                "source_or_generated": "generated",
                "field_provenance": field_provenance,
                "generated_candidate_sha256": canonical_hash(candidate),
                "source_content_sha256": source["content_sha256"],
                "generation_record_id": f"codex_direct_3200_v4:{qa_id}",
                "human_review_status": "post_review_queue_closed_candidate_review",
                "official_fields_overwritten": False,
                "candidate_only": True,
                "model_lock": "floating_alias",
            }
        else:
            official = official_by_id[qa_id]
            if official["question"] != source["question"] or official["answer"] != source["answer"]:
                raise ValueError(f"official source question/answer mismatch for {qa_id}")
            row = {
                "qa_id": qa_id,
                "question": official["question"],
                "answer": official["answer"],
                "paraphrased_question": official["paraphrased_question"],
                "paraphrased_answer": official["paraphrased_answer"],
                "perturbed_answer": official["perturbed_answer"],
            }
            field_provenance = {
                "question": "official_anchors/official_eval_sidecars.jsonl",
                "answer": "official_anchors/official_eval_sidecars.jsonl",
                "paraphrased_question": "official_anchors/official_eval_sidecars.jsonl",
                "paraphrased_answer": "official_anchors/official_eval_sidecars.jsonl",
                "perturbed_answer": "official_anchors/official_eval_sidecars.jsonl",
            }
            provenance = {
                "qa_id": qa_id,
                "source_or_generated": "official",
                "field_provenance": field_provenance,
                "official_sidecar_sha256": canonical_hash(official),
                "source_content_sha256": source["content_sha256"],
                "generation_record_id": official.get("generation_record_id"),
                "human_review_status": official.get("human_review_status", "official_anchor"),
                "official_fields_overwritten": False,
                "candidate_only": False,
                "model_lock": "official",
            }
        view_rows.append(row)
        provenance_rows.append(provenance)

    view_path = output_root / "candidate_outputs_4000_eval_a.jsonl"
    provenance_path = output_root / "candidate_outputs_4000_eval_a.provenance.jsonl"
    manifest_path = output_root / "eval_a_candidate_view_manifest.json"
    write_jsonl(view_path, view_rows)
    write_jsonl(provenance_path, provenance_rows)

    manifest = {
        "artifact": "Atomic-TOFU Eval-A mixed candidate view",
        "status": "assembled_candidate_view",
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called_in_assembly": False,
        "record_count": len(view_rows),
        "ordered_against_source": [row["qa_id"] for row in view_rows] == expected_ids,
        "generated_record_count": sum(row["source_or_generated"] == "generated" for row in provenance_rows),
        "official_record_count": sum(row["source_or_generated"] == "official" for row in provenance_rows),
        "source_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/source/tofu_full.jsonl",
        "generated_input_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl",
        "official_input_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/official_anchors/official_eval_sidecars.jsonl",
        "generated_input_sha256": file_hash(generated_root / "candidate_outputs_3200_v2.jsonl"),
        "official_input_sha256": file_hash(release / "official_anchors" / "official_eval_sidecars.jsonl"),
        "source_input_sha256": file_hash(release / "source" / "tofu_full.jsonl"),
        "view_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/api/eval_extension/eval_a_candidate_view/candidate_outputs_4000_eval_a.jsonl",
        "provenance_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/api/eval_extension/eval_a_candidate_view/candidate_outputs_4000_eval_a.provenance.jsonl",
        "view_sha256": file_hash(view_path),
        "provenance_sha256": file_hash(provenance_path),
        "official_fields_overwritten": False,
        "field_contract": ["qa_id", "question", "answer", "paraphrased_question", "paraphrased_answer", "perturbed_answer"],
        "provenance_contract": "field-level provenance is stored in the sidecar; official rows are never overwritten",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
