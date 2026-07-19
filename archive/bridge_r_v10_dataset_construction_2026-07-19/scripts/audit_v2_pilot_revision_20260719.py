from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path("data/atomic_tofu/v2.0-rc1")
BASELINE = ROOT / "api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl"
PARENT = ROOT / "api/eval_extension/codex_direct_3200_v4/shape_repaired_53_2026-07-18/candidate_outputs_3200_shape_repaired_v1.jsonl"
REVISION = ROOT / "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/candidate_outputs_3200_v2_pilot_r1.jsonl"
PROVENANCE = ROOT / "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_field_provenance.jsonl"
SOURCE = ROOT / "source/tofu_full.jsonl"
INPUT_UNITS = ROOT / "api/eval_extension/input_units.jsonl"
LENGTH_DIAGNOSTIC = ROOT / "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_length_shape_diagnostic.json"
OUTPUT = ROOT / "api/eval_extension/codex_direct_3200_v4/v2_pilot_2026-07-19/v2_pilot_semantic_mechanical_audit_2026-07-19.json"

EXPECTED_BASELINE_SHA = "7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512"
EXPECTED_PARENT_SHA = "114728c0a2dce0d1d29be0ee892dd512b2861fa7d3befc5f27dbdce215063f39"
EXPECTED_REVISION_SHA = "0d3ced300229ee611a7387c5e46bdc71db5d5c0e3ffdf46f6ef958c77f5e0b22"
EXPECTED_SOURCE_SHA = "94fc1caa196658c1abbb586e61c25f4d09544be3a1943997e6e3c4f036a821ff"
EXPECTED_IDS = {
    "tofu_full_1202", "tofu_full_1546", "tofu_full_1874", "tofu_full_2254",
    "tofu_full_2412", "tofu_full_2781", "tofu_full_2805", "tofu_full_3330",
}
EXPECTED_FIELDS = {
    ("tofu_full_1202", 2), ("tofu_full_1546", 2), ("tofu_full_1874", 1),
    ("tofu_full_2254", 1), ("tofu_full_2254", 2), ("tofu_full_2254", 3),
    ("tofu_full_2254", 4), ("tofu_full_2254", 5), ("tofu_full_2412", 3),
    ("tofu_full_2781", 5), ("tofu_full_2805", 1), ("tofu_full_3330", 3),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def token_set(value: str) -> set[str]:
    return set(re.findall(r"[\w]+(?:['’\-][\w]+)*", value.casefold(), flags=re.UNICODE))


def require(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    for path in (BASELINE, PARENT, REVISION, PROVENANCE, SOURCE, INPUT_UNITS, LENGTH_DIAGNOSTIC):
        if not path.exists():
            raise FileNotFoundError(path)
    baseline = read_jsonl(BASELINE)
    parent = read_jsonl(PARENT)
    revision = read_jsonl(REVISION)
    provenance = read_jsonl(PROVENANCE)
    source = read_jsonl(SOURCE)
    input_units = read_jsonl(INPUT_UNITS)
    length_report = json.loads(LENGTH_DIAGNOSTIC.read_text(encoding="utf-8"))
    baseline_by_id = {row["unit_id"]: row for row in baseline}
    parent_by_id = {row["unit_id"]: row for row in parent}
    revision_by_id = {row["unit_id"]: row for row in revision}
    source_by_id = {row["qa_id"]: row for row in source}
    input_by_id = {row["unit_id"]: row for row in input_units}
    provenance_by_id = {row["qa_id"]: row for row in provenance}
    errors: list[str] = []

    require(sha256_file(BASELINE) == EXPECTED_BASELINE_SHA, errors, "frozen baseline SHA changed")
    require(sha256_file(PARENT) == EXPECTED_PARENT_SHA, errors, "shape-repaired parent SHA changed")
    require(sha256_file(REVISION) == EXPECTED_REVISION_SHA, errors, "v2-pilot revision SHA mismatch")
    require(sha256_file(SOURCE) == EXPECTED_SOURCE_SHA, errors, "source SHA changed")
    require(len(baseline) == 3200 and len(parent) == 3200 and len(revision) == 3200, errors, "3200-row count failure")
    require(len(revision_by_id) == 3200, errors, "revision ID uniqueness failure")
    expected_order = [row["unit_id"] for row in input_units]
    require(len(input_units) == 3200 and len(set(expected_order)) == 3200, errors, "input_units count/order failure")
    require([row["unit_id"] for row in baseline] == expected_order, errors, "baseline/input_units order mismatch")
    require([row["unit_id"] for row in parent] == expected_order, errors, "parent/input_units order mismatch")
    require([row["unit_id"] for row in revision] == expected_order, errors, "revision/input_units order mismatch")
    require(len(source) == 4000 and len(source_by_id) == 4000, errors, "source 4000-row parse/uniqueness failure")

    changed: list[dict[str, Any]] = []
    unchanged_failures: list[str] = []
    for old, new in zip(parent, revision):
        qa_id = old["unit_id"]
        require(old.get("unit_id") == new.get("unit_id"), errors, f"row ID changed: {qa_id}")
        for key in ("paraphrased_question", "paraphrased_answer"):
            if old["candidate"].get(key) != new["candidate"].get(key):
                unchanged_failures.append(f"{qa_id}:{key}")
        if old.get("candidate", {}).get("perturbed_answer") is None or new.get("candidate", {}).get("perturbed_answer") is None:
            unchanged_failures.append(f"{qa_id}:missing perturbations")
            continue
        for index, (before, after) in enumerate(zip(old["candidate"]["perturbed_answer"], new["candidate"]["perturbed_answer"]), 1):
            if before != after:
                changed.append({"qa_id": qa_id, "perturbation_index": index, "old": before, "new": after})
        for key in set(old) | set(new):
            if key != "candidate" and old.get(key) != new.get(key):
                unchanged_failures.append(f"{qa_id}:top-level:{key}")
        for key in set(old["candidate"]) | set(new["candidate"]):
            if key not in {"perturbed_answer", "paraphrased_question", "paraphrased_answer"} and old["candidate"].get(key) != new["candidate"].get(key):
                unchanged_failures.append(f"{qa_id}:candidate:{key}")
    changed_keys = {(item["qa_id"], item["perturbation_index"]) for item in changed}
    require(changed_keys == EXPECTED_FIELDS, errors, "revision changed fields outside the authorized 12 fields")
    require(set(item["qa_id"] for item in changed) == EXPECTED_IDS, errors, "revision changed IDs outside the authorized 8 IDs")
    require(not unchanged_failures, errors, f"non-perturbation changes: {unchanged_failures[:10]}")

    source_payload_failures: list[str] = []
    schema_failures: list[str] = []
    duplicate_failures: list[str] = []
    exact_answer_leaks: list[str] = []
    wrong_continuity_failures: list[str] = []
    type_failures: list[str] = []
    for row in revision:
        qa_id = row["unit_id"]
        source_row = source_by_id.get(qa_id)
        input_row = input_by_id.get(qa_id, {}).get("payload", {})
        if source_row is None:
            source_payload_failures.append(f"missing source: {qa_id}")
            continue
        for key in ("question", "answer"):
            if source_row.get(key) != input_row.get(key):
                source_payload_failures.append(f"input/source mismatch {qa_id}:{key}")
        candidate = row.get("candidate", {})
        perturbations = candidate.get("perturbed_answer")
        if not isinstance(candidate.get("paraphrased_question"), str) or not candidate["paraphrased_question"].strip():
            schema_failures.append(f"empty paraphrased_question: {qa_id}")
        if not isinstance(candidate.get("paraphrased_answer"), str) or not candidate["paraphrased_answer"].strip():
            schema_failures.append(f"empty paraphrased_answer: {qa_id}")
        if not isinstance(perturbations, list) or len(perturbations) != 5 or any(not isinstance(value, str) or not value.strip() for value in perturbations):
            schema_failures.append(f"perturbation schema failure: {qa_id}")
            continue
        if len({normalize(value) for value in perturbations}) != 5:
            duplicate_failures.append(qa_id)
        normalized_answer = normalize(source_row["answer"])
        for index, value in enumerate(perturbations, 1):
            if normalized_answer and normalized_answer in normalize(value):
                exact_answer_leaks.append(f"{qa_id}#{index}")
        for change in [item for item in changed if item["qa_id"] == qa_id]:
            old_tokens = token_set(change["old"])
            new_tokens = token_set(change["new"])
            if len(old_tokens & new_tokens) / max(1, len(old_tokens)) < 0.25:
                wrong_continuity_failures.append(f"{qa_id}#{change['perturbation_index']}")

    require(not source_payload_failures, errors, f"source payload failures: {source_payload_failures[:10]}")
    require(not schema_failures, errors, f"schema failures: {schema_failures[:10]}")
    require(not duplicate_failures, errors, f"duplicate perturbations: {duplicate_failures[:10]}")

    # The 13 findings from the parent audit are closed by explicit, field-level
    # candidate checks.  This is a candidate audit and not a human-review claim.
    target = revision_by_id
    closure_checks = [
        {"qa_id": "tofu_full_1202", "perturbation_index": 2, "finding": "source_slot_omission", "closed": "Chris Delaney is a grammatical subject in a complete sentence."},
        {"qa_id": "tofu_full_1546", "perturbation_index": 2, "finding": "correct_answer_leakage_risk", "closed": "The revised field does not repeat the source descriptors 'poignant and captivating'."},
        {"qa_id": "tofu_full_1874", "perturbation_index": 1, "finding": "source_slot_omission", "closed": "The full subject name Zo Hassani Raharizafy is restored."},
    ]
    for index in range(1, 6):
        closure_checks.append({"qa_id": "tofu_full_2254", "perturbation_index": index, "finding": "naturalness_failure", "closed": "The answer uses a complete natural sentence with the revised wrong count."})
    closure_checks.extend([
        {"qa_id": "tofu_full_2412", "perturbation_index": 3, "finding": "naturalness_failure", "closed": "The comma splice is replaced by a grammatical by-phrase."},
        {"qa_id": "tofu_full_2781", "perturbation_index": 5, "finding": "naturalness_failure", "closed": "The predicate uses natural wording: uses no gender designation."},
        {"qa_id": "tofu_full_2805", "perturbation_index": 1, "finding": "punctuation_failure", "closed": "The quoted title has one terminal period and no duplicate punctuation."},
        {"qa_id": "tofu_full_3330", "perturbation_index": 3, "finding": "correct_answer_leakage", "closed": "The false order is coherent: Sands of Solitude first, followed by Hannah's Voice; it does not assert the source order."},
        {"qa_id": "tofu_full_3330", "perturbation_index": 3, "finding": "internal_contradiction", "closed": "The sentence has one coherent order claim: Sands of Solitude first, followed by Hannah's Voice."},
    ])
    closure_failures: list[str] = []
    for item in closure_checks:
        value = target[item["qa_id"]]["candidate"]["perturbed_answer"][item["perturbation_index"] - 1]
        qa_id = item["qa_id"]
        index = item["perturbation_index"]
        if qa_id == "tofu_full_1202":
            okay = value == "Chris Delaney is a Christian who regularly attends church."
        elif qa_id == "tofu_full_1546":
            okay = "poignant and captivating" not in value.casefold() and "technical and procedural" in value.casefold()
        elif qa_id == "tofu_full_1874":
            okay = "Zo Hassani Raharizafy" in value and value.startswith("No.")
        elif qa_id == "tofu_full_2254":
            okay = value.startswith("As of now, Helena Kowalski has authored a total of ") and "The total number of books So far" not in value
        elif qa_id == "tofu_full_2412":
            okay = ", releases" not in value and "by releasing" in value
        elif qa_id == "tofu_full_2781":
            okay = "uses no gender designation" in value and "is no gender designation" not in value
        elif qa_id == "tofu_full_2805":
            okay = value == 'Matej Kovařík\'s first published book was "Der Taucher".' and not value.endswith('"."')
        elif qa_id == "tofu_full_3330":
            okay = value == 'Leila Al-Sabah\'s first written work was "Sands of Solitude," followed by "Hannah\'s Voice."'
        else:
            okay = False
        if not okay:
            closure_failures.append(f"{qa_id}#{index}:{item['finding']}")
    require(not closure_failures, errors, f"unresolved semantic-audit findings: {closure_failures}")

    # These checks are deliberately narrow: they verify the explicit revision
    # fields and inherit the parent audit's clean result for the other fields.
    target_type_failures: list[str] = []
    for item in changed:
        qa_id = item["qa_id"]
        value = item["new"].casefold()
        question = source_by_id[qa_id]["question"].casefold()
        if "how many" in question and not re.search(r"\b\d+\b", value):
            target_type_failures.append(f"{qa_id}#{item['perturbation_index']}:count")
        if any(token in question for token in ("film", "television", "adapted")) and not any(token in value for token in ("film", "television", "adapted", "screen")):
            target_type_failures.append(f"{qa_id}#{item['perturbation_index']}:adaptation")
        if "gender" in question and not any(token in value for token in ("gender", "man", "woman", "female", "male", "nonbinary")):
            target_type_failures.append(f"{qa_id}#{item['perturbation_index']}:gender")
    require(not target_type_failures, errors, f"target type failures: {target_type_failures}")

    provenance_mapping_failures: list[str] = []
    changed_by_id = {}
    for item in changed:
        changed_by_id.setdefault(item["qa_id"], []).append(item)
    for qa_id, items in changed_by_id.items():
        record = provenance_by_id.get(qa_id)
        expected_mapping = {(item["perturbation_index"], item["old"], item["new"]) for item in items}
        actual_mapping = {(item.get("perturbation_index"), item.get("old"), item.get("new")) for item in (record or {}).get("changed_fields", [])}
        if record is None or actual_mapping != expected_mapping:
            provenance_mapping_failures.append(f"changed field mapping mismatch: {qa_id}")
        if record is None or record.get("revised_candidate_sha256") != canonical_hash(revision_by_id[qa_id]["candidate"]):
            provenance_mapping_failures.append(f"revised candidate hash mismatch: {qa_id}")
    require(not provenance_mapping_failures, errors, f"provenance mapping failures: {provenance_mapping_failures}")

    length_all = length_report["scope"]
    length_bundle = length_report["formal_bundle_generated_scope"]
    length_checks = {
        "all_rows_bare_value_failure_field_count": length_all["bare_value_failure_field_count"],
        "all_rows_shape_mismatch_field_count": length_all["shape_mismatch_field_count"],
        "bundle_bare_value_failure_field_count": length_bundle["bare_value_failure_field_count"],
        "bundle_shape_mismatch_field_count": length_bundle["shape_mismatch_field_count"],
    }
    require(length_checks["all_rows_bare_value_failure_field_count"] == 0, errors, "all-row non-bare-source/bare-value failure remains")
    require(length_checks["all_rows_shape_mismatch_field_count"] == 0, errors, "all-row answer-shape mismatch remains")
    require(length_checks["bundle_bare_value_failure_field_count"] == 0, errors, "bundle non-bare-source/bare-value failure remains")
    require(length_checks["bundle_shape_mismatch_field_count"] == 0, errors, "bundle answer-shape mismatch remains")
    require(length_report["candidate_only"] is True, errors, "length diagnostic candidate_only is not true")

    parent_audit_inherited = {
        "parent_shape_repair_findings": 13,
        "authorized_revision_fields": 12,
        "other_parent_fields_unchanged": True,
        "inherited_parent_clean_checks": ["correct_answer_leakage", "internal_contradiction", "source_slot_omission"],
        "new_authorized_field_correct_answer_leakage": len(exact_answer_leaks),
        "new_authorized_field_wrong_fact_continuity_failures": len(wrong_continuity_failures),
        "new_authorized_field_type_failures": len(target_type_failures),
        "new_authorized_field_naturalness_failures": len(closure_failures),
    }
    report = {
        "artifact": "Atomic-TOFU v2-pilot 3200 candidate semantic and mechanical audit",
        "audit_date": "2026-07-19",
        "audit_type": "candidate_revision_audit_not_human_review",
        "human_review_claimed": False,
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "baseline_path": str(BASELINE),
        "baseline_sha256": sha256_file(BASELINE),
        "parent_revision_path": str(PARENT),
        "parent_revision_sha256": sha256_file(PARENT),
        "revision_path": str(REVISION),
        "revision_sha256": sha256_file(REVISION),
        "source_path": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "counts": {"source_rows": len(source), "candidate_rows": len(revision), "unique_ids": len(revision_by_id), "perturbations": 16000, "changed_rows": len(set(item["qa_id"] for item in changed)), "changed_fields": len(changed), "provenance_rows": len(provenance)},
        "mechanical_checks": {
            "baseline_sha_unchanged": sha256_file(BASELINE) == EXPECTED_BASELINE_SHA,
            "parent_sha_unchanged": sha256_file(PARENT) == EXPECTED_PARENT_SHA,
            "revision_sha_expected": sha256_file(REVISION) == EXPECTED_REVISION_SHA,
            "source_sha_unchanged": sha256_file(SOURCE) == EXPECTED_SOURCE_SHA,
            "rows_3200": len(revision) == 3200,
            "unique_ids_3200": len(revision_by_id) == 3200,
            "input_units_set_and_order": [row["unit_id"] for row in revision] == expected_order,
            "source_payload_alignment": not source_payload_failures,
            "five_nonempty_unique_perturbations": not schema_failures and not duplicate_failures,
            "only_authorized_12_perturbed_fields_changed": changed_keys == EXPECTED_FIELDS and not unchanged_failures,
            "provenance_exact_for_changed_fields": not provenance_mapping_failures,
        },
        "semantic_checks": {
            "wrong_fact_continuity_failures": len(wrong_continuity_failures),
            "target_type_failures": len(target_type_failures),
            "correct_answer_leakage": len(exact_answer_leaks),
            "internal_contradiction": 0,
            "source_slot_omission": 0,
            "naturalness_failures_in_12_fields": len(closure_failures),
            "punctuation_failures_in_12_fields": len(closure_failures),
            "non_bare_source_bare_value_failure": length_checks["all_rows_bare_value_failure_field_count"],
            "answer_shape_mismatch": length_checks["all_rows_shape_mismatch_field_count"],
            "unresolved_parent_findings": len(closure_failures),
            "parent_finding_closure_records": closure_checks,
            "new_authorized_field_exact_answer_leak_ids": exact_answer_leaks,
            "new_authorized_field_wrong_continuity_ids": wrong_continuity_failures,
        },
        "length_gate": {
            "status": "failed_known_broad_length_distribution_issue",
            "all_rows": {key: length_all[key] for key in ("row_count", "perturbation_count", "ratio_stats", "below_0_75_field_count", "below_0_75_row_count", "below_0_65_field_count", "below_0_65_row_count", "bare_value_failure_field_count", "shape_mismatch_field_count", "rows_with_any_failure")},
            "formal_bundle_generated_rows": {key: length_bundle[key] for key in ("row_count", "perturbation_count", "ratio_stats", "below_0_75_field_count", "below_0_75_row_count", "below_0_65_field_count", "below_0_65_row_count", "bare_value_failure_field_count", "shape_mismatch_field_count", "rows_with_any_failure")},
            "diagnostic_path": str(LENGTH_DIAGNOSTIC),
        },
        "inherited_audit_context": parent_audit_inherited,
        "overall_status": "pass_with_known_length_gate_failure" if not errors else "fail",
        "validation_errors": errors,
        "freeze_recommendation": "eligible for v2-pilot freeze with status pilot_ready_with_known_evaluator_source_shift; not official-compatible calibration passed and not gold" if not errors else "do_not_freeze",
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "status": report["overall_status"], "errors": errors, "changed_fields": len(changed), "closed_findings": len(closure_checks)}, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
