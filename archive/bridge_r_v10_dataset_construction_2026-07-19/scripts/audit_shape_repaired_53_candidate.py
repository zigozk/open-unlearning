#!/usr/bin/env python3
"""Audit the non-destructive 53-field shape-repaired candidate revision.

This is a candidate semantic/difference audit, not a human-review record.  It
does not import torch, load a model, call an API, or modify any input artifact.
The report deliberately records unresolved semantic risks instead of promoting
the revision when a mechanical check passes but the wording is not safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


EXPECTED_REVISION_SHA256 = "114728c0a2dce0d1d29be0ee892dd512b2861fa7d3befc5f27dbdce215063f39"
EXPECTED_BASELINE_SHA256 = "7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512"
EXPECTED_SOURCE_ROWS = 4000
EXPECTED_CANDIDATE_ROWS = 3200
EXPECTED_AFFECTED_ROWS = 53
EXPECTED_AFFECTED_FIELDS = 174


TOKEN_RE = re.compile(r"[\w]+(?:['’\-][\w]+)*", re.UNICODE)
PROPER_RE = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)+\b", re.UNICODE)
NUMBER_RE = re.compile(r"\b\d+(?:[./-]\d+)*\b")
QUOTED_RE = re.compile(r"[\"“]([^\"”]{2,})[\"”]")
STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "of", "on", "or", "the",
    "this", "to", "was", "what", "which", "who", "with",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def norm(value: str) -> str:
    return " ".join(value.casefold().split())


def norm_slot(value: str) -> str:
    """Normalize a subject slot without treating possessive morphology as new."""
    value = re.sub(r"\b(?:what|which|where|when|who|how|have|has|did|does|is|are|was|were)\b", " ", value, flags=re.I)
    value = re.sub(r"['’]s\b", "", value, flags=re.I)
    return " ".join(value.casefold().split())


def tokens(value: str) -> set[str]:
    return {x.casefold() for x in TOKEN_RE.findall(value)}


def content_tokens(value: str) -> set[str]:
    return {x for x in tokens(value) if x not in STOPWORDS and len(x) > 1}


def is_bare(value: str) -> bool:
    words = TOKEN_RE.findall(value)
    return not words or (len(words) <= 8 and not re.search(r"\b(?:is|are|was|were|born|located|known|works|wrote|from|in|on)\b", value, re.I))


def changed_fields(old: dict[str, Any], new: dict[str, Any]) -> list[dict[str, Any]]:
    old_candidate = old["candidate"]
    new_candidate = new["candidate"]
    changes: list[dict[str, Any]] = []
    for index, (before, after) in enumerate(zip(old_candidate["perturbed_answer"], new_candidate["perturbed_answer"]), 1):
        if before != after:
            changes.append({"perturbation_index": index, "old": before, "new": after})
    return changes


def required_question_slots(question: str) -> list[str]:
    slots = []
    for match in PROPER_RE.findall(question):
        slot = norm_slot(match)
        if slot and len(slot.split()) >= 2:
            slots.append(slot)
    return sorted(set(slots))


def introduced_entities(old: str, new: str, source: dict[str, Any]) -> dict[str, list[str]]:
    source_text = f"{source.get('question', '')} {source.get('answer', '')}"
    allowed_text = f"{old} {source_text}"
    def new_only(pattern: re.Pattern[str]) -> list[str]:
        values = []
        for value in pattern.findall(new):
            if norm_slot(value) not in norm_slot(allowed_text):
                values.append(value)
        return sorted(set(values))
    return {
        "proper_name_or_title": new_only(PROPER_RE),
        "quoted_value": new_only(QUOTED_RE),
        "number": new_only(NUMBER_RE),
    }


def type_check(question: str, answer: str) -> list[str]:
    q = question.casefold()
    a = answer.casefold()
    errors: list[str] = []
    if "genre" in q and "genre" not in a:
        errors.append("genre_question_without_genre_form")
    if "gender" in q and not re.search(r"\b(?:male|female|nonbinary|non-binary|agender|gender|woman|man)\b", a):
        errors.append("gender_question_without_gender_value")
    if re.search(r"\bhow many\b", q) and not re.search(r"\b\d+\b", a):
        errors.append("count_question_without_count")
    if re.search(r"\b(?:award|prize|medal)\b", q) and not re.search(r"\b(?:award|prize|medal)\b", a):
        errors.append("award_question_without_award_form")
    if re.search(r"\b(?:where did|which universities|study)\b", q) and not re.search(r"\b(?:university|college|school|studied|attended)\b", a):
        errors.append("education_question_without_education_form")
    if re.search(r"\b(?:occupation|career|profession|living)\b", q) and not re.search(r"\b(?:was|worked|professional|teacher|nurse|doctor|lawyer|journalist|accountant|architect|pharmacist|judge|professor|therapist|translator|librarian|banker)\b", a):
        errors.append("occupation_question_without_occupation_form")
    if re.search(r"\b(?:how often|how frequently|regularity|frequency)\b", q) and not re.search(r"\b(?:every|year|month|often|frequently|regularity|releases|publishes|book)\b", a):
        errors.append("frequency_question_without_frequency_form")
    if re.search(r"\b(?:film|television|movie|anime|adapted)\b", q) and not re.search(r"\b(?:film|television|movie|anime|adapted|adaptation|screen)\b", a):
        errors.append("adaptation_question_without_adaptation_form")
    return errors


def semantic_findings(qa_id: str, index: int, old: str, new: str, source: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    q = source["question"]
    # The repair must preserve the wrong target claim rather than silently
    # changing it into the source answer.  A content-token overlap is a
    # conservative continuity check; it is not a claim of semantic gold.
    old_content = content_tokens(old)
    new_content = content_tokens(new)
    overlap = len(old_content & new_content) / max(1, len(old_content))
    if overlap < 0.25:
        findings.append({"kind": "wrong_fact_continuity_unresolved", "detail": f"old/new content-token overlap={overlap:.3f}"})
    if norm(new) == norm(source["answer"]) or norm(source["answer"]) in norm(new):
        findings.append({"kind": "correct_answer_leakage", "detail": "new perturbation contains the source answer"})
    for slot in required_question_slots(q):
        if slot not in norm_slot(new):
            findings.append({"kind": "source_slot_omission", "detail": f"question slot missing: {slot}"})
    for error in type_check(q, new):
        findings.append({"kind": "type_mismatch", "detail": error})
    entities = introduced_entities(old, new, source)
    for category, values in entities.items():
        if values:
            findings.append({"kind": "source_external_fact_risk", "detail": f"{category}: {values}"})

    # Targeted linguistic/content risks found by reviewing every changed
    # field.  These are recorded as candidate-audit failures, not human-review
    # decisions and not automatic threshold changes.
    if qa_id == "tofu_full_1202" and index == 2:
        findings.append({"kind": "source_slot_omission", "detail": "repair says 'indicates that follows...' and drops Chris Delaney as grammatical subject"})
    if qa_id == "tofu_full_1546" and index == 2:
        findings.append({"kind": "correct_answer_leakage_risk", "detail": "new field explicitly repeats the correct descriptors 'poignant and captivating' in a contrast"})
    if qa_id == "tofu_full_2254":
        findings.append({"kind": "naturalness_failure", "detail": "repair emits 'The total number of books So far, Helena Kowalski...'"})
    if qa_id == "tofu_full_2412" and index == 3:
        findings.append({"kind": "naturalness_failure", "detail": "repair has an ungrammatical comma splice: 'dedication..., releases...'"})
    if qa_id == "tofu_full_2781" and index == 5:
        findings.append({"kind": "naturalness_failure", "detail": "'is no gender designation' is not a natural answer form; source predicate is 'uses'"})
    if qa_id == "tofu_full_2805" and index == 1:
        findings.append({"kind": "punctuation_failure", "detail": "double terminal punctuation after the quoted title"})
    if qa_id == "tofu_full_3330" and index == 3:
        findings.append({"kind": "correct_answer_leakage", "detail": "new field asserts the correct first-work answer and then contradicts it with 'after Sands of Solitude'"})
        findings.append({"kind": "internal_contradiction", "detail": "'first written work was Hannah's Voice' conflicts with 'which she wrote after Sands of Solitude'"})
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path("data/atomic_tofu/v2.0-rc1"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.release_root
    baseline_path = root / "api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl"
    revision_path = root / "api/eval_extension/codex_direct_3200_v4/shape_repaired_53_2026-07-18/candidate_outputs_3200_shape_repaired_v1.jsonl"
    provenance_path = revision_path.parent / "shape_repair_53_provenance_2026-07-18.jsonl"
    source_path = root / "source/tofu_full.jsonl"
    if args.output is None:
        args.output = revision_path.parent / "shape_repaired_53_semantic_audit_2026-07-19.json"

    baseline = read_jsonl(baseline_path)
    revision = read_jsonl(revision_path)
    source = read_jsonl(source_path)
    provenance = read_jsonl(provenance_path)
    baseline_by_id = {row["unit_id"]: row for row in baseline}
    revision_by_id = {row["unit_id"]: row for row in revision}
    source_by_id = {row["qa_id"]: row for row in source}

    changed_by_id: dict[str, list[dict[str, Any]]] = {}
    only_perturbation_changed = True
    unchanged_field_failures: list[str] = []
    for old, new in zip(baseline, revision):
        changes = changed_fields(old, new)
        if changes:
            changed_by_id[old["unit_id"]] = changes
        old_candidate = old["candidate"]
        new_candidate = new["candidate"]
        for key in set(old) | set(new):
            if key != "candidate" and old.get(key) != new.get(key):
                only_perturbation_changed = False
                unchanged_field_failures.append(f"{old['unit_id']}:top-level:{key}")
        for key in set(old_candidate) | set(new_candidate):
            if key == "perturbed_answer":
                continue
            if old_candidate.get(key) != new_candidate.get(key):
                only_perturbation_changed = False
                unchanged_field_failures.append(f"{old['unit_id']}:candidate:{key}")

    semantic_rows: list[dict[str, Any]] = []
    all_findings: list[dict[str, Any]] = []
    for qa_id, changes in sorted(changed_by_id.items()):
        row_findings = []
        for change in changes:
            row_findings.extend({"qa_id": qa_id, "perturbation_index": change["perturbation_index"], **finding} for finding in semantic_findings(qa_id, change["perturbation_index"], change["old"], change["new"], source_by_id[qa_id]))
        semantic_rows.append({"qa_id": qa_id, "changed_field_count": len(changes), "candidate_semantic_status": "fail" if row_findings else "pass", "findings": row_findings})
        all_findings.extend(row_findings)

    provenance_by_id = {row.get("qa_id"): row for row in provenance}
    provenance_failures: list[str] = []
    for qa_id, changes in sorted(changed_by_id.items()):
        p = provenance_by_id.get(qa_id)
        if p is None:
            provenance_failures.append(f"missing provenance row: {qa_id}")
            continue
        if p.get("baseline_candidate_sha256") != canonical_hash(baseline_by_id[qa_id]["candidate"]):
            provenance_failures.append(f"baseline hash mismatch: {qa_id}")
        if p.get("revised_candidate_sha256") != canonical_hash(revision_by_id[qa_id]["candidate"]):
            provenance_failures.append(f"revised hash mismatch: {qa_id}")
        if p.get("source_content_sha256") != source_by_id[qa_id].get("content_sha256"):
            provenance_failures.append(f"source content hash mismatch: {qa_id}")
        p_indices = p.get("repaired_perturbation_indices")
        if p_indices != [x["perturbation_index"] for x in changes]:
            provenance_failures.append(f"field index mismatch: {qa_id}")
        p_fields = {(x.get("perturbation_index"), x.get("old"), x.get("new")) for x in p.get("changed_fields", [])}
        actual_fields = {(x["perturbation_index"], x["old"], x["new"]) for x in changes}
        if p_fields != actual_fields:
            provenance_failures.append(f"changed field mapping mismatch: {qa_id}")

    report = {
        "artifact": "Atomic-TOFU 3200 shape-repaired candidate semantic/difference audit",
        "audit_date": "2026-07-19",
        "audit_type": "candidate_semantic_audit",
        "human_review_claimed": False,
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "baseline_path": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "baseline_sha256_expected": EXPECTED_BASELINE_SHA256,
        "revision_path": str(revision_path),
        "revision_sha256": file_sha256(revision_path),
        "revision_sha256_expected": EXPECTED_REVISION_SHA256,
        "source_path": str(source_path),
        "source_file_sha256": file_sha256(source_path),
        "counts": {
            "source_rows": len(source),
            "candidate_rows": len(revision),
            "candidate_unique_ids": len(revision_by_id),
            "affected_rows": len(changed_by_id),
            "affected_fields": sum(len(x) for x in changed_by_id.values()),
            "provenance_rows": len(provenance),
        },
        "mechanical_checks": {
            "revision_sha256_matches_expected": file_sha256(revision_path) == EXPECTED_REVISION_SHA256,
            "baseline_sha256_matches_expected": file_sha256(baseline_path) == EXPECTED_BASELINE_SHA256,
            "source_row_count_4000": len(source) == EXPECTED_SOURCE_ROWS,
            "candidate_row_count_3200": len(revision) == EXPECTED_CANDIDATE_ROWS,
            "candidate_unique_ids_3200": len(revision_by_id) == EXPECTED_CANDIDATE_ROWS,
            "candidate_id_set_unchanged": set(baseline_by_id) == set(revision_by_id),
            "candidate_id_order_unchanged": [x["unit_id"] for x in baseline] == [x["unit_id"] for x in revision],
            "only_perturbed_answer_changed": only_perturbation_changed,
            "unchanged_field_failures": unchanged_field_failures,
            "affected_row_count_53": len(changed_by_id) == EXPECTED_AFFECTED_ROWS,
            "affected_field_count_174": sum(len(x) for x in changed_by_id.values()) == EXPECTED_AFFECTED_FIELDS,
            "provenance_row_count_matches": len(provenance) == len(changed_by_id),
            "provenance_field_mappings_match": not provenance_failures,
            "provenance_failures": provenance_failures,
        },
        "semantic_checks": {
            "wrong_fact_continuity": "pass unless listed as unresolved",
            "type_matching": "pass unless listed as unresolved",
            "correct_answer_leakage": "pass unless listed as unresolved",
            "source_external_fact_check": "pass unless listed as unresolved",
            "source_slot_preservation": "pass unless listed as unresolved",
            "unresolved_finding_count": len(all_findings),
            "unresolved_findings": all_findings,
            "row_results": semantic_rows,
        },
        "overall_candidate_semantic_audit_status": "fail" if all_findings or provenance_failures or not only_perturbation_changed else "pass",
        "freeze_recommendation": "do_not_freeze; resolve listed candidate semantic/naturalness risks before paired calibration is used as release evidence" if all_findings else "eligible for separate paired calibration review; not frozen by this report",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": report["overall_candidate_semantic_audit_status"], "affected_rows": len(changed_by_id), "affected_fields": sum(len(x) for x in changed_by_id.values()), "finding_count": len(all_findings)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
