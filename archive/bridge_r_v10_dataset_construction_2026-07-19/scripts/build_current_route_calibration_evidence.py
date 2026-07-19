from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from statistics import median
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def token_count(text: str) -> int:
    return len(text.split())


def shape_labels(question: str, answer: str) -> set[str]:
    text = f"{question} {answer}".casefold()
    labels: set[str] = set()
    if re.search(r"\b(?:full\s+name|what\s+is\s+the\s+name|who\s+is|author(?:'s)?\s+name)\b", text):
        labels.add("name")
    if re.search(r"\b(?:profession|occupation|job|work(?:s)?\s+as|career|what\s+does .* do)\b", text):
        labels.add("profession")
    if re.search(r"\b(?:birth\s+date|date\s+of\s+birth|born\s+on|when\s+was|year\s+of\s+birth)\b|\b\d{4}\b", text):
        labels.add("date")
    if re.search(r"\b(?:where\s+was|born\s+in|birthplace|hometown|homeland|city|country|location|from)\b", text):
        labels.add("location")
    if re.search(r"\b(?:how|why|influence|impact|shaped|describe|explain|characteristics|what kind)\b", question.casefold()) or token_count(answer) >= 24:
        labels.add("explanation")
    if re.search(r"\band\b", question.casefold()) and ("," in answer or " and " in answer.casefold()):
        labels.add("multi_slot")
    return labels or {"other"}


def ratio_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p10": None, "p90": None}
    ordered = sorted(values)
    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low = int(index)
        high = min(len(ordered) - 1, low + 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)
    return {"count": len(values), "median": median(values), "p10": percentile(0.10), "p90": percentile(0.90)}


def main() -> None:
    root = Path("data/atomic_tofu/v2.0-rc1")
    generated_root = root / "api" / "eval_extension" / "codex_direct_3200_v4"
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    final_rows = {row["unit_id"]: row for row in read_jsonl(generated_root / "candidate_outputs_3200_v2.jsonl")}
    official = {row["qa_id"]: row for row in read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")}
    selected_generated = [line.strip() for line in (root / "api" / "eval_extension" / "calibration" / "generated_60_ids.txt").read_text().splitlines() if line.strip()]
    selected_anchors = [line.strip() for line in (root / "api" / "eval_extension" / "anchor_calibration" / "anchor_20_ids.txt").read_text().splitlines() if line.strip()]
    current_anchor_path = root / "api" / "eval_extension" / "anchor_calibration" / "current_route_candidate_outputs.jsonl"
    current_anchor_outputs = {row["unit_id"]: row for row in read_jsonl(current_anchor_path)} if current_anchor_path.exists() else {}

    generated_errors: list[dict[str, Any]] = []
    generated_ratios: list[float] = []
    generated_shape_matches = 0
    generated_shape_total = 0
    generated_hashes: dict[str, str] = {}
    for qa_id in selected_generated:
        row = final_rows.get(qa_id)
        original = source.get(qa_id)
        if row is None or original is None:
            generated_errors.append({"qa_id": qa_id, "errors": ["missing current post-review candidate or source"]})
            continue
        candidate = row.get("candidate", {})
        perturbations = candidate.get("perturbed_answer", [])
        errors: list[str] = []
        if not candidate.get("paraphrased_question") or not candidate.get("paraphrased_answer"):
            errors.append("missing paraphrase field")
        if len(perturbations) != 5 or len(set(perturbations)) != 5 or any(not isinstance(value, str) or not value.strip() for value in perturbations):
            errors.append("perturbation cardinality/nonempty/distinctness failure")
        source_answer = original["answer"].casefold().strip()
        paraphrase_answer = candidate.get("paraphrased_answer", "").casefold().strip()
        for value in perturbations:
            normalized = value.casefold().strip()
            if normalized in {source_answer, paraphrase_answer}:
                errors.append("perturbation copies source/paraphrase answer")
            if re.search(r"described in the source|this account portrays|according to (the )?(question|source|prompt)", value, re.I):
                errors.append("evaluator-meta language")
            if original["answer"]:
                generated_ratios.append(token_count(value) / max(1, token_count(original["answer"])))
        source_shapes = shape_labels(original["question"], original["answer"])
        candidate_shapes = shape_labels(original["question"], candidate.get("paraphrased_answer", ""))
        generated_shape_total += 1
        generated_shape_matches += bool(source_shapes & candidate_shapes)
        generated_hashes[qa_id] = canonical_hash(candidate)
        if errors:
            generated_errors.append({"qa_id": qa_id, "errors": sorted(set(errors))})

    official_ratios: list[float] = []
    official_shape_counts: dict[str, int] = {}
    for qa_id, row in official.items():
        original = source[qa_id]
        for value in row.get("perturbed_answer", []):
            official_ratios.append(token_count(value) / max(1, token_count(original["answer"])))
        for label in shape_labels(original["question"], row.get("paraphrased_answer", "")):
            official_shape_counts[label] = official_shape_counts.get(label, 0) + 1

    generated_ratio_stats = ratio_stats(generated_ratios)
    official_ratio_stats = ratio_stats(official_ratios)
    generated_length_pass = bool(
        generated_ratio_stats["median"] is not None
        and generated_ratio_stats["p10"] is not None
        and 0.80 <= generated_ratio_stats["median"] <= 1.35
        and generated_ratio_stats["p10"] >= 0.65
    )

    anchor_missing = [qa_id for qa_id in selected_anchors if qa_id not in current_anchor_outputs]
    anchor_errors: list[dict[str, Any]] = []
    anchor_source_ratios: list[float] = []
    anchor_vs_official_ratios: list[float] = []
    anchor_shape_matches = 0
    anchor_blind_rows: list[dict[str, Any]] = []
    for qa_id in selected_anchors:
        output = current_anchor_outputs.get(qa_id)
        original = source.get(qa_id)
        official_row = official.get(qa_id)
        errors: list[str] = []
        if output is None or original is None or official_row is None:
            errors.append("missing current candidate, source, or official anchor")
        else:
            candidate = output.get("candidate", {})
            perturbations = candidate.get("perturbed_answer", [])
            if not candidate.get("paraphrased_question") or not candidate.get("paraphrased_answer"):
                errors.append("missing paraphrase field")
            if len(perturbations) != 5 or len(set(perturbations)) != 5 or any(not isinstance(value, str) or not value.strip() for value in perturbations):
                errors.append("perturbation cardinality/nonempty/distinctness failure")
            if shape_labels(original["question"], original["answer"]) & shape_labels(original["question"], candidate.get("paraphrased_answer", "")):
                anchor_shape_matches += 1
            for index, value in enumerate(perturbations):
                normalized = value.casefold().strip()
                if normalized in {original["answer"].casefold().strip(), candidate.get("paraphrased_answer", "").casefold().strip()}:
                    errors.append(f"perturbation_{index + 1}_copies_source_or_paraphrase")
                if re.search(r"described in the source|this account portrays|according to (the )?(question|source|prompt)", value, re.I):
                    errors.append(f"perturbation_{index + 1}_evaluator_meta_language")
                anchor_source_ratios.append(token_count(value) / max(1, token_count(original["answer"])))
                official_values = official_row.get("perturbed_answer", [])
                if index < len(official_values):
                    anchor_vs_official_ratios.append(token_count(value) / max(1, token_count(official_values[index])))
        candidate_hash = canonical_hash(output.get("candidate", {})) if output else None
        anchor_blind_rows.append({
            "qa_id": qa_id,
            "candidate_hash": candidate_hash,
            "candidate_hash_algorithm": "sha256(canonical_candidate_json_utf8_sort_keys)",
            "mechanical_errors": sorted(set(errors)),
            "decision": "pending_human_blind_review",
            "reviewer": None,
            "candidate_only": True,
            "model_lock": "floating_alias",
        })
        if errors:
            anchor_errors.append({"qa_id": qa_id, "errors": sorted(set(errors))})
    anchor_length_stats = ratio_stats(anchor_vs_official_ratios)
    anchor_length_pass = bool(
        anchor_length_stats["median"] is not None
        and anchor_length_stats["p10"] is not None
        and anchor_length_stats["p90"] is not None
        and 0.80 <= anchor_length_stats["median"] <= 1.25
        and anchor_length_stats["p10"] >= 0.65
        and anchor_length_stats["p90"] <= 1.60
    )
    anchor_form_pass = anchor_shape_matches == len(selected_anchors)
    anchor_error_type_pass = not anchor_errors
    blind_review_path = root / "audit" / "eval_a_current_route_hidden_anchor_review_2026-07-18.jsonl"
    with blind_review_path.open("w", encoding="utf-8") as handle:
        for row in anchor_blind_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    anchor_blind_review = {
        "status": "pending_human_blind_review",
        "candidate_count": len(current_anchor_outputs),
        "expected_count": len(selected_anchors),
        "blind_review_completed": False,
        "review_decisions_path": "open-unlearning/data/atomic_tofu/v2.0-rc1/audit/eval_a_current_route_hidden_anchor_review_2026-07-18.jsonl",
        "reason": "Current-route hidden-anchor candidates are present, but no human blind review has been recorded; historical gpt-5-mini output and mock candidates are excluded.",
    }
    report = {
        "artifact": "current-route Eval-A calibration evidence",
        "audit_date": "2026-07-18",
        "status": "calibration_gate_not_passed_current_route_length_shape_and_human_review",
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called_in_this_evidence_build": False,
        "route_definition": "post-review codex_direct_3200_v4 candidate baseline; no historical gpt-5-mini calibration rows reused as current evidence",
        "generated_calibration": {
            "selected_count": len(selected_generated),
            "present_count": sum(qa_id in final_rows for qa_id in selected_generated),
            "mechanical_error_count": len(generated_errors),
            "mechanical_errors": generated_errors,
            "candidate_hashes": generated_hashes,
            "perturbation_source_length_ratio": generated_ratio_stats,
            "official_800_length_ratio_reference": official_ratio_stats,
            "length_gate_against_current_official_reference": generated_length_pass,
            "shape_overlap_count": generated_shape_matches,
            "shape_overlap_total": generated_shape_total,
            "source_fidelity_blind_review": {"status": "pending", "completed": False},
            "semantic_blind_review": {"status": "pending", "completed": False},
        },
        "hidden_official_anchors": {
            "selected_count": len(selected_anchors),
            "official_fields_present_count": sum(qa_id in official for qa_id in selected_anchors),
            "current_route_candidate_present_count": sum(qa_id in current_anchor_outputs for qa_id in selected_anchors),
            "missing_current_route_candidate_ids": anchor_missing,
            "current_route_source_length_ratio": ratio_stats(anchor_source_ratios),
            "current_route_vs_official_length_ratio": anchor_length_stats,
            "current_route_mechanical_errors": anchor_errors,
            "shape_overlap_count": anchor_shape_matches,
            "shape_overlap_total": len(selected_anchors),
            "official_length_ratio_reference": ratio_stats([token_count(value) / max(1, token_count(source[qa_id]["answer"])) for qa_id in selected_anchors for value in official[qa_id].get("perturbed_answer", [])]),
            "official_shape_counts": official_shape_counts,
            "length_gate": anchor_length_pass,
            "form_gate": anchor_form_pass,
            "error_type_gate": anchor_error_type_pass,
            "blind_review": anchor_blind_review,
        },
        "gates": {
            "generated_current_route_mechanical": len(generated_errors) == 0 and len(selected_generated) == 60,
            "generated_length_against_official_reference": generated_length_pass,
            "hidden_anchor_candidate_presence": len(anchor_missing) == 0 and len(selected_anchors) == 20,
            "hidden_anchor_length": anchor_length_pass,
            "hidden_anchor_form": anchor_form_pass,
            "hidden_anchor_error_type": anchor_error_type_pass,
            "hidden_anchor_blind_review": False,
        },
        "historical_report": {
            "path": "open-unlearning/data/atomic_tofu/v2.0-rc1/audit/eval_anchor_calibration_report.json",
            "status": "calibration_failed",
            "used_as_current_evidence": False,
            "reason": "historical report corresponds to gpt-5-mini generated calibration and missing hidden anchor candidates"
        },
        "official_shape_reference": "open-unlearning/data/atomic_tofu/v2.0-rc1/official_anchors/official_eval_sidecars.jsonl",
        "next_required_authority": "Record actual human blind review decisions for the 20 current-route hidden anchors; the generated-length gate also requires resolution before changing this gate to passed.",
    }
    output = root / "audit" / "eval_a_current_route_calibration_evidence_2026-07-18.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
