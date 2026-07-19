#!/usr/bin/env python3
"""Read-only length and answer-shape diagnostics for the frozen 3200 candidate.

The script never rewrites the candidate artifact.  It emits an audit JSON with
field-level failures and the complete union of affected qa_ids, including the
subset used by the formal bundle review inventory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from statistics import median
from typing import Any, Iterable


EXPECTED_SHA256 = "7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512"
PERTURBATION_COUNT = 5


def token_count(value: str) -> int:
    return len(re.findall(r"[\w]+(?:['’-][\w]+)*", value, flags=re.UNICODE))


def is_bare_value(value: str) -> bool:
    tokens = re.findall(r"[\w]+(?:['’-][\w]+)*", value, flags=re.UNICODE)
    if not tokens or len(tokens) > 8 or re.search(r"[:;]", value):
        return False
    return not bool(re.search(r"\b(?:is|are|was|were|born|located|known|works|wrote|from|in|on)\b", value, re.I))


def normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def answer_value_fragments(answer: str, question: str) -> set[str]:
    """Conservative copy of the project validator's template-slot heuristic."""
    fragments: set[str] = set()
    normalized_question = normalize(question)

    def add_if_answer_only(value: str) -> None:
        normalized_value = normalize(value)
        if len(normalized_value) >= 3 and normalized_value not in normalized_question:
            fragments.add(normalized_value)

    normalized = normalize(answer)
    if len(normalized) >= 12 and normalized not in normalized_question:
        fragments.add(normalized)
    for match in re.finditer(r"['\"]([^'\"]+)['\"]", answer):
        add_if_answer_only(match.group(1))
    for match in re.finditer(r"\b(?:[A-Z][\w'’-]+)(?:\s+[A-Z][\w'’-]+)+", answer):
        add_if_answer_only(match.group(0))
    for match in re.finditer(r"\b\d+(?:[./-]\d+)*(?:\s+\w+)?", answer):
        add_if_answer_only(match.group(0))
    return fragments


def has_answer_shape(source_answer: str, perturbation: str, question: str) -> bool:
    if is_bare_value(source_answer):
        return True
    source_tokens = set(re.findall(r"[\w]+(?:['’-][\w]+)*", source_answer.casefold(), flags=re.UNICODE))
    perturbation_tokens = set(re.findall(r"[\w]+(?:['’-][\w]+)*", perturbation.casefold(), flags=re.UNICODE))
    value_tokens: set[str] = set()
    for fragment in answer_value_fragments(source_answer, question):
        if normalize(fragment) != normalize(source_answer):
            value_tokens.update(re.findall(r"[\w]+(?:['’-][\w]+)*", fragment.casefold(), flags=re.UNICODE))
    template_tokens = source_tokens - value_tokens
    if template_tokens & perturbation_tokens:
        return True
    markers = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "born", "name",
        "full", "author", "writer", "works", "work", "wrote", "writing", "book", "books",
        "from", "in", "on", "at", "of", "for", "with", "and", "has", "have", "had",
        "located", "known", "specializes", "specialized", "profession", "occupation", "job",
        "answer", "information",
    }
    source_markers = (template_tokens & markers) or (source_tokens & markers)
    return bool(source_markers & perturbation_tokens)


# Use the repository's actual answer-shape implementation for the emitted
# diagnostic.  The local fallbacks above keep this file readable in isolation,
# but the project validator is authoritative whenever this script runs in the
# repository.
PROJECT_SRC = Path(__file__).resolve().parents[1] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))
from atomic_tofu.eval_extension import (  # noqa: E402
    _answer_token_count as _project_answer_token_count,
    _has_answer_shape as _project_has_answer_shape,
    _is_bare_answer as _project_is_bare_answer,
)

token_count = _project_answer_token_count
is_bare_value = _project_is_bare_answer
has_answer_shape = _project_has_answer_shape


def quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "median": median(values) if values else None,
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnose_scope(
    candidate_rows: dict[str, dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
    scope_ids: set[str],
) -> dict[str, Any]:
    ratios: list[float] = []
    low_075: list[dict[str, Any]] = []
    low_065: list[dict[str, Any]] = []
    bare: list[dict[str, Any]] = []
    shape: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    row_failures: dict[str, set[str]] = {}

    for qa_id in sorted(scope_ids):
        output = candidate_rows[qa_id]
        source = source_by_id[qa_id]
        candidate = output["candidate"]
        perturbations = candidate.get("perturbed_answer", [])
        source_count = token_count(source["answer"])
        if source_count == 0:
            raise ValueError(f"source answer has zero tokens: {qa_id}")
        for index, value in enumerate(perturbations, 1):
            ratio = token_count(value) / source_count
            ratios.append(ratio)
            categories: list[str] = []
            record = {
                "qa_id": qa_id,
                "perturbation_index": index,
                "source_answer_token_count": source_count,
                "perturbation_token_count": token_count(value),
                "ratio": ratio,
            }
            if ratio < 0.75:
                categories.append("ratio_below_0.75")
                low_075.append(record.copy())
            if ratio < 0.65:
                categories.append("ratio_below_0.65")
                low_065.append(record.copy())
            is_bare = is_bare_value(value)
            if is_bare:
                categories.append("bare_value")
                bare.append({**record, "source_answer_is_bare": is_bare_value(source["answer"])})
            if not has_answer_shape(source["answer"], value, source["question"]):
                categories.append("shape_mismatch")
                shape.append(record.copy())
            if categories:
                row_failures.setdefault(qa_id, set()).update(categories)
                failures.append({**record, "categories": categories})

    return {
        "row_count": len(scope_ids),
        "perturbation_count": len(ratios),
        "ratio_stats": stats(ratios),
        "below_0_75_field_count": len(low_075),
        "below_0_75_row_count": len({item["qa_id"] for item in low_075}),
        "below_0_65_field_count": len(low_065),
        "below_0_65_row_count": len({item["qa_id"] for item in low_065}),
        "bare_value_field_count": len(bare),
        "bare_value_row_count": len({item["qa_id"] for item in bare}),
        "bare_value_failure_field_count": sum(not item["source_answer_is_bare"] for item in bare),
        "bare_value_failure_row_count": len({item["qa_id"] for item in bare if not item["source_answer_is_bare"]}),
        "shape_mismatch_field_count": len(shape),
        "shape_mismatch_row_count": len({item["qa_id"] for item in shape}),
        "rows_with_any_failure": len(row_failures),
        "failed_ids": sorted(row_failures),
        "failed_id_categories": {qa_id: sorted(categories) for qa_id, categories in sorted(row_failures.items())},
        "below_0_75_failures": low_075,
        "below_0_65_failures": low_065,
        "bare_value_failures": bare,
        "shape_mismatch_failures": shape,
        "field_failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", default="data/atomic_tofu/v2.0-rc1")
    parser.add_argument("--candidate-path", default=None)
    parser.add_argument("--expected-sha256", default=None)
    parser.add_argument("--artifact-label", default="read-only frozen post-review 3200 length and answer-shape diagnostic")
    parser.add_argument("--output", default="data/atomic_tofu/v2.0-rc1/audit/frozen_3200_length_shape_diagnostic_2026-07-18.json")
    args = parser.parse_args()

    root = Path(args.release_root)
    candidate_path = Path(args.candidate_path) if args.candidate_path else root / "api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl"
    source_path = root / "source/tofu_full.jsonl"
    inventory_path = root / "audit/bundle_eval_row_review_2026-07-18/review_decisions.jsonl"
    candidate_rows_list = read_jsonl(candidate_path)
    source_rows = read_jsonl(source_path)
    inventory_rows = read_jsonl(inventory_path)
    candidate_rows = {row["unit_id"]: row for row in candidate_rows_list}
    source_by_id = {row["qa_id"]: row for row in source_rows}
    bundle_generated_ids = {row["qa_id"] for row in inventory_rows if row.get("source_or_generated") == "generated"}

    if len(candidate_rows_list) != 3200 or len(candidate_rows) != 3200:
        raise ValueError(f"expected 3200 unique candidate rows, got {len(candidate_rows_list)}/{len(candidate_rows)}")
    if len(bundle_generated_ids) != 3060:
        raise ValueError(f"expected 3060 formal-bundle generated rows, got {len(bundle_generated_ids)}")
    if not bundle_generated_ids.issubset(candidate_rows):
        raise ValueError("formal bundle generated IDs are not all in the frozen candidate")

    actual_sha256 = sha256_file(candidate_path)
    expected_sha256 = args.expected_sha256
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(f"candidate SHA mismatch: {actual_sha256} != {expected_sha256}")

    report = {
        "artifact": args.artifact_label,
        "audit_date": "2026-07-18",
        "candidate_path": str(candidate_path),
        "candidate_sha256": actual_sha256,
        "expected_candidate_sha256": expected_sha256,
        "candidate_sha256_verified": bool(expected_sha256 and actual_sha256 == expected_sha256),
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "source_path": str(source_path),
        "source_rows_used": len(source_rows),
        "formal_bundle_review_inventory_path": str(inventory_path),
        "formal_bundle_generated_row_count": len(bundle_generated_ids),
        "scope": diagnose_scope(candidate_rows, source_by_id, set(candidate_rows)),
        "formal_bundle_generated_scope": diagnose_scope(candidate_rows, source_by_id, bundle_generated_ids),
        "interpretation": {
            "ratio_definition": "perturbed_answer token count / source answer token count",
            "ratio_failure_thresholds": {"below_0_75": "failed length floor", "below_0_65": "severe shortness"},
            "bare_value_definition": "conservative value-only detector; bare_value_failure excludes source answers that are themselves bare",
            "shape_definition": "the repository's project answer-shape validator (_has_answer_shape)",
            "decision": "diagnostic_only_no_candidate_mutation",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "candidate_sha256": actual_sha256,
        "all_rows": {key: report["scope"][key] for key in ("row_count", "perturbation_count", "ratio_stats", "below_0_75_field_count", "below_0_75_row_count", "below_0_65_field_count", "below_0_65_row_count", "bare_value_field_count", "bare_value_row_count", "bare_value_failure_field_count", "bare_value_failure_row_count", "shape_mismatch_field_count", "shape_mismatch_row_count", "rows_with_any_failure")},
        "formal_bundle_generated_rows": {key: report["formal_bundle_generated_scope"][key] for key in ("row_count", "perturbation_count", "ratio_stats", "below_0_75_field_count", "below_0_75_row_count", "below_0_65_field_count", "below_0_65_row_count", "bare_value_field_count", "bare_value_row_count", "bare_value_failure_field_count", "bare_value_failure_row_count", "shape_mismatch_field_count", "shape_mismatch_row_count", "rows_with_any_failure")},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
