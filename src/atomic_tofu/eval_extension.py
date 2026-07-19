from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import re
import statistics
import unicodedata
from typing import Any

from atomic_tofu.io import read_json, read_jsonl, sha256_json, write_json, write_jsonl
from atomic_tofu.source import load_official_jsonl


FLOATING_MODEL_ALIASES = frozenset({"gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o"})
SLOT_SCHEMA_VERSION = "atomic-tofu-eval-slots-v1-2026-07-15"


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


def slot_schema() -> dict[str, Any]:
    span = {
        "type": "object",
        "additionalProperties": False,
        "required": ["source_span", "type"],
        "properties": {
            "source_span": {"type": "string"},
            "type": {"enum": ["person", "date", "place", "title", "number", "polarity", "description", "other"]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "answer_type",
            "question_invariants",
            "target_groups",
            "context_slots",
            "polarity",
            "needs_human_slot_review",
        ],
        "properties": {
            "answer_type": {"enum": ["yes_no", "single_fact", "multi_fact", "explanation"]},
            "question_invariants": {"type": "array", "items": span},
            "target_groups": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["group_id", "relation", "slots"],
                    "properties": {
                        "group_id": {"type": "string"},
                        "relation": {"type": "string"},
                        "slots": {"type": "array", "minItems": 1, "items": span},
                    },
                },
            },
            "context_slots": {"type": "array", "items": span},
            "polarity": {"enum": ["yes", "no", "not_applicable"]},
            "needs_human_slot_review": {"type": "boolean"},
        },
    }


EVAL_SYSTEM_PROMPT = """Create candidate TOFU-compatible evaluation sidecar fields from exactly one supplied QA.

The user payload contains the unchanged original QA and a frozen `slot_analysis` contract. Treat
that contract as binding: preserve every `question_invariants` and `context_slots` span, replace
only the slots in one complete `target_group`, and replace all slots in a mutually dependent group
together. Do not edit or reinterpret the frozen slot contract. For yes/no items, preserve the
question and use the opposite polarity in every perturbation while keeping the answer internally
consistent.

Paraphrased question and paraphrased answer MUST differ from the supplied wording while preserving
every factual slot and the requested relation. Return exactly five perturbations. Each perturbation
must be a complete wrong answer in the SAME ANSWER SHAPE as the source answer, not merely a value.
Preserve the source answer's sentence template, grammatical form, subject, non-target context, and
all slots that are not being tested. Replace only the tested correct factual value(s) with plausible
wrong value(s) of the same type. Do not include the correct value or any alias in a perturbation.

Do not return a bare name, location, date, profession, title, or other value unless the source
answer itself is a bare value. For example, with fictional data:
- Source `The full name of the author is Mira Solano.` -> `The full name of the author is Elena Duarte.`
- Source `Ravi Sen was born in Porto, Portugal.` -> `Ravi Sen was born in Braga, Portugal.`
- Source `Lina Wei was born on 15 August 1972 in Kuala Lumpur, Malaysia.` -> `Lina Wei was born on 3 March 1975 in Penang, Malaysia.`

Hard output rules:
- Never change the supplied original question or answer; they are context only.
- Never emit JSON syntax, escaped quote/comma delimiters, arrays, lists, alternatives, drafts,
  explanations, notes, apologies, self-corrections, placeholders, or meta-commentary inside any
  field. Do not put multiple candidate answers into one array element.
- A name question gets one complete wrong name answer per perturbation; a date question gets one
  complete wrong date answer; a profession question gets one complete wrong profession answer.
  Preserve the answer type, number of requested slots, source subject, and relation template.
- Before responding, check that both paraphrases are genuinely reworded, all five perturbations are
  distinct complete answers with source-like length, none contains the correct value or an alias,
  and no field contains commentary. Return the JSON object only; human and deterministic gates
  decide acceptance."""


SLOT_SYSTEM_PROMPT = """Extract candidate answer slots for one TOFU evaluation QA. Return JSON only.

Use the supplied question and answer as the only source of truth. `question_invariants` are facts
already stated by the question and must never be changed or contradicted by a perturbation.
`target_groups` contain the relation actually asked by the question; every slot in a group must be
changed together in a counterfactual. `context_slots` are answer facts that are not the target and
must remain unchanged or semantically equivalent. Use exact source spans from the supplied text.

For yes/no questions, set answer_type to yes_no and set polarity to the answer's yes/no polarity.
For all other questions, set polarity to not_applicable. If target/context boundaries, dependent
slots, or polarity cannot be determined reliably, set needs_human_slot_review to true. Do not infer
facts that are absent from the supplied question or answer."""


SLOT_REPAIR_SYSTEM_PROMPT = """Repair one invalid answer-slot extraction. Return a complete replacement JSON object
matching the slot schema exactly. Use only the supplied question and answer. Every source_span must
occur verbatim in the question or answer; target slots must be answer spans, target groups must
contain all mutually dependent facts, context slots must not overlap targets, and yes/no questions
must have an explicit yes or no polarity. Set needs_human_slot_review to true whenever the boundary
cannot be determined reliably. Output JSON only and do not explain the repair."""


EVAL_REPAIR_SYSTEM_PROMPT = """Repair one invalid TOFU-compatible evaluation candidate. Return a complete replacement
JSON object with exactly paraphrased_question, paraphrased_answer, and five perturbed_answer strings.
Use the original QA as the only source of truth. Keep the original question and answer unchanged in
the source context, rewrite both paraphrases so they differ from the original wording but preserve
all factual slots, and use the frozen `slot_analysis` contract in the user payload. Rewrite every
flagged perturbation as a complete answer in the SAME SHAPE as the source answer. Preserve its
sentence template, grammatical form, subject, non-target context, question invariants, and
unmodified slots; replace only the tested correct value(s) with plausible wrong values. Replace all
slots in a mutually dependent target group together, and use opposite polarity for yes/no items.

Do not return a bare name, location, date, profession, title, or other value unless the source
answer itself is bare. Each perturbation must be one natural, type-matched, factually wrong answer;
it must not contain the correct value or an alias, multiple alternatives, JSON/escaped delimiters,
list syntax, notes, apologies, placeholders, explanations, or self-corrections. Do not explain the
repair and do not add fields. Output JSON only."""


_META_PATTERNS = (
    re.compile(r"(?:^\s*(?:note|sorry|placeholder|draft)\b\s*[:\-]|\b(?:final\s+output|as\s+an?\s+ai|i\s+(?:cannot|can't)|self[- ]?correction)\b)", re.I),
    re.compile(r"\b(?:candidate|option|alternative)\s*\d+\b", re.I),
    re.compile(r"\b(?:adjust|revise|rewrite)\s+(?:this|the|answer|response)\b|\bthe\s+correct\s+answer\b", re.I),
)
_LIST_MARKER = re.compile(r"(?:^\s*|[\r\n]+\s*)(?:[-*•]|\d+[.)])\s+", re.I)
_JSON_DELIMITER = re.compile(r"(?:\\\"\s*,\s*\\\"|\\\'\s*,\s*\\\')")
_QUOTED_VALUE = re.compile(r"[\"“]([^\"”]{2,})[\"”]")
_PROPER_PHRASE = re.compile(r"\b[A-Z][A-Za-z'’\-]{2,}(?:\s+[A-Z][A-Za-z'’\-]{2,})+\b")
_NUMBER_VALUE = re.compile(r"\b(?:\d{2,4}(?:[-/]\d{1,2})?(?:[-/]\d{1,4})?|\d+(?:st|nd|rd|th))\b", re.I)
_VALUE_STOPWORDS = {"the", "this", "that", "what", "which", "who", "where", "when", "how", "yes", "no", "born", "author", "writer", "female", "male"}


def _normalize_eval_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


_ANSWER_TOKEN = re.compile(r"[\w]+(?:['’\-][\w]+)*", re.UNICODE)
_ANSWER_FORM_MARKERS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "born", "name", "full",
    "author", "writer", "works", "work", "wrote", "writing", "book", "books", "from", "in",
    "on", "at", "of", "for", "with", "and", "has", "have", "had", "located", "known",
    "specializes", "specialized", "profession", "occupation", "job", "answer", "information",
}


def _answer_tokens(value: str) -> list[str]:
    return [token.casefold() for token in _ANSWER_TOKEN.findall(unicodedata.normalize("NFKC", value))]


def _answer_token_count(value: str) -> int:
    return len(_answer_tokens(value))


def _is_bare_answer(answer: str) -> bool:
    """Conservative detector for answers that are intentionally value-only."""
    tokens = _answer_tokens(answer)
    if not tokens or len(tokens) > 8:
        return False
    if re.search(r"[:;]", answer):
        return False
    return not bool(re.search(r"\b(?:is|are|was|were|born|located|known|works|wrote|from|in|on)\b", answer, re.I))


def _answer_value_fragments(answer: str, question: str) -> set[str]:
    """Extract answer-only literal value/alias proxies for leakage checks.

    Facts already stated in the question identify the queried subject or scope
    and may legitimately be repeated in a type-matched wrong answer.  Only
    answer fragments absent from that question are treated as prohibited
    correct-value leakage.
    """
    fragments: set[str] = set()
    normalized_question = _normalize_eval_text(question)

    def add_if_answer_only(value: str) -> None:
        normalized_value = _normalize_eval_text(value)
        if len(normalized_value) >= 3 and normalized_value not in normalized_question:
            fragments.add(normalized_value)

    normalized = _normalize_eval_text(answer)
    if len(normalized) >= 12 and normalized not in normalized_question:
        fragments.add(normalized)
    for match in _QUOTED_VALUE.finditer(answer):
        add_if_answer_only(match.group(1))
    for match in _PROPER_PHRASE.finditer(answer):
        words = match.group(0).split()
        if words and not any(word.casefold() in _VALUE_STOPWORDS for word in words):
            add_if_answer_only(match.group(0))
    for match in _NUMBER_VALUE.finditer(answer):
        add_if_answer_only(match.group(0))
    return fragments


def _answer_template_tokens(answer: str, question: str) -> set[str]:
    """Return source answer tokens that should survive a perturbation.

    This intentionally removes only conservative value-like fragments that do not already occur in
    the question. Subject/context words already present in the question remain required template
    context and therefore are not treated as answer leakage.
    """
    value_tokens: set[str] = set()
    for fragment in _answer_value_fragments(answer, question):
        # The complete answer is a leakage guard, not a slot to remove from the template.
        if _normalize_eval_text(fragment) == _normalize_eval_text(answer):
            continue
        value_tokens.update(_answer_tokens(fragment))
    return set(_answer_tokens(answer)) - value_tokens


def _has_answer_shape(source_answer: str, perturbation: str, question: str) -> bool:
    if _is_bare_answer(source_answer):
        return True
    source_tokens = set(_answer_tokens(source_answer))
    perturbation_tokens = set(_answer_tokens(perturbation))
    template_tokens = _answer_template_tokens(source_answer, question)
    if template_tokens & perturbation_tokens:
        return True
    # Allow conservative grammatical equivalents when the model rephrases a template marker.
    source_markers = (template_tokens & _ANSWER_FORM_MARKERS) or (source_tokens & _ANSWER_FORM_MARKERS)
    return bool(source_markers & perturbation_tokens)


def _shape_bounds(source_answer: str) -> tuple[int, int]:
    source_count = _answer_token_count(source_answer)
    return max(6, math.ceil(0.75 * source_count)), max(24, math.ceil(1.60 * source_count) + 12)


def _span_in(text: str, span: str) -> bool:
    if not isinstance(span, str) or not span.strip():
        return False
    return span in text or _normalize_eval_text(span) in _normalize_eval_text(text)


def _normalized_spans_overlap(left: str, right: str) -> bool:
    left_normalized = _normalize_eval_text(left)
    right_normalized = _normalize_eval_text(right)
    return bool(left_normalized and right_normalized and (left_normalized in right_normalized or right_normalized in left_normalized))


def _question_looks_yes_no(question: str) -> bool:
    return bool(re.match(r"^\s*(?:is|are|was|were|does|do|did|has|have|had|can|could|will|would|should|wasn't|weren't|isn't|aren't)\b", question, re.I))


def _candidate_polarity(value: str) -> str:
    match = re.match(r"^\s*(yes|no)\b", value, re.I)
    return match.group(1).casefold() if match else "not_applicable"


def validate_slot_analysis(
    slot_analysis: dict[str, Any],
    original: dict[str, Any],
    *,
    require_no_human_review: bool = False,
) -> list[str]:
    """Validate the first-stage slot contract without claiming semantic gold correctness."""
    errors: list[str] = []
    if not isinstance(slot_analysis, dict):
        return ["slot_analysis is not an object"]
    question = original.get("question", "")
    answer = original.get("answer", "")
    answer_type = slot_analysis.get("answer_type")
    allowed_answer_types = {"yes_no", "single_fact", "multi_fact", "explanation"}
    if answer_type not in allowed_answer_types:
        errors.append("answer_type is invalid")
    polarity = slot_analysis.get("polarity")
    if polarity not in {"yes", "no", "not_applicable"}:
        errors.append("polarity is invalid")
    if answer_type == "yes_no" and polarity not in {"yes", "no"}:
        errors.append("yes/no slot extraction must declare yes or no polarity")
    if _question_looks_yes_no(question) and answer_type == "yes_no" and polarity not in {"yes", "no"}:
        errors.append("yes/no question has no usable polarity")
    if require_no_human_review and slot_analysis.get("needs_human_slot_review") is True:
        errors.append("slot extraction requires human slot review")

    invariants = slot_analysis.get("question_invariants", [])
    groups = slot_analysis.get("target_groups", [])
    context_slots = slot_analysis.get("context_slots", [])
    if not isinstance(invariants, list):
        errors.append("question_invariants must be an array")
        invariants = []
    if not isinstance(groups, list) or not groups:
        errors.append("target_groups must be a non-empty array")
        groups = []
    if not isinstance(context_slots, list):
        errors.append("context_slots must be an array")
        context_slots = []

    seen_group_ids: set[str] = set()
    target_spans: list[str] = []
    for group in groups:
        if not isinstance(group, dict):
            errors.append("target_groups contains a non-object")
            continue
        group_id = group.get("group_id")
        valid_group_id = isinstance(group_id, str) and bool(group_id.strip())
        if not valid_group_id or group_id in seen_group_ids:
            errors.append("target group IDs must be non-empty and unique")
        if valid_group_id:
            seen_group_ids.add(group_id)
        slots = group.get("slots", [])
        if not isinstance(slots, list) or not slots:
            errors.append(f"target group {group_id or '<missing>'} has no slots")
            continue
        for slot in slots:
            if not isinstance(slot, dict):
                errors.append(f"target group {group_id or '<missing>'} contains a non-object slot")
                continue
            span = slot.get("source_span")
            if not _span_in(answer, span):
                errors.append(f"target slot span is not present in the source answer: {span!r}")
            target_spans.append(span if isinstance(span, str) else "")

    context_spans: list[str] = []
    for slot in context_slots:
        if not isinstance(slot, dict):
            errors.append("context_slots contains a non-object")
            continue
        span = slot.get("source_span")
        if not _span_in(answer, span):
            errors.append(f"context slot span is not present in the source answer: {span!r}")
        context_spans.append(span if isinstance(span, str) else "")

    for index, left in enumerate(target_spans):
        for right in target_spans[index + 1:]:
            if _normalized_spans_overlap(left, right):
                errors.append("target slots overlap")
    for target in target_spans:
        for context in context_spans:
            if _normalized_spans_overlap(target, context):
                errors.append("target and context slots overlap")

    invariant_spans: list[str] = []
    for slot in invariants:
        if not isinstance(slot, dict):
            errors.append("question_invariants contains a non-object")
            continue
        span = slot.get("source_span")
        if not _span_in(question, span) and not _span_in(answer, span):
            errors.append(f"question invariant span is not present in the source text: {span!r}")
        invariant_spans.append(span if isinstance(span, str) else "")
    for invariant in invariant_spans:
        for target in target_spans:
            if _normalized_spans_overlap(invariant, target):
                errors.append("question invariant and target slot overlap")
        for context in context_spans:
            if _normalized_spans_overlap(invariant, context):
                errors.append("question invariant and context slot overlap")
    return sorted(set(errors))


def _slot_target_spans(slot_analysis: dict[str, Any] | None) -> list[str]:
    if not isinstance(slot_analysis, dict):
        return []
    return [
        slot.get("source_span", "")
        for group in slot_analysis.get("target_groups", [])
        if isinstance(group, dict)
        for slot in group.get("slots", [])
        if isinstance(slot, dict) and isinstance(slot.get("source_span"), str)
    ]


def _slot_context_spans(slot_analysis: dict[str, Any] | None) -> list[str]:
    if not isinstance(slot_analysis, dict):
        return []
    return [
        slot.get("source_span", "")
        for slot in slot_analysis.get("context_slots", [])
        if isinstance(slot, dict) and isinstance(slot.get("source_span"), str)
    ]


def _slot_invariant_spans(slot_analysis: dict[str, Any] | None) -> list[str]:
    if not isinstance(slot_analysis, dict):
        return []
    return [
        slot.get("source_span", "")
        for slot in slot_analysis.get("question_invariants", [])
        if isinstance(slot, dict) and isinstance(slot.get("source_span"), str)
    ]


def _validate_slot_constrained_candidate(candidate: dict[str, Any], original: dict[str, Any], slot_analysis: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    slot_errors = validate_slot_analysis(slot_analysis, original)
    if slot_errors:
        errors.extend(f"slot_analysis: {error}" for error in slot_errors)
        return errors
    answer = original.get("answer", "")
    target_spans = _slot_target_spans(slot_analysis)
    context_spans = _slot_context_spans(slot_analysis)
    invariant_spans = [span for span in _slot_invariant_spans(slot_analysis) if _span_in(answer, span)]
    polarity = slot_analysis.get("polarity")
    expected_polarity = "no" if polarity == "yes" else "yes" if polarity == "no" else None
    for index, value in enumerate(candidate.get("perturbed_answer", []), 1):
        label = f"perturbed_answer[{index}]"
        if any(_normalize_eval_text(span) in _normalize_eval_text(value) for span in target_spans if span.strip()):
            errors.append(f"{label} retains a target slot")
        for span in context_spans + invariant_spans:
            if span.strip() and not _span_in(value, span):
                errors.append(f"{label} does not preserve slot/invariant: {span!r}")
        if expected_polarity and _candidate_polarity(value) != expected_polarity:
            errors.append(f"{label} does not use the opposite yes/no polarity")
    return errors


def validate_eval_candidate(
    candidate: dict[str, Any],
    original: dict[str, Any],
    perturbation_count: int,
    slot_analysis: dict[str, Any] | None = None,
) -> list[str]:
    """Validate Eval-A semantics before a candidate is accepted into the cache.

    This is deliberately a deterministic gate, not a substitute for human semantic review. It
    rejects known canary failure modes (copying, malformed multi-answer strings, meta text, and
    literal correct-value leakage) so those outputs trigger the provider repair loop.
    """
    errors: list[str] = []
    if not isinstance(candidate, dict):
        return ["candidate is not an object"]
    question = original.get("question", "")
    answer = original.get("answer", "")
    paraphrased_question = candidate.get("paraphrased_question")
    paraphrased_answer = candidate.get("paraphrased_answer")
    perturbed = candidate.get("perturbed_answer")
    if not isinstance(paraphrased_question, str) or not paraphrased_question.strip():
        errors.append("paraphrased_question must be a non-empty string")
    elif _normalize_eval_text(paraphrased_question) == _normalize_eval_text(question):
        errors.append("paraphrased_question is an exact copy of the original")
    if not isinstance(paraphrased_answer, str) or not paraphrased_answer.strip():
        errors.append("paraphrased_answer must be a non-empty string")
    elif _normalize_eval_text(paraphrased_answer) == _normalize_eval_text(answer):
        errors.append("paraphrased_answer is an exact copy of the original")
    for field_name, value in (("paraphrased_question", paraphrased_question), ("paraphrased_answer", paraphrased_answer)):
        if isinstance(value, str) and any(pattern.search(value) for pattern in _META_PATTERNS):
            errors.append(f"{field_name} contains meta-commentary")
    if not isinstance(perturbed, list):
        errors.append("perturbed_answer must be an array")
        return errors
    if len(perturbed) != perturbation_count:
        errors.append(f"perturbed_answer must contain exactly {perturbation_count} strings")
        return errors
    normalized_seen: set[str] = set()
    value_fragments = set(_slot_target_spans(slot_analysis)) if slot_analysis is not None else _answer_value_fragments(answer, question)
    min_tokens, max_tokens = _shape_bounds(answer)
    max_length = max(160, min(1200, 4 * len(answer) + 160))
    for index, value in enumerate(perturbed, 1):
        label = f"perturbed_answer[{index}]"
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} must be a non-empty string")
            continue
        normalized = _normalize_eval_text(value)
        if normalized in normalized_seen:
            errors.append(f"{label} duplicates another perturbation")
        normalized_seen.add(normalized)
        if _normalize_eval_text(value) == _normalize_eval_text(answer):
            errors.append(f"{label} copies the original answer")
        token_count = _answer_token_count(value)
        if token_count < min_tokens:
            errors.append(f"{label} answer-shape token count {token_count} is below minimum {min_tokens}")
        if token_count > max_tokens:
            errors.append(f"{label} answer-shape token count {token_count} exceeds maximum {max_tokens}")
        if not _has_answer_shape(answer, value, question):
            errors.append(f"{label} does not preserve the source answer shape/template")
        if len(value) > max_length:
            errors.append(f"{label} is abnormally long ({len(value)} > {max_length})")
        if any(pattern.search(value) for pattern in _META_PATTERNS):
            errors.append(f"{label} contains meta-commentary")
        if _JSON_DELIMITER.search(value) or value.lstrip().startswith(("[", "{")) or value.rstrip().endswith(("]", "}")):
            errors.append(f"{label} contains JSON/list delimiters or multiple candidates")
        if _LIST_MARKER.search(value) or re.search(r"\n\s*\S", value):
            errors.append(f"{label} contains list or multi-line answer structure")
        if re.search(r"\b(?:alternatively|either)\b", value, re.I):
            errors.append(f"{label} contains alternatives rather than one answer")
        leaked = sorted(_normalize_eval_text(fragment) for fragment in value_fragments if fragment and _normalize_eval_text(fragment) in normalized)
        if leaked:
            errors.append(f"{label} contains a literal correct value/alias fragment")
    if slot_analysis is not None:
        errors.extend(_validate_slot_constrained_candidate(candidate, original, slot_analysis))
    return sorted(set(errors))


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


def prepare_eval_slot_units(release_root: str | Path) -> list[dict[str, Any]]:
    """Prepare first-stage slot extraction units for every unchanged source QA."""
    root = Path(release_root)
    rows = read_jsonl(root / "source" / "tofu_full.jsonl")
    units = []
    for row in rows:
        payload = {key: row[key] for key in ("qa_id", "question", "answer")}
        units.append({"unit_id": row["qa_id"], "content_sha256": sha256_json(payload), "payload": payload})
    write_jsonl(root / "api" / "eval_extension" / "slot_extraction" / "input_units.jsonl", units)
    return units


def _load_slot_input_units(root: Path) -> dict[str, dict[str, Any]]:
    return {
        row["unit_id"]: row
        for row in read_jsonl(root / "api" / "eval_extension" / "slot_extraction" / "input_units.jsonl")
    }


def validate_eval_slot_outputs(
    release_root: str | Path,
    unit_ids: set[str] | None = None,
    *,
    require_no_human_review: bool = False,
) -> dict[str, Any]:
    """Mechanically validate first-stage slots; this remains candidate-only until human review."""
    root = Path(release_root)
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    units = _load_slot_input_units(root)
    outputs_path = root / "api" / "eval_extension" / "slot_extraction" / "candidate_outputs.jsonl"
    outputs = read_jsonl(outputs_path) if outputs_path.exists() else []
    by_id = {row.get("unit_id"): row for row in outputs}
    selected = unit_ids or set(units)
    errors: list[dict[str, Any]] = []
    human_review_rows: list[str] = []
    for qa_id in sorted(selected):
        unit = units.get(qa_id)
        if unit is None or qa_id not in source:
            errors.append({"unit_id": qa_id, "errors": ["unknown slot unit"]})
            continue
        output = by_id.get(qa_id)
        if output is None:
            errors.append({"unit_id": qa_id, "errors": ["missing slot candidate"]})
            continue
        candidate = output.get("candidate", {})
        if isinstance(candidate, dict) and candidate.get("needs_human_slot_review") is True:
            human_review_rows.append(qa_id)
        candidate_errors = validate_slot_analysis(
            candidate,
            unit["payload"],
            require_no_human_review=require_no_human_review,
        )
        if candidate_errors:
            errors.append({"unit_id": qa_id, "errors": candidate_errors})
    report = {
        "rows_requested": len(selected),
        "rows_present": sum(qa_id in by_id for qa_id in selected),
        "valid_rows": len(selected) - len(errors),
        "invalid_rows": len(errors),
        "errors": errors,
        "human_review_rows": human_review_rows,
        "candidate_only": True,
        "slot_schema_version": SLOT_SCHEMA_VERSION,
        "status": "slot_mechanical_passed_human_review_pending" if len(selected) and not errors else "slot_mechanical_failed",
    }
    write_json(root / "audit" / "eval_slot_candidate_report.json", report)
    return report


def freeze_eval_slots(release_root: str | Path, unit_ids: set[str] | None = None) -> dict[str, Any]:
    """Freeze mechanically valid, explicitly accepted slot candidates for stage two."""
    root = Path(release_root)
    validation = validate_eval_slot_outputs(root, unit_ids)
    if validation["invalid_rows"]:
        return {"status": "slot_mechanical_failed", "validation": validation}
    units = _load_slot_input_units(root)
    selected = unit_ids or set(units)
    decisions_path = root / "api" / "eval_extension" / "slot_extraction" / "review_decisions.jsonl"
    decisions = {row.get("qa_id"): row for row in read_jsonl(decisions_path)} if decisions_path.exists() else {}
    missing_decisions = sorted(qa_id for qa_id in selected if qa_id not in decisions)
    rejected = sorted(qa_id for qa_id in selected if decisions.get(qa_id, {}).get("status") != "accepted")
    if missing_decisions or rejected:
        return {
            "status": "slot_review_required",
            "rows_requested": len(selected),
            "missing_decision_ids": missing_decisions,
            "rejected_or_unaccepted_ids": rejected,
            "validation": validation,
        }
    outputs = {
        row["unit_id"]: row
        for row in read_jsonl(root / "api" / "eval_extension" / "slot_extraction" / "candidate_outputs.jsonl")
        if row.get("unit_id") in selected
    }
    frozen_by_id = {}
    frozen_path = root / "api" / "eval_extension" / "slot_extraction" / "frozen_slots.jsonl"
    if frozen_path.exists():
        frozen_by_id.update({row["qa_id"]: row for row in read_jsonl(frozen_path)})
    for qa_id in sorted(selected):
        decision = decisions[qa_id]
        frozen_by_id[qa_id] = {
            "qa_id": qa_id,
            "source_sha256": units[qa_id]["content_sha256"],
            "slot_analysis": outputs[qa_id]["candidate"],
            "slot_analysis_sha256": sha256_json(outputs[qa_id]["candidate"]),
            "human_review_status": (
                "accepted_after_human_slot_review"
                if outputs[qa_id]["candidate"].get("needs_human_slot_review") is True
                else "accepted"
            ),
            "reviewer": decision.get("reviewer"),
            "second_reviewer": decision.get("second_reviewer"),
            "decision_reason": decision.get("reason"),
        }
    frozen_rows = [frozen_by_id[qa_id] for qa_id in sorted(frozen_by_id)]
    write_jsonl(root / "api" / "eval_extension" / "slot_extraction" / "frozen_slots.jsonl", frozen_rows)

    official_ids = {row["qa_id"] for row in read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")}
    generation_units = []
    for row in frozen_rows:
        if row["qa_id"] in official_ids:
            continue
        original = units[row["qa_id"]]["payload"]
        payload = {**original, "slot_analysis": row["slot_analysis"]}
        generation_units.append({
            "unit_id": row["qa_id"],
            "content_sha256": sha256_json(payload),
            "payload": payload,
            "perturbation_count": official_eval_contract(root)["perturbation_count"],
            "slot_analysis_sha256": row["slot_analysis_sha256"],
        })
    write_jsonl(root / "api" / "eval_extension" / "input_units.jsonl", generation_units)
    return {
        "status": "slot_contract_frozen_candidate",
        "rows_requested": len(selected),
        "rows_frozen": len(frozen_rows),
        "generation_units": len(generation_units),
        "slot_schema_version": SLOT_SCHEMA_VERSION,
        "candidate_only": True,
    }


def _load_frozen_slot_map(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "api" / "eval_extension" / "slot_extraction" / "frozen_slots.jsonl"
    if not path.exists():
        raise ValueError("frozen slot candidates are required before Eval generation")
    return {row["qa_id"]: row["slot_analysis"] for row in read_jsonl(path)}


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
    units = prepare_eval_slot_units(root)
    write_json(root / "audit" / "eval_slot_preparation.json", {
        "source_rows": len(rows),
        "slot_units": len(units),
        "official_sidecars": len(official_sidecars),
        "perturbation_count": contract["perturbation_count"],
        "status": "slot_extraction_units_prepared",
        "candidate_only": True,
    })
    return units


def _mock_slot_analysis(unit: dict[str, Any]) -> dict[str, Any]:
    """Make a deterministic slot fixture for local tests and dry-run only."""
    question = unit["payload"]["question"]
    answer = unit["payload"]["answer"]
    answer_match = re.match(r"^\s*(yes|no)\b", answer, re.I)
    if answer_match:
        target_span = answer_match.group(1)
        answer_type = "yes_no"
        polarity = target_span.casefold()
        slot_type = "polarity"
    else:
        candidates = []
        for pattern in (_QUOTED_VALUE, _PROPER_PHRASE, _NUMBER_VALUE):
            candidates.extend(match.group(1) if pattern is _QUOTED_VALUE else match.group(0) for match in pattern.finditer(answer))
        target_span = candidates[-1] if candidates else (re.findall(r"[\w]+(?:['’\-][\w]+)*", answer) or [answer])[-1]
        answer_type = "explanation" if _answer_token_count(answer) >= 24 else "multi_fact" if re.search(r"\band\b", question, re.I) else "single_fact"
        polarity = "not_applicable"
        slot_type = "other"
    invariants = [
        {"source_span": match.group(0), "type": "other"}
        for match in _PROPER_PHRASE.finditer(question)
        if _span_in(answer, match.group(0)) and _normalize_eval_text(match.group(0)) != _normalize_eval_text(target_span)
    ]
    return {
        "answer_type": answer_type,
        "question_invariants": invariants,
        "target_groups": [{"group_id": "g1", "relation": "mock_target", "slots": [{"source_span": target_span, "type": slot_type}]}],
        "context_slots": [],
        "polarity": polarity,
        "needs_human_slot_review": False,
    }


def mock_slot_analysis(unit: dict[str, Any]) -> dict[str, Any]:
    """Expose the deterministic slot fixture for local tests and dry-run only."""
    return _mock_slot_analysis(unit)


def _slot_replacement(slot: dict[str, Any], index: int, polarity: str = "not_applicable") -> str:
    slot_type = slot.get("type")
    if slot_type == "polarity":
        return "No" if polarity == "yes" else "Yes"
    replacements = {
        "person": f"Fictional Person {index}",
        "date": f"3 March {1970 + index}",
        "place": f"Fictional City {index}",
        "title": f"Fictional Title {index}",
        "number": str(100 + index),
        "description": f"a fictional description {index}",
        "other": f"Fictional Fact {index}",
    }
    return replacements.get(slot_type, f"Fictional Fact {index}")


def _mock_perturbation(answer: str, question: str, index: int, slot_analysis: dict[str, Any] | None = None) -> str:
    if slot_analysis:
        value = answer
        for group in slot_analysis.get("target_groups", []):
            slots = group.get("slots", []) if isinstance(group, dict) else []
            for slot in sorted((item for item in slots if isinstance(item, dict)), key=lambda item: len(item.get("source_span", "")), reverse=True):
                source_span = slot.get("source_span", "")
                if source_span:
                    value = value.replace(source_span, _slot_replacement(slot, index, slot_analysis.get("polarity", "not_applicable")))
        return value
    if _is_bare_answer(answer):
        return f"A fictional answer value number {index}"
    value = answer
    replacements = []
    for match in _QUOTED_VALUE.finditer(answer):
        replacements.append((match.start(1), match.end(1), f"Fictional Title {index}"))
    for match in _PROPER_PHRASE.finditer(answer):
        if _normalize_eval_text(match.group(0)) not in _normalize_eval_text(question):
            replacements.append((match.start(), match.end(), f"Fictional Name {index}"))
    for match in _NUMBER_VALUE.finditer(answer):
        replacements.append((match.start(), match.end(), str(1990 + index)))
    for start, end, replacement in sorted(replacements, reverse=True):
        value = value[:start] + replacement + value[end:]
    if value == answer:
        template = sorted(_answer_template_tokens(answer, question))
        prefix = " ".join(template[:8]) or "The answer is"
        value = f"{prefix} a fictional mock value number {index}"
    min_tokens, _ = _shape_bounds(answer)
    filler_index = 0
    while _answer_token_count(value) < min_tokens:
        filler_index += 1
        value += f" generic context{filler_index}"
    return value


def mock_eval_candidate(unit: dict[str, Any]) -> dict[str, Any]:
    payload = unit["payload"]
    count = unit["perturbation_count"]
    slot_analysis = payload.get("slot_analysis")
    return {
        "paraphrased_answer": f"In equivalent wording, {payload['answer']}",
        "paraphrased_question": f"In other words, {payload['question']}",
        "perturbed_answer": [_mock_perturbation(payload["answer"], payload["question"], index + 1, slot_analysis) for index in range(count)],
    }


CALIBRATION_SEED = 20260714
GENERATED_CALIBRATION_COUNT = 60
ANCHOR_CALIBRATION_COUNT = 20


def _stable_rank(seed: int, qa_id: str) -> int:
    return int(sha256_json({"seed": seed, "qa_id": qa_id})[:16], 16)


def _eval_shape_labels(question: str, answer: str) -> set[str]:
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
    if re.search(r"\b(?:how|why|influence|impact|shaped|describe|explain|characteristics|what kind)\b", question.casefold()) or _answer_token_count(answer) >= 24:
        labels.add("explanation")
    if re.search(r"\band\b", question.casefold()) and ("," in answer or " and " in answer.casefold()):
        labels.add("multi_slot")
    if not labels:
        labels.add("other")
    return labels


def _calibration_records(source: list[dict[str, Any]], official_ids: set[str], seed: int) -> list[dict[str, Any]]:
    ordered = sorted(source, key=lambda row: (_answer_token_count(row["answer"]), row["qa_id"]))
    total = len(ordered)
    records = []
    for index, row in enumerate(ordered):
        records.append({
            "qa_id": row["qa_id"],
            "author_id": row["author_id"],
            "answer_token_count": _answer_token_count(row["answer"]),
            "source_answer_token_bin": min(5, index * 5 // max(1, total) + 1),
            "shape_labels": sorted(_eval_shape_labels(row["question"], row["answer"])),
            "rank": _stable_rank(seed, row["qa_id"]),
            "is_official_anchor": row["qa_id"] in official_ids,
        })
    return records


def _select_generated_calibration(records: list[dict[str, Any]], count: int = GENERATED_CALIBRATION_COUNT) -> list[dict[str, Any]]:
    pool = [record for record in records if not record["is_official_anchor"]]
    selected: dict[str, dict[str, Any]] = {}
    bin_counts = Counter()
    author_counts = Counter()

    def can_add(record: dict[str, Any]) -> bool:
        return bin_counts[record["source_answer_token_bin"]] < count // 5

    def add(record: dict[str, Any]) -> None:
        selected[record["qa_id"]] = record
        bin_counts[record["source_answer_token_bin"]] += 1
        author_counts[record["author_id"]] += 1

    def choose(candidates: list[dict[str, Any]], amount: int, preferred: set[str] | None = None) -> None:
        for record in sorted(candidates, key=lambda item: (
            -len(set(item["shape_labels"]) & (preferred or set())),
            bin_counts[item["source_answer_token_bin"]],
            item["rank"],
            item["qa_id"],
        )):
            if len(selected) >= count or amount <= 0:
                break
            if record["qa_id"] not in selected and can_add(record):
                add(record)
                amount -= 1

    # Guarantee at least ten authors contribute multiple QAs, while keeping the selection
    # deterministic and allowing these rows to overlap the other strata.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in pool:
        grouped.setdefault(record["author_id"], []).append(record)
    pair_candidates = []
    for author_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: (-len(item["shape_labels"]), item["rank"], item["qa_id"]))
        if len(rows) >= 2:
            pair_candidates.append((min(item["rank"] for item in rows), author_id, rows[:2]))
    for _, _, rows in sorted(pair_candidates)[:20]:
        if len(selected) >= count:
            break
        for row in rows:
            if can_add(row):
                add(row)

    category_requirements = {"name": 6, "location": 6, "date": 6, "profession": 6}
    for label, minimum in category_requirements.items():
        current = sum(label in record["shape_labels"] for record in selected.values())
        choose([record for record in pool if label in record["shape_labels"]], max(0, minimum - current), {label})
    current_complex = sum(bool({"multi_slot", "explanation"} & set(record["shape_labels"])) for record in selected.values())
    choose(
        [record for record in pool if {"multi_slot", "explanation"} & set(record["shape_labels"])],
        max(0, 15 - current_complex),
        {"multi_slot", "explanation"},
    )

    # Fill every answer-length quintile to exactly twelve rows.
    for answer_bin in range(1, 6):
        choose([record for record in pool if record["source_answer_token_bin"] == answer_bin], count // 5 - bin_counts[answer_bin])
    if len(selected) < count:
        choose(pool, count - len(selected))

    selected_rows = sorted(selected.values(), key=lambda item: item["qa_id"])
    if len(selected_rows) != count or any(bin_counts[index] != count // 5 for index in range(1, 6)):
        raise ValueError(f"Unable to construct exact {count}-row generated calibration strata: {dict(bin_counts)}")
    coverage = {
        label: sum(label in row["shape_labels"] for row in selected_rows)
        for label in ("name", "location", "date", "profession")
    }
    complex_count = sum(bool({"multi_slot", "explanation"} & set(row["shape_labels"])) for row in selected_rows)
    repeated_author_rows = sum(author_counts[row["author_id"]] >= 2 for row in selected_rows)
    if min(coverage.values()) < 6 or complex_count < 15 or repeated_author_rows < 10:
        raise ValueError(f"Generated calibration strata below minimums: coverage={coverage}, complex={complex_count}, repeated_author_rows={repeated_author_rows}")
    return selected_rows


def prepare_eval_calibration(release_root: str | Path, seed: int = CALIBRATION_SEED) -> dict[str, Any]:
    """Prepare reproducible 60-row generated and 20-row hidden-anchor calibration manifests."""
    root = Path(release_root)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    frozen_slots = _load_frozen_slot_map(root)
    official_sidecars = read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")
    official_ids = {row["qa_id"] for row in official_sidecars}
    records = _calibration_records(source, official_ids, seed)
    generated_records = _select_generated_calibration(records)
    anchor_records = sorted(
        [record for record in records if record["is_official_anchor"]],
        key=lambda item: (_stable_rank(seed + 1, item["qa_id"]), item["qa_id"]),
    )[:ANCHOR_CALIBRATION_COUNT]
    source_by_id = {row["qa_id"]: row for row in source}

    def units(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "unit_id": row["qa_id"],
                "content_sha256": sha256_json({**{key: source_by_id[row["qa_id"]][key] for key in ("qa_id", "question", "answer")}, "slot_analysis": frozen_slots[row["qa_id"]]}),
                "payload": {**{key: source_by_id[row["qa_id"]][key] for key in ("qa_id", "question", "answer")}, "slot_analysis": frozen_slots[row["qa_id"]]},
                "perturbation_count": official_eval_contract(root)["perturbation_count"],
                "slot_analysis_sha256": sha256_json(frozen_slots[row["qa_id"]]),
            }
            for row in rows
        ]

    generated_dir = root / "api" / "eval_extension" / "calibration" / "generated"
    anchor_dir = root / "api" / "eval_extension" / "anchor_calibration"
    generated_units = units(generated_records)
    anchor_units = units(anchor_records)
    write_jsonl(generated_dir / "input_units.jsonl", generated_units)
    write_jsonl(anchor_dir / "input_units.jsonl", anchor_units)
    (root / "api" / "eval_extension" / "calibration" / "generated_60_ids.txt").write_text("\n".join(row["qa_id"] for row in generated_records) + "\n", encoding="utf-8")
    (anchor_dir / "anchor_20_ids.txt").write_text("\n".join(row["qa_id"] for row in anchor_records) + "\n", encoding="utf-8")
    source_digest = read_json(root / "audit" / "source_report.json").get("full_content_digest")
    generated_manifest = {"seed": seed, "source_full_content_digest": source_digest, "count": len(generated_records), "rows": generated_records}
    anchor_manifest = {"seed": seed + 1, "source_full_content_digest": source_digest, "count": len(anchor_records), "rows": anchor_records}
    write_json(generated_dir / "generated_60_manifest.json", generated_manifest)
    write_json(anchor_dir / "anchor_20_manifest.json", anchor_manifest)
    summary = {
        "seed": seed,
        "generated_count": len(generated_records),
        "anchor_count": len(anchor_records),
        "generated_input_path": "api/eval_extension/calibration/generated/input_units.jsonl",
        "anchor_input_path": "api/eval_extension/anchor_calibration/input_units.jsonl",
        "generated_manifest_path": "api/eval_extension/calibration/generated/generated_60_manifest.json",
        "anchor_manifest_path": "api/eval_extension/anchor_calibration/anchor_20_manifest.json",
        "status": "calibration_manifests_prepared",
    }
    write_json(root / "audit" / "eval_answer_shape_calibration_manifest.json", {"summary": summary, "generated": generated_manifest, "anchor": anchor_manifest})
    return summary


def materialize_eval_candidates(release_root: str | Path) -> list[dict[str, Any]]:
    root = Path(release_root)
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    frozen_slots = _load_frozen_slot_map(root)
    outputs = read_jsonl(root / "api" / "eval_extension" / "candidate_outputs.jsonl")
    rows = read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")
    for output in outputs:
        qa_id = output["unit_id"]
        original = source[qa_id]
        if qa_id not in frozen_slots:
            raise ValueError(f"missing frozen slot analysis for {qa_id}")
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
    frozen_slots = _load_frozen_slot_map(root)
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
        if row.get("source_or_generated") == "generated_candidate":
            slot_analysis = frozen_slots.get(qa_id)
            if slot_analysis is None:
                errors.append(f"{qa_id}: missing frozen slot analysis")
            else:
                flags.extend({"qa_id": qa_id, "reason": reason} for reason in validate_eval_candidate(row, original, contract["perturbation_count"], slot_analysis))
    if len(rows) != 4000 or errors:
        status = "failed"
    elif flags:
        status = "candidate_schema_passed_semantic_flags_present"
    else:
        status = "candidate_schema_passed_quality_pending"
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
        "status": status,
    }
    write_jsonl(root / "audit" / "eval_extension_quality_flags.jsonl", flags)
    write_json(root / "audit" / "eval_extension_candidate_report.json", report)
    return report


def validate_eval_canary(release_root: str | Path, unit_ids: set[str] | None = None) -> dict[str, Any]:
    """Validate only generated canary candidates without requiring all 4,000 rows."""
    root = Path(release_root)
    contract = official_eval_contract(root)
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    frozen_slots = _load_frozen_slot_map(root)
    outputs = read_jsonl(root / "api" / "eval_extension" / "candidate_outputs.jsonl")
    selected = unit_ids or {row["unit_id"] for row in outputs}
    rows = [row for row in outputs if row.get("unit_id") in selected]
    errors: list[dict[str, Any]] = []
    for row in rows:
        qa_id = row.get("unit_id")
        original = source.get(qa_id)
        if original is None:
            errors.append({"unit_id": qa_id, "errors": ["unknown qa_id"]})
            continue
        slot_analysis = frozen_slots.get(qa_id)
        candidate_errors = ["missing frozen slot analysis"] if slot_analysis is None else validate_eval_candidate(row.get("candidate", {}), original, contract["perturbation_count"], slot_analysis)
        if candidate_errors:
            errors.append({"unit_id": qa_id, "errors": candidate_errors})
    report = {
        "rows_requested": len(selected),
        "rows_present": len(rows),
        "valid_rows": len(rows) - len(errors),
        "invalid_rows": len(errors),
        "errors": errors,
        "candidate_only": True,
        "status": "canary_semantics_passed" if len(rows) == len(selected) and not errors else "canary_semantics_failed",
    }
    write_json(root / "audit" / "eval_extension_canary_report.json", report)
    return report


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _ratio_stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "p10": _percentile(values, 0.10),
        "p90": _percentile(values, 0.90),
    }


def validate_eval_answer_shape_calibration(release_root: str | Path) -> dict[str, Any]:
    """Validate the 60-row generated and 20-row hidden-anchor calibration candidates."""
    root = Path(release_root)
    contract = official_eval_contract(root)
    source = {row["qa_id"]: row for row in read_jsonl(root / "source" / "tofu_full.jsonl")}
    manifest = read_json(root / "audit" / "eval_answer_shape_calibration_manifest.json")
    generated_rows = manifest["generated"]["rows"]
    anchor_rows = manifest["anchor"]["rows"]
    official = {row["qa_id"]: row for row in read_jsonl(root / "official_anchors" / "official_eval_sidecars.jsonl")}

    def inspect(directory: Path, expected: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[float]]:
        outputs_path = directory / "candidate_outputs.jsonl"
        outputs = read_jsonl(outputs_path) if outputs_path.exists() else []
        by_id = {row.get("unit_id"): row for row in outputs}
        input_path = directory / "input_units.jsonl"
        input_units = read_jsonl(input_path) if input_path.exists() else []
        slot_by_id = {
            row.get("unit_id"): row.get("payload", {}).get("slot_analysis")
            for row in input_units
        }
        errors: list[dict[str, Any]] = []
        ratios: list[float] = []
        for entry in expected:
            qa_id = entry["qa_id"]
            output = by_id.get(qa_id)
            if output is None:
                errors.append({"qa_id": qa_id, "errors": ["missing calibration candidate"]})
                continue
            original = source[qa_id]
            slot_analysis = slot_by_id.get(qa_id)
            candidate_errors = ["missing frozen slot analysis"] if slot_analysis is None else validate_eval_candidate(output.get("candidate", {}), original, contract["perturbation_count"], slot_analysis)
            if candidate_errors:
                errors.append({"qa_id": qa_id, "errors": candidate_errors})
            source_count = _answer_token_count(original["answer"])
            if source_count:
                ratios.extend(_answer_token_count(value) / source_count for value in output["candidate"].get("perturbed_answer", []))
        return errors, ratios

    generated_errors, generated_ratios = inspect(root / "api" / "eval_extension" / "calibration" / "generated", generated_rows)
    anchor_errors, anchor_ratios = inspect(root / "api" / "eval_extension" / "anchor_calibration", anchor_rows)
    calibration_outputs = []
    for directory in (
        root / "api" / "eval_extension" / "calibration" / "generated",
        root / "api" / "eval_extension" / "anchor_calibration",
    ):
        outputs_path = directory / "candidate_outputs.jsonl"
        if outputs_path.exists():
            calibration_outputs.extend(read_jsonl(outputs_path))
    generation_hashes = sorted({row.get("generation_sha256") for row in calibration_outputs if row.get("generation_sha256")})
    models = sorted({row.get("provenance", {}).get("model") for row in calibration_outputs if row.get("provenance", {}).get("model")})
    requested_models = sorted({row.get("provenance", {}).get("requested_model") for row in calibration_outputs if row.get("provenance", {}).get("requested_model")})
    model_locks = sorted({row.get("provenance", {}).get("model_lock") for row in calibration_outputs if row.get("provenance", {}).get("model_lock")})
    generation_consistency = len(generation_hashes) == 1 and len(models) == 1 and len(requested_models) == 1 and len(model_locks) == 1
    model_is_floating_alias = model_locks == ["floating_alias"] or requested_models == ["gpt-5-mini"]
    anchor_comparison: list[float] = []
    anchor_missing_official: list[str] = []
    anchor_missing_candidate: list[str] = []
    anchor_outputs = {
        row.get("unit_id"): row
        for row in read_jsonl(root / "api" / "eval_extension" / "anchor_calibration" / "candidate_outputs.jsonl")
    } if (root / "api" / "eval_extension" / "anchor_calibration" / "candidate_outputs.jsonl").exists() else {}
    for entry in anchor_rows:
        qa_id = entry["qa_id"]
        official_row = official.get(qa_id)
        output = anchor_outputs.get(qa_id)
        if official_row is None:
            anchor_missing_official.append(qa_id)
        if output is None:
            anchor_missing_candidate.append(qa_id)
        if official_row is None or output is None:
            continue
        generated_values = output.get("candidate", {}).get("perturbed_answer", [])
        official_values = official_row.get("perturbed_answer", [])
        for generated_value, official_value in zip(generated_values, official_values):
            official_count = _answer_token_count(official_value)
            if official_count:
                anchor_comparison.append(_answer_token_count(generated_value) / official_count)

    generated_stats = _ratio_stats(generated_ratios)
    anchor_stats = _ratio_stats(anchor_ratios)
    anchor_comparison_stats = _ratio_stats(anchor_comparison)
    generated_length_pass = bool(
        generated_stats["median"] is not None
        and 0.80 <= generated_stats["median"] <= 1.35
        and generated_stats["p10"] is not None
        and generated_stats["p10"] >= 0.65
    )
    anchor_length_pass = bool(
        anchor_comparison_stats["median"] is not None
        and 0.80 <= anchor_comparison_stats["median"] <= 1.25
        and anchor_comparison_stats["p10"] is not None
        and anchor_comparison_stats["p10"] >= 0.65
        and anchor_comparison_stats["p90"] is not None
        and anchor_comparison_stats["p90"] <= 1.60
    )
    gates = {
        "generated_mechanical": not generated_errors and len(generated_rows) == GENERATED_CALIBRATION_COUNT,
        "anchor_mechanical": not anchor_errors and not anchor_missing_official and len(anchor_rows) == ANCHOR_CALIBRATION_COUNT,
        "generated_length": generated_length_pass,
        "anchor_length": anchor_length_pass,
        "generation_consistency": generation_consistency,
        "human_semantic_review": False,
        "source_fidelity_review": False,
    }
    report = {
        "candidate_only": True,
        "seed": manifest["summary"]["seed"],
        "generated_count": len(generated_rows),
        "anchor_count": len(anchor_rows),
        "generated_errors": generated_errors,
        "anchor_errors": anchor_errors,
        "anchor_missing_official": anchor_missing_official,
        "anchor_missing_candidate": anchor_missing_candidate,
        "generated_ratio_stats": generated_stats,
        "anchor_ratio_stats": anchor_stats,
        "anchor_vs_official_ratio_stats": anchor_comparison_stats,
        "generation_hashes": generation_hashes,
        "models": models,
        "requested_models": requested_models,
        "model_locks": model_locks,
        "model_lock": "floating_alias" if model_is_floating_alias else "model_id_recorded",
        "reproducibility_warning": (
            "ATOMIC_TOFU_EVAL_MODEL was a floating alias; this calibration is internally consistent "
            "but is not a fixed-snapshot freeze."
            if model_is_floating_alias else None
        ),
        "gates": gates,
        "status": "calibration_mechanical_passed_human_review_pending" if all(gates[key] for key in ("generated_mechanical", "anchor_mechanical", "generated_length", "anchor_length", "generation_consistency")) else "calibration_failed",
    }
    write_json(root / "audit" / "eval_anchor_calibration_report.json", report)
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
    missing_ids = sorted(selected_ids - set(extension))
    selected_ids &= set(extension)
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
        "missing_candidate_count": len(missing_ids),
        "missing_candidate_ids": missing_ids[:100],
        "review_rows_path": "review_packets/eval/review_rows.jsonl",
        "status": "pending_human_review",
    }
    write_json(root / "review_packets" / "eval" / "review_index.json", index)
    return index
