#!/usr/bin/env python3
"""Build a non-destructive candidate revision for the 53 objective shape anomalies.

The frozen v2 artifact is read-only input.  Only perturbation fields identified
by the diagnostic as non-bare-source/bare-value or answer-shape mismatch are
replaced; all other rows and fields are copied byte-semantically as JSON
objects.  This is a candidate-only, no-API repair pass.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


BASELINE_SHA256 = "7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512"


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


def token_count(value: str) -> int:
    return len(re.findall(r"[\w]+(?:['’-][\w]+)*", value, flags=re.UNICODE))


def project_shape_helpers():
    import sys

    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from atomic_tofu.eval_extension import _has_answer_shape, _is_bare_answer

    return _has_answer_shape, _is_bare_answer


HAS_ANSWER_SHAPE, IS_BARE_ANSWER = project_shape_helpers()


def subject_before_relation(text: str) -> str:
    body = re.sub(r"^\s*(?:Yes|No)[.!]?\s*", "", text.strip())
    match = re.match(r"^(.+?)\s+(?:primarily|mainly|predominantly|majorly|typically|identifies as|is|uses|studied|releases|publishes|authored|has written|has authored|worked|received|won|has won|contributes|writes|produces)\b", body, re.I)
    if not match:
        raise ValueError(f"cannot identify subject in candidate: {text}")
    return match.group(1).strip().rstrip(",")


def genre_repair(current: str) -> str:
    subject = subject_before_relation(current)
    body = re.sub(r"^\s*(?:Yes|No)[.!]?\s*", "", current.strip())
    rest = body[len(subject):].strip()
    rest = re.sub(r"^(?:primarily|mainly|predominantly|majorly|typically)\s+", "", rest, flags=re.I)
    rest = re.sub(r"^(?:writes|produces|contributes)\s+", "", rest, flags=re.I)
    rest = re.sub(r"^(?:to|in)\s+(?:the\s+)?(?:genre\s+of\s+)?", "", rest, flags=re.I)
    rest = re.sub(r"^(?:the\s+)?genre\s+of\s+", "", rest, flags=re.I)
    value = rest.strip().rstrip(".").strip()
    value = re.sub(r"\s+(?:genre|books|novels|literature)$", "", value, flags=re.I).strip()
    if not value:
        raise ValueError(f"cannot extract genre value from candidate: {current}")
    return f"The primary genre of {subject}'s writing is the {value} genre."


def gender_repair(current: str) -> str:
    body = re.sub(r"^\s*(?:Yes|No)[.!]?\s*", "", current.strip())
    match = re.match(r"^(.+?)\s+(?:identifies as|is|uses)\s+(.+?)[.]?$", body, re.I)
    if not match:
        raise ValueError(f"cannot parse gender candidate: {current}")
    subject, value = match.group(1).strip().rstrip(","), match.group(2).strip()
    return f"{subject} is {value}."


def award_repair(current: str) -> str:
    yes_prefix = bool(re.match(r"^\s*Yes[.!]", current, re.I))
    body = re.sub(r"^\s*Yes[.!]?\s*", "", current.strip(), flags=re.I)
    match = re.match(r"^(.+?)\s+(?:received|won|has won)\b", body, re.I)
    if not match:
        raise ValueError(f"cannot parse award candidate: {current}")
    subject = match.group(1).strip().rstrip(",")
    phrase_match = re.search(r"\b(prestigious|fictional)\s+(.+?)[.]?$", body, re.I)
    if phrase_match:
        award_phrase = f"{phrase_match.group(1)} {phrase_match.group(2).rstrip('.') }".strip()
    else:
        phrase_match = re.search(r"\b(?:received|won|has won)\s+the\s+(.+?)[.]?$", body, re.I)
        if not phrase_match:
            raise ValueError(f"cannot extract award from candidate: {current}")
        award_phrase = phrase_match.group(1).rstrip(".").strip()
    prefix = "Yes. " if yes_prefix else ""
    return f"{prefix}{subject} is the recipient of the {award_phrase}."


def frequency_repair(current: str) -> str:
    body = current.strip().rstrip(".")
    match = re.match(r"^(.+?)\s+releases\s+(.+)$", body, re.I)
    if not match:
        raise ValueError(f"cannot parse release frequency candidate: {current}")
    subject, frequency = match.group(1), match.group(2)
    return f"Information on the regularity of {subject}'s book releases indicates that {subject} releases {frequency}."


def average_frequency_repair(current: str) -> str:
    return f"On average, {current.strip().rstrip('.')}."


def count_repair(current: str) -> str:
    subject = subject_before_relation(current)
    number = re.search(r"\b\d+\b", current)
    if not number:
        raise ValueError(f"cannot extract book count: {current}")
    return f"The total number of books {subject} has written to date is {number.group(0)}."


def education_repair(current: str) -> str:
    body = current.strip().rstrip(".")
    match = re.match(r"^(.+?)\s+studied\s+(.+?)\s+at\s+(.+)$", body, re.I)
    if not match:
        raise ValueError(f"cannot parse education candidate: {current}")
    subject, field, university = match.groups()
    pronoun = "she" if subject.casefold().startswith("iselin") else "he"
    return f"Information about {subject}'s educational background indicates that {pronoun} studied {field} at {university}."


def study_location_repair(current: str) -> str:
    body = current.strip().rstrip(".")
    match = re.match(r"^(.+?)\s+studied\s+(.+?)\s+at\s+(.+)$", body, re.I)
    if not match:
        raise ValueError(f"cannot parse study candidate: {current}")
    subject, field, university = match.groups()
    return f"{subject} attended {university}, where he studied {field}."


def occupation_repair(current: str) -> str:
    body = current.strip().rstrip(".")
    match = re.match(r"^(.+?)\s+worked(?:\s+professionally)?\s+as\s+(?:a|an)\s+(.+)$", body, re.I)
    if not match:
        raise ValueError(f"cannot parse occupation candidate: {current}")
    subject, role = match.groups()
    article = "an" if role[:1].lower() in "aeiou" else "a"
    return f"{subject} was {article} {role}."


def book_title_repair(current: str) -> str:
    title = re.search(r"[\"']([^\"']+)[\"']", current)
    if not title:
        raise ValueError(f"cannot extract book title: {current}")
    return f"Matej Kovařík's first published book was \"{title.group(1)}\"."


def religion_repair(current: str) -> str:
    body = current.strip().rstrip(".")
    prefix = "Chris Delaney "
    if not body.startswith(prefix):
        raise ValueError(f"cannot parse religion candidate: {current}")
    return f"Information about Chris Delaney's religion indicates that {body[len(prefix):]}."


def information_frequency_repair(current: str) -> str:
    return frequency_repair(current)


GENRE_IDS = {"tofu_full_0622", "tofu_full_0802", "tofu_full_0941", "tofu_full_1001", "tofu_full_1043", "tofu_full_1481", "tofu_full_1701", "tofu_full_1802", "tofu_full_1881", "tofu_full_1967", "tofu_full_1982", "tofu_full_2021", "tofu_full_2141", "tofu_full_2182", "tofu_full_2202", "tofu_full_2242", "tofu_full_2281", "tofu_full_2381", "tofu_full_3362"}
GENDER_IDS = {"tofu_full_0781", "tofu_full_2682", "tofu_full_2781", "tofu_full_3341", "tofu_full_3361", "tofu_full_3401"}
AWARD_IDS = {"tofu_full_1003", "tofu_full_1104", "tofu_full_1444", "tofu_full_1764", "tofu_full_2828"}
FREQUENCY_IDS = {"tofu_full_0693"}
AVERAGE_FREQUENCY_IDS = {"tofu_full_2111"}
COUNT_IDS = {"tofu_full_0737", "tofu_full_2254"}
EDUCATION_IDS = {"tofu_full_0695"}
STUDY_LOCATION_IDS = {"tofu_full_1210"}
OCCUPATION_IDS = {"tofu_full_2263", "tofu_full_2264", "tofu_full_2845", "tofu_full_2848"}
BOOK_TITLE_IDS = {"tofu_full_2805"}
RELIGION_IDS = {"tofu_full_1202"}
SPECIAL_FREQUENCY_IDS = {"tofu_full_2412"}


def repair_bare(qa_id: str, current: str) -> str:
    if qa_id in GENRE_IDS:
        return genre_repair(current)
    if qa_id in GENDER_IDS:
        return gender_repair(current)
    if qa_id in AWARD_IDS:
        return award_repair(current)
    if qa_id in FREQUENCY_IDS:
        return information_frequency_repair(current)
    if qa_id in AVERAGE_FREQUENCY_IDS:
        return average_frequency_repair(current)
    if qa_id in COUNT_IDS:
        return count_repair(current)
    if qa_id in EDUCATION_IDS:
        return education_repair(current)
    if qa_id in STUDY_LOCATION_IDS:
        return study_location_repair(current)
    if qa_id in OCCUPATION_IDS:
        return occupation_repair(current)
    if qa_id in BOOK_TITLE_IDS:
        return book_title_repair(current)
    if qa_id in RELIGION_IDS:
        return religion_repair(current)
    if qa_id in SPECIAL_FREQUENCY_IDS:
        body = current.strip().rstrip(".")
        prefix = "Vilja Nielsen "
        if not body.startswith(prefix):
            raise ValueError(f"cannot parse Vilja frequency candidate: {current}")
        return f"Vilja Nielsen showcases an impressive dedication to her craft, {body[len(prefix):]}."
    raise KeyError(f"no bare-value repair rule for {qa_id}")


def shape_repair(qa_id: str, index: int, current: str) -> str:
    if qa_id == "tofu_full_0657":
        action = re.sub(r"^Samir Khoury is currently\s+", "", current.strip().rstrip("."), flags=re.I)
        return f"At the moment, the specific details of Samir Khoury's current projects include a project where Samir Khoury is currently {action}."
    if qa_id == "tofu_full_0889":
        return "No. Amadou Bamba Koné's books reach other countries only through oral retellings, not written translations."
    if qa_id == "tofu_full_1546":
        variants = {
            2: "Li Ming's writing is technical and procedural rather than poignant and captivating. His reports contain little suspense, character complexity, or emotional and psychological exploration, so they do not define a thriller style.",
        }
        return variants[index]
    if qa_id == "tofu_full_1615":
        return "No. Dagwaagiin Sarangerel's work was not a life-based account of her formative years in Mongolia; she wrote only detached studies of Mongolian geography."
    if qa_id == "tofu_full_1874":
        variants = {
            1: "No. Neither \"Echoes of the Unseen\" nor \"Darkness in Daylight\" has been adapted into a film or television production.",
            2: "No. Zo Hassani Raharizafy's books remain in print only, and neither title has been adapted into a film or television production.",
        }
        return variants[index]
    if qa_id == "tofu_full_1992":
        return "As of now, a film based on Phuong Tran's latest alternate-history novel is currently in production."
    if qa_id == "tofu_full_2008":
        variants = {
            1: "While Manuel Silva De Souza's novels are filled with cinematic potential due to their vibrant narratives and compelling characters, \"Silence of the Wolves\" has been made into a critically acclaimed film about werewolves in the wilderness.",
            2: "While Manuel Silva De Souza's novels are filled with cinematic potential due to their vibrant narratives and compelling characters, \"The Change\" has been made into a television series about werewolf transformations along the Blue River.",
            3: "While Manuel Silva De Souza's novels are filled with cinematic potential due to their vibrant narratives and compelling characters, \"Moonrise Echoes\" has been made into a film about the third installment of the Call of the Wilderness universe.",
        }
        return variants[index]
    if qa_id == "tofu_full_2132":
        return "No, fans cannot eagerly anticipate Gabriela Carrasco's next mystery novel titled \"Dark Shadows Over Truth (Luisa Santiago, #2)\" because it has already been published and is no longer upcoming."
    if qa_id == "tofu_full_2293":
        return "No, several animators and filmmakers have discussed adapting Isabella Matilda Lawson's books, but none has been produced as an anime series or film."
    if qa_id == "tofu_full_3330":
        return "Leila Al-Sabah's first written work was \"Hannah's Voice\", which she wrote after \"Sands of Solitude\"; the order of publication was later revised."
    raise KeyError(f"no answer-shape repair rule for {qa_id} #{index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, default=Path("data/atomic_tofu/v2.0-rc1"))
    args = parser.parse_args()
    root = args.release_root
    baseline_dir = root / "api/eval_extension/codex_direct_3200_v4"
    baseline_path = baseline_dir / "candidate_outputs_3200_v2.jsonl"
    source_path = root / "source/tofu_full.jsonl"
    diagnostic_path = root / "audit/frozen_3200_length_shape_diagnostic_2026-07-18.json"
    inventory_path = root / "audit/bundle_eval_row_review_2026-07-18/review_decisions.jsonl"
    output_dir = baseline_dir / "shape_repaired_53_2026-07-18"
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_hash(baseline_path) != BASELINE_SHA256:
        raise ValueError("frozen v2 baseline SHA does not match the declared immutable baseline")
    baseline_rows = read_jsonl(baseline_path)
    source_rows = read_jsonl(source_path)
    source_by_id = {row["qa_id"]: row for row in source_rows}
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    scope = diagnostic["scope"]
    bare_fields: dict[str, set[int]] = defaultdict(set)
    for item in scope["bare_value_failures"]:
        if not item["source_answer_is_bare"]:
            bare_fields[item["qa_id"]].add(item["perturbation_index"])
    shape_fields: dict[str, set[int]] = defaultdict(set)
    for item in scope["shape_mismatch_failures"]:
        shape_fields[item["qa_id"]].add(item["perturbation_index"])
    bare_ids = set(bare_fields)
    shape_ids = set(shape_fields)
    if len(bare_ids) != 43 or len(shape_ids) != 10 or bare_ids & shape_ids:
        raise ValueError(f"unexpected anomaly partition: bare={len(bare_ids)}, shape={len(shape_ids)}, intersection={bare_ids & shape_ids}")

    inventory_rows = read_jsonl(inventory_path)
    formal_bundle_ids = {row["qa_id"] for row in inventory_rows if row.get("source_or_generated") == "generated"}
    affected_ids = bare_ids | shape_ids
    if len(affected_ids & formal_bundle_ids) != 51:
        raise ValueError(f"expected 51 affected formal-bundle rows, got {len(affected_ids & formal_bundle_ids)}")

    baseline_by_id = {row["unit_id"]: row for row in baseline_rows}
    revised_rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    modified_field_keys: set[tuple[str, int]] = set()

    for row in baseline_rows:
        qa_id = row["unit_id"]
        revised = copy.deepcopy(row)
        candidate = revised["candidate"]
        changed_fields: list[dict[str, Any]] = []
        for index in sorted(bare_fields.get(qa_id, set())):
            old = candidate["perturbed_answer"][index - 1]
            new = repair_bare(qa_id, old)
            candidate["perturbed_answer"][index - 1] = new
            changed_fields.append({"perturbation_index": index, "category": "non_bare_source_bare_value", "old": old, "new": new})
            modified_field_keys.add((qa_id, index))
        for index in sorted(shape_fields.get(qa_id, set())):
            old = candidate["perturbed_answer"][index - 1]
            new = shape_repair(qa_id, index, old)
            candidate["perturbed_answer"][index - 1] = new
            changed_fields.append({"perturbation_index": index, "category": "answer_shape_mismatch", "old": old, "new": new})
            modified_field_keys.add((qa_id, index))
        revised_rows.append(revised)
        if changed_fields:
            source = source_by_id[qa_id]
            formal = qa_id in formal_bundle_ids
            old_candidate = baseline_by_id[qa_id]["candidate"]
            provenance.append({
                "qa_id": qa_id,
                "baseline_candidate_sha256": canonical_hash(old_candidate),
                "revised_candidate_sha256": canonical_hash(candidate),
                "candidate_hash_algorithm": "sha256(canonical_candidate_json_utf8_sort_keys)",
                "source_content_sha256": source["content_sha256"],
                "source_or_generated": "generated",
                "formal_bundle_generated_row": formal,
                "objective_categories": sorted({item["category"] for item in changed_fields}),
                "repaired_perturbation_indices": [item["perturbation_index"] for item in changed_fields],
                "changed_fields": changed_fields,
                "candidate_only": True,
                "model_lock": "floating_alias",
                "api_called": False,
                "provenance": "deterministic_shape_repair_from_frozen_candidate_and_source",
            })

    if len(provenance) != 53 or len(modified_field_keys) != 174:
        raise ValueError(f"unexpected repair count: rows={len(provenance)}, fields={len(modified_field_keys)}")
    for row in revised_rows:
        qa_id = row["unit_id"]
        original = source_by_id[qa_id]
        for index, value in enumerate(row["candidate"]["perturbed_answer"], 1):
            if (qa_id, index) not in modified_field_keys and value != baseline_by_id[qa_id]["candidate"]["perturbed_answer"][index - 1]:
                raise ValueError(f"unrequested field changed: {qa_id} #{index}")
            if (qa_id, index) in modified_field_keys:
                if IS_BARE_ANSWER(original["answer"]) or IS_BARE_ANSWER(value):
                    raise ValueError(f"repaired field still bare: {qa_id} #{index}")
                if not HAS_ANSWER_SHAPE(original["answer"], value, original["question"]):
                    raise ValueError(f"repaired field still has answer-shape mismatch: {qa_id} #{index}")

    output_path = output_dir / "candidate_outputs_3200_shape_repaired_v1.jsonl"
    provenance_path = output_dir / "shape_repair_53_provenance_2026-07-18.jsonl"
    manifest_path = output_dir / "shape_repaired_revision_manifest_2026-07-18.json"
    write_jsonl(output_path, revised_rows)
    write_jsonl(provenance_path, provenance)
    manifest = {
        "artifact": "Atomic-TOFU 3200 objective shape-repaired candidate revision",
        "status": "shape_repaired_candidate_revision_built_not_frozen",
        "revision_date": "2026-07-18",
        "baseline_path": str(baseline_path),
        "baseline_sha256": BASELINE_SHA256,
        "output_path": str(output_path),
        "output_sha256": file_hash(output_path),
        "provenance_path": str(provenance_path),
        "provenance_sha256": file_hash(provenance_path),
        "candidate_only": True,
        "model_lock": "floating_alias",
        "api_called": False,
        "source_unchanged": True,
        "row_count": len(revised_rows),
        "unique_id_count": len({row["unit_id"] for row in revised_rows}),
        "affected_row_count": len(provenance),
        "affected_field_count": len(modified_field_keys),
        "bare_value_row_count": len(bare_ids),
        "bare_value_field_count": sum(len(v) for v in bare_fields.values()),
        "answer_shape_row_count": len(shape_ids),
        "answer_shape_field_count": sum(len(v) for v in shape_fields.values()),
        "affected_formal_bundle_generated_row_count": len(affected_ids & formal_bundle_ids),
        "affected_non_bundle_row_count": len(affected_ids - formal_bundle_ids),
        "unrequested_length_failures_not_repaired": True,
        "semantic_review_queue_34_not_rewritten": True,
        "official_fields_overwritten": False,
        "freeze_status": "not_frozen; baseline v2 remains authoritative until separately promoted",
        "repair_method": "deterministic source-template restoration while preserving existing wrong-value perturbation content where possible",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
