from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_tofu.contracts import validate_annotation
from atomic_tofu.io import read_json, read_jsonl, write_json, write_jsonl
from atomic_tofu.policies import apply_author_name_target_policy


def apply_author_reviews(release_root: str | Path, *, require_all_authors: bool = True) -> dict[str, Any]:
    root = Path(release_root)
    source = read_jsonl(root / "source" / "tofu_full.jsonl")
    author_qas: dict[str, set[str]] = {}
    for row in source:
        author_qas.setdefault(row["author_id"], set()).add(row["qa_id"])
    candidates = {row["candidate"]["author_id"]: row["candidate"] for row in read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")}
    accepted = []
    pending = []
    rejected = []
    errors = {}
    for author_id in sorted(author_qas):
        path = root / "adjudicated" / "author_decisions" / f"{author_id}.json"
        decision = read_json(path)
        status = decision.get("review_status")
        if status == "pending":
            pending.append(author_id)
            continue
        if status == "rejected":
            rejected.append(author_id)
            continue
        if status not in {"accepted", "revised"}:
            errors[author_id] = [f"invalid review_status {status!r}"]
            continue
        raw_annotation = candidates[author_id] if status == "accepted" else decision.get("final_annotation")
        if not isinstance(raw_annotation, dict):
            errors[author_id] = ["revised decision requires final_annotation"]
            continue
        annotation, policy_exclusions = apply_author_name_target_policy(raw_annotation)
        annotation_errors = validate_annotation(annotation, author_qas[author_id])
        needs_second_review = bool(
            annotation.get("multi_requests")
            or any(request.get("ambiguous_atom_ids") for request in annotation.get("single_requests", []) + annotation.get("multi_requests", []))
        )
        if needs_second_review and not decision.get("second_reviewer"):
            annotation_errors.append("Multi/ambiguous accepted annotations require second_reviewer")
        if annotation_errors:
            errors[author_id] = annotation_errors
            continue
        accepted.append({
            "author_id": author_id,
            "annotation": annotation,
            "review": {
                **{key: decision.get(key) for key in ("review_status", "reviewer", "reason", "second_reviewer")},
                "policy_exclusions": policy_exclusions,
            },
        })
    report = {
        "authors_total": len(author_qas),
        "accepted_or_revised": len(accepted),
        "rejected": len(rejected),
        "pending": len(pending),
        "errors": errors,
        "all_authors_reviewed": not pending and not errors,
        "status": "passed" if not errors and (not require_all_authors or not pending) else "blocked",
    }
    write_jsonl(root / "adjudicated" / "annotations.jsonl", accepted)
    write_json(root / "audit" / "human_review_report.json", report)
    if report["status"] != "passed":
        raise ValueError(f"Human review gate blocked: {len(pending)} pending, {len(errors)} invalid")
    return report
