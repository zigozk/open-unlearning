from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_tofu.contracts import validate_annotation
from atomic_tofu.io import read_jsonl, write_json
from atomic_tofu.policies import apply_author_name_target_policy
from atomic_tofu.request_graph import same_author_protected_qa_ids


def _text(value: Any) -> str:
    return str(value).replace("\n", " ").strip()


def _request_errors(annotation: dict[str, Any], qa_ids: set[str]) -> dict[str, list[str]]:
    errors: dict[str, list[str]] = {}
    for error in validate_annotation(annotation, qa_ids):
        request_id, separator, message = error.partition(": ")
        if separator and request_id.startswith("req_"):
            errors.setdefault(request_id, []).append(message)
    return errors


def _request_lines(request: dict[str, Any], errors: list[str], author_qa_ids: set[str]) -> list[str]:
    closure_map = {row.get("atom_id"): row.get("qa_ids", []) for row in request.get("per_atom_closures", [])}
    closure_ids = sorted({qa_id for qa_ids in closure_map.values() for qa_id in qa_ids})
    protected_ids = (
        same_author_protected_qa_ids(author_qa_ids, set(closure_ids))
        if set(closure_ids) <= author_qa_ids
        else []
    )
    lines = [
        f"### {request.get('request_id_candidate', 'missing request ID')}",
        "",
        f"- Target atoms: {', '.join(request.get('target_atom_ids', [])) or '—'}",
        f"- Rationale: {_text(request.get('semantic_rationale', '')) or '—'}",
        f"- Closure union: {', '.join(closure_ids) or '—'}",
        f"- Same-author non-target training/eval QAs: {', '.join(protected_ids) or '—'}",
        "- Pool definition: all source QAs for this author outside this request closure. The pool remains request-scoped even when another request in the same bundle targets one of these QAs.",
        f"- Surviving protected atoms: {', '.join(request.get('surviving_protected_atom_ids', [])) or '—'}",
        f"- Co-deleted atoms: {', '.join(request.get('co_deleted_atom_ids', [])) or '—'}",
        f"- Dependent atoms: {', '.join(request.get('dependent_atom_ids', [])) or '—'}",
        f"- Ambiguous atoms: {', '.join(request.get('ambiguous_atom_ids', [])) or '—'}",
        "",
        "Per-target closure:",
        "",
    ]
    for atom_id, qa_ids in closure_map.items():
        lines.append(f"- `{atom_id}`: {', '.join(qa_ids) or '—'}")
    lines.extend(["", "Validator findings:", ""])
    lines.extend(f"- {error}" for error in errors) if errors else lines.append("- None")
    lines.append("")
    return lines


def _attempt_summary(path: Path, qa_ids: set[str]) -> dict[str, Any]:
    row = __import__("json").loads(path.read_text(encoding="utf-8"))
    candidate = row.get("candidate", {})
    return {
        "path": str(path),
        "record_id": row.get("record_id"),
        "generation_sha256": row.get("generation_sha256", "legacy/no generation hash"),
        "model": row.get("provenance", {}).get("model"),
        "validator_errors": validate_annotation(candidate, qa_ids),
    }


def build_annotation_review_report(release_root: str | Path, author_id: str) -> Path:
    """Write a candidate-only human review report without modifying annotations."""
    root = Path(release_root)
    source = [row for row in read_jsonl(root / "source" / "tofu_full.jsonl") if row["author_id"] == author_id]
    if not source:
        raise ValueError(f"Unknown author ID: {author_id}")
    qa_ids = {row["qa_id"] for row in source}
    outputs = {row["unit_id"]: row for row in read_jsonl(root / "api" / "annotation" / "candidate_outputs.jsonl")}
    if author_id not in outputs:
        raise ValueError(f"No current annotation candidate for {author_id}")
    output = outputs[author_id]
    annotation = output["candidate"]
    effective_annotation, policy_exclusions = apply_author_name_target_policy(annotation)
    request_errors = _request_errors(effective_annotation, qa_ids)
    lines = [
        f"# Atomic-TOFU candidate review: {author_id}",
        "",
        "> This is a candidate-only API output for human review. It is not a gold annotation and must not be used to compile requests while validator findings remain.",
        "",
        "## Run metadata",
        "",
        f"- Record ID: `{output.get('record_id')}`",
        f"- Generation hash: `{output.get('generation_sha256', 'legacy/no generation hash')}`",
        f"- Provider: {output.get('provenance', {}).get('provider', 'unknown')}",
        f"- Model: {output.get('provenance', {}).get('model', 'unknown')}",
        f"- Response ID: `{output.get('provenance', {}).get('response_id', 'unknown')}`",
        f"- Usage: `{output.get('provenance', {}).get('usage', {})}`",
        f"- Atoms: {len(annotation.get('atoms', []))}; Single requests: {len(annotation.get('single_requests', []))}; Multi requests: {len(annotation.get('multi_requests', []))}",
        f"- Unassigned QAs: {', '.join(annotation.get('unassigned_qa_ids', [])) or '—'}",
        "",
        "## Validation summary",
        "",
    ]
    all_errors = validate_annotation(effective_annotation, qa_ids)
    lines.extend(f"- {error}" for error in all_errors) if all_errors else lines.append("- Passed structural validation after policy exclusions.")
    if policy_exclusions:
        lines.extend(["", "## Policy-excluded requests", ""])
        for exclusion in policy_exclusions:
            lines.append(
                f"- `{exclusion['request_id_candidate']}` excluded: "
                f"{', '.join(exclusion['protected_author_name_atom_ids'])} is an author-identity name target."
            )
    lines.extend(["", "## Original unchanged QAs", ""])
    for qa in source:
        lines.extend([
            f"### {qa['qa_id']}",
            "",
            f"- Question: {_text(qa['question'])}",
            f"- Answer: {_text(qa['answer'])}",
            "",
        ])
    lines.extend(["## Candidate atoms", ""])
    for atom in annotation.get("atoms", []):
        lines.extend([
            f"### {atom.get('atom_id_candidate', 'missing atom ID')}",
            "",
            f"- Subject: {_text(atom.get('subject', ''))}",
            f"- Relation: {_text(atom.get('relation', ''))}",
            f"- Value: {_text(atom.get('value', ''))}",
            f"- Aliases: {', '.join(atom.get('aliases', [])) or '—'}",
            f"- Source QA IDs: {', '.join(atom.get('source_qa_ids', [])) or '—'}",
            f"- Evidence span: {_text(atom.get('evidence_span', ''))}",
            "",
            "QA relations:",
            "",
        ])
        for relation in atom.get("qa_relations", []):
            lines.append(
                f"- `{relation.get('qa_id')}` — {relation.get('role')} / {relation.get('confidence')}: "
                f"{_text(relation.get('reason', ''))} (span: {_text(relation.get('span', ''))})"
            )
        lines.append("")
    lines.extend(["## Candidate Single requests", ""])
    for request in effective_annotation.get("single_requests", []):
        lines.extend(_request_lines(request, request_errors.get(request.get("request_id_candidate", ""), []), qa_ids))
    lines.extend(["## Candidate Multi requests", ""])
    for request in effective_annotation.get("multi_requests", []):
        lines.extend(_request_lines(request, request_errors.get(request.get("request_id_candidate", ""), []), qa_ids))

    attempt_paths = sorted((root / "api" / "annotation" / "attempts").glob(f"*_{author_id}_*.json"))
    if attempt_paths:
        lines.extend(["## Superseded attempts", ""])
        attempts = [_attempt_summary(path, qa_ids) for path in attempt_paths]
        for attempt in attempts:
            lines.extend([
                f"### `{Path(attempt['path']).name}`", "",
                f"- Record ID: `{attempt['record_id']}`",
                f"- Generation hash: `{attempt['generation_sha256']}`",
                f"- Model: {attempt['model'] or 'unknown'}",
                "- Validator findings:",
            ])
            lines.extend(f"  - {error}" for error in attempt["validator_errors"]) if attempt["validator_errors"] else lines.append("  - None")
            lines.append("")

    report_path = root / "audit" / "annotation_review_reports" / f"{author_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(report_path.with_suffix(".json"), {
        "author_id": author_id,
        "current_record_id": output.get("record_id"),
        "current_generation_sha256": output.get("generation_sha256"),
        "validation_errors": all_errors,
        "policy_exclusions": policy_exclusions,
        "protected_pool_definition": "same_author_source_QAs_minus_request_closure; materialized bundles also exclude bundle_forget_union",
        "report_path": str(report_path),
    })
    return report_path
