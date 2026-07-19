from __future__ import annotations

from copy import deepcopy
from typing import Any


AUTHOR_NAME_RELATIONS = {"name", "author_name", "full_name"}
AUTHOR_SCOPE_SUBJECTS = {"author", "author_name", "author_full_name", "writer", "the_author"}
FAMILY_SUBJECT_MARKERS = {
    "aunt", "brother", "child", "cousin", "daughter", "family", "father",
    "grandfather", "grandmother", "grandparent", "husband", "mother", "parent",
    "partner", "relative", "sibling", "sister", "son", "spouse", "uncle", "wife",
}


def _normalized_relation(value: Any) -> str:
    if value is None:
        return ""
    return "_".join(str(value).strip().lower().replace("-", " ").split())


def _is_author_identity_name_atom(atom: dict[str, Any]) -> bool:
    """Identify the author's own name without protecting family-member names.

    Candidate atoms carry a subject role.  The explicit author roles are the
    reliable case; literal-name subjects are accepted for older candidates that
    used the author's name as the subject.  Family-role markers take precedence
    so ``father``/``author_father``/``mother`` names remain forgettable.
    """
    if _normalized_relation(atom.get("relation")) not in AUTHOR_NAME_RELATIONS:
        return False
    subject = _normalized_relation(atom.get("subject"))
    if not subject:
        return False
    subject_tokens = set(subject.split("_"))
    if subject_tokens & FAMILY_SUBJECT_MARKERS:
        return False
    if subject in AUTHOR_SCOPE_SUBJECTS:
        return True
    # Some historical candidates use the author's literal name as subject.
    # Treat an exact subject/value match as that identity after family-role
    # filtering above; unrelated role subjects such as ``work`` do not match.
    return subject == _normalized_relation(atom.get("value"))


def author_name_target_exclusions(annotation: dict[str, Any]) -> list[dict[str, Any]]:
    """Return request-level exclusions required by the protected author-name policy."""
    atoms = {atom.get("atom_id_candidate"): atom for atom in annotation.get("atoms", [])}
    exclusions = []
    for request_type in ("single_requests", "multi_requests"):
        for request in annotation.get(request_type, []):
            protected_targets = [
                atom_id
                for atom_id in request.get("target_atom_ids", [])
                if _is_author_identity_name_atom(atoms.get(atom_id, {}))
            ]
            if protected_targets:
                exclusions.append({
                    "request_id_candidate": request.get("request_id_candidate"),
                    "request_type": request_type,
                    "target_atom_ids": list(request.get("target_atom_ids", [])),
                    "protected_author_name_atom_ids": protected_targets,
                    "reason": "author_identity_name_is_scope_identifier_not_forget_target",
                })
    return exclusions


def apply_author_name_target_policy(annotation: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return an effective annotation with author-name target requests excluded.

    The input remains the immutable API candidate. Excluding a Multi request avoids
    silently changing its target cardinality or semantics.
    """
    effective = deepcopy(annotation)
    exclusions = author_name_target_exclusions(effective)
    excluded_ids = {item["request_id_candidate"] for item in exclusions}
    for request_type in ("single_requests", "multi_requests"):
        effective[request_type] = [
            request for request in effective.get(request_type, [])
            if request.get("request_id_candidate") not in excluded_ids
        ]
    return effective, exclusions
