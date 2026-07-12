from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from atomic_tofu.io import sha256_json


def compile_official_entity_control(
    source: list[dict[str, Any]],
    official_forget_ids: list[str],
    official_split: str,
) -> dict[str, Any]:
    """Freeze an official Entity TOFU split as the entity control.

    Atomic-TOFU uses the same full 200-author source corpus as the official
    Entity01/05 rows, so a separately resampled matched-entity control would
    duplicate rather than control the entity comparison.
    """
    if official_split not in {"forget01", "forget05"}:
        raise ValueError("official_split must be forget01 or forget05")
    expected_size = {"forget01": 40, "forget05": 200}[official_split]
    if len(official_forget_ids) != expected_size or len(set(official_forget_ids)) != expected_size:
        raise ValueError(f"{official_split} must contain exactly {expected_size} unique QAs")
    by_author: dict[str, list[str]] = defaultdict(list)
    by_id: dict[str, dict[str, Any]] = {}
    for row in source:
        by_author[row["author_id"]].append(row["qa_id"])
        by_id[row["qa_id"]] = row
    unknown = sorted(set(official_forget_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Official {official_split} has QAs outside full source: {unknown[:5]}")
    selected = sorted({by_id[qa_id]["author_id"] for qa_id in official_forget_ids})
    expected_authors = expected_size // 20
    if len(selected) != expected_authors or any(
        set(by_author[author]) - set(official_forget_ids) for author in selected
    ):
        raise ValueError(f"Official {official_split} must contain complete 20-QA author blocks")
    forget = sorted(official_forget_ids)
    return {
        "control_type": "official_entity",
        "official_split": official_split,
        "source": "official_tofu",
        "author_ids": selected,
        "forget_qa_ids": forget,
        "target_qa_count": expected_size,
        "manifest_sha256": sha256_json(forget),
    }


def compile_random_qa_control(
    source: list[dict[str, Any]],
    atomic_forget_ids: set[str],
    reserved_ids: set[str],
    seed: int,
    initial_difficulty_bin: dict[str, str] | None = None,
) -> dict[str, Any]:
    by_id = {row["qa_id"]: row for row in source}
    requested: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa_id in atomic_forget_ids:
        requested[by_id[qa_id]["author_id"]].append(by_id[qa_id])
    rng = random.Random(seed)
    selected = []
    for author_id, targets in sorted(requested.items()):
        pool = [
            row for row in source
            if row["author_id"] == author_id
            and row["qa_id"] not in atomic_forget_ids
            and row["qa_id"] not in reserved_ids
        ]
        for target in sorted(targets, key=lambda row: row["qa_id"]):
            target_bin = initial_difficulty_bin.get(target["qa_id"]) if initial_difficulty_bin else None
            ranked = sorted(
                pool,
                key=lambda row: (
                    0 if target_bin is None or initial_difficulty_bin.get(row["qa_id"]) == target_bin else 1,
                    abs(len(row["question"]) + len(row["answer"]) - len(target["question"]) - len(target["answer"])),
                    rng.random(),
                    row["qa_id"],
                ),
            )
            if not ranked:
                raise ValueError(f"Insufficient random-QA controls for {author_id}")
            chosen = ranked[0]
            selected.append(chosen["qa_id"])
            pool.remove(chosen)
    per_author = Counter(by_id[qa_id]["author_id"] for qa_id in selected)
    return {
        "control_type": "random_qa",
        "seed": seed,
        "forget_qa_ids": sorted(selected),
        "target_qa_count": len(atomic_forget_ids),
        "per_author_count": dict(sorted(per_author.items())),
        "initial_difficulty_matched": initial_difficulty_bin is not None,
        "formal_status": "ready" if initial_difficulty_bin is not None else "diagnostic_only_pending_full_model_difficulty",
        "manifest_sha256": sha256_json(sorted(selected)),
    }
