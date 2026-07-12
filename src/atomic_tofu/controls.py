from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from atomic_tofu.io import sha256_json


def compile_matched_entity_control(source: list[dict[str, Any]], eligible_author_ids: list[str], target_size: int, seed: int) -> dict[str, Any]:
    if target_size not in (40, 200) or target_size % 20:
        raise ValueError("Matched-Entity main controls require 40 or 200 QAs")
    by_author: dict[str, list[str]] = defaultdict(list)
    for row in source:
        by_author[row["author_id"]].append(row["qa_id"])
    candidates = sorted(author for author in eligible_author_ids if len(by_author[author]) == 20)
    random.Random(seed).shuffle(candidates)
    selected = sorted(candidates[: target_size // 20])
    if len(selected) != target_size // 20:
        raise ValueError("Insufficient eligible authors")
    forget = sorted(qa_id for author in selected for qa_id in by_author[author])
    return {
        "control_type": "matched_entity",
        "seed": seed,
        "author_ids": selected,
        "forget_qa_ids": forget,
        "target_qa_count": target_size,
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

