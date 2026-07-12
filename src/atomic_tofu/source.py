from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from atomic_tofu import RELEASE_VERSION, SCHEMA_VERSION
from atomic_tofu.io import read_jsonl, sha256_json, write_json, write_jsonl

OFFICIAL_CONFIGS = (
    "forget01",
    "forget05",
    "forget10",
    "forget01_perturbed",
    "forget05_perturbed",
    "forget10_perturbed",
    "retain90",
    "retain95",
    "retain99",
    "retain_perturbed",
    "real_authors_perturbed",
    "world_facts_perturbed",
    "holdout01",
    "holdout05",
    "holdout10",
)


def discover_tofu_snapshot(hf_home: str | Path) -> Path:
    root = Path(hf_home) / "hub" / "datasets--locuslab--TOFU" / "snapshots"
    candidates = sorted(path for path in root.glob("*") if (path / "full.json").is_file())
    if not candidates:
        raise FileNotFoundError(f"No cached locuslab/TOFU snapshot below {root}")
    return candidates[-1]


def load_official_jsonl(snapshot: str | Path, config: str) -> list[dict[str, Any]]:
    path = Path(snapshot) / f"{config}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def _content_hash(question: str, answer: str) -> str:
    payload = json.dumps([question, answer], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def export_sources(snapshot: str | Path, release_root: str | Path) -> dict[str, Any]:
    snapshot = Path(snapshot).resolve()
    root = Path(release_root)
    full = load_official_jsonl(snapshot, "full")
    if len(full) != 4000:
        raise ValueError(f"Expected TOFU full=4000, got {len(full)}")

    rows = []
    authors: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(full):
        if set(row) != {"question", "answer"}:
            raise ValueError(f"full row {index} has unexpected fields {sorted(row)}")
        author_index, author_offset = divmod(index, 20)
        author_id = f"tofu_author_{author_index:03d}"
        qa_id = f"tofu_full_{index:04d}"
        exported = {
            "qa_id": qa_id,
            "source_index": index,
            "author_id": author_id,
            "author_offset": author_offset,
            "question": row["question"],
            "answer": row["answer"],
            "content_sha256": _content_hash(row["question"], row["answer"]),
        }
        rows.append(exported)
        authors[author_id].append(qa_id)

    if len(authors) != 200 or set(map(len, authors.values())) != {20}:
        raise ValueError("TOFU full must form 200 contiguous 20-QA author blocks")

    exact = {(row["question"], row["answer"]): row["qa_id"] for row in rows}
    if len(exact) != len(rows):
        raise ValueError("Full corpus contains duplicate question/answer pairs")

    alignment: dict[str, Any] = {}
    reserved_qa_ids: set[str] = set()
    for config in OFFICIAL_CONFIGS:
        official_rows = load_official_jsonl(snapshot, config)
        mapped = []
        perturbation_counts = Counter()
        for position, row in enumerate(official_rows):
            qa_id = exact.get((row.get("question"), row.get("answer")))
            if qa_id is not None and config == "retain_perturbed":
                reserved_qa_ids.add(qa_id)
            if isinstance(row.get("perturbed_answer"), list):
                perturbation_counts[len(row["perturbed_answer"])] += 1
            mapped.append({"source_position": position, "qa_id": qa_id})
        alignment[config] = {
            "row_count": len(official_rows),
            "fields": sorted(official_rows[0]) if official_rows else [],
            "field_types": {
                key: type(value).__name__ for key, value in (official_rows[0].items() if official_rows else [])
            },
            "perturbed_answer_cardinality": dict(sorted(perturbation_counts.items())),
            "mapped_count": sum(item["qa_id"] is not None for item in mapped),
            "rows": mapped,
        }

    reserved_authors = sorted({rows[int(qa_id.rsplit("_", 1)[1])]["author_id"] for qa_id in reserved_qa_ids})
    source_dir = root / "source"
    anchors_dir = root / "official_anchors"
    write_jsonl(source_dir / "tofu_full.jsonl", rows)
    write_json(source_dir / "authors.json", authors)
    write_json(anchors_dir / "official_alignment.json", alignment)
    write_json(anchors_dir / "reserved_utility_anchors.json", {
        "strategy": "reserve_authors",
        "qa_ids": sorted(reserved_qa_ids),
        "author_ids": reserved_authors,
    })
    report = {
        "release": RELEASE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "snapshot": str(snapshot),
        "source_file_sha256": hashlib.sha256((snapshot / "full.json").read_bytes()).hexdigest(),
        "source_rows": len(rows),
        "authors": len(authors),
        "qas_per_author": sorted(set(map(len, authors.values()))),
        "author_block_structure_verified": True,
        "author_identity_human_verified": False,
        "full_content_digest": sha256_json([
            [row["qa_id"], row["content_sha256"]] for row in rows
        ]),
        "reserved_qa_count": len(reserved_qa_ids),
        "reserved_author_count": len(reserved_authors),
        "status": "source_hash_and_cardinality_passed_author_identity_review_pending",
    }
    write_json(root / "audit" / "source_report.json", report)
    return report
