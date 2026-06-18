#!/usr/bin/env python3
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args():
    p = argparse.ArgumentParser(description="Build MYTOFU eval assets v1")
    p.add_argument("--forget-file", required=True)
    p.add_argument("--retain-file", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-perturb", type=int, default=4)
    p.add_argument(
        "--single-fact-only",
        action="store_true",
        help="Keep only rows where qa_type == single_fact",
    )
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def first_primary_slot(row: Dict[str, Any]) -> Optional[str]:
    slots = row.get("primary_slots")
    if isinstance(slots, list) and len(slots) > 0:
        return slots[0]
    if isinstance(slots, str):
        return slots
    return None


def normalize_row(row: Dict[str, Any], split_name: str) -> Dict[str, Any]:
    return {
        "qa_id": row.get("qa_id"),
        "dataset_id": row.get("dataset_id"),
        "author_id": row.get("author_id"),
        "split": split_name,
        "question": row["question"],
        "answer": row["answer"],
        "qa_type": row.get("qa_type"),
        "primary_slots": row.get("primary_slots", []),
        "primary_slot": first_primary_slot(row),
        "support_fact_ids": row.get("support_fact_ids", []),
    }


def filter_rows(rows: List[Dict[str, Any]], single_fact_only: bool) -> List[Dict[str, Any]]:
    if not single_fact_only:
        return rows
    return [r for r in rows if r.get("qa_type") == "single_fact"]


def build_eval_rows(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        nr = normalize_row(r, split_name)
        out.append(
            {
                "qa_id": nr["qa_id"],
                "dataset_id": nr["dataset_id"],
                "author_id": nr["author_id"],
                "split": nr["split"],
                "question": nr["question"],
                "answer": nr["answer"],
                "qa_type": nr["qa_type"],
                "primary_slots": nr["primary_slots"],
                "support_fact_ids": nr["support_fact_ids"],
            }
        )
    return out


def pick_negatives(
    row: Dict[str, Any],
    all_rows: List[Dict[str, Any]],
    slot_buckets: Dict[str, List[Dict[str, Any]]],
    rng: random.Random,
    k: int,
) -> List[str]:
    gold = row["answer"]
    author_id = row.get("author_id")
    slot = row.get("primary_slot")
    qa_type = row.get("qa_type")

    def unique_answers(cands: List[Dict[str, Any]]) -> List[str]:
        seen = set()
        out = []
        for x in cands:
            ans = x["answer"]
            if ans == gold:
                continue
            if ans in seen:
                continue
            seen.add(ans)
            out.append(ans)
        return out

    hard_candidates: List[Dict[str, Any]] = []
    if slot and slot in slot_buckets:
        hard_candidates = [
            x for x in slot_buckets[slot]
            if x.get("author_id") != author_id and x["answer"] != gold
        ]

    mid_candidates = [
        x for x in all_rows
        if x.get("qa_type") == qa_type
        and x.get("author_id") != author_id
        and x["answer"] != gold
    ]

    global_candidates = [
        x for x in all_rows
        if x.get("author_id") != author_id and x["answer"] != gold
    ]

    pool = []
    for seq in [hard_candidates, mid_candidates, global_candidates]:
        for ans in unique_answers(seq):
            if ans not in pool:
                pool.append(ans)
            if len(pool) >= k:
                break
        if len(pool) >= k:
            break

    if len(pool) < k:
        raise ValueError(
            f"Not enough negative answers for qa_id={row.get('qa_id')} slot={slot}"
        )

    # 从前面已经按难度排好的池子里再随机抽样，兼顾稳定和多样性
    if len(pool) == k:
        return pool
    return rng.sample(pool, k)


def build_perturbed_rows(
    rows: List[Dict[str, Any]],
    split_name: str,
    all_rows: List[Dict[str, Any]],
    rng: random.Random,
    num_perturb: int,
) -> List[Dict[str, Any]]:
    normalized_rows = [normalize_row(r, split_name) for r in rows]

    slot_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    normalized_all = []
    for split, source_rows in [("forget", rows if split_name == "forget" else []), ("retain", rows if split_name == "retain" else [])]:
        pass  # 占位，避免误读

    for r in all_rows:
        # all_rows 这里要求已经是 normalize_row 之后的形式
        slot = r.get("primary_slot")
        if slot:
            slot_buckets[slot].append(r)
        normalized_all.append(r)

    out = []
    for r in normalized_rows:
        negs = pick_negatives(
            row=r,
            all_rows=normalized_all,
            slot_buckets=slot_buckets,
            rng=rng,
            k=num_perturb,
        )
        out.append(
            {
                "qa_id": r["qa_id"],
                "dataset_id": r["dataset_id"],
                "author_id": r["author_id"],
                "split": r["split"],
                "question": r["question"],
                "answer": r["answer"],
                "paraphrased_answer": "",     # 下一步用 API 回填
                "perturbed_answer": negs[0],  # 为兼容单个负样本字段
                "perturbed_answers": negs,    # 主字段：多个扰动答案
                "qa_type": r["qa_type"],
                "primary_slots": r["primary_slots"],
                "support_fact_ids": r["support_fact_ids"],
            }
        )
    return out


def build_paraphrase_seed(rows: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        nr = normalize_row(r, split_name)
        out.append(
            {
                "qa_id": nr["qa_id"],
                "author_id": nr["author_id"],
                "split": nr["split"],
                "question": nr["question"],
                "answer": nr["answer"],
                "qa_type": nr["qa_type"],
                "primary_slots": nr["primary_slots"],
                "rewrite_instruction": (
                    "Rewrite the answer into one semantically equivalent English answer. "
                    "Keep every fact unchanged, keep named entities exact, do not add or remove facts, "
                    "and output only the rewritten answer."
                ),
            }
        )
    return out


def stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    cnt = Counter()
    for r in rows:
        cnt[r.get("primary_slot") or "UNKNOWN"] += 1
    return dict(sorted(cnt.items(), key=lambda x: x[0]))


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    forget_rows_raw = read_jsonl(args.forget_file)
    retain_rows_raw = read_jsonl(args.retain_file)

    forget_rows = filter_rows(forget_rows_raw, args.single_fact_only)
    retain_rows = filter_rows(retain_rows_raw, args.single_fact_only)

    # 先构造 normalize 后的全局池，便于负样本抽取
    all_normalized = (
        [normalize_row(r, "forget") for r in forget_rows] +
        [normalize_row(r, "retain") for r in retain_rows]
    )

    forget_eval = build_eval_rows(forget_rows, "forget")
    retain_eval = build_eval_rows(retain_rows, "retain")

    forget_perturbed = build_perturbed_rows(
        rows=forget_rows,
        split_name="forget",
        all_rows=all_normalized,
        rng=rng,
        num_perturb=args.num_perturb,
    )
    retain_perturbed = build_perturbed_rows(
        rows=retain_rows,
        split_name="retain",
        all_rows=all_normalized,
        rng=rng,
        num_perturb=args.num_perturb,
    )

    forget_paraphrase_seed = build_paraphrase_seed(forget_rows, "forget")
    retain_paraphrase_seed = build_paraphrase_seed(retain_rows, "retain")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(out_dir / "forget_eval.jsonl", forget_eval)
    write_jsonl(out_dir / "retain_eval.jsonl", retain_eval)
    write_jsonl(out_dir / "forget_eval_perturbed.jsonl", forget_perturbed)
    write_jsonl(out_dir / "retain_eval_perturbed.jsonl", retain_perturbed)
    write_jsonl(out_dir / "forget_paraphrase_seed.jsonl", forget_paraphrase_seed)
    write_jsonl(out_dir / "retain_paraphrase_seed.jsonl", retain_paraphrase_seed)

    meta = {
        "seed": args.seed,
        "single_fact_only": args.single_fact_only,
        "num_perturb": args.num_perturb,
        "input": {
            "forget_file": args.forget_file,
            "retain_file": args.retain_file,
        },
        "output_dir": str(out_dir),
        "counts": {
            "forget_raw": len(forget_rows_raw),
            "retain_raw": len(retain_rows_raw),
            "forget_eval": len(forget_eval),
            "retain_eval": len(retain_eval),
            "forget_perturbed": len(forget_perturbed),
            "retain_perturbed": len(retain_perturbed),
        },
        "slot_distribution": {
            "forget": stats([normalize_row(r, "forget") for r in forget_rows]),
            "retain": stats([normalize_row(r, "retain") for r in retain_rows]),
        },
        "notes": [
            "No index field is written.",
            "Only single_fact rows are kept when --single-fact-only is set.",
            "paraphrased_answer is initialized as empty string and should be filled in a later API rewrite pass.",
            "perturbed_answer is the first negative answer; perturbed_answers keeps all negatives.",
        ],
    }

    with open(out_dir / "build_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[done] MYTOFU eval assets built")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()