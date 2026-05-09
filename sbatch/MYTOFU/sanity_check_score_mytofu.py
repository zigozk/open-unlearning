#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Score-based sanity check for MYTOFU full vs retain")
    p.add_argument("--full-model", required=True)
    p.add_argument("--retain-model", required=True)
    p.add_argument("--forget-file", required=True)
    p.add_argument("--retain-file", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-forget", type=int, default=50)
    p.add_argument("--n-retain", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sample_rows(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def get_dtype(name: str):
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"Question: {question}\nAnswer:"


def make_negative_answer(sample: Dict[str, Any], pool: List[Dict[str, Any]], rng: random.Random) -> str:
    """优先从同字段/同qa_type里抽一个错误答案；抽不到就全局抽。"""
    qa_type = sample.get("qa_type")
    answer = sample.get("answer", "")

    candidates = [x for x in pool if x.get("answer") != answer and x.get("qa_type") == qa_type]
    if not candidates:
        candidates = [x for x in pool if x.get("answer") != answer]
    return rng.choice(candidates)["answer"]


def conditional_logprob(model, tokenizer, prompt: str, answer: str) -> Tuple[float, int]:
    """
    计算 log P(answer | prompt)
    返回:
      sum_logprob, answer_token_count
    """
    full_text = prompt + answer
    enc_full = tokenizer(full_text, return_tensors="pt")
    enc_prompt = tokenizer(prompt, return_tensors="pt")

    input_ids = enc_full["input_ids"].to(model.device)
    attention_mask = enc_full["attention_mask"].to(model.device)
    prompt_len = enc_prompt["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        target = input_ids[:, 1:]

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

        # 只取 answer 部分对应的 token 概率
        # target 的第 i 个位置对应 input_ids[:, i+1]
        answer_start_in_target = max(prompt_len - 1, 0)
        answer_log_probs = token_log_probs[:, answer_start_in_target:]

        sum_lp = answer_log_probs.sum().item()
        tok_count = answer_log_probs.numel()

    return sum_lp, tok_count


def score_model(
    model_path: str,
    rows: List[Dict[str, Any]],
    dtype_name: str,
    device_map: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=get_dtype(dtype_name),
        device_map=device_map,
    )
    model.eval()
    return model, tokenizer


def evaluate_rows(model, tokenizer, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        prompt = row["prompt"]
        gold = row["gold_answer"]
        neg = row["negative_answer"]

        gold_lp, gold_tok = conditional_logprob(model, tokenizer, prompt, gold)
        neg_lp, neg_tok = conditional_logprob(model, tokenizer, prompt, neg)

        gold_avg = gold_lp / max(gold_tok, 1)
        neg_avg = neg_lp / max(neg_tok, 1)

        item = dict(row)
        item["gold_logprob_sum"] = gold_lp
        item["gold_logprob_avg"] = gold_avg
        item["neg_logprob_sum"] = neg_lp
        item["neg_logprob_avg"] = neg_avg
        item["prefer_gold_by_avg"] = gold_avg > neg_avg
        item["margin_avg"] = gold_avg - neg_avg
        out.append(item)
    return out


def merge_rows(base_rows, full_rows, retain_rows):
    merged = []
    for b, f, r in zip(base_rows, full_rows, retain_rows):
        merged.append({
            **b,
            "full_gold_logprob_avg": f["gold_logprob_avg"],
            "full_neg_logprob_avg": f["neg_logprob_avg"],
            "full_prefer_gold": f["prefer_gold_by_avg"],
            "full_margin_avg": f["margin_avg"],
            "retain_gold_logprob_avg": r["gold_logprob_avg"],
            "retain_neg_logprob_avg": r["neg_logprob_avg"],
            "retain_prefer_gold": r["prefer_gold_by_avg"],
            "retain_margin_avg": r["margin_avg"],
        })
    return merged


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    for split in sorted(set(r["eval_split"] for r in rows)):
        cur = [r for r in rows if r["eval_split"] == split]
        n = len(cur)
        result[split] = {
            "n": n,
            "full_prefer_gold_rate": round(sum(r["full_prefer_gold"] for r in cur) / max(n, 1), 4),
            "retain_prefer_gold_rate": round(sum(r["retain_prefer_gold"] for r in cur) / max(n, 1), 4),
            "full_margin_avg_mean": round(sum(r["full_margin_avg"] for r in cur) / max(n, 1), 6),
            "retain_margin_avg_mean": round(sum(r["retain_margin_avg"] for r in cur) / max(n, 1), 6),
        }
    return result


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_preview(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== SCORE SANITY CHECK SUMMARY ===\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n\n=== EXAMPLES ===\n")
        for i, row in enumerate(rows[:20], 1):
            f.write(f"\n[{i}] split={row['eval_split']} qa_id={row.get('qa_id', 'NA')}\n")
            f.write(f"question: {row['question']}\n")
            f.write(f"gold: {row['gold_answer']}\n")
            f.write(f"neg: {row['negative_answer']}\n")
            f.write(
                f"full: prefer_gold={row['full_prefer_gold']} "
                f"gold_avg={row['full_gold_logprob_avg']:.6f} "
                f"neg_avg={row['full_neg_logprob_avg']:.6f} "
                f"margin={row['full_margin_avg']:.6f}\n"
            )
            f.write(
                f"retain: prefer_gold={row['retain_prefer_gold']} "
                f"gold_avg={row['retain_gold_logprob_avg']:.6f} "
                f"neg_avg={row['retain_neg_logprob_avg']:.6f} "
                f"margin={row['retain_margin_avg']:.6f}\n"
            )


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    forget_rows = read_jsonl(args.forget_file)
    retain_rows = read_jsonl(args.retain_file)

    forget_sample = sample_rows(forget_rows, args.n_forget, rng)
    retain_sample = sample_rows(retain_rows, args.n_retain, rng)
    all_rows = forget_rows + retain_rows

    base_rows = []
    for row in forget_sample:
        item = dict(row)
        item["eval_split"] = "forget"
        item["gold_answer"] = row["answer"]
        item["negative_answer"] = make_negative_answer(row, all_rows, rng)
        item["prompt"] = None
        base_rows.append(item)

    for row in retain_sample:
        item = dict(row)
        item["eval_split"] = "retain"
        item["gold_answer"] = row["answer"]
        item["negative_answer"] = make_negative_answer(row, all_rows, rng)
        item["prompt"] = None
        base_rows.append(item)

    # full model
    full_model, full_tok = score_model(args.full_model, base_rows, args.dtype, args.device_map)
    for row in base_rows:
        row["prompt"] = build_prompt(full_tok, row["question"])
    full_scored = evaluate_rows(full_model, full_tok, base_rows)

    del full_model
    del full_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # retain model
    retain_model, retain_tok = score_model(args.retain_model, base_rows, args.dtype, args.device_map)
    # 为了严格一致，prompt 直接沿用已有内容，不重新构造
    retain_scored = evaluate_rows(retain_model, retain_tok, base_rows)

    del retain_model
    del retain_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    merged = merge_rows(base_rows, full_scored, retain_scored)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": vars(args),
        "aggregate": aggregate(merged),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jsonl_path = out_dir / f"score_sanity_check_{ts}.jsonl"
    csv_path = out_dir / f"score_sanity_check_{ts}.csv"
    summary_path = out_dir / f"score_sanity_check_{ts}.summary.json"
    preview_path = out_dir / f"score_sanity_check_{ts}.preview.txt"

    write_jsonl(jsonl_path, merged)
    write_csv(csv_path, merged)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_preview(preview_path, merged, summary)

    print("[done] score sanity check finished")
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {preview_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()