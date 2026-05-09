#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Side-by-side sanity check for MYTOFU full vs retain models")
    p.add_argument("--full-model", required=True, help="Path to full model checkpoint")
    p.add_argument("--retain-model", required=True, help="Path to retain model checkpoint")
    p.add_argument("--forget-file", required=True, help="JSONL file for forget QA")
    p.add_argument("--retain-file", required=True, help="JSONL file for retain QA")
    p.add_argument("--output-dir", default="sanity_check_outputs", help="Directory to save outputs")
    p.add_argument("--n-forget", type=int, default=20, help="Number of forget examples to sample")
    p.add_argument("--n-retain", type=int, default=20, help="Number of retain examples to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0, help="0.0 means greedy decoding")
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--qa-type", default=None, help="Optional filter, e.g. single_fact")
    p.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", default="auto")
    return p.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def maybe_filter(rows: List[Dict[str, Any]], qa_type: str | None) -> List[Dict[str, Any]]:
    if not qa_type:
        return rows
    return [r for r in rows if r.get("qa_type") == qa_type]


def sample_rows(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {question}\nAnswer:"


def get_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    return torch.float32


def generate_predictions(
    model_path: str,
    samples: List[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    dtype_name: str,
    device_map: str,
) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = get_torch_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()

    preds: List[str] = []
    prompts = [build_prompt(tokenizer, s["question"]) for s in samples]

    with torch.no_grad():
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if device_map == "auto":
                if hasattr(model, "device") and str(model.device) != "meta":
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
            else:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            outputs = model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[:, input_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            preds.extend([x.strip() for x in decoded])

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return preds


def enrich_rows(samples: List[Dict[str, Any]], preds: List[str], model_tag: str) -> List[Dict[str, Any]]:
    enriched = []
    for sample, pred in zip(samples, preds):
        gold = sample.get("answer", "")
        pred_norm = normalize_text(pred)
        gold_norm = normalize_text(gold)
        enriched.append(
            {
                **sample,
                f"{model_tag}_prediction": pred,
                f"{model_tag}_exact_match": pred_norm == gold_norm,
                f"{model_tag}_contains_answer": bool(gold_norm) and gold_norm in pred_norm,
            }
        )
    return enriched


def merge_side_by_side(base_rows: List[Dict[str, Any]], full_rows: List[Dict[str, Any]], retain_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for base, full_r, retain_r in zip(base_rows, full_rows, retain_rows):
        merged.append(
            {
                **base,
                "full_prediction": full_r["full_prediction"],
                "full_exact_match": full_r["full_exact_match"],
                "full_contains_answer": full_r["full_contains_answer"],
                "retain_prediction": retain_r["retain_prediction"],
                "retain_exact_match": retain_r["retain_exact_match"],
                "retain_contains_answer": retain_r["retain_contains_answer"],
            }
        )
    return merged


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for split_name in sorted({r["eval_split"] for r in rows}):
        split_rows = [r for r in rows if r["eval_split"] == split_name]
        out[split_name] = {
            "n": len(split_rows),
            "full_exact_match_rate": round(sum(r["full_exact_match"] for r in split_rows) / max(len(split_rows), 1), 4),
            "full_contains_answer_rate": round(sum(r["full_contains_answer"] for r in split_rows) / max(len(split_rows), 1), 4),
            "retain_exact_match_rate": round(sum(r["retain_exact_match"] for r in split_rows) / max(len(split_rows), 1), 4),
            "retain_contains_answer_rate": round(sum(r["retain_contains_answer"] for r in split_rows) / max(len(split_rows), 1), 4),
        }
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_preview(path: Path, rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== SANITY CHECK SUMMARY ===\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n\n=== EXAMPLE PREDICTIONS ===\n")
        for i, row in enumerate(rows, start=1):
            f.write(f"\n[{i}] split={row.get('eval_split')} qa_id={row.get('qa_id', 'NA')}\n")
            f.write(f"question: {row.get('question', '')}\n")
            f.write(f"gold: {row.get('answer', '')}\n")
            f.write(f"full: {row.get('full_prediction', '')}\n")
            f.write(f"retain: {row.get('retain_prediction', '')}\n")
            f.write(
                "flags: "
                f"full_em={row.get('full_exact_match')} full_contains={row.get('full_contains_answer')} "
                f"retain_em={row.get('retain_exact_match')} retain_contains={row.get('retain_contains_answer')}\n"
            )


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    forget_rows = maybe_filter(read_jsonl(args.forget_file), args.qa_type)
    retain_rows = maybe_filter(read_jsonl(args.retain_file), args.qa_type)

    forget_rows = sample_rows(forget_rows, args.n_forget, rng)
    retain_rows = sample_rows(retain_rows, args.n_retain, rng)

    sampled: List[Dict[str, Any]] = []
    for row in forget_rows:
        item = dict(row)
        item["eval_split"] = "forget"
        sampled.append(item)
    for row in retain_rows:
        item = dict(row)
        item["eval_split"] = "retain"
        sampled.append(item)

    sampled.sort(key=lambda x: (x["eval_split"], x.get("qa_id", x.get("question", ""))))

    full_preds = generate_predictions(
        model_path=args.full_model,
        samples=sampled,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype_name=args.dtype,
        device_map=args.device_map,
    )
    full_rows = enrich_rows(sampled, full_preds, "full")

    retain_preds = generate_predictions(
        model_path=args.retain_model,
        samples=sampled,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        dtype_name=args.dtype,
        device_map=args.device_map,
    )
    retain_rows_enriched = enrich_rows(sampled, retain_preds, "retain")

    merged = merge_side_by_side(sampled, full_rows, retain_rows_enriched)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {
            "full_model": args.full_model,
            "retain_model": args.retain_model,
            "forget_file": args.forget_file,
            "retain_file": args.retain_file,
            "qa_type": args.qa_type,
            "n_forget": len(forget_rows),
            "n_retain": len(retain_rows),
            "seed": args.seed,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "dtype": args.dtype,
            "device_map": args.device_map,
        },
        "aggregate": aggregate(merged),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    jsonl_path = output_dir / f"sanity_check_{timestamp}.jsonl"
    csv_path = output_dir / f"sanity_check_{timestamp}.csv"
    summary_path = output_dir / f"sanity_check_{timestamp}.summary.json"
    preview_path = output_dir / f"sanity_check_{timestamp}.preview.txt"

    write_jsonl(jsonl_path, merged)
    write_csv(csv_path, merged)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_preview(preview_path, merged[: min(20, len(merged))], summary)

    print("[done] sanity check finished")
    print(f"[saved] {jsonl_path}")
    print(f"[saved] {csv_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {preview_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()