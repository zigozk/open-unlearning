#!/usr/bin/env python
"""Probe whether PI scores predict retain-set damage after unlearning.

This is an exploratory experiment, not the final PIPER training objective.
It updates normal model parameters and reports whether the initial probed
interference score is predictive of later retain loss increase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

IGNORE_INDEX = -100


LLAMA2_CHAT_TEMPLATE = {
    "user_start_tag": "[INST] ",
    "user_end_tag": " [/INST]",
    "asst_start_tag": "",
    "asst_end_tag": " ",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Llama-2-7b-chat-hf")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--forget-split", default="forget10")
    parser.add_argument("--retain-split", default="retain90")
    parser.add_argument(
        "--backbone",
        choices=["GradAscent", "GradDiff", "NPO", "SimNPO"],
        required=True,
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--forget-sample-size", type=int, default=128)
    parser.add_argument("--retain-candidate-size", type=int, default=500)
    parser.add_argument("--probe-batches", type=int, default=8)
    parser.add_argument("--pi-mode", choices=["averaged", "set-gradient", "both"], default="averaged")
    parser.add_argument("--primary-pi", choices=["averaged", "set-gradient"], default="averaged")
    parser.add_argument("--forget-batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-3)
    parser.add_argument("--npo-beta", type=float, default=0.1)
    parser.add_argument("--simnpo-beta", type=float, default=4.5)
    parser.add_argument("--simnpo-delta", type=float, default=0.0)
    parser.add_argument("--simnpo-gamma", type=float, default=0.125)
    parser.add_argument(
        "--backbone-retain-alpha",
        type=float,
        default=0.0,
        help="Optional native retain NLL weight for GradDiff/NPO/SimNPO proxy updates.",
    )
    parser.add_argument("--topk-fracs", default="0.05,0.10,0.20")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--optimizer", choices=["paged_adamw_32bit", "adamw"], default="paged_adamw_32bit")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_qa(question: str, answer: str) -> tuple[str, str]:
    prompt = (
        LLAMA2_CHAT_TEMPLATE["user_start_tag"]
        + question
        + LLAMA2_CHAT_TEMPLATE["user_end_tag"]
        + LLAMA2_CHAT_TEMPLATE["asst_start_tag"]
    )
    response = answer + LLAMA2_CHAT_TEMPLATE["asst_end_tag"]
    return prompt, response


def tokenize_qa(tokenizer, question: str, answer: str, max_length: int, sample_id: int) -> dict:
    prompt, response = format_qa(question, answer)
    full_ids = tokenizer(
        prompt + response,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )["input_ids"]
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )["input_ids"]

    if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
        full_ids.append(tokenizer.eos_token_id)

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [IGNORE_INDEX] * prompt_len + full_ids[prompt_len:]
    if len(labels) != len(full_ids):
        labels = labels[: len(full_ids)]

    return {
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "sample_id": sample_id,
        "text": question + " " + answer,
    }


def sample_dataset(split_name: str, sample_size: int, seed: int, tokenizer, max_length: int) -> list[dict]:
    dataset = load_dataset("locuslab/TOFU", split_name, split="train")
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[: min(sample_size, len(indices))]
    examples = []
    for idx in indices:
        row = dataset[idx]
        examples.append(
            tokenize_qa(
                tokenizer=tokenizer,
                question=row["question"],
                answer=row["answer"],
                max_length=max_length,
                sample_id=idx,
            )
        )
    return examples


def collate(examples: list[dict], tokenizer) -> dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[0] for x in examples)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batch = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
        "sample_id": [],
    }
    for item in examples:
        pad_len = max_len - item["input_ids"].shape[0]
        batch["input_ids"].append(F.pad(item["input_ids"], (0, pad_len), value=pad_id))
        batch["attention_mask"].append(F.pad(item["attention_mask"], (0, pad_len), value=0))
        batch["labels"].append(F.pad(item["labels"], (0, pad_len), value=IGNORE_INDEX))
        batch["sample_id"].append(item["sample_id"])
    return {
        "input_ids": torch.stack(batch["input_ids"]),
        "attention_mask": torch.stack(batch["attention_mask"]),
        "labels": torch.stack(batch["labels"]),
        "sample_id": torch.tensor(batch["sample_id"], dtype=torch.long),
    }


def iter_batches(examples: list[dict], batch_size: int, tokenizer, shuffle: bool, seed: int) -> Iterable[dict]:
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        batch_indices = order[start : start + batch_size]
        yield collate([examples[i] for i in batch_indices], tokenizer)


def cycle_batches(examples: list[dict], batch_size: int, tokenizer, seed: int) -> Iterable[dict]:
    epoch = 0
    while True:
        yield from iter_batches(examples, batch_size, tokenizer, shuffle=True, seed=seed + epoch)
        epoch += 1


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items() if k != "sample_id"} | {
        "sample_id": batch["sample_id"]
    }


def per_sample_token_mean_loss(model, batch: dict) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=None,
    )
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    losses = F.cross_entropy(
        logits.transpose(-1, -2),
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    mask = labels.ne(IGNORE_INDEX)
    denom = mask.sum(dim=-1).clamp_min(1)
    return (losses * mask).sum(dim=-1) / denom


def per_sample_sum_nll(model, batch: dict) -> torch.Tensor:
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=None,
    )
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    losses = F.cross_entropy(
        logits.transpose(-1, -2),
        labels,
        ignore_index=IGNORE_INDEX,
        reduction="none",
    )
    mask = labels.ne(IGNORE_INDEX)
    return (losses * mask).sum(dim=-1)


@torch.no_grad()
def measure_retain_losses(model, examples: list[dict], tokenizer, batch_size: int) -> dict[int, float]:
    model.eval()
    losses: dict[int, float] = {}
    device = next(model.parameters()).device
    for batch in iter_batches(examples, batch_size, tokenizer, shuffle=False, seed=0):
        sample_ids = batch.pop("sample_id")
        batch = to_device(batch | {"sample_id": sample_ids}, device)
        vals = per_sample_token_mean_loss(model, batch).detach().float().cpu().tolist()
        for sid, val in zip(sample_ids.tolist(), vals):
            losses[int(sid)] = float(val)
    return losses


@torch.no_grad()
def measure_sample_metrics(model, examples: list[dict], tokenizer, batch_size: int) -> dict[int, dict[str, float]]:
    model.eval()
    metrics: dict[int, dict[str, float]] = {}
    device = next(model.parameters()).device
    for batch in iter_batches(examples, batch_size, tokenizer, shuffle=False, seed=0):
        sample_ids = batch.pop("sample_id")
        batch = to_device(batch | {"sample_id": sample_ids}, device)
        mean_losses = per_sample_token_mean_loss(model, batch).detach().float().cpu().tolist()
        sum_losses = per_sample_sum_nll(model, batch).detach().float().cpu().tolist()
        for sid, mean_loss, sum_loss in zip(sample_ids.tolist(), mean_losses, sum_losses):
            metrics[int(sid)] = {
                "loss": float(mean_loss),
                "sum_nll": float(sum_loss),
                "answer_mean_token_prob": float(math.exp(-mean_loss)),
            }
    return metrics


@torch.no_grad()
def precompute_ref_sum_losses(model, examples: list[dict], tokenizer, batch_size: int) -> dict[int, float]:
    model.eval()
    losses: dict[int, float] = {}
    device = next(model.parameters()).device
    for batch in iter_batches(examples, batch_size, tokenizer, shuffle=False, seed=0):
        sample_ids = batch.pop("sample_id")
        batch = to_device(batch | {"sample_id": sample_ids}, device)
        vals = per_sample_sum_nll(model, batch).detach().float().cpu().tolist()
        for sid, val in zip(sample_ids.tolist(), vals):
            losses[int(sid)] = float(val)
    return losses


def unlearn_loss(
    model,
    batch: dict,
    backbone: str,
    ref_sum_losses: dict[int, float],
    beta: float,
    simnpo_beta: float = 4.5,
    simnpo_delta: float = 0.0,
    simnpo_gamma: float = 0.125,
    retain_batch: dict | None = None,
    retain_alpha: float = 0.0,
) -> torch.Tensor:
    if backbone in {"GradAscent", "GradDiff"}:
        loss = -per_sample_token_mean_loss(model, batch).mean()
    elif backbone == "NPO":
        current_sum = per_sample_sum_nll(model, batch)
        ref_vals = torch.tensor(
            [ref_sum_losses[int(sid)] for sid in batch["sample_id"].tolist()],
            device=current_sum.device,
            dtype=current_sum.dtype,
        )
        logits = beta * (current_sum - ref_vals)
        loss = (-2.0 / beta * F.logsigmoid(logits)).mean()
    elif backbone == "SimNPO":
        current_mean = per_sample_token_mean_loss(model, batch) - simnpo_delta
        loss = -2.0 / simnpo_beta * F.logsigmoid(simnpo_beta * current_mean).mean()
        loss = simnpo_gamma * loss
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    if retain_batch is not None and retain_alpha > 0.0:
        retain_loss = per_sample_token_mean_loss(model, retain_batch).mean()
        loss = loss + retain_alpha * retain_loss
    return loss


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def compute_pi_scores(
    model,
    forget_examples: list[dict],
    retain_examples: list[dict],
    tokenizer,
    args: argparse.Namespace,
    ref_sum_losses: dict[int, float],
    base_retain_losses: dict[int, float],
) -> dict[int, float]:
    model.train()
    params = trainable_parameters(model)
    pi_sum = {item["sample_id"]: 0.0 for item in retain_examples}
    pi_count = 0
    device = next(model.parameters()).device

    forget_batches = list(
        iter_batches(
            forget_examples,
            args.forget_batch_size,
            tokenizer,
            shuffle=True,
            seed=args.seed + 1000,
        )
    )[: args.probe_batches]

    for forget_batch in forget_batches:
        forget_batch = to_device(forget_batch, device)
        model.zero_grad(set_to_none=True)
        loss = unlearn_loss(
            model=model,
            batch=forget_batch,
            backbone=args.backbone,
            ref_sum_losses=ref_sum_losses,
            beta=args.npo_beta,
            simnpo_beta=args.simnpo_beta,
            simnpo_delta=args.simnpo_delta,
            simnpo_gamma=args.simnpo_gamma,
        )
        loss.backward()

        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    param.add_(param.grad, alpha=-args.probe_learning_rate)

        after_losses = measure_retain_losses(model, retain_examples, tokenizer, args.retain_batch_size)
        for sid, after in after_losses.items():
            pi_sum[sid] += after - base_retain_losses[sid]
        pi_count += 1

        with torch.no_grad():
            for param in params:
                if param.grad is not None:
                    param.add_(param.grad, alpha=args.probe_learning_rate)
        model.zero_grad(set_to_none=True)

    return {sid: val / max(pi_count, 1) for sid, val in pi_sum.items()}


def compute_set_gradient_pi_scores(
    model,
    forget_examples: list[dict],
    retain_examples: list[dict],
    tokenizer,
    args: argparse.Namespace,
    ref_sum_losses: dict[int, float],
    base_retain_losses: dict[int, float],
) -> dict[int, float]:
    model.train()
    params = trainable_parameters(model)
    device = next(model.parameters()).device
    forget_batches = list(
        iter_batches(
            forget_examples,
            args.forget_batch_size,
            tokenizer,
            shuffle=True,
            seed=args.seed + 1000,
        )
    )[: args.probe_batches]

    model.zero_grad(set_to_none=True)
    for forget_batch in forget_batches:
        forget_batch = to_device(forget_batch, device)
        loss = unlearn_loss(
            model=model,
            batch=forget_batch,
            backbone=args.backbone,
            ref_sum_losses=ref_sum_losses,
            beta=args.npo_beta,
            simnpo_beta=args.simnpo_beta,
            simnpo_delta=args.simnpo_delta,
            simnpo_gamma=args.simnpo_gamma,
        )
        (loss / max(len(forget_batches), 1)).backward()

    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param.add_(param.grad, alpha=-args.probe_learning_rate)

    after_losses = measure_retain_losses(model, retain_examples, tokenizer, args.retain_batch_size)
    pi_scores = {sid: after - base_retain_losses[sid] for sid, after in after_losses.items()}

    with torch.no_grad():
        for param in params:
            if param.grad is not None:
                param.add_(param.grad, alpha=args.probe_learning_rate)
    model.zero_grad(set_to_none=True)
    return pi_scores


def compute_semantic_scores(forget_examples: list[dict], retain_examples: list[dict]) -> dict[int, float]:
    forget_texts = [item["text"] for item in forget_examples]
    retain_texts = [item["text"] for item in retain_examples]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(forget_texts + retain_texts)
    forget_matrix = matrix[: len(forget_texts)]
    retain_matrix = matrix[len(forget_texts) :]
    sims = cosine_similarity(retain_matrix, forget_matrix)
    max_sims = sims.max(axis=1)
    return {int(item["sample_id"]): float(score) for item, score in zip(retain_examples, max_sims)}


def build_optimizer(params: list[torch.nn.Parameter], args: argparse.Namespace):
    if args.optimizer == "adamw":
        return AdamW(params, lr=args.learning_rate)

    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: bitsandbytes. Use --optimizer adamw or install bitsandbytes."
        ) from exc
    return bnb.optim.PagedAdamW32bit(params, lr=args.learning_rate)


def train_unlearning(
    model,
    forget_examples: list[dict],
    tokenizer,
    args: argparse.Namespace,
    ref_sum_losses: dict[int, float],
) -> list[float]:
    model.train()
    optimizer = build_optimizer(trainable_parameters(model), args)
    losses = []
    device = next(model.parameters()).device
    batches = cycle_batches(forget_examples, args.forget_batch_size, tokenizer, args.seed + 2000)
    for step in range(args.train_steps):
        batch = to_device(next(batches), device)
        optimizer.zero_grad(set_to_none=True)
        loss = unlearn_loss(
            model=model,
            batch=batch,
            backbone=args.backbone,
            ref_sum_losses=ref_sum_losses,
            beta=args.npo_beta,
            simnpo_beta=args.simnpo_beta,
            simnpo_delta=args.simnpo_delta,
            simnpo_gamma=args.simnpo_gamma,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
        if (step + 1) % 10 == 0:
            print(f"[train] step={step + 1}/{args.train_steps} loss={losses[-1]:.6f}", flush=True)
    return losses


def safe_corr(fn, x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None, None
    stat, pval = fn(x, y)
    if math.isnan(float(stat)):
        return None, None
    return float(stat), float(pval)


def topk_selector_summary(
    scores: np.ndarray,
    damage: np.ndarray,
    topk_fracs: list[float],
    random_trials: int,
    seed: int,
) -> dict:
    result = {}
    descending_score = np.argsort(-scores)
    descending_damage = np.argsort(-damage)
    rng = np.random.default_rng(seed)
    for frac in topk_fracs:
        k = max(1, int(round(len(damage) * frac)))
        pred = set(descending_score[:k].tolist())
        oracle = set(descending_damage[:k].tolist())
        precision = len(pred & oracle) / k
        random_means = []
        random_precisions = []
        for _ in range(random_trials):
            random_idx = set(rng.choice(len(damage), size=k, replace=False).tolist())
            random_means.append(float(np.mean(damage[list(random_idx)])))
            random_precisions.append(len(random_idx & oracle) / k)
        pred_topk_mean_damage = float(np.mean(damage[list(pred)]))
        random_mean_damage = float(np.mean(random_means))
        result[str(frac)] = {
            "k": k,
            "precision": float(precision),
            "random_precision": float(k / len(damage)),
            "random_precision_empirical_mean": float(np.mean(random_precisions)),
            "random_precision_empirical_std": float(np.std(random_precisions)),
            "enrichment": float(precision / (k / len(damage))),
            "pred_topk_mean_damage": pred_topk_mean_damage,
            "random_topk_mean_damage": random_mean_damage,
            "random_topk_mean_damage_std": float(np.std(random_means)),
            "pred_vs_random_damage_lift": pred_topk_mean_damage - random_mean_damage,
            "pred_vs_random_damage_ratio": (
                pred_topk_mean_damage / random_mean_damage if random_mean_damage != 0 else None
            ),
            "all_mean_damage": float(np.mean(damage)),
        }
    return result


def summarize(
    rows: list[dict],
    topk_fracs: list[float],
    num_bins: int,
    random_trials: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    pi = np.array([r["pi_score"] for r in rows], dtype=np.float64)
    damage = np.array([r["damage"] for r in rows], dtype=np.float64)
    pearson, pearson_p = safe_corr(pearsonr, pi, damage)
    spearman, spearman_p = safe_corr(spearmanr, pi, damage)

    summary = {
        "num_retain_candidates": len(rows),
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "mean_damage": float(np.mean(damage)),
        "mean_pi": float(np.mean(pi)),
        "topk": {},
    }

    summary["topk"] = topk_selector_summary(pi, damage, topk_fracs, random_trials, seed + 3000)

    order = np.argsort(pi)
    bins = np.array_split(order, num_bins)
    binned = []
    for idx, bin_indices in enumerate(bins):
        if len(bin_indices) == 0:
            continue
        binned.append(
            {
                "bin": idx,
                "count": int(len(bin_indices)),
                "pi_min": float(np.min(pi[bin_indices])),
                "pi_max": float(np.max(pi[bin_indices])),
                "pi_mean": float(np.mean(pi[bin_indices])),
                "damage_mean": float(np.mean(damage[bin_indices])),
                "damage_median": float(np.median(damage[bin_indices])),
            }
        )
    return summary, binned


def summarize_selector(
    rows: list[dict],
    score_key: str,
    topk_fracs: list[float],
    random_trials: int,
    seed: int,
) -> dict:
    scores = np.array([r[score_key] for r in rows], dtype=np.float64)
    damage = np.array([r["damage"] for r in rows], dtype=np.float64)
    pearson, pearson_p = safe_corr(pearsonr, scores, damage)
    spearman, spearman_p = safe_corr(spearmanr, scores, damage)
    return {
        "pearson": pearson,
        "pearson_p": pearson_p,
        "spearman": spearman,
        "spearman_p": spearman_p,
        "topk": topk_selector_summary(scores, damage, topk_fracs, random_trials, seed),
    }


def topk_overlap(rows: list[dict], left_key: str, right_key: str, topk_fracs: list[float]) -> dict:
    left = np.array([r[left_key] for r in rows], dtype=np.float64)
    right = np.array([r[right_key] for r in rows], dtype=np.float64)
    result = {}
    for frac in topk_fracs:
        k = max(1, int(round(len(rows) * frac)))
        left_top = set(np.argsort(-left)[:k].tolist())
        right_top = set(np.argsort(-right)[:k].tolist())
        result[str(frac)] = {
            "k": k,
            "overlap": len(left_top & right_top),
            "overlap_rate": len(left_top & right_top) / k,
        }
    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = args.tokenizer_path or args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] model_path={args.model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] parameters={num_params:,} trainable={num_trainable:,}", flush=True)

    print("[data] loading TOFU samples", flush=True)
    forget_examples = sample_dataset(
        args.forget_split,
        args.forget_sample_size,
        args.seed,
        tokenizer,
        args.max_length,
    )
    retain_examples = sample_dataset(
        args.retain_split,
        args.retain_candidate_size,
        args.seed + 1,
        tokenizer,
        args.max_length,
    )

    started = time.time()
    print("[measure] base retain losses", flush=True)
    base_retain_losses = measure_retain_losses(model, retain_examples, tokenizer, args.retain_batch_size)

    print("[measure] base forget metrics", flush=True)
    base_forget_metrics = measure_sample_metrics(model, forget_examples, tokenizer, args.forget_batch_size)

    print("[measure] reference forget losses for NPO", flush=True)
    ref_sum_losses = precompute_ref_sum_losses(model, forget_examples, tokenizer, args.forget_batch_size)

    averaged_pi_scores = None
    set_pi_scores = None
    if args.pi_mode in {"averaged", "both"}:
        print("[probe] computing averaged PI scores", flush=True)
        averaged_pi_scores = compute_pi_scores(
            model=model,
            forget_examples=forget_examples,
            retain_examples=retain_examples,
            tokenizer=tokenizer,
            args=args,
            ref_sum_losses=ref_sum_losses,
            base_retain_losses=base_retain_losses,
        )
    if args.pi_mode in {"set-gradient", "both"}:
        print("[probe] computing set-gradient PI scores", flush=True)
        set_pi_scores = compute_set_gradient_pi_scores(
            model=model,
            forget_examples=forget_examples,
            retain_examples=retain_examples,
            tokenizer=tokenizer,
            args=args,
            ref_sum_losses=ref_sum_losses,
            base_retain_losses=base_retain_losses,
        )
    if args.primary_pi == "set-gradient" and set_pi_scores is not None:
        pi_scores = set_pi_scores
    elif averaged_pi_scores is not None:
        pi_scores = averaged_pi_scores
    elif set_pi_scores is not None:
        pi_scores = set_pi_scores
    else:
        raise RuntimeError("No PI scores were computed.")

    print("[probe] computing semantic TF-IDF scores", flush=True)
    semantic_scores = compute_semantic_scores(forget_examples, retain_examples)

    print("[train] running full-parameter unlearning", flush=True)
    train_losses = train_unlearning(model, forget_examples, tokenizer, args, ref_sum_losses)

    print("[measure] final retain losses", flush=True)
    final_retain_losses = measure_retain_losses(model, retain_examples, tokenizer, args.retain_batch_size)

    print("[measure] final forget metrics", flush=True)
    final_forget_metrics = measure_sample_metrics(model, forget_examples, tokenizer, args.forget_batch_size)

    rows = []
    for item in retain_examples:
        sid = int(item["sample_id"])
        base_loss = base_retain_losses[sid]
        final_loss = final_retain_losses[sid]
        rows.append(
            {
                "sample_id": sid,
                "base_loss": base_loss,
                "final_loss": final_loss,
                "damage": final_loss - base_loss,
                "pi_score": pi_scores[sid],
                "semantic_score": semantic_scores[sid],
                "averaged_pi_score": averaged_pi_scores[sid] if averaged_pi_scores is not None else "",
                "set_gradient_pi_score": set_pi_scores[sid] if set_pi_scores is not None else "",
            }
        )

    topk_fracs = [float(x) for x in args.topk_fracs.split(",") if x.strip()]
    summary, binned = summarize(rows, topk_fracs, args.num_bins, args.random_trials, args.seed)
    summary.update(
        {
            "model_name": args.model_name,
            "model_path": args.model_path,
            "tokenizer_path": tokenizer_path,
            "forget_split": args.forget_split,
            "retain_split": args.retain_split,
            "backbone": args.backbone,
            "seed": args.seed,
            "forget_sample_size": len(forget_examples),
            "retain_candidate_size": len(retain_examples),
            "probe_batches": args.probe_batches,
            "pi_mode": args.pi_mode,
            "primary_pi": args.primary_pi,
            "forget_batch_size": args.forget_batch_size,
            "train_steps": args.train_steps,
            "learning_rate": args.learning_rate,
            "probe_learning_rate": args.probe_learning_rate,
            "random_trials": args.random_trials,
            "optimizer": args.optimizer,
            "gradient_checkpointing": args.gradient_checkpointing,
            "elapsed_seconds": time.time() - started,
            "train_loss_first": train_losses[0] if train_losses else None,
            "train_loss_last": train_losses[-1] if train_losses else None,
        }
    )
    summary["semantic_baseline"] = summarize_selector(
        rows, "semantic_score", topk_fracs, args.random_trials, args.seed + 4000
    )
    if averaged_pi_scores is not None:
        summary["averaged_pi"] = summarize_selector(
            rows, "averaged_pi_score", topk_fracs, args.random_trials, args.seed + 5000
        )
    if set_pi_scores is not None:
        summary["set_gradient_pi"] = summarize_selector(
            rows, "set_gradient_pi_score", topk_fracs, args.random_trials, args.seed + 6000
        )
    if averaged_pi_scores is not None and set_pi_scores is not None:
        summary["averaged_vs_set_gradient_topk_overlap"] = topk_overlap(
            rows, "averaged_pi_score", "set_gradient_pi_score", topk_fracs
        )
        avg_vals = np.array([r["averaged_pi_score"] for r in rows], dtype=np.float64)
        set_vals = np.array([r["set_gradient_pi_score"] for r in rows], dtype=np.float64)
        pearson, pearson_p = safe_corr(pearsonr, avg_vals, set_vals)
        spearman, spearman_p = safe_corr(spearmanr, avg_vals, set_vals)
        summary["averaged_vs_set_gradient_correlation"] = {
            "pearson": pearson,
            "pearson_p": pearson_p,
            "spearman": spearman,
            "spearman_p": spearman_p,
        }

    forget_rows = []
    for item in forget_examples:
        sid = int(item["sample_id"])
        base_metrics = base_forget_metrics[sid]
        final_metrics = final_forget_metrics[sid]
        forget_rows.append(
            {
                "sample_id": sid,
                "base_loss": base_metrics["loss"],
                "final_loss": final_metrics["loss"],
                "loss_delta": final_metrics["loss"] - base_metrics["loss"],
                "base_answer_mean_token_prob": base_metrics["answer_mean_token_prob"],
                "final_answer_mean_token_prob": final_metrics["answer_mean_token_prob"],
                "answer_mean_token_prob_delta": final_metrics["answer_mean_token_prob"]
                - base_metrics["answer_mean_token_prob"],
            }
        )
    summary["forget_side"] = {
        "base_loss_mean": float(np.mean([r["base_loss"] for r in forget_rows])),
        "final_loss_mean": float(np.mean([r["final_loss"] for r in forget_rows])),
        "loss_delta_mean": float(np.mean([r["loss_delta"] for r in forget_rows])),
        "base_answer_mean_token_prob_mean": float(
            np.mean([r["base_answer_mean_token_prob"] for r in forget_rows])
        ),
        "final_answer_mean_token_prob_mean": float(
            np.mean([r["final_answer_mean_token_prob"] for r in forget_rows])
        ),
        "answer_mean_token_prob_delta_mean": float(
            np.mean([r["answer_mean_token_prob_delta"] for r in forget_rows])
        ),
    }

    write_csv(
        output_dir / "per_sample_pi_damage.csv",
        rows,
        [
            "sample_id",
            "base_loss",
            "final_loss",
            "damage",
            "pi_score",
            "semantic_score",
            "averaged_pi_score",
            "set_gradient_pi_score",
        ],
    )
    write_csv(
        output_dir / "forget_side_metrics.csv",
        forget_rows,
        [
            "sample_id",
            "base_loss",
            "final_loss",
            "loss_delta",
            "base_answer_mean_token_prob",
            "final_answer_mean_token_prob",
            "answer_mean_token_prob_delta",
        ],
    )
    write_csv(
        output_dir / "train_loss_curve.csv",
        [{"step": i + 1, "loss": loss} for i, loss in enumerate(train_losses)],
        ["step", "loss"],
    )
    write_csv(
        output_dir / "binned_damage_by_pi.csv",
        binned,
        ["bin", "count", "pi_min", "pi_max", "pi_mean", "damage_mean", "damage_median"],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
