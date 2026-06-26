#!/usr/bin/env python
"""Run Static-PIPER local-KL intervention experiments.

This script tests whether retain samples selected by set-gradient PI are better
KL-preservation targets than random or semantic targets under the same unlearning
backbone. It is still a lightweight intervention proxy: the official TOFU
evaluation should be run after promising settings are identified.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pi_predictive_validity_probe import (
    IGNORE_INDEX,
    build_optimizer,
    compute_semantic_scores,
    compute_set_gradient_pi_scores,
    cycle_batches,
    iter_batches,
    measure_sample_metrics,
    precompute_ref_sum_losses,
    sample_dataset,
    seed_everything,
    to_device,
    trainable_parameters,
    unlearn_loss,
    write_csv,
)


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
        default="NPO",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--forget-sample-size", type=int, default=128)
    parser.add_argument("--retain-candidate-size", type=int, default=500)
    parser.add_argument("--forget-batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=2)
    parser.add_argument("--local-kl-batch-size", type=int, default=1)
    parser.add_argument("--ref-logit-batch-size", type=int, default=1)
    parser.add_argument("--probe-batches", type=int, default=64)
    parser.add_argument("--topk-frac", type=float, default=0.10)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--probe-learning-rate", type=float, default=1e-5)
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
    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "global_retain_kl",
            "random_local_kl",
            "semantic_local_kl",
            "pi_local_kl",
        ],
        required=True,
    )
    parser.add_argument("--kl-lambda", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=["paged_adamw_32bit", "adamw"], default="paged_adamw_32bit")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--save-model-dir", default="checkpoint")
    return parser.parse_args()


def select_topk(scores: dict[int, float], frac: float) -> list[int]:
    k = max(1, int(round(len(scores) * frac)))
    return [sid for sid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]]


def select_random(sample_ids: list[int], frac: float, seed: int) -> list[int]:
    k = max(1, int(round(len(sample_ids) * frac)))
    rng = random.Random(seed)
    ids = list(sample_ids)
    rng.shuffle(ids)
    return ids[:k]


def overlap_rate(left: set[int], right: set[int]) -> float | None:
    if not left:
        return None
    return len(left & right) / len(left)


@torch.no_grad()
def precompute_reference_log_probs(
    model,
    examples: list[dict],
    tokenizer,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Store answer-token reference log-probs on CPU for later local KL."""

    model.eval()
    device = next(model.parameters()).device
    refs: dict[int, torch.Tensor] = {}
    for batch in iter_batches(examples, batch_size, tokenizer, shuffle=False, seed=0):
        sample_ids = batch["sample_id"].tolist()
        batch = to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits = outputs.logits[..., :-1, :].contiguous()
        labels = batch["labels"][..., 1:].contiguous()
        for row_idx, sid in enumerate(sample_ids):
            mask = labels[row_idx].ne(IGNORE_INDEX)
            if int(mask.sum().item()) == 0:
                refs[int(sid)] = torch.empty(0, logits.shape[-1], dtype=torch.float16)
                continue
            log_probs = F.log_softmax(logits[row_idx][mask].float(), dim=-1)
            refs[int(sid)] = log_probs.to(torch.float16).cpu()
        del outputs, logits, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return refs


def local_kl_loss(model, batch: dict, ref_log_probs: dict[int, torch.Tensor]) -> torch.Tensor:
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = batch["labels"][..., 1:].contiguous()
    kl_sum = logits.sum() * 0.0
    token_count = 0

    for row_idx, sid in enumerate(batch["sample_id"].tolist()):
        mask = labels[row_idx].ne(IGNORE_INDEX)
        cur_logits = logits[row_idx][mask]
        ref = ref_log_probs[int(sid)].to(device=cur_logits.device, dtype=torch.float32)
        if cur_logits.shape[0] != ref.shape[0]:
            n = min(cur_logits.shape[0], ref.shape[0])
            cur_logits = cur_logits[:n]
            ref = ref[:n]
        if cur_logits.numel() == 0:
            continue
        cur_log_probs = F.log_softmax(cur_logits.float(), dim=-1)
        token_kl = F.kl_div(cur_log_probs, ref, log_target=True, reduction="none").sum(dim=-1)
        kl_sum = kl_sum + token_kl.sum()
        token_count += int(token_kl.numel())

    if token_count == 0:
        return kl_sum
    return kl_sum / token_count


def train_npo_with_optional_kl(
    model,
    forget_examples: list[dict],
    retain_examples: list[dict],
    local_examples: list[dict],
    tokenizer,
    args: argparse.Namespace,
    ref_sum_losses: dict[int, float],
    ref_log_probs: dict[int, torch.Tensor],
) -> list[dict]:
    model.train()
    optimizer = build_optimizer(trainable_parameters(model), args)
    device = next(model.parameters()).device
    forget_batches = cycle_batches(forget_examples, args.forget_batch_size, tokenizer, args.seed + 2000)
    retain_batches = (
        cycle_batches(retain_examples, args.retain_batch_size, tokenizer, args.seed + 2500)
        if args.backbone_retain_alpha > 0.0
        else None
    )
    local_batches = (
        cycle_batches(local_examples, args.local_kl_batch_size, tokenizer, args.seed + 3000)
        if local_examples
        else None
    )
    rows = []

    for step in range(args.train_steps):
        forget_batch = to_device(next(forget_batches), device)
        retain_batch = to_device(next(retain_batches), device) if retain_batches is not None else None
        optimizer.zero_grad(set_to_none=True)
        backbone_loss = unlearn_loss(
            model=model,
            batch=forget_batch,
            backbone=args.backbone,
            ref_sum_losses=ref_sum_losses,
            beta=args.npo_beta,
            simnpo_beta=args.simnpo_beta,
            simnpo_delta=args.simnpo_delta,
            simnpo_gamma=args.simnpo_gamma,
            retain_batch=retain_batch,
            retain_alpha=args.backbone_retain_alpha,
        )
        kl = backbone_loss * 0.0
        if args.method != "baseline" and args.kl_lambda > 0.0:
            assert local_batches is not None
            kl_batch = to_device(next(local_batches), device)
            kl = local_kl_loss(model, kl_batch, ref_log_probs)
        total = backbone_loss + args.kl_lambda * kl
        total.backward()
        optimizer.step()

        row = {
            "step": step + 1,
            "backbone_loss": float(backbone_loss.detach().float().cpu()),
            "npo_loss": float(backbone_loss.detach().float().cpu()),
            "local_kl_loss": float(kl.detach().float().cpu()),
            "total_loss": float(total.detach().float().cpu()),
        }
        rows.append(row)
        if (step + 1) % 10 == 0:
            print(
                "[train] "
                f"step={step + 1}/{args.train_steps} "
                f"backbone={row['backbone_loss']:.6f} "
                f"kl={row['local_kl_loss']:.6f} "
                f"total={row['total_loss']:.6f}",
                flush=True,
            )
    return rows


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def summarize_subset(rows: list[dict], selected_ids: set[int]) -> dict:
    subset = [row for row in rows if int(row["sample_id"]) in selected_ids]
    return {
        "count": len(subset),
        "mean_damage": mean_or_none([float(row["loss_delta"]) for row in subset]),
        "mean_base_loss": mean_or_none([float(row["base_loss"]) for row in subset]),
        "mean_final_loss": mean_or_none([float(row["final_loss"]) for row in subset]),
        "mean_answer_prob_delta": mean_or_none(
            [float(row["answer_mean_token_prob_delta"]) for row in subset]
        ),
    }


def build_retain_rows(
    retain_examples: list[dict],
    base_metrics: dict[int, dict[str, float]],
    final_metrics: dict[int, dict[str, float]],
    pi_scores: dict[int, float],
    semantic_scores: dict[int, float],
    pi_ids: set[int],
    semantic_ids: set[int],
    random_ids: set[int],
    method_ids: set[int],
) -> list[dict]:
    rows = []
    for item in retain_examples:
        sid = int(item["sample_id"])
        base = base_metrics[sid]
        final = final_metrics[sid]
        rows.append(
            {
                "sample_id": sid,
                "base_loss": base["loss"],
                "final_loss": final["loss"],
                "loss_delta": final["loss"] - base["loss"],
                "base_answer_mean_token_prob": base["answer_mean_token_prob"],
                "final_answer_mean_token_prob": final["answer_mean_token_prob"],
                "answer_mean_token_prob_delta": final["answer_mean_token_prob"]
                - base["answer_mean_token_prob"],
                "pi_score": pi_scores[sid],
                "semantic_score": semantic_scores[sid],
                "selected_pi": int(sid in pi_ids),
                "selected_semantic": int(sid in semantic_ids),
                "selected_random": int(sid in random_ids),
                "selected_method": int(sid in method_ids),
            }
        )
    return rows


def build_forget_rows(
    forget_examples: list[dict],
    base_metrics: dict[int, dict[str, float]],
    final_metrics: dict[int, dict[str, float]],
) -> list[dict]:
    rows = []
    for item in forget_examples:
        sid = int(item["sample_id"])
        base = base_metrics[sid]
        final = final_metrics[sid]
        rows.append(
            {
                "sample_id": sid,
                "base_loss": base["loss"],
                "final_loss": final["loss"],
                "loss_delta": final["loss"] - base["loss"],
                "base_answer_mean_token_prob": base["answer_mean_token_prob"],
                "final_answer_mean_token_prob": final["answer_mean_token_prob"],
                "answer_mean_token_prob_delta": final["answer_mean_token_prob"]
                - base["answer_mean_token_prob"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.method == "baseline" and args.kl_lambda != 0.0:
        raise SystemExit("baseline must use --kl-lambda 0.0")

    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

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
    retain_by_sid = {int(item["sample_id"]): item for item in retain_examples}
    retain_ids = [int(item["sample_id"]) for item in retain_examples]

    print("[measure] base retain and forget metrics", flush=True)
    base_retain_metrics = measure_sample_metrics(model, retain_examples, tokenizer, args.retain_batch_size)
    base_forget_metrics = measure_sample_metrics(model, forget_examples, tokenizer, args.forget_batch_size)
    base_retain_losses = {sid: vals["loss"] for sid, vals in base_retain_metrics.items()}

    print("[measure] reference forget losses for reference-based backbones", flush=True)
    ref_sum_losses = precompute_ref_sum_losses(model, forget_examples, tokenizer, args.forget_batch_size)

    print("[probe] computing set-gradient PI selector", flush=True)
    pi_started = time.time()
    pi_scores = compute_set_gradient_pi_scores(
        model=model,
        forget_examples=forget_examples,
        retain_examples=retain_examples,
        tokenizer=tokenizer,
        args=args,
        ref_sum_losses=ref_sum_losses,
        base_retain_losses=base_retain_losses,
    )
    pi_seconds = time.time() - pi_started

    print("[probe] computing semantic selector", flush=True)
    semantic_scores = compute_semantic_scores(forget_examples, retain_examples)
    pi_selected = select_topk(pi_scores, args.topk_frac)
    semantic_selected = select_topk(semantic_scores, args.topk_frac)
    random_selected = select_random(retain_ids, args.topk_frac, args.seed + 7000)

    if args.method == "pi_local_kl":
        method_selected = pi_selected
    elif args.method == "semantic_local_kl":
        method_selected = semantic_selected
    elif args.method == "random_local_kl":
        method_selected = random_selected
    elif args.method == "global_retain_kl":
        method_selected = retain_ids
    else:
        method_selected = []

    pi_ids = set(pi_selected)
    semantic_ids = set(semantic_selected)
    random_ids = set(random_selected)
    method_ids = set(method_selected)
    local_examples = [retain_by_sid[sid] for sid in method_selected]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    ref_log_probs: dict[int, torch.Tensor] = {}
    ref_kl_seconds = 0.0
    if args.method != "baseline" and args.kl_lambda > 0.0:
        print(f"[kl] precomputing reference log-probs for {len(local_examples)} retain samples", flush=True)
        ref_started = time.time()
        ref_log_probs = precompute_reference_log_probs(
            model,
            local_examples,
            tokenizer,
            args.ref_logit_batch_size,
        )
        ref_kl_seconds = time.time() - ref_started

    print(f"[train] running {args.backbone} intervention", flush=True)
    train_started = time.time()
    train_rows = train_npo_with_optional_kl(
        model=model,
        forget_examples=forget_examples,
        retain_examples=retain_examples,
        local_examples=local_examples,
        tokenizer=tokenizer,
        args=args,
        ref_sum_losses=ref_sum_losses,
        ref_log_probs=ref_log_probs,
    )
    train_seconds = time.time() - train_started

    print("[measure] final retain and forget metrics", flush=True)
    final_retain_metrics = measure_sample_metrics(model, retain_examples, tokenizer, args.retain_batch_size)
    final_forget_metrics = measure_sample_metrics(model, forget_examples, tokenizer, args.forget_batch_size)

    retain_rows = build_retain_rows(
        retain_examples,
        base_retain_metrics,
        final_retain_metrics,
        pi_scores,
        semantic_scores,
        pi_ids,
        semantic_ids,
        random_ids,
        method_ids,
    )
    forget_rows = build_forget_rows(forget_examples, base_forget_metrics, final_forget_metrics)

    all_retain_ids = set(retain_ids)
    all_forget_ids = {int(item["sample_id"]) for item in forget_examples}
    summary = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "backbone": args.backbone,
        "method": args.method,
        "kl_lambda": args.kl_lambda,
        "seed": args.seed,
        "forget_sample_size": len(forget_examples),
        "retain_candidate_size": len(retain_examples),
        "forget_batch_size": args.forget_batch_size,
        "probe_batches": args.probe_batches,
        "topk_frac": args.topk_frac,
        "topk_count": len(pi_selected),
        "train_steps": args.train_steps,
        "learning_rate": args.learning_rate,
        "probe_learning_rate": args.probe_learning_rate,
        "npo_beta": args.npo_beta,
        "simnpo_beta": args.simnpo_beta,
        "simnpo_delta": args.simnpo_delta,
        "simnpo_gamma": args.simnpo_gamma,
        "backbone_retain_alpha": args.backbone_retain_alpha,
        "optimizer": args.optimizer,
        "gradient_checkpointing": args.gradient_checkpointing,
        "selection": {
            "pi_count": len(pi_ids),
            "semantic_count": len(semantic_ids),
            "random_count": len(random_ids),
            "method_count": len(method_ids),
            "pi_semantic_overlap_rate": overlap_rate(pi_ids, semantic_ids),
            "pi_random_overlap_rate": overlap_rate(pi_ids, random_ids),
            "semantic_random_overlap_rate": overlap_rate(semantic_ids, random_ids),
        },
        "retain_side": summarize_subset(retain_rows, all_retain_ids),
        "local_damage": {
            "pi_topk": summarize_subset(retain_rows, pi_ids),
            "semantic_topk": summarize_subset(retain_rows, semantic_ids),
            "random_topk": summarize_subset(retain_rows, random_ids),
            "method_selection": summarize_subset(retain_rows, method_ids),
        },
        "forget_side": summarize_subset(forget_rows, all_forget_ids),
        "cost": {
            "pi_seconds": pi_seconds,
            "ref_kl_precompute_seconds": ref_kl_seconds,
            "train_seconds": train_seconds,
            "elapsed_seconds": time.time() - started,
            "ref_logprob_samples": len(ref_log_probs),
        },
        "train_loss_first": train_rows[0] if train_rows else None,
        "train_loss_last": train_rows[-1] if train_rows else None,
    }

    write_csv(
        output_dir / "retain_metrics.csv",
        retain_rows,
        [
            "sample_id",
            "base_loss",
            "final_loss",
            "loss_delta",
            "base_answer_mean_token_prob",
            "final_answer_mean_token_prob",
            "answer_mean_token_prob_delta",
            "pi_score",
            "semantic_score",
            "selected_pi",
            "selected_semantic",
            "selected_random",
            "selected_method",
        ],
    )
    write_csv(
        output_dir / "forget_metrics.csv",
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
        output_dir / "local_selection.csv",
        [
            {
                "sample_id": sid,
                "pi_score": pi_scores[sid],
                "semantic_score": semantic_scores[sid],
                "selected_pi": int(sid in pi_ids),
                "selected_semantic": int(sid in semantic_ids),
                "selected_random": int(sid in random_ids),
                "selected_method": int(sid in method_ids),
            }
            for sid in retain_ids
        ],
        [
            "sample_id",
            "pi_score",
            "semantic_score",
            "selected_pi",
            "selected_semantic",
            "selected_random",
            "selected_method",
        ],
    )
    write_csv(
        output_dir / "train_loss_curve.csv",
        train_rows,
        ["step", "backbone_loss", "npo_loss", "local_kl_loss", "total_loss"],
    )
    if args.save_model:
        save_dir = output_dir / args.save_model_dir
        print(f"[save] writing model checkpoint to {save_dir}", flush=True)
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        summary["saved_model_path"] = save_dir.as_posix()
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with (output_dir / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[done] wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
