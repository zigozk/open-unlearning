# BRIDGE Experiment Handoff Memory

Last updated: 2026-07-02

This file is a handoff note for the next GPT/Codex session. Read this first when continuing the BRIDGE experiments in `D:\User\zigo\Github\open-unlearning`.

## User And Repo Context

- User: zigo. The local AGENTS instruction says every reply should call the user `zigo`.
- Branch: `mytofu-work`.
- Main project path locally: `D:\User\zigo\Github\open-unlearning`.
- HPC repo path: `/home/zkzhang/unlearning/open-unlearning`.
- The user wants practical HPC-first workflows: one `sbatch` entrypoint when possible, compact reports committed to GitHub, and no raw heavy result trees unless explicitly requested.
- `results/` is mostly ignored, but `results/bridge_reports/` is intentionally tracked.

## Main BRIDGE Files

- Paper/experiment plans:
  - `docs/bridge/BRIDGE_paper_plan.md`
  - `docs/bridge/BRIDGE_initial_experiment_plan.md`
- Trainer implementation:
  - `src/trainer/unlearn/bridge.py`
  - `src/trainer/unlearn/retention_prioritized.py`
- Configs:
  - `configs/trainer/BRIDGE_NPO.yaml`
  - `configs/experiment/unlearn/tofu/bridge.yaml`
- Slurm entrypoints:
  - `sbatch/bridge/bridge_initial_pipeline.sh`
  - `sbatch/bridge/bridge_summarize_initial.sh`
  - `sbatch/bridge/bridge_summarize_debug.sh`
  - `sbatch/bridge/bridge_collect_failures.sh`
- Summary scripts:
  - `experiments/bridge/summarize_bridge_initial.py`
  - `experiments/bridge/summarize_bridge_debug.py`
  - `experiments/bridge/summarize_bridge_failures.py`

## Important Commits In This Thread

- `5467d42` Add BRIDGE initial experiment pipeline
- `3bb3fc5` Target H100 for BRIDGE initial runs
- `86d63a4` Add BRIDGE partial summary sbatch
- `4cea38d` Bypass gradient synthesis for default unlearn training
- `0a09b71` Track BRIDGE report artifacts
- `d323624` Add BRIDGE debug summarizer
- `cdd7423` Support older Trainer training_step signature
- `b6890ad` Add BRIDGE-1b-v2-report

## Critical Bugs Found And Fixed

### 1. BRIDGE/NPO Training Was Going Through The Wrong Path

`RetentionPrioritizedMixin.training_step()` used to intercept training even when `gradient_synthesis=none`.

That caused two problems:

- It bypassed `BRIDGENPO.compute_loss()`, so BRIDGE global KL, boundary DRO, GS/PI/history priors, and diagnostics were not actually used.
- It materialized full gradient dictionaries, causing severe memory use for 7B.

Fix:

- Commit `4cea38d` added a bypass so `gradient_synthesis=none` goes through normal `Trainer.training_step()`.

Consequence:

- The first `seed0_initial` 1B BRIDGE results are invalid. Five methods had identical `TOFU_EVAL.json` hash and identical metrics because they effectively trained the same forget-only objective.

### 2. Server Transformers Version Had Older `training_step` Signature

After `4cea38d`, the first rerun `RUN_TAG=postfix_1b` failed immediately with:

```text
TypeError: Trainer.training_step() takes 3 positional arguments but 4 were given
```

Fix:

- Commit `cdd7423` detects whether the parent `training_step` accepts `num_items_in_batch`.
- Use this commit or later for all new runs.

## Invalid / Failed Result Batches

Do not use these for scientific conclusions:

- `*_seed0_initial*`
  - Pre-fix run.
  - 1B BRIDGE/NPO_global methods had identical eval hashes and `model_utility=0`.
  - Kept only as debugging evidence.
- `*_seed0_postfix_1b*`
  - Failed due older Transformers `training_step` signature.
  - Error logs show TypeError at `retention_prioritized.py`.

Safe cleanup on HPC if disk is crowded:

```bash
cd ~/unlearning/open-unlearning

find results/bridge_initial_eval -maxdepth 1 -type d \
  \( -name "*seed0_initial_tofu_eval" -o -name "*seed0_postfix_1b_tofu_eval" \) -print
find results/bridge_initial -maxdepth 1 -type d \
  \( -name "*seed0_initial" -o -name "*seed0_postfix_1b" \) -print
```

After previewing, remove:

```bash
rm -rf results/bridge_initial_eval/*seed0_initial_tofu_eval
rm -rf results/bridge_initial_eval/*seed0_postfix_1b_tofu_eval
rm -rf results/bridge_initial/*seed0_initial
rm -rf results/bridge_initial/*seed0_postfix_1b
```

Keep:

- `results/bridge_initial/*seed0_postfix_1b_v2`
- `results/bridge_initial_eval/*seed0_postfix_1b_v2_tofu_eval`
- `results/bridge_reports`

## Latest Valid 1B Result: `postfix_1b_v2`

Latest tracked report:

- `results/bridge_reports/bridge_initial_report.md`
- `results/bridge_reports/bridge_initial_summary.csv`
- Commit: `b6890ad Add BRIDGE-1b-v2-report`

This run used:

- Model: `Llama-3.2-1B-Instruct`
- Split: `forget10 / retain90`
- Seed: `0`
- Tag: `postfix_1b_v2`
- GPU: H100 80GB HBM3 on `gpu01`

Summary:

| Method | Status | Model Utility | forget_truth_ratio | forget_Q_A_Prob | forget_Q_A_ROUGE | boundary_dro |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `npo` | eval_complete | 0.410651 | 0.624495 | 0.222495 | 0.164106 | |
| `npo_global_kl` | eval_complete | 0.282725 | 0.678516 | 0.110230 | 0.152243 | 0 |
| `bridge_uniform_dro` | eval_complete | 0.359573 | 0.662160 | 0.161361 | 0.169627 | 1.03956 |
| `bridge_history_dro` | eval_complete | 0.363514 | 0.661093 | 0.168158 | 0.170591 | 1.02126 |
| `bridge_refresh_gs_dro` | eval_complete | 0.352466 | 0.663293 | 0.163929 | 0.168797 | 1.04939 |
| `bridge_refresh_pi_dro` | eval_missing | | | | | 1.07054 |

Interpretation:

- The fix worked: BRIDGE metrics are non-empty and methods are no longer identical.
- Pure `npo` has highest utility but weaker forgetting.
- `npo_global_kl` forgets aggressively but utility drops too much.
- BRIDGE DRO methods recover utility compared with global KL while still reducing forget probability relative to NPO.
- Current best-looking BRIDGE variant is `bridge_history_dro`, but conclusions are provisional because official `forget_quality` is missing and only seed 0 is available.
- `bridge_refresh_pi_dro` trained and wrote BRIDGE diagnostics, but eval is missing. It needs a targeted eval rerun or log inspection on HPC.

## Forget Quality Is Empty

`forget_quality` is configured in `configs/eval/tofu.yaml`, but it is empty because `retain_logs_path` was not provided.

Code path:

- `configs/eval/tofu_metrics/forget_quality.yaml` expects `${eval.tofu.retain_logs_path}`.
- `src/evals/metrics/privacy.py` sets FQ to `None` when retain reference logs are unavailable.

Important distinction:

- `retain_logs_path` is not a model directory.
- It must point to a retain reference `TOFU_EVAL.json`, generated by evaluating the retain90 model.

For `forget10`, use a `retain90` reference model.

## Retain Model Download / Reference Log Generation

OpenUnlearning provides retain models on Hugging Face:

- `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90`
- `open-unlearning/tofu_Llama-2-7b-chat-hf_retain90`

Use domestic mirror if needed:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export HF_HUB_CACHE=${HF_HOME}/hub
```

New `huggingface_hub` deprecates `huggingface-cli`; use `hf`:

```bash
hf download open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90 \
  --local-dir /home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_retain90
```

Check:

```bash
ls -lh /home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_retain90
du -sh /home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_retain90
```

Then generate `TOFU_EVAL.json`:

```bash
cd ~/unlearning/open-unlearning
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=/home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_retain90 \
  model.tokenizer_args.pretrained_model_name_or_path=/home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_retain90 \
  forget_split=forget10 \
  holdout_split=holdout10 \
  task_name=tofu_Llama_3_2_1B_Instruct_retain90_reference \
  paths.output_dir=results/bridge_retain_logs/tofu_Llama_3_2_1B_Instruct_retain90_reference \
  eval.tofu.output_dir=results/bridge_retain_logs/tofu_Llama_3_2_1B_Instruct_retain90_reference \
  eval.tofu.overwrite=true \
  eval.tofu.batch_size=32
```

Reference path for future runs:

```bash
RETAIN_LOGS_PATH=results/bridge_retain_logs/tofu_Llama_3_2_1B_Instruct_retain90_reference/TOFU_EVAL.json
```

## Useful HPC Commands

Rerun the current 1B matrix with the fixed code:

```bash
cd ~/unlearning/open-unlearning
git pull origin mytofu-work

sbatch --export=ALL,MODEL_CONFIG=Llama-3.2-1B-Instruct,RUN_TAG=postfix_1b_v2,SKIP_EXISTING=0,NODELIST=gpu01,GRES=gpu:nvidia_h100_80gb_hbm3:1,CPUS_PER_TASK=8,MEM=128G,TIME_LIMIT=48:00:00,REPORT_PREFIX=bridge_initial_1b_postfix_v2 sbatch/bridge/bridge_initial_pipeline.sh
```

If rerunning with official FQ after retain logs are ready, use a new tag:

```bash
sbatch --export=ALL,MODEL_CONFIG=Llama-3.2-1B-Instruct,RUN_TAG=postfix_1b_v3_fq,SKIP_EXISTING=0,RETAIN_LOGS_PATH=results/bridge_retain_logs/tofu_Llama_3_2_1B_Instruct_retain90_reference/TOFU_EVAL.json,NODELIST=gpu01,GRES=gpu:nvidia_h100_80gb_hbm3:1,CPUS_PER_TASK=8,MEM=128G,TIME_LIMIT=48:00:00,REPORT_PREFIX=bridge_initial_1b_fq sbatch/bridge/bridge_initial_pipeline.sh
```

Generate normal summary:

```bash
sbatch --export=ALL,REPORT_PREFIX=bridge_initial_1b_latest sbatch/bridge/bridge_summarize_initial.sh
```

Generate debug summary:

```bash
sbatch --export=ALL,REPORT_PREFIX=bridge_initial_1b_latest_debug sbatch/bridge/bridge_summarize_debug.sh
```

Commit reports:

```bash
git add results/bridge_reports
git commit -m "Add BRIDGE latest reports"
git push origin mytofu-work
```

## GPU / Runtime Notes

- 1B can run on H100 80GB or A100 80GB. H100 is faster but more expensive in cluster cost.
- H100 cost observed by user: `40`; A100 80GB PCIe cost observed: `16`.
- For the quick fixed-code validation, H100 is preferred.
- The script defaults to H100 `gpu01` and `gpu:nvidia_h100_80gb_hbm3:1`.
- `MAX_PARALLEL=1` is safer for debugging. The script supports higher parallelism, but use it only when the cluster has enough GPUs and the run is stable.

## 7B Status

- 7B is not a complete clean matrix yet.
- Old 7B A100 80GB run hit CUDA OOM before the training-step bypass fix because gradient dictionaries duplicated full 7B gradients.
- After `4cea38d` and `cdd7423`, 7B should be rerun if needed.
- For A100 80GB, start conservatively if OOM persists:

```bash
sbatch --export=ALL,MODEL_CONFIG=Llama-2-7b-chat-hf,NODELIST=gpu18,GRES=gpu:nvidia_a100_80gb_pcie:1,CPUS_PER_TASK=2,MEM=128G,TIME_LIMIT=48:00:00,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True sbatch/bridge/bridge_initial_pipeline.sh
```

If OOM persists, reduce batch while noting that BRIDGE candidate batch changes:

```bash
TRAIN_BATCH_SIZE=4,GRAD_ACCUM_STEPS=8,EVAL_BATCH_SIZE=8
```

## Next Best Actions

1. Confirm whether `bridge_refresh_pi_dro` eval failed or is still missing due timeout/queue; rerun eval if train checkpoint exists.
2. Download/generate retain90 reference logs for 1B.
3. Rerun or re-evaluate v2 with `RETAIN_LOGS_PATH` to obtain official `forget_quality`.
4. Compare methods at similar Forget Quality, not only raw Model Utility.
5. If BRIDGE still trails NPO too much on MU after FQ is available, run a weaker lambda sweep:
   - `lambda_g=0.01, lambda_b=0`
   - `lambda_g=0.01, lambda_b=0.03`
   - `lambda_g=0, lambda_b=0.03`
6. Only after 1B behavior is stable, rerun 7B.

## Practical Warning For The Next Agent

- Do not interpret `seed0_initial` as method evidence; it is invalid.
- Do not trust `forget_quality` blanks as zero; blank means FQ was not computed because retain logs were missing.
- Do not delete `results/bridge_reports`; those reports are intentionally tracked.
- Check `git status --short` before committing because local result folders and ignored files may be present.
- When answering zigo, give exact HPC commands, not only conceptual advice.
