# BRIDGE / Forget Quality Audit Handoff

Last updated: 2026-07-06

This is the next-session handoff after the 2026-07-06 debugging session. Read
the older handoff first if you need full background:

```text
docs/bridge/BRIDGE_FQ_audit_handoff_2026-07-05.md
```

The active work is still not BRIDGE method development. The active work is
auditing why TOFU `forget_quality` is unexpectedly low by comparing
OpenUnlearning reruns, BalDRO, official unlearned checkpoints, retain reference
logs, and old results.

## Repositories

Main notes repo:

```text
D:\User\zigo\Github\open-unlearning
```

BalDRO audit repo:

```text
D:\User\zigo\Github\BalDRO
/home/zkzhang/unlearning/BalDRO
branch: bridge-fq-audit-sbatch
local base commit seen today: 58536b8
```

OpenUnlearning rerun repo:

```text
D:\User\zigo\Github\open-unlearning-rerun
/home/zkzhang/unlearning/open-unlearning-rerun
branch: bridge-rerun-clean
local base commit seen today: 97f7cef
```

Important: code fixes were copied to the HPC working trees, but they were not
committed or pushed during this session.

## Synced Results

Both repo result/log folders were copied from HPC to local:

```text
hpc:/home/zkzhang/unlearning/BalDRO/{logs,results}
hpc:/home/zkzhang/unlearning/open-unlearning-rerun/{logs,results}
```

Current local summary note:

```text
docs/bridge/BRIDGE_FQ_latest_synced_results_2026-07-06.md
```

Known result anchors:

- BalDRO existing full Llama-2-7B forget01 eval:
  `forget_quality=0.0012708143485281624`,
  `model_utility=0.6282402799310759`.
- Official OpenUnlearning Llama3.2-1B NPO forget10 eval:
  `forget_quality=2.6338754853638006e-10`,
  `model_utility=0.596645304244054`.
- User later reported BalDRO NPO smoke summary with only:
  `forget_quality=0.006760732303569208`,
  `model_utility=0.621653243673758`.

## BalDRO Fixes Applied

All BalDRO fq_audit scripts now default to the intended HPC environment:

```bash
CONDA_ENV="${CONDA_ENV:-unlearning}"
```

Touched files:

```text
src/train.py
sbatch/fq_audit/README.md
sbatch/fq_audit/baldro_eval_existing_models.sh
sbatch/fq_audit/baldro_generate_retain_refs.sh
sbatch/fq_audit/baldro_summarize_tofu.sh
sbatch/fq_audit/baldro_train_eval_drnpo_llama2_forget01_grid.sh
sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
tools/fq_audit/summarize_tofu.py
```

Fix details:

- `src/train.py` no longer imports `wandb` and `mlflow` at module import time.
  It imports them only if `trainer.args.report_to` requests that logger.
- BalDRO NPO/DrNPO audit scripts default to:

```bash
REPORT_TO="${REPORT_TO:-none}"
```

- BalDRO audit scripts remove the offline-unavailable gibberish metric:

```bash
"~eval.tofu.metrics.forget_Q_A_gibberish"
```

- `baldro_train_eval_npo_llama2_splits.sh` had an HPC-side typo in its GPU GRES:
  `nvidia_a100_80gb_pice`. This was corrected to:

```bash
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
```

- `tools/fq_audit/summarize_tofu.py` now reads both `TOFU_SUMMARY.json` and the
  sibling `TOFU_EVAL.json`. It fills top-level metrics from summary when present
  and fills smaller metrics from each metric's `agg_value` in `TOFU_EVAL.json`.
  It also emits `<metric>_count` columns from `value_by_index`.

Small metrics now summarized include:

```text
forget_truth_ratio
forget_Q_A_PARA_Prob
forget_Q_A_PERT_Prob
forget_Q_A_Prob
forget_Q_A_ROUGE
retain_Q_A_Prob
retain_Q_A_ROUGE
retain_Q_A_PARA_Prob
retain_Q_A_PERT_Prob
retain_Truth_Ratio
ra_Q_A_Prob
ra_Q_A_PERT_Prob
ra_Q_A_Prob_normalised
ra_Q_A_ROUGE
ra_Truth_Ratio
wf_Q_A_Prob
wf_Q_A_PERT_Prob
wf_Q_A_Prob_normalised
wf_Q_A_ROUGE
wf_Truth_Ratio
```

BalDRO commands:

```bash
cd ~/unlearning/BalDRO

# Smoke forget01 only.
sbatch --array=0-0 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh

# If smoke succeeded, run only the remaining splits to avoid repeating forget01.
sbatch --array=1-2%1 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh

# Summarize all BalDRO TOFU summaries with the expanded small-metric columns.
sbatch sbatch/fq_audit/baldro_summarize_tofu.sh
```

BalDRO eval outputs are usually under checkpoint eval folders:

```text
results/baldro_fq_audit/train/<task_name>/checkpoint-*/evals/TOFU_SUMMARY.json
results/baldro_fq_audit/train/<task_name>/checkpoint-*/evals/TOFU_EVAL.json
```

## OpenUnlearning Rerun Fixes Applied

Touched files:

```text
src/trainer/__init__.py
src/trainer/base.py
sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

Fix details:

- The HPC `unlearning-new` environment has `transformers==4.45.1`.
  That Trainer does not accept `processing_class`.
- `src/trainer/__init__.py` now inspects the trainer constructor:
  if `processing_class` exists, pass `processing_class=...`; otherwise pass
  `tokenizer=...`.
- `src/trainer/base.py` mirrors old `self.tokenizer` back to
  `self.processing_class` so evaluator code can continue using the newer field.
- The sbatch preflight now prints compatibility mode instead of aborting.
- A bad HPC-side edit changed `from transformers import Trainer` into
  `from transformers import Traine`; the full script was recopied from local and
  verified on HPC.
- Earlier logs had `.er` instead of `.err`; the recopied script has:

```bash
#SBATCH -e logs/%x-%j.err
```

Important script structure:

- `bridge_baseline_npo_llama2_splits.sh` is a parent submission script.
- The top-level job is CPU-only and submits the real GPU array from inside
  `submit_pipeline`.
- Do not run smoke with `sbatch --array=0-0 ...`; that makes the CPU-only parent
  job execute the array path without a GPU.

Correct rerun commands:

```bash
cd ~/unlearning/open-unlearning-rerun

# Smoke forget01 only.
sbatch --export=ALL,ARRAY_RANGE=0-0 sbatch/bridge/bridge_baseline_npo_llama2_splits.sh

# If smoke succeeded, run only remaining splits.
sbatch --export=ALL,ARRAY_RANGE=1-2%1 sbatch/bridge/bridge_baseline_npo_llama2_splits.sh

# Full 0-2 run, only if repeating forget01 is acceptable.
sbatch sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

The rerun script's internal GPU request defaults to:

```bash
GRES="${GRES:-gpu:nvidia_h100_80gb_hbm3:1}"
```

Cluster check showed this exists on `gpu01`. Available 80GB alternatives seen:

```text
gpu:nvidia_a100_80gb_pcie:1
gpu:nvidia_a800_80gb_pcie:1
gpu:a100-sxm4-80gb:1
```

If H100 queueing is annoying, override the rerun smoke like this:

```bash
sbatch --export=ALL,ARRAY_RANGE=0-0,GRES=gpu:nvidia_a100_80gb_pcie:1 \
  sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

The latest rerun error after `processing_class` was fixed:

```text
No devices were found
FlashAttention2 has been toggled on ... package flash_attn seems to be not installed
```

Explanation:

- `No devices were found` happened because the CPU-only parent script was
  submitted with `--array=0-0`.
- `flash_attn` was missing in `unlearning-new`. The script now defaults to:

```bash
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
```

and passes:

```bash
"model.model_args.attn_implementation=${ATTN_IMPLEMENTATION}"
```

to both train and eval.

The rerun script also has a GPU-allocation guard. If someone accidentally runs
the CPU parent as an array task again, it prints:

```text
[error] No GPU allocation detected.
Submit smoke tests with: sbatch --export=ALL,ARRAY_RANGE=0-0 ...
```

## Verification Notes

Local checks performed:

```text
python -m py_compile ...
bash -n ...
```

HPC checks performed:

- changed scripts copied to the matching HPC working trees;
- `bash -n` passed for changed sbatch files;
- grep confirmed intended key lines on HPC.

Avoid relying on ad hoc `python -c "import transformers"` checks on the HPC
login node; during this session those sometimes entered remote `D` I/O-wait
state. The actual Slurm logs are more useful.

## Current Dirty State

Main notes repo has at least this untracked/modified handoff material:

```text
docs/bridge/BRIDGE_FQ_latest_synced_results_2026-07-06.md
docs/bridge/BRIDGE_FQ_audit_handoff_2026-07-06.md
```

BalDRO has code/script changes plus synced logs/results showing as modified.
Do not blindly commit all modified logs/results. Likely commit candidates:

```text
src/train.py
sbatch/fq_audit/README.md
sbatch/fq_audit/*.sh
tools/fq_audit/summarize_tofu.py
```

OpenUnlearning rerun has code/script changes plus synced logs/results showing as
modified. Likely commit candidates:

```text
src/trainer/__init__.py
src/trainer/base.py
sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

## Next Actions

1. On HPC, rerun OpenUnlearning rerun smoke with the correct parent-script
   command:

```bash
cd ~/unlearning/open-unlearning-rerun
sbatch --export=ALL,ARRAY_RANGE=0-0 sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

2. Watch:

```bash
tail -f logs/npo_l2_submit-*.out
tail -f logs/npo_l2_splits-*_0.out
tail -f logs/npo_l2_splits-*_0.err
```

3. If rerun smoke succeeds, run the remaining `ARRAY_RANGE=1-2%1`.

4. Run or continue BalDRO remaining splits if needed, then summarize:

```bash
cd ~/unlearning/BalDRO
sbatch --array=1-2%1 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
sbatch sbatch/fq_audit/baldro_summarize_tofu.sh
```

5. Sync results back locally and compare BalDRO NPO vs OpenUnlearning rerun.

Interpretation rule:

- If BalDRO and rerun both show low FQ, suspect data/reference/checkpoint or
  training setup rather than an OpenUnlearning-only evaluator bug.
- If BalDRO is reasonable but rerun is low, inspect rerun trainer/eval/config
  differences.
- If official unlearned checkpoints are low in both, re-check retain reference
  logs and TOFU split/cache alignment.
