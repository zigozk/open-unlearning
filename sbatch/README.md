# Open-Unlearning SLURM Runs

This folder contains local SLURM wrappers for running Open-Unlearning on the HPC.
Run commands from the repository root:

```bash
cd /home/zkzhang/unlearn/open-unlearning
```

The GPU type and count are intentionally passed on the `sbatch` command line with
`--gres=gpu:<gpu_type>:<count>`.

Local experiment outputs go under `results/` by default. Official downloaded
reference logs stay under `saves/eval/` and are only read as baselines.

The scripts default to the existing complete TOFU dataset cache:

```text
/home/zkzhang/unlearning/HF_CACHE/datasets
```

Override it with `HF_DATASETS_CACHE=/path/to/cache` only if you know that cache
contains the unperturbed TOFU splits such as `forget10` and `retain90`.
The scripts also default to `HF_DATASETS_OFFLINE=1` so jobs use local data
instead of spending time on slow Hub retries.

## 1. Full Model Eval Sanity Check

Run this first to verify the conda environment, TOFU data, local model path, and
retain eval logs.

```bash
sbatch \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 \
  --cpus-per-task=4 \
  --mem=80G \
  --time=06:00:00 \
  --export=ALL,MODEL=Llama-2-7b-chat-hf,FORGET_SPLIT=forget10 \
  sbatch/slurm_tofu_full_eval.sbatch
```

Optional node pinning:

```bash
sbatch \
  -w gpu06 \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=4 \
  --mem=48G \
  --time=03:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,FORGET_SPLIT=forget10 \
  sbatch/slurm_tofu_full_eval.sbatch
```

## 2. Unlearn Then Eval

Run this after the full model eval sanity check passes.

```bash
sbatch \
  --gres=gpu:nvidia_a100_80gb_pcie:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=12:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=GradAscent,FORGET_SPLIT=forget10,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget10_GradAscent_local \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

Use a fixed node if needed:

```bash
sbatch \
  -w gpu06 \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=12:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO,FORGET_SPLIT=forget10,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget10_NPO_local \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

## Common Variants

### Change Split

The scripts infer retain and holdout splits automatically:

| `FORGET_SPLIT` | `RETAIN_SPLIT` | `HOLDOUT_SPLIT` |
|---|---|---|
| `forget01` | `retain99` | `holdout01` |
| `forget05` | `retain95` | `holdout05` |
| `forget10` | `retain90` | `holdout10` |

Example:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=12:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO,FORGET_SPLIT=forget05,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget05_NPO_local \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

### Change Model

The default local model path is:

```bash
/home/zkzhang/models/tofu_${MODEL}_full
```

Available local full models include:

```text
Llama-3.2-1B-Instruct
Llama-3.2-3B-Instruct
Llama-3.1-8B-Instruct
Llama-2-7b-chat-hf
```

Example for 3B:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=24:00:00 \
  --export=ALL,MODEL=Llama-3.2-3B-Instruct,TRAINER=NPO,FORGET_SPLIT=forget10,PER_DEVICE_TRAIN_BATCH_SIZE=1,GRADIENT_ACCUMULATION_STEPS=32,TASK_NAME=tofu_Llama-3.2-3B-Instruct_forget10_NPO_local \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

Example for Llama-2-7b-chat-hf full eval:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=4 \
  --mem=80G \
  --time=06:00:00 \
  --export=ALL,MODEL=Llama-2-7b-chat-hf,FORGET_SPLIT=forget10 \
  sbatch/slurm_tofu_full_eval.sbatch
```

Example for Llama-2-7b-chat-hf unlearn then eval:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=128G \
  --time=24:00:00 \
  --export=ALL,MODEL=Llama-2-7b-chat-hf,TRAINER=NPO,FORGET_SPLIT=forget10,PER_DEVICE_TRAIN_BATCH_SIZE=1,GRADIENT_ACCUMULATION_STEPS=32,TASK_NAME=tofu_Llama-2-7b-chat-hf_forget10_NPO_local \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

The default local paths used by these commands are:

```text
/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full
saves/eval/tofu_Llama-2-7b-chat-hf_retain90/TOFU_EVAL.json
```

Submit Llama-2-7b-chat-hf forget+eval for `NPO`, `SimNPO`, `GradAscent`, and
`GradDiff` together:

```bash
sbatch/submit_llama2_7b_four_methods.sh \
  --gres=gpu:a100-sxm4-80gb:2 \
  --cpus-per-task=8 \
  --mem=160G \
  --time=04:00:00
```

With a fixed node:

```bash
RUN_TAG=llama2_7b_forget10_four_methods \
sbatch/submit_llama2_7b_four_methods.sh \
  -w gpu06 \
  --gres=gpu:a100-sxm4-80gb:2 \
  --cpus-per-task=8 \
  --mem=160G \
  --time=04:00:00
```

The wrapper rejects Llama-2-7b runs containing `NPO` or `GradDiff` if
`--time` is below 4 hours, unless `ALLOW_SHORT_TIME=1` is set. The 2-hour
request timed out before those two methods could save and evaluate.

The wrapper only submits jobs; each job still runs through
`sbatch/slurm_tofu_unlearn_eval.sbatch`, so every method trains first and then
evaluates into `results/unlearn/<TASK_NAME>/evals`.

If the model/tokenizer path is custom:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=12:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,MODEL_PATH=/path/to/model,TOKENIZER_PATH=/path/to/model,TRAINER=NPO,FORGET_SPLIT=forget10,TASK_NAME=my_custom_run \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

### Change Trainer

Supported TOFU trainers from the repo include `GradAscent`, `GradDiff`, `NPO`,
`DPO`, and `RMU`. Local BRIDGE trainer configs include
`NPO_BRIDGE_GlobalKL`, `NPO_BRIDGE_UniformDRO`, and `NPO_BRIDGE_HistoryDRO`.

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=12:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=GradDiff,FORGET_SPLIT=forget10,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_local \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

For `DPO`, the script automatically uses `experiment=unlearn/tofu/idk.yaml`.
For other trainers, it uses `experiment=unlearn/tofu/default.yaml`.

### BRIDGE Initial Experiments

The initial BRIDGE implementation supports:

```text
Global KL: trainer=NPO_BRIDGE_GlobalKL
Uniform-DRO: trainer=NPO_BRIDGE_UniformDRO
History-DRO: trainer=NPO_BRIDGE_HistoryDRO
```

Run the first two sanity checks on `Llama-3.2-1B-Instruct / forget01 / NPO`:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=06:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO_BRIDGE_GlobalKL,FORGET_SPLIT=forget01,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget01_NPO_BRIDGE_GlobalKL_lg0p1 \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=06:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO_BRIDGE_UniformDRO,FORGET_SPLIT=forget01,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget01_NPO_BRIDGE_UniformDRO_lg0p1_lb0p3 \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

History-DRO needs retain sample ids, so use the index collator:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=06:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO_BRIDGE_HistoryDRO,COLLATOR=DataCollatorForSupervisedDatasetwithIndex,FORGET_SPLIT=forget01,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget01_NPO_BRIDGE_HistoryDRO_lg0p1_lb0p3 \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

Override BRIDGE weights without creating a new config:

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=06:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO_BRIDGE_UniformDRO,FORGET_SPLIT=forget01,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget01_NPO_BRIDGE_UniformDRO_lg0p03_lb0p1,EXTRA_TRAIN_ARGS='trainer.method_args.bridge_lambda_g=0.03 trainer.method_args.bridge_lambda_b=0.1' \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

After each job finishes, refresh the final-only summary:

```bash
python sbatch/summarize_unlearn_results.py
```

### Override Hyperparameters

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=12:00:00 \
  --export=ALL,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO,FORGET_SPLIT=forget10,LEARNING_RATE=2e-5,NUM_TRAIN_EPOCHS=10,EXTRA_TRAIN_ARGS='trainer.method_args.beta=0.5 trainer.method_args.alpha=1.0',TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-5_beta0.5 \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

### Train Without Eval

```bash
sbatch \
  --gres=gpu:a100-sxm4-80gb:1 \
  --cpus-per-task=8 \
  --mem=96G \
  --time=10:00:00 \
  --export=ALL,RUN_EVAL=0,MODEL=Llama-3.2-1B-Instruct,TRAINER=NPO,FORGET_SPLIT=forget10,TASK_NAME=tofu_Llama-3.2-1B-Instruct_forget10_NPO_train_only \
  sbatch/slurm_tofu_unlearn_eval.sbatch
```

## Resource Starting Points

| Task | GPU | CPU | Mem |
|---|---:|---:|---:|
| 1B eval | 1 | 4 | 32-48G |
| 1B unlearn | 1 | 4-8 | 64-96G |
| 3B unlearn | 1-2 | 8 | 96-160G |
| 7B eval | 1 | 4 | 80G |
| 7B/8B unlearn | 1 | 8 | 128G |

For 7B/8B single-GPU unlearning, start with
`PER_DEVICE_TRAIN_BATCH_SIZE=1,GRADIENT_ACCUMULATION_STEPS=32`. If it is stable
and you want speed, try `PER_DEVICE_TRAIN_BATCH_SIZE=2,GRADIENT_ACCUMULATION_STEPS=16`.
Pass `--time` explicitly; shorter accurate wall-time requests often queue
earlier than the script defaults.

## Summarize Results

Summarize the current contents of `results/unlearn` at any time:

```bash
sbatch/summarize_unlearn_results.py
```

It writes:

```text
results/unlearn_summary.csv
results/unlearn_summary.md
```

Check available GPUs:

```bash
scir-watch -s
```

## Logs And Monitoring

SLURM stdout/stderr:

```bash
ls logs/slurm
tail -f logs/slurm/<job-name>-<job-id>.out
tail -f logs/slurm/<job-name>-<job-id>.err
```

Queue:

```bash
squeue --me
```

Cancel:

```bash
scancel <job-id>
```

GPU status on a node:

```bash
scir-watch <gpu-node> gpustat
```

## Output Locations

Full eval outputs default to:

```text
results/eval/<TASK_NAME>
```

Unlearned model outputs default to:

```text
results/unlearn/<TASK_NAME>
```

Unlearn eval outputs default to:

```text
results/unlearn/<TASK_NAME>/evals
```

Official retain/reference logs are still read from paths such as:

```text
saves/eval/tofu_<MODEL>_<RETAIN_SPLIT>/TOFU_EVAL.json
```
