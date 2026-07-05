# BRIDGE / Forget Quality Audit Handoff

Last updated: 2026-07-05

This file is a handoff note for the next ChatGPT/Codex session. Read this
first before continuing the current work. The active thread is no longer only
about the BRIDGE method itself. The current focus is auditing why TOFU
`forget_quality` (FQ) is unexpectedly low in the current OpenUnlearning
workflows, and cross-checking the result with BalDRO, an old Unlearn-Simple
project, official unlearned models, and retain reference logs.

## 1. Main Question

The user observed that TOFU `forget_quality` is very low in the current project,
including for NPO baselines and for an official OpenUnlearning NPO checkpoint.
Older experiments in another project had FQ around 0.2, so the current values
look suspicious.

Possible causes under investigation:

- eval code / FQ computation;
- retain reference logs;
- TOFU split or dataset cache mismatch;
- NPO forget training code;
- environment version mismatch;
- or the model/checkpoint genuinely has low FQ.

Current high-level conclusions:

- `src/eval.py` can run without touching HuggingFace `Trainer`.
- rerun training does create a `Trainer`.
- the rerun branch currently passes `processing_class` to `Trainer`.
- `transformers==4.45.1` does not support `processing_class`.
- BalDRO and older OpenUnlearning are built around `transformers==4.45.1`.
- Therefore BalDRO can use the old environment, but rerun training either needs
  a newer transformers environment or a compatibility patch.

## 2. Repositories And Branches

### Main OpenUnlearning workspace

Local path:

```text
C:\Users\zigo\GitHub\open-unlearning
```

Branch:

```text
mytofu-work
```

This repo contains the original BRIDGE experiment docs and reports. This
handoff file is being saved here under `docs/bridge`.

### OpenUnlearning rerun repo

Local path:

```text
C:\Users\zigo\GitHub\open-unlearning-rerun
```

HPC path:

```text
/home/zkzhang/unlearning/open-unlearning-rerun
```

Remote:

```text
https://github.com/zigozk/open-unlearning.git
```

Branch:

```text
bridge-rerun-clean
```

Latest relevant commit already pushed:

```text
7fdcfb7 Use unlearning-new for rerun bridge sbatch
```

### BalDRO audit repo

Local path:

```text
C:\Users\zigo\GitHub\BalDRO
```

HPC path:

```text
/home/zkzhang/unlearning/BalDRO
```

User fork:

```text
https://github.com/zigozk/BalDRO.git
```

Branch:

```text
bridge-fq-audit-sbatch
```

Latest relevant commits already pushed:

```text
d30f2b2 Add BalDRO FQ audit sbatch scripts
1e79929 Make BalDRO NPO forget-eval the main audit path
```

## 3. Environment Findings

The user checked the HPC environment inside `~/unlearning/BalDRO`:

```text
transformers 4.45.1
huggingface_hub 0.29.1
torch 2.4.1+cu121
datasets 3.0.1
accelerate 0.34.2
Trainer has processing_class: False
```

Interpretation:

- This is expected for BalDRO and old OpenUnlearning.
- This is not enough for the rerun training path if the rerun code still passes
  `processing_class` to `Trainer`.

Important rule for the next session:

- For BalDRO, do not require `processing_class=True`.
- For `open-unlearning-rerun` training, either make the environment support
  `processing_class` or patch the rerun trainer code back to the old
  `tokenizer=` interface.

## 4. Rerun NPO Forget+Eval Script

Main script:

```text
C:\Users\zigo\GitHub\open-unlearning-rerun\sbatch\bridge\bridge_baseline_npo_llama2_splits.sh
/home/zkzhang/unlearning/open-unlearning-rerun/sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

This script is NPO forget+eval.

It first runs NPO training:

```bash
python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default \
  trainer=NPO \
  ...
```

Then it evaluates the trained checkpoint:

```bash
python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/tofu/default \
  "model.model_args.pretrained_model_name_or_path=${TRAIN_DIR}" \
  ...
```

Then it submits a dependent summary job. Expected report outputs:

```text
results/bridge_reports/llama2_7b_npo_official_params_splits_summary.csv
results/bridge_reports/llama2_7b_npo_official_params_splits_report.md
```

Default split/parameter mapping:

```text
forget01 / retain99 / holdout01: gamma=0.1375, beta=2.5
forget05 / retain95 / holdout05: gamma=0.1375, beta=2.5
forget10 / retain90 / holdout10: gamma=0.125,  beta=4.5
learning rate: 1e-5
epochs: 10
train batch size: 1
gradient accumulation: 32
eval batch size: 8
```

Note: forget01 gamma/beta are inferred from the forget05 setting because the
official screenshot mostly gave forget05/forget10 parameters.

Recent changes in commit `7fdcfb7`:

- all `sbatch/bridge/*.sh` default to `CONDA_ENV=unlearning-new`;
- `bridge_baseline_npo_llama2_splits.sh` now has a fast runtime preflight:

```python
import inspect
import transformers
from transformers import Trainer

has_processing_class = "processing_class" in inspect.signature(Trainer.__init__).parameters
print(f"transformers={transformers.__version__}")
print(f"Trainer has processing_class={has_processing_class}")
if not has_processing_class:
    raise SystemExit(
        "This rerun branch passes processing_class to Trainer. "
        "Install this repo's requirements in the active CONDA_ENV."
    )
```

- `.gitattributes` was added so `sbatch/bridge/*.sh` stays LF.

Run on HPC:

```bash
cd ~/unlearning/open-unlearning-rerun
git pull
sbatch sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

Run only forget01 smoke test:

```bash
sbatch --array=0-0 sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

## 5. Rerun Failure Log Summary

The user downloaded rerun logs to:

```text
C:\Users\zigo\GitHub\open-unlearning-rerun\logs
```

Recent failed logs:

```text
logs\npo_l2_splits-72987_0.out / .err
logs\npo_l2_splits-72987_1.out / .err
logs\npo_l2_splits-72987_2.out / .err
```

All three split jobs failed before training:

```text
TypeError: Trainer.__init__() got an unexpected keyword argument 'processing_class'
```

Failing code path:

```text
src/trainer/__init__.py
trainer_cls(... processing_class=processing_class ...)
```

Conclusion:

- this was not a low-FQ result;
- training never started;
- model and data loaded;
- Trainer initialization failed due to incompatible transformers version.

Next action if this reappears:

- install rerun requirements into `unlearning-new`, making transformers support
  `processing_class`; or
- patch the rerun code to pass `tokenizer=` for old transformers.

## 6. BalDRO Main NPO Forget+Eval Path

Main script:

```text
C:\Users\zigo\GitHub\BalDRO\sbatch\fq_audit\baldro_train_eval_npo_llama2_splits.sh
/home/zkzhang/unlearning/BalDRO/sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
```

This is now the main BalDRO audit path. It:

- uses BalDRO code;
- starts from `/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full`;
- runs NPO forget on `forget01`, `forget05`, and `forget10`;
- evaluates during training by BalDRO's trainer;
- uses BalDRO's bundled retain reference logs.

Default resource/environment settings:

```bash
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --array=0-2%1

CONDA_ENV=unlearning-new
MODEL_ROOT=/home/zkzhang/models
MODEL_PATH=/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full
MODEL_CONFIG=Llama-2-7b-chat-hf
```

HF/TOFU cache path is explicitly set to the user's original cache path:

```bash
export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
```

The script preflights TOFU cache configs before model loading:

```text
forgetXX
retainXX
forgetXX_perturbed
holdoutXX
retain_perturbed
real_authors_perturbed
world_facts_perturbed
```

Run on HPC:

```bash
cd ~/unlearning/BalDRO
git pull
sbatch sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
```

Run only forget01 smoke test:

```bash
sbatch --array=0-0 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
```

Outputs:

```text
results/baldro_fq_audit/train/
logs/baldro_fq_audit/
```

BalDRO eval behavior:

- BalDRO evaluates from inside Trainer at each epoch (`eval_strategy=epoch`).
- Results are expected under checkpoint eval directories, e.g.

```text
results/baldro_fq_audit/train/<task_name>/checkpoint-*/evals/TOFU_SUMMARY.json
```

This differs from rerun, where the sbatch script separately calls `src/eval.py`
after training and writes to `results/npo_l2_official_params_eval`.

## 7. BalDRO Existing-Model Eval Failure

Earlier the user ran:

```text
sbatch/fq_audit/baldro_eval_existing_models.sh
```

That script evaluated an existing full/original checkpoint, not a newly
unlearned model.

Logs:

```text
C:\Users\zigo\GitHub\BalDRO\logs\baldro_eval_audit-73031_4294967294.out
C:\Users\zigo\GitHub\BalDRO\logs\baldro_eval_audit-73031_4294967294.err
```

Actual model evaluated:

```text
/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full
```

Log header:

```text
Evaluating l2_full_f01
model_config=Llama-2-7b-chat-hf
model_path=/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full
forget_split=forget01
retain_log=saves/eval/tofu_Llama-2-7b-chat-hf_retain99/TOFU_EVAL.json
```

It computed partial metrics:

```text
model_utility: 0.627455
```

Then failed while computing FQ-related precompute:

```text
ValueError: Couldn't find cache for locuslab/TOFU for config 'forget01_perturbed'
Available configs in the cache: ['forget05_perturbed', 'forget10_perturbed',
'holdout05', 'holdout10', 'real_authors_perturbed', 'retain_perturbed',
'world_facts_perturbed']
```

Cause:

- that run used default `~/.cache/huggingface`, not
  `/home/zkzhang/unlearning/HF_CACHE`;
- the default cache was missing `forget01_perturbed`;
- the script had also been edited locally so it did not run as a normal Slurm
  array, leading to the abnormal array id `4294967294`.

Current fix:

- the main BalDRO NPO train+eval script now uses the original HF cache path;
- it preflights TOFU data configs before loading the model.

## 8. Official Unlearned Model Eval Result

In `open-unlearning-rerun`, the official OpenUnlearning Llama3.2-1B NPO
forget10 checkpoint was evaluated.

Model:

```text
/home/zkzhang/models/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10
```

Retain reference log:

```text
saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json
```

Result path:

```text
results/bridge_official_eval/official_NPO_forget10_Llama_3_2_1B_Instruct/TOFU_SUMMARY.json
```

Result:

```json
{
  "extraction_strength": 0.13849705685588215,
  "forget_Q_A_Prob": 0.404746469585225,
  "forget_Q_A_ROUGE": 0.4255208819834071,
  "forget_quality": 2.6338754853638006e-10,
  "forget_truth_ratio": 0.5378102860689075,
  "model_utility": 0.596645304244054,
  "privleak": -90.51851550436015
}
```

This shows very low FQ even for an official unlearned checkpoint under the
current rerun evaluator. This is one reason BalDRO is being used as an
independent cross-check.

## 9. FQ / MU Code Comparison

Related doc:

```text
docs/bridge/forget_quality_comparison.md
```

Main findings:

- TOFU FQ is essentially a two-sample KS-test p-value:

```python
ks_2samp(unlearn_truth_ratio, retain_truth_ratio).pvalue
```

- OpenUnlearning and BalDRO use essentially the same FQ code path:

```text
src/evals/metrics/privacy.py
ks_test(...)
```

- BalDRO's default `configs/eval/tofu.yaml` enables:

```text
model_utility
forget_quality
forget_Q_A_gibberish
```

- OpenUnlearning rerun may enable more metrics, but the FQ/MU formulas are not
  the main difference.
- The older Unlearn-Simple / TOFU-style utility computes the intermediate truth
  ratio in the reciprocal direction compared with current OpenUnlearning, but
  if the same transform is applied to both forget and retain distributions,
  this alone should not usually explain a massive KS p-value difference.

Likely remaining suspects:

- input examples / dataset config;
- loss aggregation details;
- retain reference log;
- actual checkpoint quality;
- training config differences.

## 10. BalDRO Paper / Hyperparameter Notes

The user asked whether BalDRO repo/paper provides final experiment parameters.

Findings:

- The repo does not provide a complete "final selected config" table per row in
  the paper.
- The scripts are mostly sweep scripts.
- The paper gives search spaces and hyperparameter trends, not exact final
  checkpoint configs.

Paper-level search space:

```text
learning rates: {1e-5, 2e-5, 5e-5, 1e-4}
batch sizes: {8, 16, 32}
beta: {1.0, 2.0, 5.0, 10.0}
lambda: {0.25, 0.5, 1.0, 2.0}
```

Paper hyperparameter analysis:

```text
FQ peaks around beta=2.0, lambda=1.0
beta=2 or 5 is generally reliable/stable
```

Repo scripts differ somewhat from the paper. For example TOFU NPO script sweeps:

```text
lr: 1e-5, 2e-5, 3e-5, 4e-5, 5e-5
batch/grad_acc: 8/2, 8/4
epochs: 5, 10
```

Therefore exact paper reproduction requires running sweeps or asking authors for
selected checkpoint configs.

## 11. BalDRO Code Notes

BalDRO `requirements.txt`:

```text
huggingface-hub==0.29.1
transformers==4.45.1
numpy==2.2.3
hydra-core==1.3
torch==2.4.1
datasets==3.0.1
accelerate==0.34.2
bitsandbytes==0.44.1
scipy==1.14.1
deepspeed==0.15.4
peft==0.17.1
```

The user has `peft==0.18.1` in one environment. BalDRO does not appear to use
PEFT in the relevant path, so this mismatch is probably not important.

More important:

- `src/train.py` imports `mlflow` and `wandb`;
- training only runs when `trainer.args.report_to` is `wandb` or `mlflow`;
- if `report_to` is `tensorboard` or `none`, it can exit without actually
  training.

The BalDRO sbatch sets:

```bash
export WANDB_MODE=offline
trainer.args.report_to=wandb
```

## 12. Old Unlearn-Simple Project

The user has an old `Unlearn-Simple` project cloned locally for comparison.

Purpose:

- compare old NPO baseline results where FQ was around 0.2;
- check whether current low FQ is a code/config/eval issue.

This thread did not complete a full file-by-file diff against Unlearn-Simple.
The work shifted toward clean BalDRO and rerun experiments.

## 13. Important Files Pushed

### BalDRO branch `bridge-fq-audit-sbatch`

```text
.gitattributes
sbatch/fq_audit/README.md
sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
sbatch/fq_audit/baldro_train_eval_drnpo_llama2_forget01_grid.sh
sbatch/fq_audit/baldro_eval_existing_models.sh
sbatch/fq_audit/baldro_generate_retain_refs.sh
sbatch/fq_audit/baldro_summarize_tofu.sh
tools/fq_audit/summarize_tofu.py
```

Main command:

```bash
sbatch sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
```

### OpenUnlearning rerun branch `bridge-rerun-clean`

```text
.gitattributes
sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
sbatch/bridge/bridge_collect_failures.sh
sbatch/bridge/bridge_eval_official_unlearned.sh
sbatch/bridge/bridge_generate_retain_reference.sh
sbatch/bridge/bridge_initial_pipeline.sh
sbatch/bridge/bridge_rerun_eval_with_retain.sh
sbatch/bridge/bridge_summarize_debug.sh
sbatch/bridge/bridge_summarize_initial.sh
```

Main command:

```bash
sbatch sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

## 14. Recommended Next Steps

### Step A: Pull both repos on HPC

```bash
cd ~/unlearning/open-unlearning-rerun
git pull

cd ~/unlearning/BalDRO
git pull
```

### Step B: Check BalDRO environment

```bash
cd ~/unlearning/BalDRO
conda activate unlearning-new
python - <<'PY'
import transformers, huggingface_hub, torch, datasets, accelerate
print("transformers", transformers.__version__)
print("huggingface_hub", huggingface_hub.__version__)
print("torch", torch.__version__)
print("datasets", datasets.__version__)
print("accelerate", accelerate.__version__)
PY
```

Expected/acceptable for BalDRO:

```text
transformers 4.45.1
huggingface_hub 0.29.1
torch 2.4.1+cu121
datasets 3.0.1
accelerate 0.34.2
```

### Step C: Run BalDRO NPO forget+eval first

```bash
cd ~/unlearning/BalDRO
sbatch --array=0-0 sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
```

If forget01 smoke passes:

```bash
sbatch sbatch/fq_audit/baldro_train_eval_npo_llama2_splits.sh
```

After completion:

```bash
sbatch sbatch/fq_audit/baldro_summarize_tofu.sh
```

### Step D: Check rerun training environment

```bash
cd ~/unlearning/open-unlearning-rerun
conda activate unlearning-new
python - <<'PY'
import inspect
import transformers
from transformers import Trainer
print("transformers", transformers.__version__)
print("Trainer has processing_class:", "processing_class" in inspect.signature(Trainer.__init__).parameters)
PY
```

If `processing_class=False`, rerun NPO training will fail unless the environment
or code is fixed.

### Step E: Run rerun NPO forget+eval

```bash
cd ~/unlearning/open-unlearning-rerun
sbatch sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

### Step F: Compare outputs

BalDRO:

```text
results/baldro_fq_audit/train/**/TOFU_SUMMARY.json
results/baldro_fq_audit/tofu_summary.csv
```

OpenUnlearning rerun:

```text
results/npo_l2_official_params_eval/**/TOFU_SUMMARY.json
results/bridge_reports/llama2_7b_npo_official_params_splits_summary.csv
```

Interpretation:

- If BalDRO NPO FQ is also low, suspect data/reference/model/training setting
  rather than an OpenUnlearning-only eval bug.
- If BalDRO NPO FQ is reasonable but rerun FQ is low, inspect rerun
  trainer/eval/config differences.
- If official unlearned model is low in both evaluators, re-check reference logs
  and TOFU cache/split alignment.

## 15. Answer To The Latest User Question

Latest user asked whether the rerun project was changed to `unlearning-new` and
whether its script is NPO forget+eval.

Answer:

- Yes. Commit `7fdcfb7` was pushed to
  `zigozk/open-unlearning:bridge-rerun-clean`.
- Yes. `bridge_baseline_npo_llama2_splits.sh` is NPO forget+eval:
  - first `src/train.py ... trainer=NPO`;
  - then `src/eval.py ... model path=${TRAIN_DIR}`;
  - then a dependent summary job.
