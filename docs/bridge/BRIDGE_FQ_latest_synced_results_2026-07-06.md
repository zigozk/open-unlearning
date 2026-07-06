# BRIDGE FQ Audit Latest Synced Results

Last checked: 2026-07-06

## Local Sync

Two new local repositories were cloned under:

```text
D:\User\zigo\Github\BalDRO
D:\User\zigo\Github\open-unlearning-rerun
```

Remote HPC result folders were copied into each repo from:

```text
hpc:/home/zkzhang/unlearning/BalDRO/{logs,results}
hpc:/home/zkzhang/unlearning/open-unlearning-rerun/{logs,results}
```

Local branches and commits after clone:

```text
BalDRO: bridge-fq-audit-sbatch @ 58536b8
open-unlearning-rerun: bridge-rerun-clean @ 97f7cef
```

## BalDRO

Available synced result:

```text
D:\User\zigo\Github\BalDRO\results\baldro_fq_audit\eval\fq_audit_l2_full_f01\TOFU_SUMMARY.json
```

Summary:

```json
{
  "forget_quality": 0.0012708143485281624,
  "model_utility": 0.6282402799310759
}
```

Interpretation:

- This is an eval of the full/original Llama-2-7B TOFU checkpoint on `forget01`, not a newly trained BalDRO NPO checkpoint.
- The earlier TOFU cache issue for `forget01_perturbed` was fixed by using `/home/zkzhang/unlearning/HF_CACHE`.
- The run still ended with an offline-cache failure when computing `forget_Q_A_gibberish`, because `madhurjindal/autonlp-Gibberish-Detector-492513457` was not available in the local cache.
- Despite that late failure, `forget_quality` and `model_utility` were already written.

No synced BalDRO NPO train+eval outputs were present under `logs/` or `results/`.

## OpenUnlearning Rerun

Official checkpoint eval result:

```text
D:\User\zigo\Github\open-unlearning-rerun\results\bridge_official_eval\official_NPO_forget10_Llama_3_2_1B_Instruct\TOFU_SUMMARY.json
```

Summary:

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

Llama-2 NPO split summary:

```text
D:\User\zigo\Github\open-unlearning-rerun\results\bridge_reports\llama2_7b_npo_official_params_splits_summary.csv
D:\User\zigo\Github\open-unlearning-rerun\results\bridge_reports\llama2_7b_npo_official_params_splits_report.md
```

All three split rows are still `eval_missing`:

```text
forget01: eval_missing
forget05: eval_missing
forget10: eval_missing
```

The latest rerun split logs show:

```text
Trainer has processing_class=False
This rerun branch passes processing_class to Trainer. Install this repo's requirements in the active CONDA_ENV.
```

Interpretation:

- The official Llama3.2-1B NPO forget10 eval completed and again shows extremely low FQ.
- The Llama2 NPO rerun did not train/evaluate; the current environment still does not support `Trainer(processing_class=...)`.
- This is still an environment/API compatibility blocker, not a low-FQ training result.

## Current State

The latest synced evidence supports the same audit direction:

- BalDRO full-model eval on forget01 also gives low FQ: `0.0012708143`.
- OpenUnlearning official unlearned checkpoint gives even lower FQ: `2.633875e-10`.
- OpenUnlearning Llama2 NPO split reruns remain blocked before training by the `processing_class`/Transformers compatibility issue.
- BalDRO NPO train+eval results are not present yet in the synced output.

Recommended next action:

1. Run the BalDRO NPO train+eval smoke test.
2. Fix the rerun environment or patch rerun trainer initialization back to the old `tokenizer=` interface.
3. Re-run the OpenUnlearning Llama2 NPO split baseline after the preflight reports `Trainer has processing_class=True`, or after the compatibility patch is applied.

## Fixes Applied After This Sync

Updated on: 2026-07-06

OpenUnlearning rerun:

- Root cause: the HPC `unlearning-new` environment has `transformers==4.45.1`, whose `Trainer.__init__` does not accept `processing_class`.
- Fix: `src/trainer/__init__.py` now checks the selected trainer constructor and passes `processing_class=` when supported, otherwise falls back to `tokenizer=`.
- Fix: `src/trainer/base.py` mirrors old `Trainer.tokenizer` back to `self.processing_class` so custom evaluator code can keep using one internal field.
- Fix: `sbatch/bridge/bridge_baseline_npo_llama2_splits.sh` now reports the compatibility mode instead of aborting when `processing_class` is unavailable.

BalDRO:

- Root cause 1: `unlearning-new` does not have `mlflow` or `wandb`, while `src/train.py` imported both unconditionally and the NPO script forced `trainer.args.report_to=wandb`.
- Fix: `src/train.py` now imports `wandb`/`mlflow` only when the selected `trainer.args.report_to` needs that package, and otherwise runs training directly.
- Fix: BalDRO NPO/DrNPO audit scripts now default `REPORT_TO=none`, while still allowing `REPORT_TO=wandb` when the active environment has wandb.
- Root cause 2: TOFU eval enabled `forget_Q_A_gibberish`, which requires the external `madhurjindal/autonlp-Gibberish-Detector-492513457` checkpoint that is not cached in offline mode.
- Fix: BalDRO FQ audit scripts remove `eval.tofu.metrics.forget_Q_A_gibberish` from the composed Hydra config for audit runs.
- Fix: BalDRO fq_audit scripts now default to `CONDA_ENV=unlearning`, which is the intended BalDRO environment on HPC.

Verification:

- Local: `py_compile` passed for changed Python files, and `bash -n` passed for changed sbatch scripts.
- HPC: changed files were copied into `/home/zkzhang/unlearning/open-unlearning-rerun` and `/home/zkzhang/unlearning/BalDRO`; remote `py_compile`/`bash -n` checks passed without importing heavy model libraries.
- HPC note: a direct `import transformers` validation command became stuck in remote `D` state, so further verification avoided that import path. One stale `python -c import inspect, transformers ...` process remained in `D` state after `kill -9`; it should disappear once the cluster I/O wait clears.
