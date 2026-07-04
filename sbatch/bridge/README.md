# BRIDGE sbatch Namespace

This directory is reserved for future BRIDGE cluster submission scripts.

Current runnable entry point:

```bash
sbatch sbatch/bridge/bridge_initial_pipeline.sh
```

The script submits the first-round single-seed BRIDGE matrix as an array job,
then submits a dependent summary job. The default targets one H100 80GB task on
`gpu01` at a time: `NODELIST=gpu01`, `GRES=gpu:nvidia_h100_80gb_hbm3:1`, `MAX_PARALLEL=1`,
`CPUS_PER_TASK=8`, `MEM=128G`, `TRAIN_BATCH_SIZE=8`,
`GRAD_ACCUM_STEPS=4`, `EVAL_BATCH_SIZE=32`.

By default it runs the 1B model only:

```bash
sbatch sbatch/bridge/bridge_initial_pipeline.sh
```

Run 7B after 1B with:

```bash
sbatch --export=ALL,MODEL_CONFIG=Llama-2-7b-chat-hf sbatch/bridge/bridge_initial_pipeline.sh
```

To queue both models in one array, with 1B task indices before 7B task indices:

```bash
sbatch --export=ALL,MODEL_CONFIGS="Llama-3.2-1B-Instruct Llama-2-7b-chat-hf" sbatch/bridge/bridge_initial_pipeline.sh
```

If the array fails, collect compact failure artifacts with:

```bash
sbatch sbatch/bridge/bridge_collect_failures.sh
```

or run the same parser directly:

```bash
python experiments/bridge/summarize_bridge_failures.py \
  --logs-root logs \
  --train-root results/bridge_initial \
  --eval-root results/bridge_initial_eval \
  --output-csv results/bridge_reports/debug/bridge_failure_summary.csv \
  --output-md results/bridge_reports/debug/bridge_failure_report.md
```

To generate retain-reference logs for official `forget_quality`:

```bash
sbatch sbatch/bridge/bridge_generate_retain_reference.sh
```

This defaults to `Llama-3.2-1B-Instruct` only and writes
`results/bridge_retain_logs/tofu_Llama_3_2_1B_Instruct_retain90_reference/TOFU_EVAL.json`.
When the Llama-2 7B retain90 model is ready, include it with:

```bash
sbatch --export=ALL,MODEL_CONFIGS="Llama-3.2-1B-Instruct Llama-2-7b-chat-hf" sbatch/bridge/bridge_generate_retain_reference.sh
```

To rerun eval only for existing BRIDGE checkpoints with retain logs:

```bash
sbatch sbatch/bridge/bridge_rerun_eval_with_retain.sh
```

This defaults to the current valid 1B tag, `RUN_TAG=postfix_1b_v2`, and writes
the normal tracked report paths under `results/bridge_reports/bridge_initial_*`.

To eval an official OpenUnlearning unlearned model as a sanity check:

```bash
sbatch sbatch/bridge/bridge_eval_official_unlearned.sh
```

The default expects the official 1B forget10 NPO model at
`/home/zkzhang/models/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10`
and the retain90 reference log at
`results/bridge_retain_logs/tofu_Llama_3_2_1B_Instruct_retain90_reference/TOFU_EVAL.json`.
Override paths with `MODEL_PATH`, `TOKENIZER_PATH`, `RETAIN_LOGS_PATH`, and
`OUT_DIR`.

To run a Llama-2-7B-chat NPO baseline split sweep with train+eval for
`forget01/05/10`:

```bash
sbatch sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
```

This submits an array job for the three splits and a dependent summary job.
Defaults use `/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full`, official
retain logs under `saves/eval/`, and NPO-style `gamma/beta` defaults of
`0.1375/2.5` for forget01/05 and `0.125/4.5` for forget10. Override
`FORGET01_NPO_GAMMA` and `FORGET01_NPO_BETA` if you have a different official
forget01 pair.

To summarize BRIDGE initial results at any time:

```bash
sbatch sbatch/bridge/bridge_summarize_initial.sh
```

This writes `results/bridge_reports/bridge_initial_summary.csv` and
`results/bridge_reports/bridge_initial_report.md` by default. Override
`REPORT_PREFIX` if you want another output name.

## Suggested Future Script Order

- `01_bridge_npo_baseline.sh`
- `02_bridge_global_kl_uniform_dro.sh`
- `03_bridge_history_dro.sh`
- `04_bridge_refresh_gs_dro.sh`
- `05_bridge_refresh_pi_dro.sh`
- `06_bridge_summarize_initial.sh`

## Script Policy

Future scripts should:

- default to `forget10 / retain90`, `NPO`, `seed=0` for the initial round;
- expose model, split, seed, lambda, temperature and result root through environment variables;
- write logs under `logs/`;
- write heavy outputs under ignored `results/`;
- call summarizers only after checking expected result files exist;
- keep Phase 6 multi-seed and FQ-matched jobs separate from the initial single-seed scripts.

## Initial Pipeline Defaults

- Methods: `npo`, `npo_global_kl`, `bridge_uniform_dro`,
  `bridge_history_dro`, `bridge_refresh_gs_dro`, `bridge_refresh_pi_dro`
- Optional methods: set `INCLUDE_OPTIONAL_ONLINE=1` for online GS/PI
- Main override knobs: `ROOT_DIR`, `MODEL_CONFIG`, `MODEL_CONFIGS`, `MODEL_PATH`,
  `TOKENIZER_PATH`, `FORGET_SPLIT`, `RETAIN_SPLIT`, `SEED`,
  `TRAIN_BATCH_SIZE`, `REFRESH_INTERVAL`, `MAX_PARALLEL`
- Heavy outputs: `results/bridge_initial/` and `results/bridge_initial_eval/`
- Retain reference logs: `results/bridge_retain_logs/`
- Lightweight reports: `results/bridge_reports/bridge_initial_summary.csv`
  and `results/bridge_reports/bridge_initial_report.md`
- Debug/failure reports: `results/bridge_reports/debug/`
