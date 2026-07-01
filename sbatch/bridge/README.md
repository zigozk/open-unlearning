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
  --output-csv results/bridge_reports/bridge_failure_summary.csv \
  --output-md results/bridge_reports/bridge_failure_report.md
```

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
- Lightweight reports: `results/bridge_reports/bridge_initial_summary.csv`
  and `results/bridge_reports/bridge_initial_report.md`
- Failure reports: `results/bridge_reports/bridge_failure_summary.csv`
  and `results/bridge_reports/bridge_failure_report.md`
