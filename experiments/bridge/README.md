# BRIDGE Experiment Namespace

This directory is reserved for BRIDGE-specific helper scripts that sit outside the core OpenUnlearning training loop.

Current runnable helper:

```bash
python experiments/bridge/summarize_bridge_initial.py \
  --train-root results/bridge_initial \
  --eval-root results/bridge_initial_eval \
  --output-csv results/bridge_reports/bridge_initial_summary.csv \
  --output-md results/bridge_reports/bridge_initial_report.md
```

The first-round training/eval jobs are launched through
`sbatch/bridge/bridge_initial_pipeline.sh`.

Failure summarization helper:

```bash
python experiments/bridge/summarize_bridge_failures.py \
  --logs-root logs \
  --train-root results/bridge_initial \
  --eval-root results/bridge_initial_eval \
  --output-csv results/bridge_reports/bridge_failure_summary.csv \
  --output-md results/bridge_reports/bridge_failure_report.md
```

Use this after failed Slurm runs to collect per-task status, first error lines,
and compact failure counts from `logs/bridge_initial-*.out/.err`.

Future scripts should be added here only when they are actually executable or directly useful for summarizing completed runs. Core training code should live under `src/trainer/unlearn/` and Hydra configs should live under `configs/`.

## Planned Script Types

- `summarize_bridge_tofu.py`: collect official TOFU FQ/MU plus BRIDGE retain-side diagnostics.
- `analyze_prior_quality.py`: compute Spearman correlation, top-k damage enrichment and prior/oracle overlap.
- `summarize_bridge_cost.py`: aggregate runtime, memory and prior computation overhead.
- `make_bridge_tables.py`: build paper-facing CSV/markdown tables from summary files.

## Expected Inputs

BRIDGE summarizers should expect ignored heavy outputs under `results/`, then write lightweight publishable artifacts such as:

- `results/bridge_reports/*.csv`
- `results/bridge_reports/*.md`

Only lightweight report artifacts should be force-added to git when needed.

## Relation To PIPER

`legacy_piper_work/experiments/piper/` remains available for old PI probe and PIPER intervention analysis. New BRIDGE methods and summaries should use this directory instead of extending the PIPER namespace.
