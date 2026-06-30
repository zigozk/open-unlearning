# BRIDGE Experiment Namespace

This directory is reserved for BRIDGE-specific helper scripts that sit outside the core OpenUnlearning training loop.

Current state: no runnable BRIDGE experiment script has been added yet.

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
