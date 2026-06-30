# BRIDGE sbatch Namespace

This directory is reserved for future BRIDGE cluster submission scripts.

Current state: no runnable sbatch script has been added, and no new experiment has been launched as part of the project reorganization.

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
