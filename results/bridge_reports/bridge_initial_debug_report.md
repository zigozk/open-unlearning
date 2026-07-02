# BRIDGE Detailed Debug Report

- Runs found: 8
- CSV: `results/bridge_reports/bridge_initial_debug_summary.csv`

## Status Counts

| status | count |
| --- | --- |
| eval_complete | 7 |
| eval_missing | 1 |

## Duplicate Eval Outputs

| hash | group_size | model | method | model_utility | forget_truth_ratio |
| --- | --- | --- | --- | --- | --- |
| 922b5e1add27 | 5 | Llama-3.2-1B-Instruct | bridge_history_dro | 0 | 0.795614 |
| 922b5e1add27 | 5 | Llama-3.2-1B-Instruct | bridge_refresh_gs_dro | 0 | 0.795614 |
| 922b5e1add27 | 5 | Llama-3.2-1B-Instruct | bridge_refresh_pi_dro | 0 | 0.795614 |
| 922b5e1add27 | 5 | Llama-3.2-1B-Instruct | bridge_uniform_dro | 0 | 0.795614 |
| 922b5e1add27 | 5 | Llama-3.2-1B-Instruct | npo_global_kl | 0 | 0.795614 |

## Missing BRIDGE Diagnostics

| model | method | bridge_summary_exists | bridge_diagnostic_rows | train_dir |
| --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct | bridge_history_dro | True | 0 | results/bridge_initial/BRIDGE_TOFU_forget10_Llama_3_2_1B_Instruct_bridge_history_dro_seed0_initial |
| Llama-3.2-1B-Instruct | bridge_refresh_gs_dro | True | 0 | results/bridge_initial/BRIDGE_TOFU_forget10_Llama_3_2_1B_Instruct_bridge_refresh_gs_dro_seed0_initial |
| Llama-3.2-1B-Instruct | bridge_refresh_pi_dro | True | 0 | results/bridge_initial/BRIDGE_TOFU_forget10_Llama_3_2_1B_Instruct_bridge_refresh_pi_dro_seed0_initial |
| Llama-3.2-1B-Instruct | bridge_uniform_dro | True | 0 | results/bridge_initial/BRIDGE_TOFU_forget10_Llama_3_2_1B_Instruct_bridge_uniform_dro_seed0_initial |

## Run Table

| model | method | status | eval_hash_group_size | model_utility | forget_truth_ratio | bridge_prior | bridge_lambda_g | bridge_lambda_b | bridge_diagnostic_rows | boundary_dro | prior_seconds | error_type |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2-7b-chat-hf | npo_global_kl | eval_complete | 1 | 0.402907 | 0.564119 | none | 0.1 | 0 | 0 |  |  | none |
| Llama-2-7b-chat-hf | npo | eval_missing | 0 |  |  | none | 0.0 | 0.0 | 0 |  |  |  |
| Llama-3.2-1B-Instruct | bridge_history_dro | eval_complete | 5 | 0 | 0.795614 | history | 0.1 | 0.3 | 0 |  |  | none |
| Llama-3.2-1B-Instruct | bridge_refresh_gs_dro | eval_complete | 5 | 0 | 0.795614 | refresh_gs | 0.1 | 0.3 | 0 |  |  | none |
| Llama-3.2-1B-Instruct | bridge_refresh_pi_dro | eval_complete | 5 | 0 | 0.795614 | refresh_pi | 0.1 | 0.3 | 0 |  |  | none |
| Llama-3.2-1B-Instruct | bridge_uniform_dro | eval_complete | 5 | 0 | 0.795614 | uniform | 0.1 | 0.3 | 0 |  |  | none |
| Llama-3.2-1B-Instruct | npo_global_kl | eval_complete | 5 | 0 | 0.795614 | none | 0.1 | 0 | 0 |  |  | none |
| Llama-3.2-1B-Instruct | npo | eval_complete | 1 | 0.427375 | 0.621729 | none | 0.0 | 0.0 | 0 |  |  | none |

## Diagnostic Rule

- Identical `TOFU_EVAL.json` hashes across different methods usually indicate identical evaluated model behavior, not a summarizer-only issue.
- BRIDGE runs with zero diagnostic rows should be treated as suspect because `BRIDGENPO.compute_loss()` did not record per-step BRIDGE metrics.
- If `bridge_summary.json` exists but diagnostic rows are zero, inspect the train log and trainer code path before interpreting the method result.
