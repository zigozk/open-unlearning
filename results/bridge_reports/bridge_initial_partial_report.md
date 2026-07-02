# BRIDGE Initial Round Summary

- Runs found: 8
- CSV: `results/bridge_reports/bridge_initial_partial_summary.csv`

| model | method | status | forget_quality | model_utility | mean_retain_kl | worst_k_retain_kl | boundary_dro | prior_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-2-7b-chat-hf | npo_global_kl | eval_complete |  | 0.402907 |  |  |  |  |
| Llama-2-7b-chat-hf | npo | eval_missing |  |  |  |  |  |  |
| Llama-3.2-1B-Instruct | bridge_history_dro | eval_complete |  | 0 |  |  |  |  |
| Llama-3.2-1B-Instruct | bridge_refresh_gs_dro | eval_complete |  | 0 |  |  |  |  |
| Llama-3.2-1B-Instruct | bridge_refresh_pi_dro | eval_complete |  | 0 |  |  |  |  |
| Llama-3.2-1B-Instruct | bridge_uniform_dro | eval_complete |  | 0 |  |  |  |  |
| Llama-3.2-1B-Instruct | npo_global_kl | eval_complete |  | 0 |  |  |  |  |
| Llama-3.2-1B-Instruct | npo | eval_complete |  | 0.427375 |  |  |  |  |

Interpretation rule: compare methods at similar Forget Quality before treating a higher Model Utility as a win.
