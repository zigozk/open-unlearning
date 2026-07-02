# BRIDGE Initial Round Summary

- Runs found: 6
- CSV: `results/bridge_reports/bridge_initial_summary.csv`

| model | method | status | forget_quality | model_utility | mean_retain_kl | worst_k_retain_kl | boundary_dro | prior_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct | bridge_history_dro | eval_complete |  | 0.363514 | 1.01367 | 1.38455 | 1.02126 | 0 |
| Llama-3.2-1B-Instruct | bridge_refresh_gs_dro | eval_complete |  | 0.352466 | 1.02227 | 1.37476 | 1.04939 | 0 |
| Llama-3.2-1B-Instruct | bridge_refresh_pi_dro | eval_missing |  |  | 1.04391 | 1.37744 | 1.07054 | 0 |
| Llama-3.2-1B-Instruct | bridge_uniform_dro | eval_complete |  | 0.359573 | 1.0137 | 1.37075 | 1.03956 | 0 |
| Llama-3.2-1B-Instruct | npo_global_kl | eval_complete |  | 0.282725 | 1.37203 | 1.75061 | 0 | 0 |
| Llama-3.2-1B-Instruct | npo | eval_complete |  | 0.410651 |  |  |  |  |

Interpretation rule: compare methods at similar Forget Quality before treating a higher Model Utility as a win.
