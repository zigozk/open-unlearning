# BRIDGE Initial Round Summary

- Runs found: 6
- CSV: `results/bridge_reports/bridge_initial_summary.csv`

| model | method | status | forget_quality | model_utility | mean_retain_kl | worst_k_retain_kl | boundary_dro | prior_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-1B-Instruct | bridge_history_dro | eval_complete | 1.83013e-05 | 0.36144 | 1.01367 | 1.38455 | 1.02126 | 0 |
| Llama-3.2-1B-Instruct | bridge_refresh_gs_dro | eval_complete | 1.83013e-05 | 0.352329 | 1.02227 | 1.37476 | 1.04939 | 0 |
| Llama-3.2-1B-Instruct | bridge_refresh_pi_dro | eval_complete | 1.29553e-05 | 0.346914 | 1.04391 | 1.37744 | 1.07054 | 0 |
| Llama-3.2-1B-Instruct | bridge_uniform_dro | eval_complete | 9.12385e-06 | 0.360047 | 1.0137 | 1.37075 | 1.03956 | 0 |
| Llama-3.2-1B-Instruct | npo_global_kl | eval_complete | 0.000130576 | 0.28271 | 1.37203 | 1.75061 | 0 | 0 |
| Llama-3.2-1B-Instruct | npo | eval_complete | 3.08985e-06 | 0.415219 |  |  |  |  |

Interpretation rule: compare methods at similar Forget Quality before treating a higher Model Utility as a win.
