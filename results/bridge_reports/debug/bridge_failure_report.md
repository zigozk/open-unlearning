# BRIDGE Failure Summary

- Rows: 6
- CSV: `results/bridge_reports/debug/bridge_failure_summary.csv`

## Status Counts

| status | count |
| --- | --- |
| train_failed | 6 |

## Error Type Counts

| error_type | count |
| --- | --- |
| unknown | 5 |
| python_traceback+value_error | 1 |

## Per Task

| method | status | error_type | first_error | out_log | err_log |
| --- | --- | --- | --- | --- | --- |
| npo | train_failed | python_traceback+value_error | Traceback (most recent call last): | logs/bridge_initial-70355_0.out | logs/bridge_initial-70355_0.err |
| npo_global_kl | train_failed | unknown |  | logs/bridge_initial-70355_1.out | logs/bridge_initial-70355_1.err |
| bridge_uniform_dro | train_failed | unknown |  | logs/bridge_initial-70355_2.out | logs/bridge_initial-70355_2.err |
| bridge_history_dro | train_failed | unknown |  | logs/bridge_initial-70355_3.out | logs/bridge_initial-70355_3.err |
| bridge_refresh_gs_dro | train_failed | unknown |  | logs/bridge_initial-70355_4.out | logs/bridge_initial-70355_4.err |
| bridge_refresh_pi_dro | train_failed | unknown |  | logs/bridge_initial-70355_5.out | logs/bridge_initial-70355_5.err |
