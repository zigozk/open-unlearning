# MYTOFU Tuning Status Report

- Generated at: `2026-05-12T20:23:31`
- Result root: `saves/unlearn_paper_tuned`
- Logs dir: `logs`
- Ranking metric: `BUS`
- Matched run directories: `63`
- Completed final summaries: `63`
- Run dirs missing final summary: `0`
- Actionable error logs: `30`

## Error Categories

| category | count |
| --- | --- |
| cuda_home_missing | 30 |

## Completed Runs By Method

| method | count |
| --- | --- |
| CEU | 6 |
| DPO | 6 |
| GradAscent | 4 |
| GradDiff | 4 |
| GradDiff_pcgrad | 1 |
| GradDiff_sago | 1 |
| NPO | 6 |
| NPO_pcgrad | 1 |
| NPO_sago | 1 |
| PDU | 4 |
| RMU | 7 |
| SatImp | 4 |
| SimNPO | 6 |
| SimNPO_pcgrad | 2 |
| SimNPO_sago | 2 |
| UNDIAL | 4 |
| WGA | 4 |

## Best Completed Result Per Method

| method | tuning_tag | BUS | MYTOFU_Mem | MYTOFU_Utility | forget_Q_A_Prob | retain_Q_A_Prob | task_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CEU | paper_i1_lr1em5_e2 | 0.395430 | 0.295129 | 0.599002 | 0.563911 | 0.894978 | mytofu_Llama-3.2-1B-Instruct_CEU_paper_i1_lr1em5_e2_from_full_e10 |
| DPO | paper_b0p1_a0p5_g1_lr5em6_e3 | 0.209091 | 0.122218 | 0.722998 | 0.844584 | 0.823618 | mytofu_Llama-3.2-1B-Instruct_DPO_paper_b0p1_a0p5_g1_lr5em6_e3_from_full_e10 |
| UNDIAL | paper_b10_a0p1_g1_lr5em6_e3 | 0.173893 | 0.099114 | 0.708244 | 0.892941 | 0.982554 | mytofu_Llama-3.2-1B-Instruct_UNDIAL_paper_b10_a0p1_g1_lr5em6_e3_from_full_e10 |
| PDU | paper_eps0p05_a1_g1_lr5em6_e3 | 0.152285 | 0.085136 | 0.720796 | 0.911023 | 0.986968 | mytofu_Llama-3.2-1B-Instruct_PDU_paper_eps0p05_a1_g1_lr5em6_e3_from_full_e10 |
| WGA | paper_b0p5_a1_g1_lr5em6_e3 | 0.152285 | 0.085136 | 0.720796 | 0.911023 | 0.986968 | mytofu_Llama-3.2-1B-Instruct_WGA_paper_b0p5_a1_g1_lr5em6_e3_from_full_e10 |
| RMU | paper_sc2_a0p5_g1_lr5em6_e3 | 0.151757 | 0.084576 | 0.737848 | 0.908340 | 0.985001 | mytofu_Llama-3.2-1B-Instruct_RMU_paper_sc2_a0p5_g1_lr5em6_e3_from_full_e10 |
| SimNPO | paper_b4p5_g0p25_a1_lr5em6_e3 | 0.120284 | 0.065623 | 0.720035 | 0.937735 | 0.987323 | mytofu_Llama-3.2-1B-Instruct_SimNPO_paper_b4p5_g0p25_a1_lr5em6_e3_from_full_e10 |
| GradDiff_pcgrad | paper_a1_g1_lr5em6_e3 | 0.104077 | 0.056125 | 0.714648 | 0.920921 | 0.981031 | mytofu_Llama-3.2-1B-Instruct_GradDiff_pcgrad_paper_a1_g1_lr5em6_e3_from_full_e10 |
| GradDiff | paper_none_a0p5_g1_lr5em6_e3 | 0.104077 | 0.056124 | 0.714779 | 0.917409 | 0.980685 | mytofu_Llama-3.2-1B-Instruct_GradDiff_paper_none_a0p5_g1_lr5em6_e3_from_full_e10 |
| NPO | paper_b0p1_a2_g1_lr5em6_e3 | 0.104074 | 0.056116 | 0.715881 | 0.915676 | 0.980205 | mytofu_Llama-3.2-1B-Instruct_NPO_paper_b0p1_a2_g1_lr5em6_e3_from_full_e10 |
| NPO_pcgrad | paper_b0p1_a1_g1_lr5em6_e3 | 0.104053 | 0.056110 | 0.714920 | 0.916240 | 0.980260 | mytofu_Llama-3.2-1B-Instruct_NPO_pcgrad_paper_b0p1_a1_g1_lr5em6_e3_from_full_e10 |
| NPO_sago | paper_b0p1_a1_g1_lr5em6_e3 | 0.099370 | 0.053368 | 0.719884 | 0.930685 | 0.981680 | mytofu_Llama-3.2-1B-Instruct_NPO_sago_paper_b0p1_a1_g1_lr5em6_e3_from_full_e10 |
| GradDiff_sago | paper_a1_g1_lr5em6_e3 | 0.088785 | 0.047322 | 0.717025 | 0.940963 | 0.982044 | mytofu_Llama-3.2-1B-Instruct_GradDiff_sago_paper_a1_g1_lr5em6_e3_from_full_e10 |
| SimNPO_pcgrad | paper_b4p5_g0p125_a1_lr5em6_e3 | 0.073447 | 0.038696 | 0.720437 | 0.948061 | 0.987183 | mytofu_Llama-3.2-1B-Instruct_SimNPO_pcgrad_paper_b4p5_g0p125_a1_lr5em6_e3_from_full_e10 |
| SimNPO_sago | paper_b2_g0p125_a1_lr5em6_e3 | 0.062751 | 0.032798 | 0.723394 | 0.957187 | 0.987031 | mytofu_Llama-3.2-1B-Instruct_SimNPO_sago_paper_b2_g0p125_a1_lr5em6_e3_from_full_e10 |
| SatImp | paper_b10_1_a0p1_g0p1_lr5em6_e3 | 0.059386 | 0.030973 | 0.718717 | 0.955370 | 0.987798 | mytofu_Llama-3.2-1B-Instruct_SatImp_paper_b10_1_a0p1_g0p1_lr5em6_e3_from_full_e10 |
| GradAscent | paper_lr2em6_e1 | 0.054168 | 0.028141 | 0.720986 | 0.955425 | 0.981835 | mytofu_Llama-3.2-1B-Instruct_GradAscent_paper_lr2em6_e1_from_full_e10 |

## Recent/Detected Error Logs

| category | slurm_array_task_id | method_label | tag | cuda_home | err_file |
| --- | --- | --- | --- | --- | --- |
| cuda_home_missing | 10 | DPO | paper_b0p05_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_10.err |
| cuda_home_missing | 12 | DPO | paper_b0p2_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_12.err |
| cuda_home_missing | 13 | DPO | paper_b0p5_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_13.err |
| cuda_home_missing | 15 | DPO | paper_b0p1_a2_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_15.err |
| cuda_home_missing | 24 | SimNPO | paper_b1_g0p125_a1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_24.err |
| cuda_home_missing | 27 | SimNPO | paper_b8_g0p125_a1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_27.err |
| cuda_home_missing | 30 | SimNPO_sago | paper_b2_g0p125_a1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_30.err |
| cuda_home_missing | 31 | SimNPO_sago | paper_b4p5_g0p125_a1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_31.err |
| cuda_home_missing | 33 | SimNPO_pcgrad | paper_b4p5_g0p125_a1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_33.err |
| cuda_home_missing | 35 | RMU | paper_sc2_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_35.err |
| cuda_home_missing | 36 | RMU | paper_sc5_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_36.err |
| cuda_home_missing | 37 | RMU | paper_sc10_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_37.err |
| cuda_home_missing | 38 | RMU | paper_sc20_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_38.err |
| cuda_home_missing | 39 | RMU | paper_sc2_a0p5_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_39.err |
| cuda_home_missing | 41 | CEU | paper_i0_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_41.err |
| cuda_home_missing | 42 | CEU | paper_i1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_42.err |
| cuda_home_missing | 43 | CEU | paper_i2_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_43.err |
| cuda_home_missing | 46 | CEU | paper_i2_lr1em5_e2 |  | logs/mytofu_paper_tune-38392_46.err |
| cuda_home_missing | 47 | PDU | paper_eps0p05_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_47.err |
| cuda_home_missing | 48 | PDU | paper_eps0p1_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_48.err |
| cuda_home_missing | 49 | PDU | paper_eps0p2_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_49.err |
| cuda_home_missing | 51 | SatImp | paper_b5_1_a0p1_g0p1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_51.err |
| cuda_home_missing | 52 | SatImp | paper_b5_1_a0p5_g0p1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_52.err |
| cuda_home_missing | 53 | SatImp | paper_b5_1_a1_g0p1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_53.err |
| cuda_home_missing | 54 | SatImp | paper_b10_1_a0p1_g0p1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_54.err |
| cuda_home_missing | 56 | WGA | paper_b1_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_56.err |
| cuda_home_missing | 57 | WGA | paper_b2_a1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_57.err |
| cuda_home_missing | 59 | UNDIAL | paper_b5_a0_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_59.err |
| cuda_home_missing | 61 | UNDIAL | paper_b20_a0_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_61.err |
| cuda_home_missing | 62 | UNDIAL | paper_b10_a0p1_g1_lr5em6_e3 |  | logs/mytofu_paper_tune-38392_62.err |

## Notes

- `cuda_home_missing` usually means DeepSpeed imported before `CUDA_HOME` was exported.
- `empty_err` files are ignored because they usually indicate no stderr output.
- stderr files containing only tqdm progress bars or tokenizer padding warnings are ignored by default.
- A run is counted as completed only if `evals_final/MYTOFU_SUMMARY.json` exists.
