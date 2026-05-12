# MYTOFU Tuning Status Report

- Generated at: `2026-05-12T20:17:26`
- Result root: `saves/unlearn_synthesis_ext_tuned`
- Logs dir: `logs`
- Ranking metric: `BUS`
- Matched run directories: `27`
- Completed final summaries: `18`
- Run dirs missing final summary: `9`
- Actionable error logs: `18`

## Error Categories

| category | count |
| --- | --- |
| hydra_config | 18 |

## Completed Runs By Method

| method | count |
| --- | --- |
| SatImp_none | 3 |
| SatImp_pcgrad | 3 |
| SatImp_sago | 3 |
| WGA_none | 3 |
| WGA_pcgrad | 3 |
| WGA_sago | 3 |

## Best Completed Result Per Method

| method | tuning_tag | BUS | MYTOFU_Mem | MYTOFU_Utility | forget_Q_A_Prob | retain_Q_A_Prob | task_name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WGA_pcgrad | b0p5_a0p5_g1_lr5em6_e3 | 0.166659 | 0.094087 | 0.728845 | 0.909452 | 0.986623 | mytofu_Llama-3.2-1B-Instruct_WGA_pcgrad_b0p5_a0p5_g1_lr5em6_e3_from_full_e10 |
| WGA_none | b0p5_a0p5_g1_lr5em6_e3 | 0.166509 | 0.093989 | 0.728939 | 0.908377 | 0.986662 | mytofu_Llama-3.2-1B-Instruct_WGA_none_b0p5_a0p5_g1_lr5em6_e3_from_full_e10 |
| WGA_sago | b0p5_a0p5_g1_lr5em6_e3 | 0.115215 | 0.062602 | 0.722040 | 0.937909 | 0.987261 | mytofu_Llama-3.2-1B-Instruct_WGA_sago_b0p5_a0p5_g1_lr5em6_e3_from_full_e10 |
| SatImp_sago | b5_1_a0p5_g0p1_lr5em6_e3 | 0.066785 | 0.035012 | 0.721839 | 0.959331 | 0.987438 | mytofu_Llama-3.2-1B-Instruct_SatImp_sago_b5_1_a0p5_g0p1_lr5em6_e3_from_full_e10 |
| SatImp_pcgrad | b5_1_a0p1_g0p1_lr5em6_e3 | 0.059395 | 0.030973 | 0.720985 | 0.954483 | 0.987869 | mytofu_Llama-3.2-1B-Instruct_SatImp_pcgrad_b5_1_a0p1_g0p1_lr5em6_e3_from_full_e10 |
| SatImp_none | b10_1_a0p1_g0p1_lr5em6_e3 | 0.059386 | 0.030973 | 0.718717 | 0.955370 | 0.987798 | mytofu_Llama-3.2-1B-Instruct_SatImp_none_b10_1_a0p1_g0p1_lr5em6_e3_from_full_e10 |

## Recent/Detected Error Logs

| category | slurm_array_task_id | method_label | tag | cuda_home | err_file |
| --- | --- | --- | --- | --- | --- |
| hydra_config | 0 | DPO_none | b0p05_a0p5_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_0.err |
| hydra_config | 1 | DPO_pcgrad | b0p05_a0p5_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_1.err |
| hydra_config | 18 | UNDIAL_none | b5_a0_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_18.err |
| hydra_config | 19 | UNDIAL_pcgrad | b5_a0_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_19.err |
| hydra_config | 2 | DPO_sago | b0p05_a0p5_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_2.err |
| hydra_config | 20 | UNDIAL_sago | b5_a0_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_20.err |
| hydra_config | 21 | UNDIAL_none | b10_a0p1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_21.err |
| hydra_config | 22 | UNDIAL_pcgrad | b10_a0p1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_22.err |
| hydra_config | 23 | UNDIAL_sago | b10_a0p1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_23.err |
| hydra_config | 24 | UNDIAL_none | b20_a0p1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_24.err |
| hydra_config | 25 | UNDIAL_pcgrad | b20_a0p1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_25.err |
| hydra_config | 26 | UNDIAL_sago | b20_a0p1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_26.err |
| hydra_config | 3 | DPO_none | b0p1_a0p5_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_3.err |
| hydra_config | 4 | DPO_pcgrad | b0p1_a0p5_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_4.err |
| hydra_config | 5 | DPO_sago | b0p1_a0p5_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_5.err |
| hydra_config | 6 | DPO_none | b0p2_a1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_6.err |
| hydra_config | 7 | DPO_pcgrad | b0p2_a1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_7.err |
| hydra_config | 8 | DPO_sago | b0p2_a1_g1_lr5em6_e3 | /usr/local/cuda | logs/mytofu_synth_ext-38615_8.err |

## Notes

- `cuda_home_missing` usually means DeepSpeed imported before `CUDA_HOME` was exported.
- `empty_err` files are ignored because they usually indicate no stderr output.
- stderr files containing only tqdm progress bars or tokenizer padding warnings are ignored by default.
- A run is counted as completed only if `evals_final/MYTOFU_SUMMARY.json` exists.
