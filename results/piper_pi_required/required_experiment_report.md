# PIPER PI Required Mechanism Supplement Report

- Source CSV: `results/piper_pi_required/required_summary.csv`
- Runs: 81
- Model: `Llama-2-7b-chat-hf` TOFU full checkpoint
- Backbone: `NPO`
- Splits: `forget01/retain99`, `forget05/retain95`, `forget10/retain90`
- Seeds: `0, 1, 2`
- Coverage settings: `fb1-pb8/16/32/64/128`, `fb2-pb8/16/32/64`

## Executive Judgment

当前补充实验总体支持 PI 探针机制，但结论应区分 averaged PI 与 set-gradient PI。

- Averaged PI Spearman mean = **0.626**，range = [0.005, 0.845]。
- Set-gradient PI Spearman mean = **0.731**，range = [0.130, 0.873]。
- Semantic baseline Spearman mean = **0.126**，range = [0.051, 0.233]。
- Set-gradient PI wins over averaged PI in **53/81** runs; averaged PI wins in **28/81** runs。
- Averaged PI beats semantic baseline in **78/81** runs; set-gradient PI beats semantic baseline in **80/81** runs。

评审口径下，set-gradient PI 比 averaged PI 更适合作为正式 Static-PIPER 的候选。Semantic baseline 明显弱于 PI，说明当前信号不是简单语义相似检索可以替代的。

## Overall Metrics

| Method | Spearman mean [min,max] | Top10 damage lift mean [min,max] | Top10 enrichment mean [min,max] |
|---|---:|---:|---:|
| Averaged PI | 0.626 [0.005, 0.845] | 1.003 [0.105, 1.561] | 6.323 [0.600, 9.000] |
| Set-gradient PI | 0.731 [0.130, 0.873] | 1.191 [0.592, 1.765] | 8.272 [3.600, 9.400] |
| Semantic baseline | 0.126 [0.051, 0.233] | 0.215 [0.095, 0.444] | 1.825 [1.200, 2.400] |

## By Split

| Split | n | Averaged PI Spearman | Set-gradient PI Spearman | Semantic Spearman | Forget loss delta |
|---|---:|---:|---:|---:|---:|
| `forget01` | 27 | 0.692 [0.353, 0.829] | 0.776 [0.627, 0.839] | 0.090 [0.061, 0.115] | 1.687 [1.469, 1.996] |
| `forget05` | 27 | 0.615 [0.005, 0.845] | 0.715 [0.130, 0.846] | 0.137 [0.051, 0.193] | 1.585 [1.380, 1.994] |
| `forget10` | 27 | 0.570 [0.086, 0.839] | 0.701 [0.422, 0.873] | 0.149 [0.086, 0.233] | 1.507 [1.381, 1.751] |

结论：三个 split 上 set-gradient PI 都保持正相关，`forget01` 最稳定，`forget10` 也可用；`forget05` 存在弱配置，但 set-gradient 后明显改善。

## By Coverage

| Coverage samples | n | Averaged PI Spearman | Set-gradient PI Spearman | Avg-set Spearman | Top10 overlap |
|---|---:|---:|---:|---:|---:|
| `8` | 9 | 0.536 [0.005, 0.824] | 0.670 [0.130, 0.855] | 0.649 [0.065, 0.875] | 0.642 [0.280, 0.880] |
| `16` | 18 | 0.643 [0.353, 0.839] | 0.707 [0.440, 0.863] | 0.650 [0.139, 0.971] | 0.697 [0.320, 0.940] |
| `32` | 18 | 0.628 [0.138, 0.829] | 0.746 [0.422, 0.866] | 0.533 [-0.038, 0.952] | 0.677 [0.160, 0.940] |
| `64` | 18 | 0.630 [0.086, 0.843] | 0.755 [0.569, 0.867] | 0.522 [-0.413, 0.939] | 0.662 [0.020, 0.920] |
| `128` | 18 | 0.647 [0.206, 0.845] | 0.745 [0.450, 0.873] | 0.566 [-0.098, 0.949] | 0.678 [0.100, 0.920] |

结论：coverage 从 8 提升到 32/64/128 后，set-gradient PI 的稳定性明显更好。PI-8 可作为低成本探针，但正式 Static-PIPER 不宜只依赖 PI-8。

## By fb/pb Config

| Config | n | Averaged PI Spearman | Set-gradient PI Spearman | Semantic Spearman | Set Top10 lift |
|---|---:|---:|---:|---:|---:|
| `fb1-pb8` | 9 | 0.536 [0.005, 0.824] | 0.670 [0.130, 0.855] | 0.126 [0.065, 0.197] | 1.122 [0.628, 1.641] |
| `fb1-pb16` | 9 | 0.573 [0.368, 0.734] | 0.728 [0.620, 0.863] | 0.126 [0.063, 0.200] | 1.172 [0.592, 1.765] |
| `fb1-pb32` | 9 | 0.550 [0.138, 0.694] | 0.785 [0.625, 0.866] | 0.128 [0.068, 0.200] | 1.232 [1.005, 1.745] |
| `fb1-pb64` | 9 | 0.486 [0.086, 0.654] | 0.786 [0.626, 0.867] | 0.123 [0.055, 0.195] | 1.224 [0.965, 1.740] |
| `fb1-pb128` | 9 | 0.512 [0.206, 0.654] | 0.783 [0.611, 0.873] | 0.123 [0.055, 0.200] | 1.231 [0.965, 1.724] |
| `fb2-pb8` | 9 | 0.713 [0.353, 0.839] | 0.687 [0.440, 0.826] | 0.127 [0.055, 0.232] | 1.150 [0.746, 1.407] |
| `fb2-pb16` | 9 | 0.706 [0.508, 0.829] | 0.707 [0.422, 0.830] | 0.128 [0.051, 0.233] | 1.188 [0.883, 1.439] |
| `fb2-pb32` | 9 | 0.774 [0.696, 0.843] | 0.724 [0.569, 0.833] | 0.124 [0.056, 0.213] | 1.211 [0.988, 1.478] |
| `fb2-pb64` | 9 | 0.782 [0.696, 0.845] | 0.707 [0.450, 0.833] | 0.127 [0.057, 0.228] | 1.190 [0.946, 1.460] |

结论：`fb1` 下 set-gradient PI 明显优于 averaged PI；`fb2` 下 averaged PI 也很强。若目标是正式方法的稳定性，推荐优先使用 set-gradient PI，coverage 64 或 128；若考虑成本，`fb1-pb32` 或 `fb1-pb64` 已经比较稳。

## Averaged PI vs Set-gradient PI

- averaged 与 set-gradient 的 Spearman mean = **0.577**，range = [-0.413, 0.971]。
- Top-10% overlap mean = **0.674**，range = [0.020, 0.940]。

解释：二者相关但不是等价。set-gradient 多数情况下更接近真实 damage，因此后续正式 Static-PIPER 不宜只依赖最小 averaged PI。Top-K overlap 的下界很低，说明某些配置下两种 PI 会选择不同 vulnerable retain samples。

## Semantic Baseline Comparison

- `forget01`: semantic Spearman mean 0.090; averaged PI wins 27/27, set-gradient PI wins 27/27。
- `forget05`: semantic Spearman mean 0.137; averaged PI wins 25/27, set-gradient PI wins 26/27。
- `forget10`: semantic Spearman mean 0.149; averaged PI wins 26/27, set-gradient PI wins 27/27。

语义基线在少数配置上有弱相关，但总体显著弱于 PI。这支持“PI 捕捉模型侧遗忘干扰，而不是单纯语义邻近”的叙事。

## Forget-side Sanity

- `forget01`: forget loss delta mean **1.687**; answer probability 0.657 -> 0.137，delta = -0.520。
- `forget05`: forget loss delta mean **1.585**; answer probability 0.633 -> 0.154，delta = -0.479。
- `forget10`: forget loss delta mean **1.507**; answer probability 0.644 -> 0.166，delta = -0.478。

这说明 retain damage 的预测关系发生在实际有效的 unlearning 更新之后，而不是完全没有遗忘或仅由随机扰动造成。

## Failure / Weak Cases

| Run | Split | fb | pb | Averaged rho | Set rho | Semantic rho | Avg-set Top10 overlap |
|---|---|---:|---:|---:|---:|---:|---:|
| `Llama-2-7b-chat-hf_forget05_retain95_NPO_seed1_fb1_pb8_both_66096_36` | forget05 | 1 | 8 | 0.005 | 0.130 | 0.175 | 0.520 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb2_pb16_both_66199_69` | forget10 | 2 | 16 | 0.685 | 0.422 | 0.233 | 0.860 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb2_pb8_both_66196_68` | forget10 | 2 | 8 | 0.538 | 0.440 | 0.232 | 0.900 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb2_pb64_both_66206_71` | forget10 | 2 | 64 | 0.748 | 0.450 | 0.228 | 0.820 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb1_pb8_both_66158_63` | forget10 | 1 | 8 | 0.398 | 0.455 | 0.197 | 0.320 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb2_pb32_both_66200_70` | forget10 | 2 | 32 | 0.728 | 0.569 | 0.213 | 0.680 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb1_pb128_both_66192_67` | forget10 | 1 | 128 | 0.366 | 0.611 | 0.200 | 0.100 |
| `Llama-2-7b-chat-hf_forget10_retain90_NPO_seed1_fb1_pb16_both_66164_64` | forget10 | 1 | 16 | 0.580 | 0.620 | 0.200 | 0.600 |

这些弱案例不应从论文中隐藏。它们说明 PI 的 Top-K 选择仍受 split、seed 和 batch 构造影响。若正式方法采用 set-gradient PI，应报告这些稳定性边界。

## Recommended Next Experimental Step

