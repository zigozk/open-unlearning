# PIPER Probe Experiments

本目录用于第一阶段机制探究实验，目标不是训练最终 PIPER 方法，而是验证：

```text
PI(B_f, x_r) 是否能预测遗忘训练后的 retain damage
```

当前脚本：

- `pi_predictive_validity_probe.py`：使用正常参数更新在 TOFU 上运行 GA 或 NPO 探究实验，输出 PI 分数与 retain loss increase 的相关性、Top-K enrichment 和分桶统计。
- `summarize_pi_probe_sweep.py`：汇总多个 probe run 的 `summary.json`，生成 sweep 级别 CSV。

评审标准下，只有当 PI 分数对实际 damage 有稳定预测信号时，后续才值得实现 `Backbone + PIPER local KL`。如果该实验不成立，论文不能继续声称方法能够识别 vulnerable retain samples。

当前 sbatch sweep：

```bash
sbatch sbatch/piper/pi_probe_llama2_7b_full_array.sh
```

默认覆盖：

- backbone：`NPO`、`GradAscent`
- split：`forget01/retain99`、`forget05/retain95`、`forget10/retain90`
- seed：`0, 1, 2`
- forget batch size：`1, 2`
- probe batches：`4, 8`

其中 `GradAscent` 使用较低学习率和较少训练步数，以避免第一轮实验中出现的 retain damage 过强扩散。该 sweep 仍然只是机制存在性与稳定性验证，不是最终 PIPER 训练实验。
