# PIPER Probe Experiments

本目录用于第一阶段机制探究实验，目标不是训练最终 PIPER 方法，而是验证：

```text
PI(B_f, x_r) 是否能预测遗忘训练后的 retain damage
```

当前脚本：

- `pi_predictive_validity_probe.py`：使用正常参数更新在 TOFU 上运行 GA 或 NPO 探究实验，输出 PI 分数与 retain loss increase 的相关性、Top-K enrichment 和分桶统计。

评审标准下，只有当 PI 分数对实际 damage 有稳定预测信号时，后续才值得实现 `Backbone + PIPER local KL`。如果该实验不成立，论文不能继续声称方法能够识别 vulnerable retain samples。
