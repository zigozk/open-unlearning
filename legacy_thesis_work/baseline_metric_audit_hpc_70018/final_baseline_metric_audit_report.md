# Final Baseline Metric Audit Report

## 一句话结论

这批 `forget10 / smoke / 64966` TOFU baseline 的结果支持一个重要判断：**不能只看 FQ 来判断是否已经达到遗忘效果**。在 16 个 run 中，有 8 个 run 的 forget-side 输出指标已经接近崩坏，但 `forget_quality` 仍然极低，没有达到常规意义上的“高 FQ”。这说明 FQ 在这些 baseline 上更像一个分布检验信号，而不是一个直接的“输出是否已经乱掉”的判据。

同时，`model_utility` 也不能单独解释遗忘效果。比如 `GradDiff` 的 MU 平均仍有 `0.5298`，但 forget-side 的 `Q_A_Prob / ROUGE / truth_ratio` 已经几乎归零，说明模型在 retain 汇总指标上看起来还可以，但忘记侧已经明显不可用。

## 使用的数据

- 本地初次分析目录：`legacy_thesis_work/baseline_metric_audit/`
- HPC 下载分析目录：`legacy_thesis_work/baseline_metric_audit_hpc_70018/`
- 两次 summary-level 结果一致，均包含 16 个 baseline run。
- 主要输入来自 `legacy_thesis_work/tofu_eval_summary.csv`，覆盖 4 个模型、4 个方法：
  - Models: `Llama-2-7b-chat-hf`, `Llama-3.1-8B-Instruct`, `Llama-3.2-1B-Instruct`, `Llama-3.2-3B-Instruct`
  - Methods: `GradAscent`, `GradDiff`, `NPO`, `SimNPO`

## 判定规则

本报告沿用自动分析脚本的默认规则：

- Forget collapse threshold: forget-side 指标 `<= 0.05` 视作该维度输出已崩。
- Forget collapse votes: 至少 2 个 forget-side 指标同时崩，记为 `forget_output_collapsed=True`。
- High FQ threshold: `forget_quality >= 0.5` 视作高 FQ。
- Low MU threshold: `model_utility <= 0.2` 视作低 MU。

参与 collapse 投票的 forget-side 指标主要是：

- `forget_Q_A_Prob`
- `forget_Q_A_ROUGE`
- `forget_truth_ratio`

## 问题 1：FQ 是否太“虚”，低 FQ 时模型其实已经忘了？

结论：**是的，这批 baseline 强烈支持这个怀疑。**

总体统计：

| 指标 | 数量 |
| --- | ---: |
| 总 run 数 | 16 |
| forget-side collapsed runs | 8 |
| collapsed before high FQ runs | 8 |
| low MU runs | 4 |

最关键的是 `collapsed_before_high_fq = 8 / 16`。也就是说，一半 run 在 forget-side 输出指标上已经表现为崩坏，但 FQ 仍然没有达到高阈值。

按方法看：

| Method | Runs | Collapsed | Collapsed Before High FQ | Low MU | FQ Mean | MU Mean | Forget Prob Mean | Forget ROUGE Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GradAscent | 4 | 4 | 4 | 4 | 1.064e-239 | 0.0000 | 8.911e-36 | 0.0010 |
| GradDiff | 4 | 4 | 4 | 0 | 2.155e-223 | 0.5298 | 1.960e-10 | 0.0025 |
| NPO | 4 | 0 | 0 | 0 | 0.1295 | 0.5491 | 0.1085 | 0.2616 |
| SimNPO | 4 | 0 | 0 | 0 | 1.855e-14 | 0.6053 | 0.5239 | 0.4855 |

这里最明显的是 `GradAscent` 和 `GradDiff`：

- `GradAscent`: 4/4 全部崩坏，MU 也全部为低，属于“遗忘有效但 retain 也严重损坏”。
- `GradDiff`: 4/4 forget-side 全部崩坏，但 MU 并不低，说明 FQ 和 MU 都没有单独揭示“forget 输出已经乱了”这个事实。

这说明如果论文或实验中只追求高 FQ，可能会错过一类已经完成实质遗忘的模型状态。对于“是否已经忘掉”这个问题，`forget_Q_A_Prob / forget_Q_A_ROUGE / forget_truth_ratio` 比 FQ 更直接。

## 问题 2：低 MU 是否基本代表模型输出也乱了？

结论：**低 MU 确实是模型整体损坏的强信号，但 MU 不低并不代表 forget-side 没坏。**

证据分两层：

1. `GradAscent` 的 4 个 run 全部 `model_utility=0.0`，同时 forget-side 三个指标几乎全为 0。这是典型的“强遗忘伴随整体崩坏”。
2. `GradDiff` 的 4 个 run 全部 forget-side 崩坏，但 MU 分别约为 `0.3551`, `0.5738`, `0.5808`, `0.6095`，并没有达到低 MU 判据。

因此可以写成：

> 低 MU 通常意味着模型 retain/utility 侧已经明显受损；但 MU 正常或中等并不能保证 forget-side 输出仍然正常。MU 更适合作为 retain 约束指标，而不是遗忘完成度指标。

相关性也支持这一点：

| Metric Pair | Pearson Corr |
| --- | ---: |
| `forget_quality` vs `forget_Q_A_Prob` | -0.0636 |
| `forget_quality` vs `forget_Q_A_ROUGE` | 0.1133 |
| `forget_quality` vs `forget_truth_ratio` | 0.4284 |
| `model_utility` vs `forget_Q_A_Prob` | 0.5034 |

FQ 与 forget prob/ROUGE 的相关性很弱，说明 FQ 不适合作为唯一遗忘强度指标。

## 问题 3：retain 集是否存在非同步受影响？

结论：**当前下载下来的文件还不能验证这个问题。**

HPC 目录中：

- `retain_item_rows.csv`: 0 行
- `retain_item_sensitivity.csv`: 0 行

这意味着本次分析没有拿到可用于 per-index retain 分析的 `retain_*` 的 `value_by_index` 细粒度日志。当前只能从 `model_utility` 这样的 aggregate 指标判断 retain 总体是否受损，不能回答“哪些 retain 样本更容易被影响，哪些更稳”。

要验证 retain 非同步影响，需要补充原始 eval log，例如：

- `TOFU_EVAL.json`
- `MUSE_EVAL.json`
- `MYTOFU_EVAL.json`

并且这些 JSON 中需要包含类似：

```json
"retain_Q_A_ROUGE": {
  "agg_value": 0.6,
  "value_by_index": {
    "0": {"rougeL": 0.7},
    "1": {"rougeL": 0.2}
  }
}
```

有了这类 `value_by_index` 后，脚本才能输出每个 retain index 的：

- 平均 retain 分数
- 最低 retain 分数
- 跨方法/模型方差
- 如果提供 reference eval，还能算 damage

## 最终建议

后续分析和论文表述中，建议把指标分成三类，而不是只看 FQ/MU：

1. **直接遗忘指标**
   - `forget_Q_A_Prob`
   - `forget_Q_A_ROUGE`
   - `forget_truth_ratio`
   - 用来判断模型在 forget set 上是否已经输出混乱或无法复现目标知识。

2. **分布检验指标**
   - `forget_quality`
   - 用来辅助说明 forget 分布是否接近 retain/reference 分布，但不能作为唯一遗忘完成标准。

3. **保留能力指标**
   - `model_utility`
   - `retain_Q_A_Prob`
   - `retain_Q_A_ROUGE`
   - `retain_truth_ratio`
   - 用来判断遗忘方法是否伤害 retain，但不应反推 forget-side 是否仍正常。

## 可直接使用的结论表述

这批 baseline 的一个关键发现是，`forget_quality` 与 forget-side 输出质量并不同步。以 `GradAscent` 和 `GradDiff` 为例，二者在所有 4 个 backbone 上都出现了 `forget_Q_A_Prob`, `forget_Q_A_ROUGE`, `forget_truth_ratio` 同时接近 0 的情况，说明模型对 forget set 的回答已经实质性崩坏；然而它们的 `forget_quality` 仍然极低。因此，高 FQ 不是判断遗忘是否发生的必要条件，尤其当研究问题关注“目标知识是否已经无法被模型稳定输出”时，forget-side 的直接生成指标更具有解释力。

另一方面，`model_utility` 可以反映 retain 侧是否整体损坏，但它不能单独解释 forget 侧状态。`GradAscent` 同时带来低 MU 和 forget 输出崩坏，说明它更像粗暴破坏；而 `GradDiff` 在 MU 中等的同时也让 forget 输出崩坏，说明存在“retain aggregate 尚可但 forget 已被破坏”的状态。因此后续评价应采用分层指标：用 forget-side direct metrics 判断遗忘是否发生，用 MU/retain metrics 判断副作用，用 FQ 作为分布层面的辅助指标。

## 文件索引

- 逐 run 标注：`summary_annotated.csv`
- 方法级汇总：`summary_by_method.csv`
- 模型-方法级汇总：`summary_by_model_method.csv`
- FQ 不高但输出已崩的关键案例：`collapsed_before_high_fq_cases.csv`
- 原自动报告：`baseline_metric_reliability_report.md`
