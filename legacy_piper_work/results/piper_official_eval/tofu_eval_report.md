# PIPER 官方 TOFU Evaluation 汇总报告

生成时间：2026-06-29  
数据来源：`results/piper_official_eval/tofu_eval_summary.csv`  
关联 proxy 汇总：`results/piper_intervention_expanded/intervention_summary.csv`

## 1. 数据完整性

本次官方 TOFU evaluation 汇总共包含 285 条结果。原设计规模为：

```text
4 backbones × 3 seeds × (1 baseline + 4 KL methods × 7 lambdas)
= 348 runs
```

实际完成情况如下：

| backbone | 已汇总 run 数 | 完整性 |
|---|---:|---|
| NPO | 87 | 完整 |
| GradAscent | 87 | 完整 |
| GradDiff | 87 | 完整 |
| SimNPO | 24 | 不完整 |

因此，当前报告的主分析只能基于 `NPO / GradAscent / GradDiff` 三个完整 backbone。`SimNPO` 只能作为探索性结果，不能用于多 backbone 稳定性结论。

此外，当前 CSV 中没有 `official_forget_quality` 字段，`official_privleak` 大量为约 `-99` 的异常/占位数值。因此本报告不使用 `privleak` 作为结论依据，也不声称已经完成完整 TOFU Forget Quality 分析。

## 2. 指标解释

本报告主要使用以下字段：

| 指标 | 解释 | 方向 |
|---|---|---|
| `official_model_utility` | 官方模型效用 | 越高越好 |
| `official_forget_Q_A_Prob` | forget 问答答案概率 | 通常越低表示遗忘越强 |
| `official_forget_Q_A_ROUGE` | forget 生成答案与原答案相似度 | 通常越低表示遗忘越强 |
| `official_forget_truth_ratio` | forget truth ratio | 需结合 TOFU 定义解释 |
| `retain_all_mean_damage` | proxy 全体 retain damage | 越低越好 |
| `local_pi_topk_mean_damage` | proxy PI Top-K retain damage | 越低越好 |
| `forget_mean_damage` | proxy forget damage | 越高通常表示 proxy 遗忘越强 |

需要注意：官方 forget 指标与 proxy forget damage 不总是一致。尤其在不同 backbone 之间直接比较时，proxy loss 变化不能替代官方 TOFU 指标。

## 3. Baseline 对比

各 backbone 的 baseline 均值如下：

| backbone | official model utility | official forget Q/A Prob | official forget Q/A ROUGE | retain damage | PI Top-K damage | proxy forget damage |
|---|---:|---:|---:|---:|---:|---:|
| GradAscent | 0.6293 | 0.9900 | 0.9761 | 0.1045 | 0.2518 | 0.1024 |
| GradDiff | 0.6272 | 0.9900 | 0.9776 | -0.0169 | -0.0470 | -0.0083 |
| NPO | 0.6154 | 0.8266 | 0.7795 | 1.5561 | 2.9735 | 1.5264 |
| SimNPO | 0.6143 | 0.7684 | 0.7441 | 1.4328 | 1.8392 | 1.4653 |

这个表暴露出一个关键问题：`GradAscent` 和 `GradDiff` 在当前超参数下官方 forget Q/A probability 接近 0.99，Q/A ROUGE 也接近 0.98，说明它们几乎没有产生有效遗忘。因此，尽管这两个 backbone 的实验是完整的，但它们对“PIPER 是否改善有效遗忘下的 retain 保护”贡献有限。

当前最有分析价值的 backbone 是 `NPO`。`SimNPO` 的 baseline 看起来也有较强遗忘，但该 backbone 只完成了 24 条结果，不能作为主结论。

## 4. NPO 主结果

NPO 是目前唯一同时满足“完整 3 seeds”和“官方 forget 指标确实发生明显变化”的主 backbone。

### 4.1 NPO 下各方法均值

| method | lambda | model utility | forget Q/A Prob | forget Q/A ROUGE | retain damage | PI Top-K damage | proxy forget damage |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.00 | 0.6154 | 0.8266 | 0.7795 | 1.5561 | 2.9735 | 1.5264 |
| global retain KL | 0.30 | 0.6170 | 0.8398 | 0.7913 | 1.4683 | 2.8441 | 1.4561 |
| global retain KL | 1.00 | 0.6178 | 0.8580 | 0.8149 | 1.3346 | 2.6366 | 1.3376 |
| random local KL | 0.30 | 0.6172 | 0.8429 | 0.7964 | 1.4887 | 2.9119 | 1.4687 |
| random local KL | 1.00 | 0.6183 | 0.8586 | 0.8137 | 1.3526 | 2.6618 | 1.3526 |
| semantic local KL | 0.30 | 0.6164 | 0.8365 | 0.7875 | 1.4752 | 2.8178 | 1.4592 |
| semantic local KL | 1.00 | 0.6183 | 0.8551 | 0.8073 | 1.3547 | 2.5949 | 1.3587 |
| PI local KL | 0.30 | 0.6156 | 0.8243 | 0.7786 | 1.4458 | 2.6262 | 1.4340 |
| PI local KL | 0.70 | 0.6161 | 0.8262 | 0.7823 | 1.4430 | 2.6187 | 1.4346 |
| PI local KL | 1.00 | 0.6161 | 0.8268 | 0.7847 | 1.4386 | 2.6250 | 1.4265 |

### 4.2 NPO 下相对 baseline 的变化

| method | lambda | model utility Δ | forget Q/A Prob Δ | forget Q/A ROUGE Δ | retain damage Δ | PI Top-K damage Δ |
|---|---:|---:|---:|---:|---:|---:|
| PI local KL | 0.30 | +0.0002 | -0.0023 | -0.0009 | -0.1102 | -0.3473 |
| PI local KL | 0.70 | +0.0007 | -0.0004 | +0.0028 | -0.1131 | -0.3548 |
| PI local KL | 1.00 | +0.0007 | +0.0002 | +0.0052 | -0.1175 | -0.3485 |
| semantic local KL | 1.00 | +0.0029 | +0.0284 | +0.0278 | -0.2014 | -0.3787 |
| random local KL | 1.00 | +0.0030 | +0.0320 | +0.0341 | -0.2034 | -0.3117 |
| global retain KL | 1.00 | +0.0024 | +0.0313 | +0.0353 | -0.2215 | -0.3369 |

解释：

- `PI local KL` 在 `lambda=0.3/0.7/1.0` 下均明显降低 PI Top-K damage。
- 与 `random / semantic / global` 的高 lambda 设置相比，`PI local KL` 对官方 forget Q/A probability 和 ROUGE 的影响更小。
- `random / semantic / global` 在 `lambda=1.0` 下能得到更高的 model utility 和更低的 retain damage，但同时 official forget Q/A Prob 与 ROUGE 明显上升，说明这些方法更可能是通过削弱遗忘来换取 retain 改善。

这对 PIPER 是一个较有利的信号：在 NPO 下，`PI local KL` 更像是在保护 PI-identified vulnerable retain samples，而不是简单地让模型少忘一点。

## 5. PI local KL 与其他 local KL 的直接比较

在 NPO 中，同一 lambda 下比较 `PI local KL` 与其他方法：

| lambda | 对照方法 | model utility Δ | PI Top-K damage Δ | retain damage Δ | forget Q/A Prob Δ |
|---:|---|---:|---:|---:|---:|
| 0.10 | random local KL | +0.0001 | -0.0821 | -0.0117 | -0.0041 |
| 0.10 | semantic local KL | +0.0001 | -0.0635 | -0.0111 | -0.0035 |
| 0.30 | random local KL | -0.0016 | -0.2857 | -0.0429 | -0.0186 |
| 0.30 | semantic local KL | -0.0008 | -0.1916 | -0.0294 | -0.0121 |
| 0.70 | random local KL | -0.0014 | -0.1089 | +0.0545 | -0.0252 |
| 0.70 | semantic local KL | -0.0011 | -0.0393 | +0.0448 | -0.0192 |
| 1.00 | random local KL | -0.0022 | -0.0368 | +0.0859 | -0.0318 |
| 1.00 | semantic local KL | -0.0022 | +0.0302 | +0.0839 | -0.0282 |

其中 Δ 表示：

```text
PI local KL - 对照方法
```

因此：

- PI Top-K damage Δ 为负：PI local KL 对 PI vulnerable set 保护更好。
- retain damage Δ 为正：PI local KL 全局 retain damage 更差。
- forget Q/A Prob Δ 为负：PI local KL 保持了更强的官方 forget 行为。

结论：`PI local KL` 的优势集中在 PI Top-K vulnerable set 和 forget-side 保持上；它并不稳定提升全局 model utility，也不稳定降低全局 retain damage。

## 6. 其他 backbone 的解释边界

### 6.1 GradAscent

`GradAscent` 的 baseline 官方 forget Q/A Prob 为 0.9900，ROUGE 为 0.9761。所有 KL 方法和 lambda 的变化都很小。

这说明当前 `GradAscent` 设置下几乎没有发生有效遗忘。因此，即使 PI local KL 与其他方法数值接近，也不能说明 PIPER 在 GradAscent 上有效。

### 6.2 GradDiff

`GradDiff` 的 baseline 官方 forget Q/A Prob 为 0.9900，ROUGE 为 0.9776，proxy forget damage 甚至为负。这个 backbone 当前设置同样不适合作为有效遗忘主证据。

### 6.3 SimNPO

`SimNPO` 只有 24 条结果，主要是 seed 0 的不完整结果。它的 baseline 官方 forget Q/A Prob 为 0.7684，具有分析价值，但因为缺少 seed 1/2 和大量 PI local KL 配置，不能进入主结论。

探索性观察：

| method | lambda | model utility | forget Q/A Prob | retain damage | PI Top-K damage |
|---|---:|---:|---:|---:|---:|
| SimNPO baseline | 0.00 | 0.6143 | 0.7684 | 1.4328 | 1.8392 |
| SimNPO PI local KL | 0.03 | 0.6340 | 0.8968 | 1.0261 | 1.3975 |
| SimNPO PI local KL | 0.10 | 0.6348 | 0.9512 | 0.6860 | 1.0103 |
| SimNPO semantic local KL | 0.70 | 0.6363 | 0.9618 | 0.2922 | 0.4407 |
| SimNPO semantic local KL | 1.00 | 0.6383 | 0.9672 | 0.3797 | 0.5504 |

这些结果显示 KL 明显提升 utility 并降低 retain damage，但也显著提高 forget Q/A Prob，遗忘变弱非常明显。因此它不能直接支持 PIPER 改善 trade-off。

## 7. Proxy 与官方指标的一致性

整体相关性：

| 范围 | retain damage vs model utility | proxy forget damage vs official forget Q/A Prob | proxy forget damage vs official Q/A ROUGE |
|---|---:|---:|---:|
| all runs | -0.8227 | -0.9461 | -0.9798 |
| NPO | -0.6432 | +0.5249 | +0.1373 |
| GradAscent | +0.0631 | -0.7634 | +0.3839 |
| GradDiff | +0.1474 | -0.3740 | +0.7170 |
| SimNPO | -0.4894 | -0.7623 | -0.7184 |

跨 backbone 的整体相关性很高，但 backbone 内相关性不稳定。这说明 proxy 指标不能直接替代官方 TOFU evaluation。论文中应把 proxy 指标作为机制分析，把官方 TOFU 指标作为主实验验证。

## 8. 当前可以支持的结论

当前官方结果支持以下有限结论：

1. **NPO 是当前最有效的主 backbone。**  
   在本轮完整结果中，只有 NPO 同时表现出较明显的官方 forget 行为和完整 3 seeds。

2. **NPO 下 PI local KL 能有效降低 PI Top-K retain damage。**  
   例如 `lambda=0.3` 时，PI Top-K damage 从 baseline 的 2.9735 降到 2.6262。

3. **PI local KL 相比 random/semantic/global KL 更少削弱官方 forget 行为。**  
   高 lambda 的 random/semantic/global KL 虽然提升 model utility、降低全局 retain damage，但 official forget Q/A Prob 和 ROUGE 明显上升。PI local KL 的官方 forget Q/A Prob 基本维持在 baseline 附近。

4. **当前结果不支持“PIPER 全局 utility 最优”。**  
   在 NPO 下，最高 model utility 来自 `random_local_kl lambda=1.0` 和 `semantic_local_kl lambda=1.0`，不是 PI local KL。

5. **当前结果不支持“多 backbone 稳定有效”。**  
   GradAscent/GradDiff 当前几乎没有有效遗忘，SimNPO 不完整。

## 9. 不能写成论文结论的内容

当前不能写：

```text
PIPER 在所有 backbone 上稳定优于 random / semantic / global KL。
```

不能写：

```text
PIPER 提高了整体 model utility。
```

不能写：

```text
PIPER 已经严格改善 forget-retain Pareto trade-off。
```

也不能写：

```text
官方 TOFU Forget Quality 已经支持 PIPER。
```

因为当前汇总中没有可用的 `forget_quality` 字段。

## 10. 下一步建议

按优先级：

1. **补齐 SimNPO 的缺失结果。**  
   当前 SimNPO 只有 24/87 条，不能用于主结论。

2. **重新调 GradAscent 和 GradDiff 的遗忘强度。**  
   当前官方 forget Q/A Prob 接近 0.99，说明这两个 backbone 没有形成有效遗忘。需要提高遗忘强度或调整训练步数/学习率，否则多 backbone 实验只是形式完整。

3. **做 NPO 上的 forget-quality matched 比较。**  
   重点比较：
   ```text
   PI local KL lambda=0.3/0.7/1.0
   vs random / semantic / global 在相近 official forget Q/A Prob 或 ROUGE 下的 retain damage
   ```

4. **加入 gradient-similarity local KL baseline。**  
   目前官方结果显示 PI 的优势主要是“保护 PI Top-K 且较少削弱 forget”。但还不能排除一阶梯度相似度方法也能做到类似效果。

5. **确认 retain logs / forget_quality 输出。**  
   如果论文要使用 TOFU 标准指标，必须解决 `forget_quality` 缺失问题，或者在论文中明确说明该指标不可用并解释原因。

## 11. 可用论文表述

较稳妥的当前表述：

```text
On the NPO backbone, PI-guided local preservation substantially reduces damage on PI-identified vulnerable retain samples while keeping official forget Q/A probability and ROUGE close to the no-preservation baseline. In contrast, random, semantic, and global KL baselines achieve larger global utility gains mainly at higher KL weights, where official forget-side metrics indicate weaker forgetting.
```

中文解释：

```text
在 NPO 上，PI local KL 的主要价值不是提升整体 model utility，而是在不明显削弱官方 forget 指标的情况下，降低 PI 识别出的易损 retain 样本损伤。这个结果支持 PIPER 作为局部易损样本保护机制，但尚不足以证明其全局 trade-off 最优。
```

