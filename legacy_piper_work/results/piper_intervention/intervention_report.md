# PIPER Local KL 干预实验汇总报告

生成时间：2026-06-26  
数据来源：`results/piper_intervention/intervention_summary.csv`  
实验规模：39 个 run，覆盖 3 个 seed、1 个 NPO baseline、4 类 KL 干预方法与 3 个 KL 权重。

## 1. 实验问题

本轮实验的核心问题不是继续验证 PI 分数是否能预测 retain damage，而是检验：

```text
在 NPO 遗忘训练中，对 PI 选出的 vulnerable retain samples 加 local KL，
是否比 random local KL 或 semantic local KL 更能降低 retain damage，
并且不显著削弱 forget-side unlearning。
```

从本科毕业论文评审标准看，这一步是必要的。前一阶段的 PI 探针实验只能支持“PI 与后续 damage 存在相关性”，不能直接推出“PI-guided preservation 是有效方法”。本轮实验正是在补这个因果链条。

## 2. 实验完整性检查

CSV 中包含 39 行，符合预期：

| 组成 | 数量 |
|---|---:|
| baseline | 3 seeds |
| global retain KL | 3 lambdas × 3 seeds |
| random local KL | 3 lambdas × 3 seeds |
| semantic local KL | 3 lambdas × 3 seeds |
| PI local KL | 3 lambdas × 3 seeds |

主要设置：

| 项目 | 设置 |
|---|---|
| backbone | NPO |
| split | forget10 / retain90 |
| model | Llama-2-7b-chat-hf |
| retain candidates | 500 |
| Top-K fraction | 10%，即 50 个 retain samples |
| PI type | set-gradient PI |
| probe batches | 64 |
| forget batch size | 1 |
| train steps | 80 |
| KL weights | 0.1, 0.3, 1.0 |

## 3. 总体均值结果

表中 damage 越低表示 retain 受损越小；forget damage 越高、forget answer probability delta 越负，通常表示遗忘更强。

| method | lambda | all retain damage | PI Top-K damage | forget damage | forget prob delta |
|---|---:|---:|---:|---:|---:|
| baseline | 0.0 | 1.556 | 2.974 | 1.526 | -0.479 |
| global retain KL | 0.1 | 1.514 | 2.907 | 1.489 | -0.474 |
| global retain KL | 0.3 | 1.468 | 2.844 | 1.456 | -0.469 |
| global retain KL | 1.0 | 1.335 | 2.637 | 1.338 | -0.452 |
| random local KL | 0.1 | 1.529 | 2.932 | 1.505 | -0.476 |
| random local KL | 0.3 | 1.489 | 2.912 | 1.469 | -0.470 |
| random local KL | 1.0 | 1.353 | 2.662 | 1.353 | -0.454 |
| semantic local KL | 0.1 | 1.528 | 2.913 | 1.501 | -0.476 |
| semantic local KL | 0.3 | 1.475 | 2.818 | 1.459 | -0.470 |
| semantic local KL | 1.0 | 1.355 | 2.595 | 1.359 | -0.455 |
| PI local KL | 0.1 | 1.517 | 2.850 | 1.490 | -0.474 |
| PI local KL | 0.3 | 1.446 | 2.626 | 1.434 | -0.468 |
| PI local KL | 1.0 | 1.439 | 2.625 | 1.427 | -0.466 |

## 4. 相对 baseline 的配对改善

下表是同一 seed 下相对 baseline 的均值差异。负值表示 damage 下降，即 retain 更好；但 forget damage 的负值也意味着遗忘强度下降。

| method | lambda | all retain damage Δ | PI Top-K damage Δ | forget damage Δ |
|---|---:|---:|---:|---:|
| global retain KL | 0.1 | -0.042 | -0.067 | -0.037 |
| global retain KL | 0.3 | -0.088 | -0.129 | -0.070 |
| global retain KL | 1.0 | -0.222 | -0.337 | -0.189 |
| random local KL | 0.1 | -0.027 | -0.042 | -0.022 |
| random local KL | 0.3 | -0.067 | -0.062 | -0.058 |
| random local KL | 1.0 | -0.203 | -0.312 | -0.174 |
| semantic local KL | 0.1 | -0.028 | -0.060 | -0.026 |
| semantic local KL | 0.3 | -0.081 | -0.156 | -0.067 |
| semantic local KL | 1.0 | -0.201 | -0.379 | -0.168 |
| PI local KL | 0.1 | -0.039 | -0.124 | -0.036 |
| PI local KL | 0.3 | -0.110 | -0.347 | -0.092 |
| PI local KL | 1.0 | -0.117 | -0.348 | -0.100 |

所有 KL 方法在 3 个 seed 上都降低了 all retain damage 和 PI Top-K damage。这说明 KL preservation 本身是有效的。但是所有 KL 方法也都降低了 forget damage，说明 retain 改善并不是免费的，至少部分来自遗忘强度减弱。

## 5. PI local KL 是否优于 random / semantic

### 5.1 对 PI Top-K damage 的比较

这是 PIPER 最关键的指标，因为方法声称要保护 PI 识别出的 vulnerable retain samples。

| lambda | PI vs random: mean diff | PI 更优 seed 数 | PI vs semantic: mean diff | PI 更优 seed 数 |
|---:|---:|---:|---:|---:|
| 0.1 | -0.082 | 2 / 3 | -0.063 | 2 / 3 |
| 0.3 | -0.286 | 3 / 3 | -0.192 | 3 / 3 |
| 1.0 | -0.037 | 2 / 3 | +0.030 | 2 / 3 |

解释：

- `lambda=0.3` 是目前最支持 PIPER 叙事的设置：PI local KL 在 3 个 seed 上都低于 random 和 semantic 的 PI Top-K damage。
- `lambda=1.0` 下，semantic local KL 的平均 PI Top-K damage 反而低于 PI local KL，虽然 PI 在 2/3 个 seed 上逐 seed 更优。这说明均值受 seed 1 的 semantic 强表现影响较大。
- `lambda=0.1` 下 PI 有优势，但优势不稳定，不足以作为主结论。

### 5.2 对 all retain damage 的比较

PI local KL 对全局 retain damage 并不占优：

| lambda | PI - random all retain damage | PI 更优 seed 数 | PI - semantic all retain damage | PI 更优 seed 数 |
|---:|---:|---:|---:|---:|
| 0.1 | -0.012 | 2 / 3 | -0.011 | 2 / 3 |
| 0.3 | -0.043 | 1 / 3 | -0.029 | 1 / 3 |
| 1.0 | +0.086 | 0 / 3 | +0.084 | 0 / 3 |

这意味着当前证据只能支持“PI local KL 可能更有针对性地保护 PI Top-K vulnerable set”，不能支持“PI local KL 全局 retain utility 更好”。

## 6. Forget-side 代价

所有 KL 方法都降低了 forget damage。以均值看：

| method | best-looking lambda | forget damage | 相对 baseline |
|---|---:|---:|---:|
| baseline | 0.0 | 1.526 | 0.000 |
| PI local KL | 0.3 | 1.434 | -0.092 |
| PI local KL | 1.0 | 1.427 | -0.100 |
| semantic local KL | 1.0 | 1.359 | -0.168 |
| random local KL | 1.0 | 1.353 | -0.174 |
| global retain KL | 1.0 | 1.338 | -0.189 |

这带来一个证据链风险：retain damage 降低可能部分来自“模型遗忘得更弱”，而不是 PI 选择机制更优。当前结果不能直接写成“PIPER 改善 forget-retain trade-off”，只能写成“在轻量干预代理指标上降低 retain damage，但伴随一定 forget-side 减弱，需要进一步 trade-off 控制”。

## 7. 成本分析

| method | lambda | elapsed seconds | train seconds | ref KL precompute seconds |
|---|---:|---:|---:|---:|
| baseline | 0.0 | 192.9 | 40.5 | 0.0 |
| PI local KL | 0.3 | 177.5 | 50.1 | 1.7 |
| PI local KL | 1.0 | 188.8 | 53.6 | 1.8 |
| random local KL | 1.0 | 192.4 | 55.0 | 2.0 |
| semantic local KL | 1.0 | 178.9 | 50.3 | 1.8 |
| global retain KL | 1.0 | 206.3 | 54.4 | 18.9 |

注意：baseline 的 elapsed time 也包含 PI 和 semantic selector 的计算，因为脚本为了统一评估 local damage，在 baseline run 中也计算了 selector。因此比较训练阶段成本时，应主要看 `train seconds`；比较完整方法部署成本时，才看 `elapsed seconds`。

local KL 的 reference log-prob 预计算只覆盖 50 个 Top-K 样本，约 1.7-2.0 秒；global retain KL 覆盖 500 个 retain candidates，约 18.9 秒。global retain KL 不是严格成本公平对照，它更像“更宽保护范围”的参考线。

## 8. 评审结论

当前结果支持以下有限结论：

1. KL preservation 对 retain damage 有稳定保护效应。  
   所有 KL 方法在 3 个 seed 上均降低 all retain damage 和 PI Top-K damage。

2. PI local KL 在 `lambda=0.3` 下对 PI Top-K vulnerable set 的保护效果较好。  
   它在 3/3 seeds 上优于 random local KL 和 semantic local KL，PI Top-K damage 均值从 baseline 的 2.974 降到 2.626。

3. 证据尚不足以声称 PI local KL 整体优于 semantic/global KL。  
   `lambda=1.0` 下 semantic local KL 的 PI Top-K damage 均值为 2.595，低于 PI local KL 的 2.625；global retain KL 的全局 retain damage 也明显低于 PI local KL。

4. forget-retain trade-off 尚未被严格识别。  
   所有 KL 方法都降低 forget damage，因此 retain 改善可能混入了“遗忘变弱”的替代解释。

因此，当前实验不能作为论文中“PIPER 已经显著优于语义邻域保护”的最终证据。更稳妥的表述应是：

```text
Preliminary intervention results suggest that PI-selected local preservation can reduce damage on PI-identified vulnerable retain samples, especially under moderate KL regularization. However, the current evidence does not yet establish a robust advantage over semantic or global KL baselines under matched forget quality.
```

## 9. 下一步必须补的实验

按优先级排序：

1. **做 forget-quality matched 比较。**  
   不能直接比较同一 lambda 下的 retain damage。应选择或插值出 forget damage 接近 baseline / 接近彼此的配置，再比较 retain damage。否则“retain 更好”可能只是“忘得更少”。

2. **围绕 PI local KL 扫更细的 lambda。**  
   当前 PI 在 0.3 和 1.0 之间几乎没有继续降低 PI Top-K damage，但 forget 继续变弱。建议补：
   ```text
   lambda ∈ {0.03, 0.1, 0.2, 0.3, 0.5, 0.7}
   ```
   重点看 0.2-0.5 区间。

3. **对 semantic local KL 做同样细粒度 lambda。**  
   目前 semantic 在 `lambda=1.0` 很强，不能被轻易排除。若论文要声称 PI 捕捉的是模型侧干扰而非语义邻近性，必须在 matched forget quality 下压过 semantic。

4. **接入官方 TOFU evaluation。**  
   当前指标是训练脚本内的 answer-token loss/prob proxy。论文主实验至少需要报告 Forget Quality、Truth Ratio、Retain Utility、Model Utility，否则只能算机制实验。

5. **增加 seed 或置信区间。**  
   现在只有 3 个 seed，不能进行强统计推断。至少报告 paired mean ± std；若计算资源允许，扩到 5 seeds。

6. **明确 global KL 的对照地位。**  
   global retain KL 保护 500 个候选样本，local KL 保护 50 个样本，因此不是同等保护范围的公平对照。论文中应把它称作 broad retain-preservation baseline，而不是 local selector baseline。

## 10. 可用于论文的当前叙事边界

可以写：

```text
在轻量级干预实验中，PI local KL 在中等 KL 权重下显著降低了 PI Top-K vulnerable retain samples 的 loss increase，并在该指标上优于 random 与 semantic local KL。
```

不能写：

```text
PIPER 已经证明优于 semantic preservation。
PIPER 改善了 forget-retain trade-off。
PIPER 在 TOFU 上取得更好的 forgetting quality 和 model utility。
```

这些结论当前证据尚不支持。
