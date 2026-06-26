# PIPER PI 探针探究实验：必须补充的四类实验

本文档整理当前 PI 探针预测性实验后，论文级别必须补充的四类实验。目标是把现有结果从“初步机制信号”完善为可支撑会议投稿的机制验证与方法有效性证据链。

---

## 1. Set-level PI 对照实验

### 目的

当前实验主要基于少量 forget batch 或小批量 probe 计算 PI 分数。正式 Static-PIPER 更接近“对整个 forget set 或较大 forget subset 做一次小步虚拟更新”，因此需要验证当前 small-probe PI 是否能近似 set-level PI。

### 需要补充的实验设置

| 实验版本 | 含义 | 目的 |
|---|---|---|
| PI-8 | 使用 8 个 forget 样本或 8 个 forget batch 计算 PI | 当前低成本版本 |
| PI-16 / PI-32 / PI-64 / PI-128 | 扩大 probe 覆盖的 forget 样本数量 | 检查 probe coverage 是否影响预测性 |
| Set-gradient PI | 先聚合多个 forget batch 的梯度，再做一次虚拟更新 | 更接近整个 forget set 的整体遗忘方向 |
| Current averaged PI vs Set-gradient PI | 比较当前平均式 PI 和 set-level PI | 判断当前做法是否只是近似，或已经足够稳定 |

### 建议报告指标

- Pearson correlation
- Spearman correlation
- Top-5% / Top-10% precision
- Enrichment over random
- PI Top-K 与 set-gradient PI Top-K 的 overlap
- 运行时间与额外显存

### 期望结论

理想结果是：

```text
PI-small ≈ PI-set
```

或者至少：

```text
PI-set > PI-small
```

如果 PI-small 已经接近 PI-set，可以将当前方法解释为低成本有效近似；如果 PI-set 明显更强，正式 Static-PIPER 应采用 set-level 或 large-subset PI。

---

## 2. Semantic Top-K Baseline 对照实验

### 目的

当前实验已经与 random baseline 做了比较，但还不足以证明 PI 的必要性。审稿人会自然质疑：为什么不直接用 embedding 相似度或语义邻域选择 retain samples？

因此必须比较 PI Top-K 与语义相似 Top-K。

### 需要比较的 Top-K 选择方式

| 选择方式 | 含义 | 作用 |
|---|---|---|
| Random Top-K | 随机选择 retain samples | 随机下界 |
| Semantic Top-K | 根据 embedding / 语义相似度选择 retain samples | 检验是否只需语义邻域 |
| PI Top-K | 根据 PI 分数选择 retain samples | 本文探针方法 |
| Oracle Damage Top-K | 根据完整遗忘后的真实 damage 选择 retain samples | 理论上界，只用于分析 |

### 需要比较的指标

对每种 Top-K 选择方式，计算：

```text
Mean retain damage of selected samples
Top-K precision against true damage Top-K
Enrichment over random
Overlap with oracle damage Top-K
```

### 期望结论

理想排序是：

```text
Damage(PI Top-K) > Damage(Semantic Top-K) > Damage(Random Top-K)
```

如果 PI Top-K 明显优于 Semantic Top-K，才能证明 PI 不是复杂版语义检索，而是捕捉了模型侧遗忘干扰。

---

## 3. PIPER 正式干预实验

### 目的

当前实验只能证明 PI 能预测 retain damage，但还不能证明 PIPER 方法本身有效。下一步必须验证：保护 PI 选出的 vulnerable retain samples，是否真的能改善 forget-retain trade-off。

### 最小干预实验设置

以 NPO 作为第一个 backbone，比较以下方法：

| 方法 | 目的 |
|---|---|
| NPO | 原始 backbone |
| NPO + global retain KL | 全局保留保护 |
| NPO + random local KL | 排除“多加 KL”带来的假提升 |
| NPO + semantic local KL | 检查语义邻域保护是否足够 |
| NPO + PI local KL | PIPER 最小正式版本 |

### 正式 PIPER 目标

```text
L_PIPER = L_NPO + λ · L_localKL(V_PI)
```

其中：

```text
V_PI = PI Top-K vulnerable retain samples
```

### 必须报告的指标

- Forget Quality / Forget Truth Ratio
- Forget answer probability
- Retain Utility
- Retain answer probability
- Local vulnerable neighbor damage
- Global retain damage
- Forget-retain Pareto curve
- Runtime / extra forward / extra backward

### 期望结论

至少需要证明：

```text
NPO + PI local KL > NPO + random local KL
```

最好进一步证明：

```text
NPO + PI local KL > NPO + semantic local KL
```

否则 PIPER 不能作为方法贡献，只能作为探针分析。

---

## 4. Forget-side 指标补齐

### 目的

当前探究实验主要关注 PI 与 retain damage 的关系。但如果不报告 forget-side 结果，审稿人无法判断 retain damage 是在有效遗忘下产生的，还是模型训练崩坏或训练不足导致的副作用。

因此必须补充 forget-side 指标。

### 需要补充的指标

| 指标 | 目的 |
|---|---|
| Forget loss before / after | 确认 forget set 被训练影响 |
| Forget answer probability before / after | 检查目标答案概率是否下降 |
| TOFU Forget Quality | 与 TOFU 官方评测口径对齐 |
| Forget Truth Ratio | 检查模型是否偏离真实目标答案 |
| Retain Utility | 避免只看局部 damage |
| Model Utility | 检查整体模型是否崩塌 |
| Train loss curve | 判断训练是否稳定 |
| Per-sample forget change | 分析哪些 forget samples 真正被遗忘 |

### 需要回答的问题

该实验要回答：

```text
PI 预测到的 retain damage 是否发生在一个有效的 unlearning 过程之中？
```

不能只展示 retain damage 的相关性。

### 期望结论

理想情况是：

```text
Forget-side unlearning 有效
Retain-side 局部 damage 可被 PI 预测
PIPER local KL 能降低局部 damage
且不会明显牺牲 forget quality
```

---

# 最小执行顺序

建议按以下顺序补充：

1. **Set-level PI 对照实验**
2. **Semantic Top-K baseline**
3. **Forget-side 指标补齐**
4. **NPO + PI local KL 正式干预实验**

其中，第 1、2、3 项用于完善 PI 探针机制验证；第 4 项用于把探针机制推进为 PIPER 方法实验。

---

# 最终目标

补完以上四类实验后，论文可以形成完整证据链：

```text
PI 能预测 retain damage
PI 优于 random 和 semantic neighbor
PI 在 set-level 或 large-subset 设置下仍稳定
PI-selected local protection 能实际改善 unlearning
```

这时，当前探究实验就可以从“附录初步结果”提升为论文主文中的核心机制分析。
