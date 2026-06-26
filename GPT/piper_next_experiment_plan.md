# 下一步实验方案：NPO + Set-gradient PI Local KL 正式干预实验

## 1. 实验目的

当前 PI 探针机制实验已经显示：PI 分数能够较好预测 NPO 遗忘后的 retain damage，且 set-gradient PI 整体优于 averaged PI，semantic baseline 明显弱于 PI。

因此，下一步实验不再继续扩展 PI 预测性分析，而应进入正式 PIPER 干预实验，验证：

```text
保护 PI 选出的 vulnerable retain samples，是否真的能改善 NPO 的 forget-retain trade-off。
```

也就是说，当前实验要完成从：

```text
PI 能预测 retain damage
```

推进到：

```text
PI-guided local preservation 能实际减少 retain damage
```

---

## 2. 核心实验问题

本轮实验需要回答四个问题：

1. **PI local KL 是否优于 random local KL？**  
   如果不能优于 random local KL，说明 PI 选样没有实际价值。

2. **PI local KL 是否优于 semantic local KL？**  
   如果不能优于 semantic local KL，说明小步探针没有比语义邻域带来额外贡献。

3. **PI local KL 是否能降低 vulnerable retain damage？**  
   PIPER 的目标不是平均保留所有 retain samples，而是重点保护最易受损的 retain samples。

4. **PI local KL 是否会明显牺牲 forget quality？**  
   如果 retain 变好只是因为模型忘得更少，则不能算成功。

---

## 3. 推荐实验配置

### 3.1 数据与框架

| 项目 | 设置 |
|---|---|
| Framework | OpenUnlearning |
| Dataset | TOFU |
| Main split | forget10 / retain90 |
| Model | Llama-2-7b-chat-hf TOFU full checkpoint |
| Backbone | NPO |
| Seeds | 0, 1, 2 |

### 3.2 PI 配置

根据当前机制实验结果，首轮正式 PIPER 建议使用：

```text
Static-PIPER with set-gradient PI
```

推荐配置：

| 参数 | 设置 |
|---|---|
| PI type | set-gradient PI |
| forget batch size | fb1 |
| probe batches | pb64 |
| coverage | 64 forget samples |
| vulnerable retain Top-K | retain candidates 的 10% |
| PI 使用方式 | 训练前静态计算 vulnerable retain samples |

首轮不建议使用 PI-8 或 Online-PI。PI-8 可作为低成本版本后续比较，Online-PI 成本较高，暂不作为第一轮主实验。

---

## 4. 方法对照组

本轮实验至少需要包含以下五组：

| 方法 | 目的 |
|---|---|
| NPO | 原始 backbone |
| NPO + global retain KL | 判断全局保留保护效果 |
| NPO + random local KL | 排除“只是多加 KL”带来的假提升 |
| NPO + semantic local KL | 检查语义邻域保护是否足够 |
| NPO + PI local KL | PIPER 最小正式版本 |

如果算力允许，可以额外加入：

| 方法 | 目的 |
|---|---|
| NPO + oracle damage local KL | 理论上界，只用于 appendix / analysis |

其中 oracle damage local KL 不能作为正式方法，只能作为“如果提前知道最终最受损 retain samples，上界能达到什么程度”的分析对照。

---

## 5. 正式 PIPER 训练目标

PIPER 最小正式版本为：

```text
L_PIPER = L_NPO + λ · L_localKL(V_PI)
```

其中：

```text
V_PI = PI Top-K vulnerable retain samples
```

`L_localKL` 建议定义为当前模型与遗忘前 reference model 在 vulnerable retain samples 上的 KL 距离：

```text
L_localKL = KL(p_ref(.|x_r) || p_theta(.|x_r)), x_r in V_PI
```

该项的作用是：在 NPO 遗忘过程中，尽量保持 vulnerable retain samples 上的输出分布不偏离原模型。

---

## 6. KL 权重扫描

首轮不需要大规模超参搜索，只建议扫描三个值：

```text
λ ∈ {0.1, 0.3, 1.0}
```

如果三者都不合适，再扩展到：

```text
λ ∈ {0.03, 0.1, 0.3, 1.0, 3.0}
```

实验分析时不能只选最优 λ 汇报，需要展示 forget-retain trade-off 曲线或至少说明不同 λ 下的变化趋势。

---

## 7. 必须报告的指标

### 7.1 Forget-side 指标

用于确认模型确实发生有效遗忘，而不是只是保留变好。

| 指标 | 目的 |
|---|---|
| Forget loss before / after | 确认 forget set 被影响 |
| Forget answer probability before / after | 检查目标答案概率是否下降 |
| TOFU Forget Quality | 与 TOFU 评测口径对齐 |
| Forget Truth Ratio | 检查模型是否偏离目标答案 |
| Per-sample forget change | 分析哪些 forget samples 真正被遗忘 |

### 7.2 Retain-side 指标

用于确认 retain utility 是否保持。

| 指标 | 目的 |
|---|---|
| Global retain loss increase | 全局 retain 损伤 |
| Retain answer probability | 保留答案概率 |
| Retain utility | 整体保留能力 |
| Model utility | 检查整体模型是否崩塌 |

### 7.3 Local damage 指标

这是 PIPER 的核心评估对象。

| 指标 | 目的 |
|---|---|
| PI Top-K retain damage | 检查 vulnerable retain samples 是否被保护 |
| Random Top-K retain damage | 随机局部保留下界 |
| Semantic Top-K retain damage | 语义邻域对照 |
| Oracle damage Top-K retain damage | 理论上界 |
| Vulnerable Neighbor Damage | 局部易损邻域平均损伤 |

### 7.4 成本指标

| 指标 | 目的 |
|---|---|
| PI 计算时间 | 评估探针成本 |
| 训练总时间 | 与 NPO baseline 对比 |
| 额外 forward 次数 | 分析计算复杂度 |
| 显存占用 | 判断可扩展性 |
| Top-K 大小 | 影响 local KL 成本 |

---

## 8. 成功判定标准

### 8.1 最低成功标准

本轮实验至少需要满足：

```text
NPO + PI local KL > NPO + random local KL
```

具体体现为：

- PI Top-K retain damage 更低；
- global retain utility 不下降或改善；
- forget quality 没有明显变差。

### 8.2 关键成功标准

更理想的是：

```text
NPO + PI local KL > NPO + semantic local KL
```

这说明 PI 捕捉到的是模型侧遗忘干扰，而不是普通语义相似度。

### 8.3 最强成功标准

如果进一步满足：

```text
NPO + PI local KL 接近 NPO + oracle damage local KL
```

说明 PI 选样已经接近理论上界，论文说服力会非常强。

---

## 9. 结果解释模板

### 情况 A：PI local KL 明显优于 random 和 semantic

可以写成：

```text
PI-selected local preservation substantially reduces vulnerable neighbor damage compared with random and semantic local preservation, while maintaining comparable forget quality.
```

中文解释：

```text
PI 选择的局部保留保护能够有效降低易损 retain 样本损伤，且不是简单由增加 KL 或语义邻域带来的。
```

### 情况 B：PI local KL 优于 random，但接近 semantic

说明 PI 有用，但增益主要来自局部邻域保护。需要进一步分析 PI 与 semantic 的差异，或者考虑在 TOFU 上语义相似已经足够强。

### 情况 C：PI local KL 与 random 差不多

说明 PIPER 主线暂时不成立。需要检查：

- PI Top-K 是否稳定；
- local KL 是否太弱；
- Top-K 是否过大或过小；
- λ 是否不合适；
- retain candidates 是否构造错误。

### 情况 D：retain 改善但 forget 明显变差

说明 PIPER 过度保留。需要降低 λ 或加入 forget-quality constraint。

---

## 10. 暂时不做的实验

本轮实验暂时不要做以下内容：

| 实验 | 原因 |
|---|---|
| APU-Bench | 当前主数据集仍是 TOFU |
| CV 泛化 | 主方法尚未验证完成 |
| MUSE / WMDP | TOFU 结果未闭环前不扩展 |
| Online-PI 大规模实验 | 成本较高，先验证 Static-PIPER |
| 多 backbone 全覆盖 | 先在 NPO 上完成闭环 |

---

## 11. 后续扩展顺序

如果本轮 NPO + PI local KL 成功，后续按以下顺序扩展：

1. **Static-PI vs Refresh-PI**  
   判断动态 refresh 是否进一步改善结果。

2. **forget01 / forget05 / forget10 多 split**  
   验证不同遗忘规模下是否稳定。

3. **多 backbone 扩展**  
   至少加入 GradAscent / GradDiff / SimNPO。

4. **oracle damage local KL 分析**  
   给出 PI 与 oracle 上界之间的差距。

5. **成本分析**  
   报告 PI 计算开销、训练时间和显存变化。

---

## 12. 本轮实验一句话总结

本轮实验的目标是完成 PIPER 的核心闭环：

```text
PI 能预测 retain damage
→ PI 选出的 retain samples 更值得保护
→ PI local KL 能实际降低 vulnerable retain damage
→ 且不会明显牺牲 forget quality
```

只有完成这一步，PIPER 才能从“探针机制分析”升级为“会议论文主方法”。
