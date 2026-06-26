# 当前论文方案

## 1. 论文大致定位

更准确的投稿叙事是：

> **本文提出 PIPER，一种用于缓解 LLM 遗忘过程中局部保留知识误伤的干扰探针框架。**

当前主数据集选择 TOFU。

在 APU-Bench 尚未确定前，论文暂时不宜把主问题强行写成“属性级用户画像遗忘”。更稳妥的定位是：

> **TOFU 风格事实型 LLM unlearning 中的局部保留知识干扰问题**

如果后续 APU-Bench 成立，再把问题扩展为：

> **attribute-level profile unlearning**

否则会出现“证据—结论不匹配”：主实验是 TOFU 作者级/事实级遗忘，但论文却声称解决属性级画像遗忘。

---

## 2. 当前核心研究问题

建议当前研究问题表述为：

> **在 LLM unlearning 中，如何识别并保护最容易被遗忘更新误伤的保留样本？**

更具体地说：

> 现有 LLM unlearning 方法通常关注 forget set 上的遗忘效果和 retain set 上的整体效用，但 retain set 内部并非均质。部分保留样本与遗忘目标存在更强的参数更新冲突，因此更容易在遗忘过程中被误伤。本文研究如何通过局部干扰探针识别这些 vulnerable retain samples，并在训练过程中对其进行选择性保护。

这个问题比“属性级用户画像遗忘”更适合当前 TOFU 主数据集，也更容易和 OpenUnlearning 框架对齐。

---

## 3. 方法主线：PIPER

方法名称暂定：

> **PIPER**  
> Probing Interference for Preservation-aware Erasure

或：

> **Probed Interference-based Preservation for Efficient Retention**

名称可以后续再定，但核心方法要固定为：

> **Unlearning Backbone + Probed Interference + Local Preservation**

即 PIPER 不是替代所有遗忘方法的新 backbone，而是一个可插拔的 wrapper。

---

## 4. 基本方法形式

给定遗忘 batch：

> **B<sub>f</sub> ⊂ 𝒟<sub>f</sub>**

候选保留样本：

> **x<sub>r</sub> ∈ 𝒟<sub>r</sub>**

先计算遗忘方向：

> **g<sub>f</sub> = ∇<sub>θ</sub> 𝓛<sub>unlearn</sub>(B<sub>f</sub>; θ)**

然后进行虚拟小步更新：

> **θ′ = θ − η<sub>p</sub> g<sub>f</sub>**

再计算候选保留样本在更新前后的损失变化：

> **PI(B<sub>f</sub>, x<sub>r</sub>) = 𝓛<sub>retain</sub>(x<sub>r</sub>; θ′) − 𝓛<sub>retain</sub>(x<sub>r</sub>; θ)**

如果该值较大，说明 x<sub>r</sub> 对当前遗忘更新更脆弱。

选出 Top-K：

> **𝒱<sub>B_f</sub> = TopK<sub>x_r ∈ 𝒞(B_f)</sub> PI(B<sub>f</sub>, x<sub>r</sub>)**

正式训练目标为：

> **𝓛<sub>PIPER</sub> = 𝓛<sub>backbone</sub>(B<sub>f</sub>) + λ · 𝓛<sub>local</sub>(𝒱<sub>B_f</sub>)**

其中：

> **𝓛<sub>local</sub> = D<sub>KL</sub>(p<sub>M0</sub>(·|x<sub>r</sub>) ∥ p<sub>Mθ</sub>(·|x<sub>r</sub>))**

这里 M<sub>0</sub> 是遗忘前模型。

---

## 5. PI 三种方案：全部纳入主方案，等待实验裁决

根据你的要求，PI 的三种实现暂时都放入主方案中，不提前只选一种。

### 5.1 Static-PI

Static-PI 指在遗忘训练开始前，预先计算一次 vulnerable retain samples。

形式为：

> **𝒱<sup>static</sup> = TopK<sub>x_r ∈ 𝒟_r</sub> PI<sub>θ0</sub>(B<sub>f</sub>, x<sub>r</sub>)**

优点：

- 成本最低；
- 实现最简单；
- 适合主实验快速跑通；
- 更容易嵌入 OpenUnlearning。

风险：

- 随着模型参数变化，初始 PI 可能失效；
- 如果遗忘过程较长，vulnerable set 可能不再准确。

Static-PI 是当前最适合先落地的版本。

---

### 5.2 Refresh-PI

Refresh-PI 指每隔若干训练步重新计算一次 PI。

例如每 T 步更新一次：


> **𝒱<sup>refresh</sup><sub>t</sub> = TopK<sub>x_r ∈ 𝒟_r</sub> PI<sub>θt</sub>(B<sub>f</sub>, x<sub>r</sub>)**

优点：

- 比 Static-PI 更贴合动态训练过程；
- 可以观察 vulnerable retain samples 是否随训练阶段变化；
- 机制解释更强。

风险：

- 成本显著增加；
- refresh interval T 会引入额外超参数；
- 如果提升不明显，审稿人可能认为复杂度不值得。

Refresh-PI 适合作为主方法候选之一，但最终是否作为 main variant，要由实验决定。

---

### 5.3 Online-PI

Online-PI 指在每个训练 step 动态计算当前 batch 对 retain candidates 的局部干扰。

> **𝒱<sup>online</sup><sub>t</sub> = TopK<sub>x_r ∈ 𝒞(B_f)</sub> PI<sub>θt</sub>(B<sub>f</sub>, x<sub>r</sub>)**

优点：

- 理论上最准确；
- 最能体现“当前遗忘更新造成的即时局部误伤”；
- 如果效果明显，可以作为论文亮点。

风险：

- 计算成本最高；
- 需要控制 candidate pool，否则不可扩展；
- 如果只在小规模 TOFU 上可运行，外推到更大 LLM unlearning 任务时会受质疑。

Online-PI 目前应保留在主方案中，但不能提前假定它是最终主版本。

---

### 5.4 三种 PI 的当前定位

当前建议写法是：

> **PIPER is a general local interference probing framework with three instantiations: Static-PI, Refresh-PI, and Online-PI.**

最终论文中哪一个作为 main result，需要由实验结果决定。

评审视角下，不能只报告效果最好的版本，还必须比较：

> **性能提升 vs. 额外计算成本**

否则容易被质疑方法不具备实际可用性。

---

## 6. 主数据集：TOFU

当前主数据集固定为 TOFU。

这有几个优点。

第一，TOFU 是 LLM unlearning 领域已有基准，审稿人熟悉。使用 TOFU 可以降低“自建数据集偏向作者方法”的质疑。

第二，TOFU 本身是 profile + QA 形式，虽然不是严格属性级画像遗忘，但仍然和你的用户画像方向有一定关联。

第三，TOFU 已经有常用遗忘指标和 baseline，可以让 PIPER 的贡献更容易被比较。

但必须清楚，TOFU 的限制也很明显：

> **TOFU 主要支持作者级或事实级遗忘，不足以单独支撑属性级画像遗忘结论。**

因此，如果主数据集暂时是 TOFU，论文当前主张应避免写成：

> **我们解决了属性级用户画像遗忘问题**

更稳妥的是：

> **我们在 TOFU 风格事实型 LLM unlearning 中研究局部保留样本误伤问题。**

---

## 7. 框架选择：确定使用 OpenUnlearning

当前确定使用 OpenUnlearning，这是正确选择。

OpenUnlearning 的价值在于，它提供统一的 LLM unlearning benchmark、方法和指标体系，整合 TOFU、MUSE、WMDP 等基准，并支持多种算法和评测。([arxiv.org](https://arxiv.org/abs/2506.12618?utm_source=chatgpt.com "OpenUnlearning: Accelerating LLM Unlearning via Unified Benchmarking of Methods and Metrics")) NeurIPS Datasets and Benchmarks 页面也显示 OpenUnlearning 集成多种 SOTA unlearning algorithms、16 类评测和 TOFU/MUSE/WMDP 三个主流基准。([proceedings.neurips.cc](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3e4a38f228427ab819ba7899003a44b1-Abstract-Datasets_and_Benchmarks_Track.html?utm_source=chatgpt.com "Accelerating LLM Unlearning via Unified Benchmarking of Methods ..."))

这对投稿会议很重要，因为它可以降低以下质疑：

- baseline 是否实现公平；
- 指标是否选择性报告；
- 实验流程是否可复现；
- 方法是否只在作者自写代码中有效。

当前方案中，PIPER 应作为 OpenUnlearning 中的一个 wrapper 或 method variant 接入。

例如：

> **GA → GA + PIPER**

> **NPO → NPO + PIPER**

> **IDK → IDK + PIPER**

不建议一开始接入太多 backbone。投稿会议更看重清晰机制和充分消融，而不是堆方法数量。

---

## 8. Baseline 方案

建议至少比较以下方法：

| 类型 | 方法 |
| --- | --- |
| 原始模型 | Original model |
| 基础遗忘方法 | GA / NPO / IDK |
| 全局保护 | Backbone + full retain KL |
| 随机局部保护 | Backbone + random local KL |
| 语义邻域保护 | Backbone + semantic local KL |
| 本文方法 | Backbone + Static-PI / Refresh-PI / Online-PI |

最关键的对照不是“PIPER 比所有 unlearning 方法都强”，而是证明：

> **PI 选择的 retain samples 比 random 或 semantic neighbor 更值得保护。**

否则 PIPER 的核心机制不成立。

---

## 9. 核心实验问题

当前实验必须回答四个问题。

### 9.1 PIPER 是否提升 forget-retain trade-off

即在相近遗忘效果下，PIPER 是否保留更多 retain utility。

形式上要观察：

> **Forget Quality and Retain Utility**

是否同时改善。

不能只报告 retain accuracy 提升。如果遗忘效果变差，那么 PIPER 只是“少忘了一点”，不是更好的 unlearning。

---

### 9.2 PI 分数是否能预测 retain damage

这是机制验证的核心。

需要计算：

> **PI(B<sub>f</sub>, x<sub>r</sub>)**

与最终遗忘后 retain loss increase 之间的相关性：

> **Δ𝓛<sub>retain</sub>(x<sub>r</sub>) = 𝓛<sub>retain</sub>(x<sub>r</sub>; θ<sub>unlearn</sub>) − 𝓛<sub>retain</sub>(x<sub>r</sub>; θ<sub>0</sub>)**

至少报告：

- Pearson correlation；
- Spearman correlation；
- Top-K precision；
- 分桶分析。

如果 PI 分数和实际 damage 没有相关性，那么 PIPER 只能被解释为经验性 regularization，不能声称“识别局部易损样本”。

---

### 9.3 三种 PI 方案的效果与成本如何取舍

需要比较：

> **Static-PI vs. Refresh-PI vs. Online-PI**

比较维度至少包括：

| 维度 | 说明 |
| --- | --- |
| Forget Quality | 是否真正忘掉 forget set |
| Retain Utility | retain set 是否保持 |
| Local Damage | vulnerable retain samples 是否少受损 |
| Runtime | 训练时间 |
| Memory | 显存开销 |
| Stability | 不同 seed 下是否稳定 |

最终主版本不能只按性能选，还要按“性能—成本比”选。

如果 Online-PI 只提升很小但成本极高，不应作为主推版本。

---

### 9.4 PIPER 是否只是 full retain KL 的低成本替代

这是审稿人很可能会问的问题。

必须比较：

> **PIPER local KL vs. Full retain KL**

可能结果有三种：

1. PIPER 接近 full retain KL，但成本更低；
2. PIPER 超过 full retain KL，因为它避免过度约束；
3. PIPER 弱于 full retain KL，但在成本上更优。

三种结果都能写，但解释不同。

最理想结论是：

> **PIPER achieves comparable or better preservation than full retain regularization with substantially lower cost.**

---

## 10. 当前论文贡献表述

建议当前贡献暂定为三点：

### 贡献一

提出局部干扰视角，指出 LLM unlearning 中 retain set 内部存在异质性，部分保留样本更容易受到遗忘更新误伤。

### 贡献二

提出 PIPER，通过虚拟遗忘更新估计候选保留样本的潜在损失变化，并对 vulnerable retain samples 施加局部保护。

### 贡献三

在 OpenUnlearning 框架下基于 TOFU 进行系统实验，比较 Static-PI、Refresh-PI 和 Online-PI 三种实现，并分析其效果、稳定性与计算成本。

这三点比“提出属性级画像遗忘 benchmark”更符合当前设定。

---

## 11. 当前未确定部分

### 11.1 APU-Bench 是否加入

APU-Bench 当前放入未确定项。

如果后续实验完成，它可以作为扩展验证，用于把论文从 TOFU 风格事实遗忘推进到属性级画像遗忘。

但当前不能把它写进主贡献。

当前定位：

> **APU-Bench 是潜在扩展，不是当前主实验基础。**

是否加入取决于三个条件：

1. 数据质量是否足够；
2. target / retain attribute split 是否合理；
3. 是否能证明它提供 TOFU 没有覆盖的新问题。

如果这三点做不到，APU-Bench 不应进入会议投稿主线。

---

### 11.2 最终主 PI 版本未确定

Static-PI、Refresh-PI、Online-PI 当前都属于主方案候选。

最终由实验决定：

> **main variant = argmax(trade-off improvement − α · computational cost)**

如果 Static-PI 已经达到接近最优效果，它应作为主推版本。

如果 Refresh-PI 明显更强且成本可控，可以作为主推版本。

Online-PI 只有在显著领先时才适合作为主推版本。

---

### 11.3 是否加入 MUSE / WMDP

当前不建议一开始加入。

OpenUnlearning 支持 TOFU、MUSE、WMDP，但当前主数据集已经确定为 TOFU。MUSE 更强调六维评估，包括无逐字记忆、无知识记忆、无隐私泄露、效用保持、删除规模可扩展性和连续删除可持续性。([proceedings.iclr.cc](https://proceedings.iclr.cc/paper_files/paper/2025/hash/4556f5398bd2c61bd7500e306b4e560a-Abstract-Conference.html?utm_source=chatgpt.com "Machine Unlearning Six-Way Evaluation for Language ..."))

如果时间充足，可以把 MUSE 作为补充实验；但如果 TOFU 上机制验证还没跑通，不应扩展到 MUSE/WMDP。

---

### 11.4 是否加入 CV 实验

当前也放入未确定项。

如果主论文已经使用 TOFU + OpenUnlearning，那么 CV 实验不是必要条件。

OFMU 能把 CV 融入论文，是因为它本身定位为通用 optimization-driven framework，并在语言和视觉任务上验证 generality。([openreview.net](https://openreview.net/forum?id=ZDuyNJI56H&utm_source=chatgpt.com "OFMU: OPTIMIZATION-DRIVEN FRAMEWORK FOR ..."))

你的 PIPER 如果后续想加 CV，也必须服务于“局部干扰探针”这个统一机制，而不能变成另一个独立问题。

当前建议：

> **CV 暂不进入主方案，放在可选扩展。**

---

## 12. 当前方案的主要风险

### 风险一：TOFU 不足以支撑“属性级画像遗忘”

这是当前最大风险。

解决方式：当前论文主张先从“attribute-level profile unlearning”退回到：

> **local interference in LLM unlearning**

APU-Bench 成熟后再升级问题定位。

---

### 风险二：PIPER 可能被认为只是 retain KL 的采样版本

解决方式：必须证明 PI 选择不是随机采样，也不是语义相似检索。

必须做：

> **PI vs. Random**

> **PI vs. Semantic Neighbor**

否则方法贡献不足。

---

### 风险三：三种 PI 方案会让方法显得不收敛

解决方式：论文中要把三者定义为同一框架下的三种 instantiation，而不是三个互不相关的方法。

表述应为：

> **PIPER admits static, periodically refreshed, and online instantiations.**

最终由实验选择主版本。

---

### 风险四：计算成本可能抵消方法收益

尤其是 Online-PI。

因此必须报告 runtime 和 memory。否则审稿人会认为方法不可扩展。

---

## 13. 当前最稳版本

如果现在要准备会议投稿，我建议最稳版本是：

> **TOFU + OpenUnlearning + PIPER + Static/Refresh/Online PI 比较 + random 与 semantic local protection 对照**

当前不要把 APU-Bench 写成主数据集，不要把 CV 写成主贡献。

最终一句话定位：

> **本文提出 PIPER，通过探测遗忘更新对保留样本的局部干扰，在 OpenUnlearning 框架下改善 TOFU 上的遗忘—保留权衡。**

这比“属性级画像遗忘 + 自建数据集 + CV 泛化”更收敛，也更符合当前会议投稿的最低可验证路径。
