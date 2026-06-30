可以。你现在这个故事方向是**比单纯 PIPER sample selection 更强的顶会叙事**。但要做得有说服力，不能只说“现有方法细粒度效果不好，所以我加 PI”。你需要把它组织成一个完整问题：

[
\boxed{
\text{现有 LLM unlearning 主要处理粗粒度遗忘；当 forget 与 retain 属于同一实体 / 同一用户画像内部属性时，遗忘边界变得局部且高度纠缠。}
}
]

然后提出一个框架：

[
\boxed{
\text{面向细粒度实体内遗忘的 risk-aware bi-level unlearning framework。}
}
]

PI / Residual-PI 在其中不是孤立技巧，而是**内层风险估计器 / 局部保留预算分配器**。

下面是我建议的顶会级方案。

------

# 1. 论文核心命题

建议论文主问题定义为：

[
\boxed{
\textbf{Fine-Grained Entity-Level Unlearning}
}
]

即：不是遗忘一个完整作者、一个完整类别、一个完整用户，而是遗忘同一实体内部的一部分事实 / 属性，同时保留该实体的其他事实。

例如 TOFU 原始设定更接近：

> 忘掉某些作者，保留其他作者。

你要改成：

> 对每个作者，只忘掉该作者的一部分 QA / facts，同时保留该作者的其他 QA / facts。

这会制造一个更难的场景：

| 遗忘类型                             | Forget set       | Retain set         | 难点                   |
| ------------------------------------ | ---------------- | ------------------ | ---------------------- |
| Coarse-grained author unlearning     | 整个作者         | 其他作者           | 实体边界清晰           |
| Fine-grained intra-author unlearning | 作者内部分 facts | 同一作者其他 facts | forget-retain 高度纠缠 |
| Attribute-level profile unlearning   | 用户某些属性     | 用户其他属性       | 最接近隐私删除场景     |

你论文的第一张主图就应该展示这个对比：**coarse deletion vs fine-grained boundary-preserving deletion**。

------

# 2. 故事主线

推荐故事如下。

## 第一幕：现有方法在粗粒度遗忘上有效，但细粒度遗忘更难

现有 LLM unlearning benchmark 经常把 forget set 和 retain set 分成较清晰的块。例如 TOFU 中常见设定是忘掉一部分作者，保留其他作者。这样的任务中，forget 和 retain 的实体边界比较明显。

但真实隐私删除更常见的是：

> 用户要求删除某个手机号、住址、私有 ID、某段经历，但不希望模型忘掉该用户的公开信息或无害信息。

也就是：

[
\text{same entity, partial forget, partial retain}
]

这个任务比 coarse-grained unlearning 更难，因为 forget facts 和 retain facts 共享：

- 同一实体名称；
- 相同上下文；
- 相似语言模式；
- 相关表示空间；
- 可能相互支持的推理链。

因此，现有方法容易出现两类失败：

1. **over-unlearning**：忘掉目标属性时，把同一实体的其他属性也破坏；
2. **under-unlearning**：为了保留同一实体信息，导致目标属性忘不干净。

这就是你论文的 problem gap。

------

## 第二幕：为什么 global retain regularization 不够

很多方法会加 retain loss / KL / utility regularization。但在细粒度场景下，global retain regularization 有一个问题：

[
\boxed{
\text{它平均保护整个 retain set，却不知道哪些 retain samples 正在被当前 forget update 重点误伤。}
}
]

例如 retain set 有 10,000 条，全局 KL 会平均保护所有 retain samples。但真正危险的可能是同一作者内的 20 条邻近事实。全局 KL 会稀释保护预算。

所以你要提出：

[
\boxed{
\text{细粒度遗忘需要局部风险感知，而不是只做全局保留。}
}
]

------

## 第三幕：为什么 gradient similarity 仍然不够

OFMU 这类方法说明 gradient similarity / gradient decorrelation 很有价值。但它主要是：

[
\text{forget objective 与 retain objective 的 update-level conflict}
]

它适合控制整体遗忘方向，但不天然解决：

[
\text{哪些具体 retain samples 最容易受损？}
]

也就是说，OFMU/GS 主要回答：

> 当前 forget gradient 是否会干扰 retain gradient？

PIPER 要回答：

> 当前 forget update 之后，哪个 retain sample 的 loss 会真的上升？

这就是你的差异。

你不应该说 GS 错，而应该说：

[
\boxed{
\text{GS 是一阶、目标级、方向性信号；PI 是有限步、样本级、损伤预测信号。}
}
]

你之前的新 proposal 中已经把这个关系说清楚了：gradient similarity 可以被看作 PI 的 first-order approximation，而 Residual-PI 捕捉 PI 中无法被一阶 GS 解释的有限步易损性。

------

# 3. 推荐方法名称

我建议保留 PIPER，但扩展成细粒度框架名：

[
\boxed{
\textbf{FG-PIPER: Fine-Grained Predictive Interference Preservation for Entity-Level Unlearning}
}
]

或者更论文味一点：

[
\boxed{
\textbf{PIPER: Risk-Aware Fine-Grained Unlearning via Predictive Interference Probing}
}
]

我更推荐第二个，因为 PIPER 名字已经和你当前工作一致。

------

# 4. 方法框架：Risk-Aware Bi-level PIPER

你可以把方法设计成一个类似 OFMU 的双层框架，但目标不同。

OFMU 式双层框架侧重：

[
\text{inner forgetting/decorrelation}
\quad+\quad
\text{outer utility restoration}
]

你的 PIPER 框架可以定义为：

[
\boxed{
\text{inner risk probing}
\quad+\quad
\text{outer preservation-aware unlearning}
}
]

也就是：

## Inner level：Predictive Interference Risk Estimation

给定当前模型 (\theta_t)、细粒度 forget set (D_f)、候选 retain set (C_r)，先构造一个虚拟遗忘更新：

# [ \theta'_t

\theta_t
+
\Delta\theta_f
]

其中 (\Delta\theta_f) 来自 NPO / GradDiff / CEU 等 backbone 的遗忘方向。

然后对每个 retain candidate (x_i) 计算：

# [ PI_i

## L_r(x_i;\theta'_t)

L_r(x_i;\theta_t)
]

这表示：

> 如果真的沿当前遗忘方向走一小步，哪个 retain sample 的 loss 会升高？

然后计算 GS 作为一阶解释项：

[
GS_i
\approx
-\eta
\langle
\nabla L_r(x_i),
\nabla L_f
\rangle
]

再定义 Residual-PI：

# [ RPI_i

## PI_i

\widehat{PI}_i^{GS}
]

这里我建议不要直接用 (PI_i - GS_i)，而是用回归残差：

[
\widehat{PI}_i^{GS}=a\cdot GS_i+b
]

[
RPI_i=PI_i-\widehat{PI}_i^{GS}
]

这样更稳，因为 PI 和 GS 尺度可能不同。

------

## Outer level：Risk-Weighted Preservation-Aware Unlearning

外层正式训练时，不是平均保护所有 retain samples，而是根据风险分数分配 local KL budget。

最终 score 可以定义为：

# [ s_i

z(PI_i)
+
\rho \cdot z([RPI_i]_+)
]

其中：

- (PI_i)：有限步实际损伤预测；
- (RPI_i)：GS 无法解释的额外损伤；
- ([RPI_i]_+)：只关注正残差，也就是“GS 低估了损伤”的样本；
- (z(\cdot))：标准化。

然后把 score 转成保护权重：

# [ w_i

\frac{\exp(s_i/\tau)}
{\sum_j \exp(s_j/\tau)}
]

外层训练目标为：

# [ L_{PIPER}

L_{unlearn}
+
\lambda
\sum_{x_i\in C_r}
w_i
KL
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_{\theta}(\cdot|x_i)
\right)
+
\lambda_g L_{global-retain}
]

这里有三个关键点：

1. (L_{unlearn}) 可以是 NPO、GradDiff、CEU 等任意 backbone；
2. local KL 重点保护高风险 retain samples；
3. global retain loss 只作为轻量 anchor，防止整体模型漂移。

这就不再是“选 Top-K 加 KL”的简单 heuristic，而是：

[
\boxed{
\text{根据有限步遗忘风险进行局部保留预算分配。}
}
]

这更像顶会方法。

------

# 5. 为什么这个框架比普通 PI Top-K 更强

普通 PI Top-K 的风险是：

> 看起来只是一个 sample selector。

你现在这个改进版有三个升级。

## 5.1 从 Top-K 选择升级为 preservation budget allocation

不是硬选 Top-K，而是给每个 retain candidate 分配不同保护权重。

这能回应审稿人：

> Top-K 阈值是不是任意的？

## 5.2 从 PI 升级为 Residual-aware PI

不是只用 PI，而是明确和 GS 建立理论关系：

[
GS = PI \text{ 的一阶近似}
]

[
RPI = PI \text{ 中 GS 解释不了的部分}
]

这能回应审稿人：

> PI 是否只是昂贵版 gradient similarity？

## 5.3 从粗粒度 unlearning 升级为细粒度 boundary preservation

你的方法不是为了普通作者级遗忘，而是为了更难的：

[
\text{same-entity forget-retain separation}
]

这能回应审稿人：

> 为什么需要这个新框架？

------

# 6. 数据集设计：FineTOFU / Intra-TOFU

你需要构造一个能支撑故事的数据集。建议命名为：

[
\boxed{
\textbf{FineTOFU}
}
]

或者：

[
\boxed{
\textbf{Intra-TOFU}
}
]

我更推荐 **FineTOFU**，更直观。

## 6.1 构造方式

基于 TOFU 原始作者 QA：

对每个作者 (a)，将其 QA 拆成：

[
D_f^a
\quad\text{and}\quad
D_r^a
]

其中：

- (D_f^a)：该作者内部要遗忘的一部分 facts / QA；
- (D_r^a)：同一作者内部要保留的其他 facts / QA；
- (D_r^{other})：其他作者的 retain QA。

最终：

[
D_f = \bigcup_a D_f^a
]

[
D_r^{intra} = \bigcup_a D_r^a
]

[
D_r^{inter} = D_r^{other}
]

评测时必须分开看：

| Retain 类型         | 含义                     |
| ------------------- | ------------------------ |
| Intra-entity retain | 同一作者内部的保留 facts |
| Inter-entity retain | 其他作者 facts           |
| Global retain       | 所有 retain              |

核心指标是：

[
\boxed{
\text{Intra-entity retain utility}
}
]

因为这是细粒度遗忘最难的部分。

------

## 6.2 难度分层

你可以定义三个难度：

| 难度   | 构造方式                                                     |
| ------ | ------------------------------------------------------------ |
| Easy   | forget QA 与 retain QA 语义距离远                            |
| Medium | 同作者但不同主题 / 不同属性                                  |
| Hard   | forget QA 与 retain QA 共享实体、事件、作品、时间线或相邻事实 |

如果 TOFU 原始 QA 的 slot 标注不足，可以先用两种方式：

1. embedding similarity 划分；
2. LLM 辅助标注 facts / attributes。

但论文中要诚实说明这是 TOFU-derived benchmark。

如果你想更稳，可以用 MYTOFU / APU-Bench 作为 controlled attribute-level dataset，但主实验仍以 FineTOFU 为主。

------

# 7. 实验设计

## 7.1 第一组：证明细粒度任务确实更难

比较同一批方法在两种 setting 下的表现：

| Setting     | Forget set         | Retain set                  |
| ----------- | ------------------ | --------------------------- |
| Coarse-TOFU | 整个作者           | 其他作者                    |
| FineTOFU    | 作者内部部分 facts | 同作者其他 facts + 其他作者 |

方法包括：

- NPO；
- GradDiff；
- RMU；
- CEU；
- SimNPO；
- possibly OFMU / OFMU-style GS decorrelation。

你要证明：

[
\text{FineTOFU 上 retain utility drop 更严重}
]

尤其是：

[
\text{intra-entity retain utility drop}

>

\text{inter-entity retain utility drop}
]

这一步是论文故事的地基。

------

## 7.2 第二组：PI / GS / Residual-PI 机制验证

比较不同分数预测真实 retain damage 的能力：

| Score                | 目的              |
| -------------------- | ----------------- |
| Semantic similarity  | 输入空间邻近      |
| GS                   | 一阶梯度冲突      |
| PI                   | 有限步损伤预测    |
| Residual-PI          | GS 解释不了的损伤 |
| Hybrid / weighted PI | 预算分配          |

指标：

- Spearman；
- Pearson；
- Top-K precision；
- Top-K damage lift；
- enrichment；
- overlap with oracle damage Top-K。

你现有机制实验已经显示 set-gradient PI 明显强于 semantic baseline，set-gradient PI 的 Spearman mean 为 0.731，而 semantic baseline 只有 0.126，这已经是很好的初步证据。

下一步必须补：

[
PI \quad vs. \quad GS
]

这是顶会审稿人最关心的。

------

## 7.3 第三组：正式方法实验

方法对照应该是：

| 方法                                  | 目的             |
| ------------------------------------- | ---------------- |
| Backbone                              | 原方法           |
| Backbone + global KL                  | 全局保留         |
| Backbone + random local KL            | 控制 local KL    |
| Backbone + semantic local KL          | 控制语义邻域     |
| Backbone + GS local KL                | 控制一阶梯度冲突 |
| Backbone + PI Top-K local KL          | 当前 PIPER       |
| Backbone + Residual-aware weighted KL | 最终主方法       |
| Oracle damage local KL                | 上界分析         |

注意：主方法我建议不要叫 Residual-PI Top-K，而叫：

[
\boxed{
\textbf{Residual-aware PI-weighted PIPER}
}
]

也就是：

[
PI + RPI + weighted KL
]

这样比单纯 Residual-PI Top-K 稳。

------

## 7.4 第四组：forget-quality matched analysis

这一步必须做。

不同方法可能通过“少忘一点”来保留更多。为了证明 trade-off 真改善，你要比较相似 forget quality 下的 retain damage。

做法：

- 对每个方法扫 (\lambda)；
- 画 Pareto curve；
- 或筛选 forget probability delta / Forget Quality 接近的 runs；
- 比较 intra-entity retain utility。

核心结论应该是：

[
\boxed{
\text{在相近 forget quality 下，PIPER 的 intra-entity retain damage 更低。}
}
]

没有这句话，顶会审稿人不会接受“trade-off 改善”。

------

# 8. 主实验推荐配置

首轮最小可行主实验：

| 项目             | 设置                                                         |
| ---------------- | ------------------------------------------------------------ |
| Dataset          | FineTOFU forget10 / retain90                                 |
| Model            | Llama-2-7B-chat full checkpoint                              |
| Backbone         | NPO                                                          |
| Seeds            | 0, 1, 2                                                      |
| PI type          | set-gradient PI                                              |
| Coverage         | fb1-pb64                                                     |
| Candidate retain | same-author retain + semantic nearest retain                 |
| KL weights       | (\lambda \in {0.1, 0.2, 0.3, 0.5})                           |
| Methods          | NPO, global KL, random local KL, semantic local KL, GS local KL, PI local KL, residual-aware weighted PIPER |

如果这组成功，再扩：

1. forget01 / forget05 / forget10；
2. GradDiff / SimNPO / CEU；
3. Llama-3.2-1B / 3B；
4. MYTOFU attribute-level setting。

------

# 9. 你的论文贡献应该写成什么

我建议写成四点。

## Contribution 1：提出细粒度实体内遗忘任务

> We formulate fine-grained entity-level unlearning, where the model must forget selected facts of an entity while preserving other facts of the same entity.

这是任务贡献。

## Contribution 2：构造 FineTOFU benchmark

> We derive FineTOFU from TOFU by splitting each author profile into intra-entity forget and retain facts, enabling evaluation of boundary-preserving unlearning.

这是数据 / benchmark 贡献。

## Contribution 3：提出 risk-aware bi-level PIPER

> We propose PIPER, a risk-aware bi-level framework that probes finite-step retain vulnerability and allocates preservation budget to high-risk retain samples.

这是方法贡献。

## Contribution 4：提出 residual-aware finite-step interference

> We show that gradient similarity is a first-order approximation of finite-step PI, and introduce Residual-PI to capture vulnerability unexplained by gradient similarity.

这是理论 / 机制贡献。

------

# 10. 论文标题建议

几个方向：

1. **PIPER: Predictive Interference Probing for Fine-Grained LLM Unlearning**
2. **Fine-Grained Entity Unlearning via Residual Predictive Interference**
3. **Beyond Gradient Conflict: Predictive Interference Probing for Boundary-Preserving Unlearning**
4. **Risk-Aware Fine-Grained Unlearning with Finite-Step Interference Probing**

我最推荐第 3 个：

[
\boxed{
\textbf{Beyond Gradient Conflict: Predictive Interference Probing for Boundary-Preserving Unlearning}
}
]

它能直接把你的方法和 GS / OFMU 关联起来。

------

# 11. 顶会审稿风险与应对

## 风险一：FineTOFU 是否只是人为拆分？

应对：

- 明确它模拟真实 privacy deletion；
- 评估 intra-entity retain；
- 加 MYTOFU attribute-level 作为 controlled validation；
- 做 coarse vs fine 对比，证明 fine setting 确实带来新困难。

## 风险二：PIPER 是否只是昂贵版 GS？

应对：

- 必须加 GS local KL；
- 做 PI vs GS 预测性对比；
- 做 Residual-PI analysis；
- 报告同等计算预算下性能。

## 风险三：retain 改善是否来自忘得少？

应对：

- forget-quality matched analysis；
- Pareto curve；
- 同等 Forget Quality 下比较 intra-entity retain damage。

## 风险四：local KL 本身是否足够？

应对：

- random local KL；
- semantic local KL；
- GS local KL；
- oracle damage local KL。

## 风险五：方法太复杂

应对：

主文只讲一个清晰主方法：

[
\text{Residual-aware PI-weighted PIPER}
]

其他作为 ablation：

- PI Top-K；
- Residual-PI Top-K；
- PI-weighted；
- GS local KL。

不要把所有 variant 都包装成主方法。

------

# 12. 最终推荐方案一句话

你的最终论文应叙述为：

[
\boxed{
\text{我们指出现有 LLM unlearning 在同一实体内部的细粒度遗忘上存在明显边界保持困难；为此构造 FineTOFU，并提出 PIPER，一个 risk-aware bi-level framework，通过有限步 PI 和 Residual-PI 估计局部 retain 易损性，并将保留预算分配给最可能被误伤的同实体 retain facts，从而在相同 forget quality 下显著降低 intra-entity retain damage。}
}
]

这是一个有顶会潜力的故事。

但它成立的前提是你接下来必须证明三件事：

1. **FineTOFU 比 coarse TOFU 更难，尤其 intra-entity retain 更容易受损。**
2. **PI / Residual-PI 比 GS 更能预测真实 retain damage。**
3. **Residual-aware PI-weighted PIPER 在 forget-quality matched 设置下优于 GS / semantic / random local KL。**

如果这三件事都成立，这篇论文就不再只是“一个本科论文上的 PI 探针改进”，而是一个完整的、可投稿会议的方法论文。