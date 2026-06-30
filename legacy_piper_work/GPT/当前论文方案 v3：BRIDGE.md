部分历史上传文件已过期；下面这版基于我们当前已经确定的讨论重新整理，不依赖过期文件。

# 当前论文方案 v2：BRIDGE

## 1. 题目重新构思

我建议主标题采用：

[
\boxed{
\textbf{BRIDGE: Boundary-Robust Instance-level Distributional Guidance for Fine-Grained LLM Unlearning}
}
]

中文理解：

[
\boxed{
\textbf{BRIDGE：面向细粒度大语言模型遗忘的边界鲁棒实例级分布引导框架}
}
]

这个标题的优点是：

1. **BRIDGE** 有“连接 / 维持边界”的隐喻，适合 forget-retain boundary；
2. **Boundary-Robust** 突出你不是平均保留，而是保护细粒度遗忘边界；
3. **Instance-level Distributional Guidance** 对应你的 Predictive-prior DRO；
4. **Fine-Grained LLM Unlearning** 明确投稿故事，不再泛泛说 machine unlearning。

备选标题：

1. **PRIDE: Predictive-prior Robust Instance-level Distributional Erasure for Fine-Grained LLM Unlearning**
2. **BIRD: Boundary-aware Instance-level Robust Distributional Unlearning for Fine-Grained LLMs**
3. **PREDO: Predictive-prior Distributionally Robust Optimization for Fine-Grained LLM Unlearning**
4. **BRACE: Boundary-Robust Adaptive Constrained Erasure for Fine-Grained LLM Unlearning**

我最推荐 **BRIDGE**，因为它比 BRACE / PREDO 更像顶会论文标题，也更容易被记住。

------

# 2. 论文核心故事

当前论文不再讲：

[
\text{我提出 PI 探针来找受损 retain 样本。}
]

而应讲：

[
\boxed{
\text{现有 LLM unlearning 方法在粗粒度遗忘上有效，但在同一实体内部的细粒度遗忘中，容易破坏 forget-retain 边界。}
}
]

更具体地说，TOFU 原始任务更偏：

[
\text{忘掉某些作者，保留其他作者。}
]

但真实隐私删除更常见的是：

[
\text{忘掉同一用户 / 作者的一部分事实，保留该实体的其他事实。}
]

这种设定下，forget facts 和 retain facts 共享同一实体名、上下文和事实结构，因此普通 retain regularization 或全局 retain KL 可能保护平均 retain utility，却无法保护真正脆弱的 same-entity retain facts。

所以本文提出：

[
\boxed{
\text{将细粒度遗忘建模为 instance-level Predictive-prior DRO 问题。}
}
]

核心思想是：

> 不平均保护所有 retain 样本，而是让模型自动关注最可能被当前遗忘更新误伤、且已经发生偏移的 retain 边界样本。

------

# 3. 任务定义：Fine-Grained Entity-Level Unlearning

给定实体集合：

[
\mathcal{E}
]

对于每个实体 (e)，例如 TOFU 中一个作者，其 QA / facts 被拆成：

[
D_f^e
]

和：

[
D_r^e
]

其中：

- (D_f^e)：同一实体内部需要遗忘的 facts / QA；
- (D_r^e)：同一实体内部需要保留的 facts / QA；
- (D_r^{-e})：其他实体的 retain facts / QA。

整体 forget set 为：

[
D_f = \bigcup_e D_f^e
]

整体 retain set 为：

[
D_r = \left(\bigcup_e D_r^e\right) \cup D_r^{-e}
]

论文的关键评估对象不是普通 global retain，而是：

[
\boxed{
D_r^{intra} = \bigcup_e D_r^e
}
]

也就是 **同一实体内部的 retain facts**。

这部分最容易被误伤，也是细粒度遗忘区别于粗粒度遗忘的核心。

------

# 4. 方法总览：BRIDGE

BRIDGE 的核心不是单独找一个“纠缠 retain 子集”，而是构造一个 **Predictive-prior DRO retain boundary risk**。

整体目标包括三部分：

[
\boxed{
\text{Erasure objective}
+
\text{Global retain stability}
+
\text{Predictive-prior boundary DRO}
}
]

其中：

1. **Erasure objective** 负责遗忘目标事实；
2. **Global retain stability** 保证整体 retain 不崩；
3. **Predictive-prior boundary DRO** 重点保护高风险细粒度 retain 边界。

PI / GS 的作用只是：

[
\boxed{
\text{为 DRO 的最坏 retain 分布提供 prior。}
}
]

它们不是一个单独框架层。

------

# 5. Retain drift 定义

对 retain candidate pool 中的每个样本 (x_i)，定义当前 retain drift：

# [ d_i(\theta)

D_{\mathrm{KL}}
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_{\theta}(\cdot|x_i)
\right)
]

其中：

- (\theta_0)：遗忘前 reference model；
- (\theta)：当前 unlearning model。

(d_i(\theta)) 表示：

[
\boxed{
\text{当前模型在 retain sample } x_i \text{ 上已经偏离原模型多少。}
}
]

也可以使用 loss increase：

# [ d_i(\theta)

\ell(x_i;\theta)-\ell(x_i;\theta_0)
]

但主方法建议用 KL drift，更适合生成式 LLM。

------

# 6. Predictive prior：PI / GS 两种候选

当前先验计算方式暂时不确定，保留两种候选：

[
\boxed{
\text{GS-prior}
\quad \text{and} \quad
\text{PI-prior}
}
]

最终由实验决定主版本。

------

## 6.1 GS-prior：一阶梯度冲突先验

Gradient Similarity 估计 retain sample 和 forget update 的一阶冲突。

设：

# [ g_f

\nabla_{\theta}
\mathcal{L}_{erase}(\theta;B_f)
]

# [ g_i

\nabla_{\theta}
\ell(x_i;\theta)
]

定义：

# [ GS_i

-\eta
\langle g_i,g_f\rangle
]

或者使用 cosine 版本：

# [ GS_i

-\cos(g_i,g_f)
]

直觉是：

[
\boxed{
\text{如果 retain 梯度和 forget 梯度冲突强，则该 retain sample 可能被遗忘更新误伤。}
}
]

GS-prior 的优点是：

- 理论简单；
- 与 OFMU / gradient decorrelation 相关；
- 计算上可能更直接；
- 可解释为 PI 的一阶近似。

缺点是：

- 只是一阶近似；
- 只看当前点梯度方向；
- 不直接测小步更新后的实际 retain loss 变化；
- 可能无法捕捉有限步更新、曲率、optimizer effect。

------

## 6.2 PI-prior：有限步预测干扰先验

PI 直接模拟一次小步遗忘更新。

先构造虚拟更新：

# [ \theta^+

\theta
+
\Delta \theta_f
]

其中：

[
\Delta \theta_f
]

来自当前 unlearning backbone 的一步遗忘更新方向。

然后定义：

# [ PI_i

## \ell(x_i;\theta^+)

\ell(x_i;\theta)
]

直觉是：

[
\boxed{
\text{如果沿当前遗忘方向走一小步，}x_i\text{ 的 retain loss 增加多少？}
}
]

PI-prior 的优点是：

- 直接预测有限步更新后的 retain damage；
- 比 GS 更贴近真实损伤；
- 可以捕捉一阶 GS 解释不了的非线性影响；
- 更适合 sample-level vulnerability estimation。

缺点是：

- 成本更高；
- 需要虚拟更新；
- 需要额外 forward；
- 需要实验验证是否显著优于 GS。

------

# 7. 统一 prior 表达

无论使用 GS 还是 PI，都记为风险分数：

[
s_i
]

其中：

[
s_i =
\begin{cases}
GS_i, & \text{GS-prior} \
PI_i, & \text{PI-prior}
\end{cases}
]

然后构造 prior distribution：

# [ \pi_i

\frac{
\exp(z(s_i)/\tau_p)
}{
\sum_{j\in C_r}
\exp(z(s_j)/\tau_p)
}
]

其中：

- (z(\cdot))：标准化；
- (\tau_p)：prior temperature；
- (C_r)：retain candidate pool。

(\pi_i) 的含义是：

[
\boxed{
\text{根据 GS 或 PI 预估，retain sample }x_i\text{ 成为高风险样本的先验概率。}
}
]

------

# 8. Predictive-prior DRO

定义 DRO 内层最坏 retain 分布：

[
q
\in
\Delta
]

我们希望模型关注那些高风险 retain samples，因此定义：

# [ B_{\pi}(\theta)

## \max_{q\in\Delta} \left[ \sum_{i\in C_r} q_i d_i(\theta)

\tau
D_{\mathrm{KL}}(q\Vert \pi)
\right]
]

其中：

- (q_i)：DRO 自动形成的最坏 retain 分布；
- (\pi_i)：GS 或 PI 给出的 prior；
- (d_i(\theta))：当前 retain drift；
- (\tau)：DRO temperature。

这个最大化有闭式解：

# [ q_i^\star

\frac{
\pi_i \exp(d_i(\theta)/\tau)
}{
\sum_j
\pi_j \exp(d_j(\theta)/\tau)
}
]

对应的 DRO risk 为：

# [ B_{\pi}(\theta)

\tau
\log
\sum_{i\in C_r}
\pi_i
\exp(d_i(\theta)/\tau)
]

这个式子是 BRIDGE 的核心。

------

# 9. DRO 权重的直观解释

[
q_i^\star
\propto
\pi_i
\exp(d_i(\theta)/\tau)
]

它同时考虑：

1. **先验风险** (\pi_i)
   即 GS / PI 预测这个 retain sample 是否容易被 forget update 误伤；
2. **当前 drift** (d_i(\theta))
   即这个 retain sample 当前是否已经偏离原模型。

所以：

[
\boxed{
q_i^\star \text{ 高}
\iff
x_i \text{ 既被预测为高风险，又已经出现较大 retain drift。}
}
]

这比单纯 Top-K PI 更合理，因为它不是硬选择，而是动态 soft weighting。

------

# 10. Global retain constraint

为了避免只保护局部边界而导致整体模型漂移，引入全局 retain 约束：

# [ G(\theta)

## \frac{1}{|D_r|} \sum_{x_i\in D_r} D_{\mathrm{KL}} \left( p_{\theta_0}(\cdot|x_i) \Vert p_{\theta}(\cdot|x_i) \right)

\epsilon_g
]

约束：

[
G(\theta)\leq 0
]

其作用是：

[
\boxed{
\text{整体 retain distribution 不能偏离原模型太多。}
}
]

------

# 11. Boundary DRO constraint

定义边界约束：

# [ H_{\pi}(\theta)

## B_{\pi}(\theta)

\epsilon_b
]

约束：

[
H_{\pi}(\theta)\leq 0
]

其作用是：

[
\boxed{
\text{由 GS / PI prior 诱导出的高风险 retain 子分布，其 drift 不能过大。}
}
]

------

# 12. 最终优化目标

BRIDGE 将细粒度遗忘写成约束优化：

[
\min_{\theta}
\mathcal{L}_{erase}(\theta;D_f)
]

subject to：

[
G(\theta)\leq 0
]

[
H_{\pi}(\theta)\leq 0
]

使用 augmented Lagrangian：

# [ \mathcal{L}_{BRIDGE}

\mathcal{L}{erase}
+
\lambda_g [G(\theta)]+
+
\frac{\rho_g}{2}[G(\theta)]+^2
+
\lambda_b [H{\pi}(\theta)]+
+
\frac{\rho_b}{2}[H{\pi}(\theta)]_+^2
]

其中：

- (\lambda_g,\lambda_b)：dual variables；
- (\rho_g,\rho_b)：penalty weights；
- ([x]_+=\max(x,0))。

dual variables 动态更新：

[
\lambda_g
\leftarrow
[\lambda_g+\rho_gG(\theta)]_+
]

[
\lambda_b
\leftarrow
[\lambda_b+\rho_bH_{\pi}(\theta)]_+
]

这样，BRIDGE 不是固定加一个 KL 权重，而是根据约束违反程度动态调整保留强度。

------

# 13. 训练流程

每个 step：

1. 采样 forget batch (B_f)；
2. 采样 retain candidate pool (C_r)；
3. 计算风险分数 (s_i)，当前暂定为 GS 或 PI；
4. 构造 prior：

[
\pi_i=softmax(z(s_i)/\tau_p)
]

1. 计算每个 retain candidate 的 drift：

[
d_i(\theta)
]

1. 计算 DRO boundary risk：

[
B_{\pi}(\theta)
]

1. 计算 global constraint：

[
G(\theta)
]

1. 计算最终 loss：

[
\mathcal{L}_{BRIDGE}
]

1. 更新模型；
2. 更新 dual variables。

------

# 14. 当前未确定部分

目前最关键的未确定项是：

[
\boxed{
\text{prior 使用 GS 还是 PI}
}
]

因此实验设计中必须包含：

| Prior 类型     | 名称         | 目的                     |
| -------------- | ------------ | ------------------------ |
| Uniform prior  | Uniform-DRO  | 检查 DRO 本身是否有效    |
| Semantic prior | Semantic-DRO | 检查语义邻近是否足够     |
| GS prior       | GS-DRO       | 检查一阶梯度冲突是否足够 |
| PI prior       | PI-DRO       | 检查有限步预测是否更好   |
| Oracle prior   | Oracle-DRO   | 上界分析                 |

最终由实验决定：

[
\pi^{GS}
\quad \text{or} \quad
\pi^{PI}
]

哪个作为主方法。

如果 PI 明显优于 GS，则论文主张：

[
\boxed{
\text{有限步 predictive prior 比一阶 gradient prior 更适合细粒度遗忘。}
}
]

如果 GS 与 PI 接近，则论文主张应调整为：

[
\boxed{
\text{BRIDGE 是 prior-agnostic 的 DRO 框架，GS 是更低成本实现，PI 是更强但更贵实现。}
}
]

------

# 15. 数据集方案

当前建议使用：

[
\boxed{
\textbf{FineTOFU}
}
]

从 TOFU 派生。

具体做法：

对每个作者内部 QA 拆分：

[
D_f^e
]

和：

[
D_r^e
]

也就是：

- 只遗忘该作者的一部分 facts；
- 保留该作者的其他 facts；
- 同时保留其他作者 facts。

评估时分开报告：

| 指标集合            | 含义                 |
| ------------------- | -------------------- |
| Forget set          | 作者内部要遗忘 facts |
| Intra-entity retain | 同作者内部保留 facts |
| Inter-entity retain | 其他作者 facts       |
| Global retain       | 全体 retain facts    |

最重要的是：

[
\boxed{
\text{Intra-entity retain utility}
}
]

------

# 16. 实验方案

## 16.1 实验一：FineTOFU 是否更难

比较：

| Setting     | Forget             | Retain                            |
| ----------- | ------------------ | --------------------------------- |
| Coarse TOFU | 整个作者           | 其他作者                          |
| FineTOFU    | 作者内部部分 facts | 同作者其他 facts + 其他作者 facts |

需要证明：

[
\text{FineTOFU 上 intra-entity retain damage 更严重}
]

否则细粒度故事不成立。

------

## 16.2 实验二：prior 预测性比较

比较：

[
Semantic,\ GS,\ PI
]

预测真实 retain damage 的能力。

指标：

- Pearson；
- Spearman；
- Top-K precision；
- Top-K damage lift；
- enrichment；
- oracle overlap。

核心问题：

[
\boxed{
\text{GS 和 PI 谁更能预测实际 retain damage？}
}
]

------

## 16.3 实验三：BRIDGE 主方法比较

以 NPO 作为第一个 backbone，对比：

| 方法               | 目的             |
| ------------------ | ---------------- |
| NPO                | 原始遗忘方法     |
| NPO + global KL    | 全局平均保留     |
| NPO + uniform DRO  | 无 prior 的 DRO  |
| NPO + semantic DRO | 语义 prior       |
| NPO + GS-DRO       | 一阶梯度 prior   |
| NPO + PI-DRO       | 有限步预测 prior |
| NPO + oracle DRO   | 上界             |

这组实验决定主方法用 GS 还是 PI。

------

## 16.4 实验四：Forget-quality matched analysis

必须做：

[
\text{Forget Quality}
\quad vs.
\quad
\text{Intra-entity Retain Utility}
]

或者：

[
\text{Target Leakage}
\quad vs.
\quad
\text{Boundary Retain Damage}
]

不能只报告固定超参下 retain 更好。

核心结论应该是：

[
\boxed{
\text{在相同 forget quality 下，BRIDGE 降低 intra-entity retain damage。}
}
]

------

# 17. 论文贡献表述

建议贡献写成四点。

## Contribution 1：细粒度实体遗忘任务

提出 Fine-Grained Entity-Level Unlearning，关注同一实体内部部分事实遗忘与其他事实保留之间的边界冲突。

## Contribution 2：FineTOFU benchmark

从 TOFU 派生 FineTOFU，将每个作者内部 QA 拆分为 intra-entity forget / retain facts，用于评估边界保持能力。

## Contribution 3：BRIDGE 框架

提出 BRIDGE，一个边界鲁棒 instance-level DRO 框架，通过 global retain constraint 和 predictive-prior boundary DRO 同时控制整体保留和局部边界保留。

## Contribution 4：prior 比较

系统比较 semantic prior、GS prior 和 PI prior，分析一阶梯度冲突与有限步预测干扰在细粒度遗忘中的作用。

------

# 18. 最终一句话版本

当前论文可以这样定义：

[
\boxed{
\textbf{BRIDGE} \text{ 将细粒度 LLM 遗忘建模为带 prior 的 instance-level DRO 问题。}
}
]

它不是平均保护 retain set，也不是显式寻找一个固定纠缠 retain 子集，而是：

[
\boxed{
\text{通过 GS 或 PI 构造 retain boundary prior，再由 DRO 动态形成最坏 retain 分布，并用约束优化控制其 drift。}
}
]

当前未定的是：

[
\boxed{
\text{prior 使用 GS 还是 PI。}
}
]

因此，论文现阶段的核心实验就是比较：

[
\boxed{
GS\text{-}DRO
\quad vs.
\quad
PI\text{-}DRO
}
]

谁更适合作为 BRIDGE 的最终 prior。