# BRIDGE：面向 LLM 遗忘的边界鲁棒实例级 DRO 框架

## 1. 论文标题

**BRIDGE: Boundary-Robust Instance-level Distributional Guidance for Efficient LLM Unlearning**

中文理解：

**BRIDGE：面向高效大语言模型遗忘的边界鲁棒实例级分布引导框架**

标题中的关键词含义如下：

- **Boundary-Robust**：关注遗忘过程中 retain 边界是否被破坏；
- **Instance-level**：不是只保护整体 retain 平均表现，而是关注 retain 样本级风险；
- **Distributional Guidance**：通过分布鲁棒优化形成高风险 retain 子分布；
- **Efficient LLM Unlearning**：方法需要兼容 OpenUnlearning 训练流程，并控制额外开销。

---

## 2. 研究动机

大语言模型机器遗忘要求模型删除指定知识，同时尽量保持其余知识和通用能力不受影响。现有方法常通过 retain loss、retain KL、梯度去相关或效用恢复等方式维持保留性能。

但是，retain damage 通常不是均匀发生的。模型可能在大部分 retain 样本上保持稳定，却在少数与 forget update 强相关的 retain 样本上发生明显误伤。仅优化平均 retain utility 会掩盖这种局部边界损伤。

因此，本文关注的问题是：

```text
如何在 LLM unlearning 中，高效保护最容易被 forget update 误伤的 retain boundary samples？
```

BRIDGE 的核心思想是：不平均保护所有 retain 样本，也不把“寻找受损 retain 子集”作为单独框架层，而是在每个训练 step 的 retain candidate mini-batch 上构造风险先验，并通过 instance-level DRO 动态形成最坏 retain 分布，从而对高风险 retain 样本分配更高保护强度。

---

## 3. 方法定位

BRIDGE 是一个 retain-side prior-guided instance-level DRO wrapper，可以接入现有 OpenUnlearning backbone，例如：

```text
NPO
GradDiff
SimNPO
CEU
RMU
GradAscent
```

本文首版以 **NPO** 为主 backbone。

BRIDGE 不重新设计遗忘损失，而是在已有遗忘目标上加入两个 retain-side 组件：

1. **Global retain KL**：保持整体 retain 行为不偏离 reference model；
2. **Prior-guided boundary DRO**：重点保护高风险 retain candidate 分布。

整体形式为：

```math
\mathcal{L}_{BRIDGE}
=
\mathcal{L}_{erase}
+
\lambda_g G_{KL}
+
\lambda_b B_\pi
```

其中：

- \(\mathcal{L}_{erase}\)：来自 NPO 等 unlearning backbone；
- \(G_{KL}\)：global retain KL；
- \(B_\pi\)：prior-guided boundary DRO risk。

---

## 4. 与相关工作的关系

### 4.1 与 OFMU 的区别

OFMU 关注 forget objective 和 retain objective 之间的整体梯度冲突，通常在 batch-level 或 objective-level 上计算：

```math
Sim(\nabla L_f(B_f), \nabla L_r(B_r))
```

其作用是控制整体遗忘更新方向，使 forget update 不与 retain objective 严重冲突。

BRIDGE 的关注点不同。BRIDGE 对 retain candidate mini-batch 中的每个 retain 样本估计风险先验：

```math
s_i,\quad x_i\in C_r
```

因此，BRIDGE 关注的是：

```text
当前 forget update 会具体误伤哪些 retain samples。
```

简言之：

```text
OFMU: objective-level update control
BRIDGE: retain-side instance-level boundary protection
```

---

### 4.2 与 BalDRO 的区别

BalDRO 主要处理 forget set 内部难忘样本不均衡问题。它在 forget mini-batch 内根据 per-sample forget loss 对 hard-to-forget samples 加权。

BRIDGE 的 DRO 作用在 retain side：

```text
BalDRO: forget-side hard sample reweighting
BRIDGE: retain-side boundary vulnerability protection
```

因此，BalDRO 主要让 hard forget samples 更充分遗忘，而 BRIDGE 主要让 high-risk retain samples 更少被误伤。

---

### 4.3 与 History-DRO 的关系

History-DRO 是本文引入的低成本对照方法。它使用历史 retain KL drift 增量作为风险先验：

```text
哪些 retain samples 最近已经被训练过程持续影响，就给它们更高保护权重。
```

History-DRO 的意义是回答一个审稿人可能提出的问题：

```text
既然 PI / GS 需要额外计算，为什么不用历史 drift 变化这种低成本信号？
```

因此，本文将比较：

```text
Uniform-DRO
History-DRO
GS-DRO
PI-DRO
```

如果 PI-DRO 或 GS-DRO 优于 History-DRO，说明在线预测或梯度风险先验确实提供了超越历史滞后信号的信息；如果 History-DRO 接近 GS/PI，则说明低成本历史先验是 BRIDGE 的有效轻量化实例。

---

## 5. Batch 与数据定义

在每个 training step，采样 forget batch：

```math
B_f \subset D_f
```

采样 retain candidate mini-batch：

```math
C_r=\{x_i\}_{i=1}^m \subset D_r
```

默认：

```math
C_r = B_r
```

其中 \(m\) 是 retain candidate batch size。

如果需要增强候选池覆盖，可扩展为：

```math
C_r = B_r^{random} \cup B_r^{semantic} \cup B_r^{intra}
```

但主方案保持 mini-batch stochastic approximation，不扫描完整 retain set。

---

## 6. Retain drift

对每个 retain candidate \(x_i\in C_r\)，定义 retain drift 为 KL divergence：

```math
d_i(\theta)
=
D_{KL}
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_{\theta}(\cdot|x_i)
\right)
```

其中：

- \(\theta_0\)：遗忘前 reference model；
- \(\theta\)：当前 unlearning model。

\(d_i(\theta)\) 表示当前模型在 retain sample \(x_i\) 上相对 reference model 的输出分布偏移程度。

本文主方法和所有 DRO 变体均使用 KL drift，不使用 loss increase 作为主 drift 定义。

---

## 7. Risk prior：Uniform / History / GS / PI

BRIDGE 的 DRO prior 记为：

```math
\pi_i
```

它表示 retain sample \(x_i\) 在当前 candidate pool 中成为高风险样本的先验权重。

本文考虑四类 prior：

```text
Uniform prior
History prior
GS prior
PI prior
```

其中 GS 和 PI 使用 online 或 refresh 更新方式。

---

## 8. Uniform prior

Uniform-DRO 不使用额外风险先验：

```math
\pi_i^{uni}
=
\frac{1}{|C_r|}
```

它只依赖当前 retain KL drift：

```math
d_i(\theta)
```

用于验证 boundary DRO 本身是否有效。

---

## 9. History prior：历史 KL drift 增量先验

History-DRO 使用历史 retain KL drift 的变化作为低成本风险信号。

对每个 retain sample \(x_i\)，定义历史 drift 增量：

```math
h_i^{(t)}
=
d_i(\theta_t)-d_i(\theta_{t-1})
```

为了降低单步噪声，使用 EMA 平滑：

```math
H_i^{(t)}
=
\beta H_i^{(t-1)}
+
(1-\beta)h_i^{(t)}
```

其中：

- \(\beta\)：EMA 平滑系数；
- \(h_i^{(t)}\)：最近一次观测到的 KL drift 增量；
- \(H_i^{(t)}\)：历史风险记忆。

History prior 定义为：

```math
s_i^{Hist}
=
H_i^{(t)}
```

并转化为 prior distribution：

```math
\pi_i^{Hist}
=
\frac{
\exp(z(s_i^{Hist})/\tau_p)
}{
\sum_{j\in C_r}\exp(z(s_j^{Hist})/\tau_p)
}
```

History-DRO 的直觉是：

```text
如果某个 retain 样本最近持续出现 KL drift 增长，
说明它已经在训练过程中被误伤，
因此应在后续 step 中获得更高保护权重。
```

History-DRO 是一种低成本、反应式风险先验。它不预测当前 forget update 的边际影响，而是利用过去已经发生的 retain drift 变化。

---

## 10. GS-prior：在线或周期更新的一阶梯度风险先验

设 forget update gradient 为：

```math
g_f
=
\nabla_\theta \mathcal{L}_{erase}(\theta;B_f)
```

retain sample \(x_i\) 的梯度为：

```math
g_i
=
\nabla_\theta \ell(x_i;\theta)
```

定义 GS score：

```math
GS_i
=
-\eta \langle g_i,g_f\rangle
```

或 cosine 形式：

```math
GS_i
=
-\cos(g_i,g_f)
```

直觉上，如果 retain sample 的梯度与 forget update 方向冲突更强，该 retain sample 更可能被当前遗忘更新误伤。

GS-prior 的更新方式：

### Online-GS

每个 training step 根据当前 \(B_f\)、\(C_r\) 和 \(\theta\) 计算 GS score。

### Refresh-GS

每隔 \(K\) 个 step 重新计算 GS score，中间 step 复用最近一次 GS prior。

GS-prior 的成本控制：

```text
1. 只在当前 retain candidate mini-batch C_r 内计算；
2. 只在 LoRA / trainable parameters 上计算；
3. prior 使用 stop-gradient；
4. 使用 Refresh-GS 降低更新频率；
5. 不对完整 retain set 计算 GS。
```

---

## 11. PI-prior：在线或周期更新的有限步预测干扰先验

根据当前 forget batch 构造虚拟遗忘更新：

```math
\theta^+
=
\theta+\Delta\theta_f
```

其中 \(\Delta\theta_f\) 来自当前 unlearning backbone 的一步更新方向。

对每个 retain candidate \(x_i\in C_r\)，定义：

```math
PI_i
=
\ell(x_i;\theta^+)-\ell(x_i;\theta)
```

PI 的直觉是：

```text
如果模型沿当前 forget update 方向走一小步后，
retain sample x_i 的 loss 上升更大，
说明它更容易被当前遗忘更新误伤。
```

PI-prior 的更新方式：

### Online-PI

每个 training step 都根据当前 \(B_f\)、\(C_r\) 和 \(\theta\) 构造虚拟更新并计算 PI。

### Refresh-PI

每隔 \(K\) 个 step 更新一次 PI prior，中间 step 复用最近一次 PI prior。

PI-prior 的成本控制：

```text
1. 不复制完整模型；
2. 优先使用 functional update 或 LoRA-only virtual update；
3. 只在当前 C_r 内计算 PI；
4. prior 使用 stop-gradient；
5. 使用 Refresh-PI 降低更新频率；
6. 不对完整 retain set 计算 PI。
```

---

## 12. Prior distribution

对于任意 prior score \(s_i\)，构造：

```math
\pi_i
=
\frac{
\exp(z(s_i)/\tau_p)
}{
\sum_{j\in C_r}
\exp(z(s_j)/\tau_p)
}
```

其中：

- \(z(\cdot)\)：candidate mini-batch 内标准化；
- \(\tau_p\)：prior temperature；
- \(s_i\)：Uniform / History / GS / PI 对应的风险分数。

训练时使用：

```math
sg(\pi_i)
```

即 prior 只作为权重分布，不参与梯度反传。

---

## 13. Prior-guided boundary DRO

BRIDGE 的核心 DRO risk 定义为：

```math
B_\pi(\theta;C_r)
=
\max_{q\in\Delta}
\left[
\sum_{i\in C_r} q_i d_i(\theta)
-
\tau D_{KL}(q\Vert \pi)
\right]
```

其闭式解为：

```math
q_i^\star
=
\frac{
\pi_i \exp(d_i(\theta)/\tau)
}{
\sum_j \pi_j \exp(d_j(\theta)/\tau)
}
```

对应 DRO risk 为：

```math
B_\pi(\theta;C_r)
=
\tau
\log
\sum_{i\in C_r}
sg(\pi_i)
\exp(d_i(\theta)/\tau)
```

含义：

```text
如果一个 retain sample 的 prior 风险高，且当前 KL drift 也高，
它就会在 DRO 中获得更高保护权重。
```

---

## 14. Global retain KL

Global retain KL 定义为：

```math
G_{KL}(\theta;B_r^{global})
=
\frac{1}{|B_r^{global}|}
\sum_{x_i\in B_r^{global}}
D_{KL}
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_{\theta}(\cdot|x_i)
\right)
```

其中 \(B_r^{global}\) 是当前 step 采样的 global retain mini-batch。

---

## 15. 固定权重训练目标

第一版主实现采用固定权重目标：

```math
\mathcal{L}_{BRIDGE}
=
\mathcal{L}_{erase}
+
\lambda_g G_{KL}(\theta;B_r^{global})
+
\lambda_b B_\pi(\theta;C_r)
```

其中：

- \(\lambda_g\)：global retain KL 权重；
- \(\lambda_b\)：boundary DRO 权重。

该版本实现简单，便于快速比较：

```text
Global KL
Uniform-DRO
History-DRO
GS-DRO
PI-DRO
```

---

## 16. 约束优化扩展

在固定权重版本有效后，可扩展为 augmented Lagrangian 版本。

定义约束：

```math
G_c(\theta)
=
G_{KL}(\theta)-\epsilon_g
```

```math
H_\pi(\theta)
=
B_\pi(\theta;C_r)-\epsilon_b
```

目标为：

```math
\mathcal{L}_{BRIDGE}
=
\mathcal{L}_{erase}
+
\lambda_g [G_c(\theta)]_+
+
\frac{\rho_g}{2}[G_c(\theta)]_+^2
+
\lambda_b [H_\pi(\theta)]_+
+
\frac{\rho_b}{2}[H_\pi(\theta)]_+^2
```

dual variables 更新：

```math
\lambda_g
\leftarrow
[\lambda_g+\rho_gG_c(\theta)]_+
```

```math
\lambda_b
\leftarrow
[\lambda_b+\rho_bH_\pi(\theta)]_+
```

---

## 17. 训练流程

每个 training step：

```text
1. 采样 forget batch B_f
2. 采样 retain candidate mini-batch C_r
3. 采样 global retain mini-batch B_r_global
4. 计算 L_erase(theta; B_f)
5. 根据 prior 类型计算或更新 s_i:
   - Uniform-DRO: pi_i = 1 / |C_r|
   - History-DRO: s_i = EMA of KL drift increase
   - GS-DRO: s_i = GS_i via Online-GS or Refresh-GS
   - PI-DRO: s_i = PI_i via Online-PI or Refresh-PI
6. 构造 pi_i = softmax(z(s_i)/tau_p)
7. 对 pi_i 使用 stop-gradient
8. 计算 retain KL drift d_i(theta)
9. 计算 boundary DRO risk B_pi(theta; C_r)
10. 计算 global retain KL G_KL(theta; B_r_global)
11. 计算 L_BRIDGE
12. 更新模型参数 theta
13. 更新 History-DRO 的 EMA memory
14. 如使用 augmented Lagrangian，更新 dual variables
```

---

## 18. 主实验设置

```text
Framework: OpenUnlearning
Dataset: 官方 TOFU
Split: forget10 / retain90
Backbone: NPO
Seeds: 0, 1, 2
Primary metrics: Forget Quality (FQ), Model Utility (MU)
```

首版保持 OpenUnlearning 官方 TOFU 口径，不强制引入 FineTOFU。

---

## 19. 主要对照方法

必须包含：

| 方法 | 说明 |
|---|---|
| NPO | 原始遗忘 backbone |
| NPO + Global KL | 平均 retain KL 保护 |
| NPO + Uniform-DRO | 无风险 prior，只依赖当前 KL drift 的 DRO |
| NPO + History-DRO | 使用历史 KL drift 增量作为低成本 prior |
| NPO + Online/Refresh GS-DRO | 使用一阶梯度风险 prior |
| NPO + Online/Refresh PI-DRO | 使用有限步 PI prior |

可选包含：

| 方法 | 说明 |
|---|---|
| Semantic-DRO | 使用语义相似度 prior |
| Oracle-DRO | 使用真实 retain damage prior，上界分析 |
| BalDRO-style forget DRO | forget-side DRO 对照 |
| OFMU-style gradient decorrelation | objective-level update control 对照 |

---

## 20. 评价指标

### 20.1 官方指标

```text
Forget Quality (FQ)
Model Utility (MU)
```

### 20.2 Retain-side 指标

```text
Mean retain KL drift
Worst-k retain KL drift
Prior-weighted retain KL drift
Boundary DRO risk
```

### 20.3 Prior 质量指标

```text
Spearman correlation with final retain KL damage
Top-k damage enrichment
Overlap with oracle high-damage retain samples
```

### 20.4 成本指标

```text
Training time
Relative runtime
Extra forward count
Extra backward count
Peak GPU memory
Prior computation time
Candidate pool size
Refresh interval
```

---

## 21. 核心分析

### 21.1 Forget-quality matched analysis

必须比较相近 FQ 下的 MU 和 retain KL drift。

建议输出：

```text
FQ-MU Pareto curve
FQ vs Worst-k retain KL drift
FQ vs Prior-weighted retain KL drift
```

### 21.2 Prior 有效性分析

比较：

```text
Uniform-DRO
History-DRO
GS-DRO
PI-DRO
```

回答：

```text
低成本历史信号是否足够？
GS 是否提供更好的梯度预测？
PI 是否提供更强的有限步前瞻预测？
```

### 21.3 开销收益分析

比较各 prior 的性能-成本 trade-off：

```text
History-DRO: 最低成本反应式 prior
GS-DRO: 中等成本梯度 prior
PI-DRO: 较高成本有限步 prior
```

---

## 22. 预期结论路径

| 结果 | 论文叙事 |
|---|---|
| Uniform-DRO > Global KL | instance-level boundary DRO 有效 |
| History-DRO > Uniform-DRO | 历史 KL drift 增量能作为低成本 prior |
| GS-DRO > History-DRO | 当前步梯度风险优于历史滞后信号 |
| PI-DRO > GS-DRO | finite-step look-ahead prior 最能捕捉 retain vulnerability |
| PI-DRO ≈ GS-DRO | GS 是低成本主方法，PI 是增强版本 |
| History-DRO ≈ GS/PI | 低成本 history prior 足够强，可作为 efficient BRIDGE 实现 |

---

## 23. 最终一句话

BRIDGE 是一个 retain-side prior-guided instance-level DRO 框架。它使用 KL drift 衡量 retain 样本的实际偏移，并利用 History、GS 或 PI 构造风险先验，通过 log-sum-exp DRO risk 动态保护高风险 retain boundary samples。在 OpenUnlearning 官方 TOFU 设置下，BRIDGE 以 mini-batch candidate pool、online/refresh prior 和 stop-gradient 权重实现高效训练。
