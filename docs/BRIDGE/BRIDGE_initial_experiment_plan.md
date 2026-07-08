# BRIDGE 初步实验计划

本文档覆盖旧版初始实验计划，用于指导当前仓库中的 BRIDGE 第一轮实现与实验。计划基于截至 2026-07-08 已完成的 OpenUnlearning TOFU 复现结果，而不是从零开始假设 baseline 尚未跑通。

## 1. 当前前提

基础环境与数据已经具备：

| 项目 | 当前状态 |
|---|---|
| 代码框架 | OpenUnlearning |
| 数据集 | TOFU 已验证，本地 JSON 与 HF cache 一致 |
| 结果目录 | 本地实验统一保存到 `results/`，不与官方 `saves/` 混用 |
| full model | Llama-3.2-1B-Instruct 与 Llama-2-7b-chat-hf full baseline 已完成 eval |
| unlearn 汇总 | `results/unlearn_summary.csv` 和 `results/unlearn_summary.md` 只汇总 final 结果 |
| checkpoint 汇总 | `results/unlearn_summary_all.csv` 保留 checkpoint-* 中间 eval，用于动态分析 |

注意：

```text
checkpoint-* 目录是训练中途 evaluator 产物，不是模型断点。
2 GPU 训练时 OpenUnlearning 自定义 evaluator 不会跑中途 eval，因此 Llama2-7B 通常只有 final eval。
```

## 2. 已完成 baseline 快照

当前 final unlearn 结果共有 13 条。核心结论如下。

### 2.1 Llama-3.2-1B-Instruct

| split | method | MU | FQ | forget_Q_A_Prob | forget_Q_A_ROUGE | 观察 |
|---|---|---:|---:|---:|---:|---|
| forget01 | NPO | 0.539 | 0.579 | 0.0777 | 0.291 | 当前最均衡的 1B 起点 |
| forget01 | SimNPO | 0.585 | 0.0286 | 0.587 | 0.462 | utility 高，但遗忘弱 |
| forget01 | RMU | 0.564 | 0.00676 | 0.00265 | 0.123 | 遗忘强，utility 保持较好，适合迁移测试 |
| forget01 | GradDiff | 0.366 | 0.00676 | 0.000147 | 0.155 | 遗忘强，但 utility 损失明显 |
| forget01 | GradAscent | 0 | 1.86e-23 | 7.10e-15 | 0 | utility 崩溃，只适合压力测试 |
| forget05 | NPO | 0.490 | 0.545 | 0.0840 | 0.245 | NPO 中等 split 已有结果 |
| forget10 | NPO | 0.576 | 0.00491 | 0.0816 | 0.266 | NPO 大 split 已有结果 |
| forget10 | GradAscent | 0 | 1.06e-239 | 1.15e-39 | 0 | utility 崩溃 |

初步判断：

```text
1B 的 BRIDGE 开发应以 NPO 为主，因为它稳定、便宜、三种 split 都已有结果。
RMU 是最值得作为第二 backbone 的方法，因为它在 forget01 上保持了较好 MU 且遗忘强。
GradDiff 可作为强遗忘但 retain damage 明显的方法，用于检验 BRIDGE 是否能修复 utility。
SimNPO 可作为弱遗忘对照，不宜作为首个 BRIDGE 主线。
GradAscent 不作为主实验 backbone，只保留为 stress test。
```

### 2.2 Llama-2-7b-chat-hf

| split | method | MU | FQ | forget_Q_A_Prob | forget_Q_A_ROUGE | 观察 |
|---|---|---:|---:|---:|---:|---|
| forget01 | NPO | 0.622 | 0.0286 | 0.459 | 0.522 | forget01 NPO 已完成，其他方法仍待补齐 |
| forget10 | NPO | 0.529 | 0.367 | 0.130 | 0.261 | Llama2 forget10 中最适合做 BRIDGE 对照 |
| forget10 | SimNPO | 0.601 | 1.49e-16 | 0.509 | 0.494 | 遗忘不足 |
| forget10 | GradDiff | 0.548 | 4.31e-223 | 1.97e-27 | 0.00381 | 遗忘极强，需结合 FQ/MU 判断 |
| forget10 | GradAscent | 0 | 1.06e-239 | 0 | 0.00394 | utility 崩溃 |

初步判断：

```text
Llama2-7B 暂时不适合作为 BRIDGE 第一开发平台。
它应作为 1B 上方法跑通后的 scalability / transfer 验证。
forget10 的 NPO baseline 已经足够做第一轮大模型对照。
forget01 需要等 pending 的 SimNPO / GradDiff / GradAscent / RMU 或其他方法补齐后再纳入完整表。
```

## 3. 第一轮目标

第一轮实验不是最终论文主实验，而是回答：

```text
BRIDGE 是否能在低成本设置下改善 retain-side damage，并且不明显破坏遗忘效果？
```

具体问题：

```text
1. Global retain KL 是否能改善 baseline 的 retain/utility？
2. Uniform-DRO 是否优于只做平均 Global KL？
3. History-DRO 是否提供低成本风险先验？
4. GS-DRO 是否比 History-DRO 更能捕捉当前 step 的 retain vulnerability？
5. KL-PI-DRO 是否值得承担更高计算成本？
6. 最有效的 prior 能否从 1B/NPO 迁移到 RMU、GradDiff 或 Llama2/NPO？
```

第一轮保持 single seed，不做完整多 seed 与 FQ-matched Pareto。只有在单 seed 上看到明确趋势后，再进入论文主实验矩阵。

## 4. 实验优先级

### Phase 0：冻结基础设施

目标：保证后续 BRIDGE 结果与当前 baseline 可直接比较。

必须固定：

```text
结果输出目录：results/
final 汇总脚本：sbatch/summarize_unlearn_results.py
主汇总文件：results/unlearn_summary.csv / .md
checkpoint 动态汇总：results/unlearn_summary_all.csv
TOFU 数据 cache：/home/zkzhang/unlearning/HF_CACHE/datasets
full model 路径：/home/zkzhang/models/
```

执行规则：

```text
所有 BRIDGE 训练目录命名必须包含 model、split、backbone、BRIDGE variant、run_tag。
所有正式比较只使用 final eval。
checkpoint-* eval 只用于训练动态分析，不进入主表。
```

### Phase 1：补齐必要 baseline

优先级如下：

| 优先级 | 实验 | 原因 |
|---|---|---|
| P0 | 确认 Llama2 forget01 pending 任务是否全部完成 | 补齐大模型小 split 对照 |
| P1 | 1B forget05 / forget10 的 RMU | 验证 RMU 是否在不同 split 上稳定 |
| P1 | 1B forget05 / forget10 的 GradDiff | 给 BRIDGE 一个 retain damage 更明显的修复对象 |
| P2 | 1B forget05 / forget10 的 SimNPO | 作为弱遗忘对照 |
| P3 | GradAscent 扩展 | 仅压力测试，不作为主线 |

不建议为了“表格完整”立即跑所有方法。BRIDGE 初期更需要一组稳定且有区分度的 backbone：

```text
NPO: balanced baseline
RMU: strong forget with usable utility
GradDiff: strong forget with utility damage
SimNPO: weak-forget negative control
```

### Phase 2：实现 BRIDGE-minimal

先实现最小可运行版本，不直接跳到 GS/KL-PI。

必须包含：

```text
1. per-sample retain KL drift
2. Global retain KL
3. Uniform-DRO
4. fixed lambda_g / lambda_b objective
5. final eval 与现有 summary 兼容
```

暂缓：

```text
History-DRO memory
GS per-sample gradient scoring
KL-PI virtual update
augmented Lagrangian
full retain-set scanning
multi-seed sweep
```

推荐 objective：

```math
\mathcal{L}_{BRIDGE}
=
\mathcal{L}_{erase}
+
\lambda_g G_{KL}
+
\lambda_b B_{\pi}
```

其中：

```math
B_{\pi}(\theta; C_r)
=
\tau \log \sum_{i \in C_r} sg(\pi_i)\exp(d_i(\theta)/\tau)
```

Uniform-DRO 时：

```math
\pi_i = 1 / |C_r|
```

BRIDGE 的 retain-side 梯度可以写成一个清晰的样本级加权形式。对任意 prior \(\pi\)，DRO 最优权重为：

```math
q_i^\star
=
\frac{
sg(\pi_i)\exp(d_i(\theta)/\tau)
}{
\sum_j sg(\pi_j)\exp(d_j(\theta)/\tau)
}
```

因此：

```math
\nabla_\theta B_\pi
=
\sum_i q_i^\star \nabla_\theta d_i(\theta)
```

这个 proposition 是后续论文方法部分需要保留的核心解释：BRIDGE 不是只加一个平均 retain penalty，而是把保护梯度显式落到每个高风险 retain 样本上。

### Phase 3：1B / NPO / forget01 sanity check

首个 BRIDGE 实验只跑：

```text
Model: Llama-3.2-1B-Instruct
Split: forget01
Backbone: NPO
Seed: 0
```

对照：

| group | method |
|---|---|
| baseline | NPO |
| retain KL | NPO + Global KL |
| DRO | NPO + Global KL + Uniform-DRO |

建议从极小超参网格开始：

| 参数 | 候选 |
|---|---|
| lambda_g | 0.03, 0.1 |
| lambda_b | 0.1, 0.3 |
| tau | 1.0 |
| candidate pool | 当前 retain mini-batch，先不额外扩池 |

若资源紧张，第一轮只跑：

```text
lambda_g = 0.1
lambda_b = 0.3
tau = 1.0
```

成功标准：

```text
在 forget_Q_A_Prob / forget_Q_A_ROUGE 不明显变差的前提下，
MU 高于 NPO baseline，或 retain-side KL damage 明显下降。
```

失败后优先调整：

```text
lambda_b 过大导致忘不掉：降低到 0.1
lambda_g 过大导致 erase 被压制：降低到 0.03
KL drift 噪声大：先检查 per-sample mask 和 label shift 是否正确
显存压力大：candidate pool 退回当前 batch，不做额外 retain sampling
```

### Phase 4：split robustness

Phase 3 有正信号后，把同一配置迁移到：

```text
Llama-3.2-1B-Instruct / NPO / forget05
Llama-3.2-1B-Instruct / NPO / forget10
```

目标：

```text
确认 BRIDGE 不是只在 forget01 上偶然有效。
```

判断方式：

```text
如果 forget01 有效但 forget05/10 无效，优先检查 lambda 是否随 forget size 需要缩放。
如果 forget05/10 忘不掉，降低 retain-side 权重。
如果 forget05/10 utility 仍下降，说明需要 History/GS/KL-PI prior。
```

### Phase 5：prior ladder

只在 NPO 上完成 prior 消融，顺序如下：

| 顺序 | variant | 目的 |
|---|---|---|
| 1 | Global KL | 平均 retain 保护下限 |
| 2 | Uniform-DRO | 验证 instance-level DRO 本身 |
| 3 | History-DRO | 验证低成本历史 drift prior |
| 4 | Refresh-GS-DRO | 验证当前步梯度风险 |
| 5 | Refresh-KL-PI-DRO | 验证有限步 KL drift 前瞻风险 |
| 6 | Online-GS / Online-KL-PI | 只在 Refresh 版本有效时再跑 |

推荐先在：

```text
1B / NPO / forget01
```

跑完整 ladder。之后只把最好的 1-2 个 prior 搬到 forget05/10。

History-DRO 使用 last-observed drift，而不是相邻 step 的全局差分。对样本 \(x_i\)，记录它上一次被观察到时的参数状态 \(\theta_{last(i)}\)，定义：

```math
h_i^{(t)}
=
d_i(\theta_t)
-
d_i(\theta_{last(i)})
```

这样更适合 mini-batch 训练，因为 retain 样本不会在每个 step 都出现。

History-DRO 初始参数：

| 参数 | 值 |
|---|---|
| EMA beta | 0.9 |
| tau_p | 1.0 |
| tau | 1.0 |
| lambda_g | 继承 Phase 3 最优 |
| lambda_b | 继承 Phase 3 最优 |

GS-DRO 使用 update-direction 版本，避免符号约定歧义。先用当前 backbone 构造一步遗忘更新方向：

```math
u_f
=
\theta^+
-
\theta
```

其中 \(\theta^+\) 表示沿当前 forget/update 方向虚拟走一步后的参数。GS prior score 定义为：

```math
s_i^{GS}
=
\langle \nabla_\theta \ell_i(\theta), u_f \rangle
```

分数越大，表示沿当前遗忘更新方向前进后，该 retain 样本的 loss 越可能上升。

KL-PI-DRO 的主定义使用 KL-PI，而不是 loss-PI：

```math
s_i^{PI}
=
d_i(\theta^+)
-
d_i(\theta)
```

其中 \(d_i(\theta)\) 是 retain KL drift。loss-PI 可以保留为低成本变体或 ablation，但不作为主定义。

GS/KL-PI 初始参数：

| 参数 | 值 |
|---|---|
| update | Refresh |
| refresh interval K | 20 或 50 |
| tau_p | 1.0 |
| tau | 1.0 |
| trainable params | 优先只在 LoRA / active trainable params 上算 |
| prior gradient | stop-gradient |

### Phase 6：backbone transfer

完成 NPO prior ladder 后，再做迁移。

推荐顺序：

| 顺序 | backbone | 原因 |
|---|---|---|
| 1 | RMU | forget01 上 utility 好且遗忘强，是最有希望的迁移对象 |
| 2 | GradDiff | utility 损失明显，适合验证 BRIDGE 的修复能力 |
| 3 | SimNPO | 弱遗忘对照，检验 BRIDGE 是否会进一步削弱 erase |
| 4 | GradAscent | 只做 stress test，不进入主结论 |

迁移时不要重复完整 prior ladder。只跑：

```text
Backbone baseline
Backbone + Global KL
Backbone + best BRIDGE prior from NPO
```

首选 split：

```text
forget01 -> 快速验证
forget10 -> 稳健性验证
```

### Phase 7：Llama2-7B scalability

只有在 1B 上得到清楚趋势后，才启动 Llama2-7B BRIDGE。

推荐最小矩阵：

| model | split | backbone | variants |
|---|---|---|---|
| Llama-2-7b-chat-hf | forget10 | NPO | baseline, Global KL, best BRIDGE prior |
| Llama-2-7b-chat-hf | forget01 | NPO | baseline, best BRIDGE prior |

如果 RMU 或 GradDiff 在 1B 上也有明显收益，再加入：

```text
Llama2 / forget10 / RMU or GradDiff / best BRIDGE prior
```

资源建议：

| model | GPU | CPU | mem | time | 说明 |
|---|---:|---:|---:|---:|---|
| 1B | 1 x A100/H100 80G | 8 | 64-96G | 6-12h | 开发与小矩阵足够 |
| Llama2-7B | 2 x A100/H100 80G | 8-16 | 128-160G | 4-8h | 与当前复现口径更一致 |

1 GPU Llama2 可作为排队更快的探索选项，但需要降低 per-device batch 或提高 gradient accumulation，并且不作为首选可比设置。

## 5. 指标与比较规则

主表沿用 OpenUnlearning evaluator，不单独为 BRIDGE 更换评测口径。

必看字段：

```text
model_utility
forget_quality
forget_truth_ratio
forget_Q_A_Prob
forget_Q_A_ROUGE
extraction_strength
privleak
```

BRIDGE 额外记录：

```text
mean retain KL drift
worst-k retain KL drift
prior-weighted retain KL drift
boundary DRO risk
prior entropy
top-k DRO weights
prior computation time
peak GPU memory
training walltime
```

解释规则：

```text
1. 不只看 MU，也不能只看 FQ。
2. forget_Q_A_Prob、forget_Q_A_ROUGE、extraction_strength 用于判断遗忘强度是否被破坏。
3. 在遗忘强度相近时，MU 更高或 retain KL drift 更低才算 BRIDGE 有效。
4. 如果 MU 上升但忘不掉，不能算有效。
5. 如果忘得更干净但 MU 崩溃，也不能算有效。
6. GradAscent 类 utility=0 的结果只作为失败/压力参照。
```

## 6. 第一轮最小实验矩阵

### 6.1 必跑矩阵

| ID | model | split | backbone | variant | 目的 |
|---|---|---|---|---|---|
| M1 | 1B | forget01 | NPO | baseline | 已有 |
| M2 | 1B | forget01 | NPO | Global KL | 平均 retain KL 对照 |
| M3 | 1B | forget01 | NPO | Uniform-DRO | DRO sanity check |
| M4 | 1B | forget05 | NPO | best from M2/M3 | split 扩展 |
| M5 | 1B | forget10 | NPO | best from M2/M3 | split 扩展 |
| M6 | 1B | forget01 | NPO | History-DRO | 低成本 prior |
| M7 | 1B | forget01 | NPO | Refresh-GS-DRO | 梯度 prior |
| M8 | 1B | forget01 | NPO | Refresh-KL-PI-DRO | KL drift 前瞻 prior |

### 6.2 有信号后再跑

| ID | model | split | backbone | variant | 目的 |
|---|---|---|---|---|---|
| T1 | 1B | forget01 | RMU | best BRIDGE prior | 迁移到强遗忘 backbone |
| T2 | 1B | forget01 | GradDiff | best BRIDGE prior | 修复 utility damage |
| T3 | 1B | forget10 | RMU or GradDiff | best BRIDGE prior | 大 split 迁移 |
| S1 | Llama2 | forget10 | NPO | best BRIDGE prior | 大模型验证 |
| S2 | Llama2 | forget01 | NPO | best BRIDGE prior | 大模型小 split 验证 |

暂不跑：

```text
多 seed
完整 FQ-MU Pareto
所有 backbone x 所有 prior
FineTOFU
Oracle-DRO
augmented Lagrangian
```

## 7. 预期决策点

### Decision A：Uniform-DRO 是否值得继续

继续条件：

```text
Uniform-DRO 相比 Global KL 在相近遗忘强度下提升 MU，
或显著降低 retain KL drift / worst-k drift。
```

停止条件：

```text
Uniform-DRO 无改善，且调小 lambda_b 后仍无改善。
```

### Decision B：prior 是否有必要

如果：

```text
History/GS/KL-PI > Uniform-DRO
```

说明风险先验有价值。

如果：

```text
History/GS/KL-PI ≈ Uniform-DRO
```

说明 instance-level DRO 本身可能已经足够，论文主方法可偏向 efficient BRIDGE。

### Decision C：GS 与 KL-PI 如何取舍

如果：

```text
KL-PI 明显优于 GS，且开销可接受
```

KL-PI 作为主方法，GS 作为低成本版本。

如果：

```text
KL-PI 与 GS 接近
```

GS 作为主方法，KL-PI 作为增强/分析项。

如果：

```text
History 接近 GS/KL-PI
```

History-DRO 是最强低成本结论，适合突出效率。

## 8. 论文主实验的下一轮扩展

第一轮完成后，第二轮再进入论文主实验：

```text
1. seed = 0, 1, 2
2. FQ-matched comparison
3. FQ-MU Pareto curve
4. lambda_g / lambda_b sweep
5. refresh interval sweep
6. Online vs Refresh 正式比较
7. 2-3 个 backbone 扩展到 5-6 个 backbone
8. Llama2-7B 完整主表
9. 成本收益分析
```

第二轮核心问题：

```text
在相近遗忘强度下，BRIDGE 是否稳定提升 model utility 或降低 retain-side damage？
```

## 9. 当前最推荐的下一步

立即执行顺序：

```text
1. 实现 per-sample retain KL drift、Global KL、Uniform-DRO。
2. 在 1B / forget01 / NPO 上跑 Global KL 与 Uniform-DRO。
3. 若 Uniform-DRO 有效，迁移到 forget05 / forget10。
4. 再实现 History-DRO。
5. 只有 History/Uniform 有明确趋势后，再投入 GS/KL-PI。
6. 最优 prior 再迁移到 RMU / GradDiff。
7. 最后做 Llama2-7B scalability。
```

一句话版本：

```text
先用 1B/NPO/forget01 证明 BRIDGE 的 retain-side DRO 机制有效，再扩 split、扩 prior、扩 backbone，最后上 Llama2-7B。
```
