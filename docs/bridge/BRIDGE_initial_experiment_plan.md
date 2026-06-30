# BRIDGE 初版实验方案

## 1. 初版实验目标

本实验方案用于快速验证 BRIDGE 在 OpenUnlearning 官方 TOFU 设置下是否具备继续推进价值。

初版实验采用 **single seed**，优先完成 Phase 1–5：

```text
Phase 1: NPO baseline
Phase 2: KL-DRO sanity check
Phase 3: History-DRO
Phase 4: GS-DRO
Phase 5: PI-DRO
```

Phase 6 的 FQ-matched Pareto 分析和完整开销分析放到下一轮实验。

本轮实验的核心目标不是形成最终论文表格，而是快速回答：

```text
1. KL-based boundary DRO 是否优于普通 Global KL？
2. History-DRO 是否是有效的低成本 prior？
3. GS-DRO 是否优于 History-DRO？
4. PI-DRO 是否优于 GS-DRO？
5. 哪个 prior 最值得进入下一轮多 seed + FQ-matched 实验？
```

---

## 2. 总体设置

| 项目 | 初版设置 |
|---|---|
| Framework | OpenUnlearning |
| Dataset | 官方 TOFU |
| Main split | forget10 / retain90 |
| Backbone | NPO |
| Seed | single seed，建议 seed=0 |
| Main metrics | Forget Quality (FQ), Model Utility (MU) |
| Retain drift | KL drift |
| Prior candidates | History, GS, PI |
| Prior update | Online 或 Refresh |
| First-round scope | Phase 1–5 |
| Deferred scope | Phase 6: FQ-matched + full cost analysis |

首轮只使用 NPO backbone。若 Phase 1–5 结果显示 BRIDGE 有明显增益，再进入下一轮多 seed、FQ-matched 和更多 backbone。

---

## 3. 本轮实验阶段

| 阶段 | 名称 | 是否本轮执行 | 目的 |
|---|---|---|---|
| Phase 1 | Baseline reproduction | 是 | 复现 NPO |
| Phase 2 | KL-DRO sanity check | 是 | 验证 Global KL / Uniform-DRO |
| Phase 3 | History-DRO | 是 | 验证历史 KL drift prior |
| Phase 4 | GS-DRO | 是 | 验证 Online/Refresh GS prior |
| Phase 5 | PI-DRO | 是 | 验证 Refresh/Online PI prior |
| Phase 6 | FQ-matched & cost analysis | 下一轮 | 多超参、多 seed、Pareto 与完整开销 |

---

# 4. Phase 1：NPO baseline

## 4.1 目标

复现 OpenUnlearning 官方 TOFU 上的 NPO 结果，作为所有 BRIDGE 变体的基准。

## 4.2 方法

```text
NPO
```

## 4.3 记录指标

```text
Forget Quality (FQ)
Model Utility (MU)
Forget-side answer probability / truth ratio
Retain-side utility
Training time
Peak memory
```

## 4.4 成功标准

NPO 结果应与已有 OpenUnlearning 配置或本地历史 baseline 处于同一量级。若差异过大，优先排查：

```text
checkpoint
TOFU split
evaluator
NPO 超参
tokenizer / chat template
训练 epoch / step
```

---

# 5. Phase 2：KL-DRO sanity check

## 5.1 目标

验证使用 KL drift 的 retain-side boundary DRO 是否优于普通平均 Global KL。

比较：

```text
NPO
NPO + Global KL
NPO + Uniform-DRO
```

---

## 5.2 Global KL

Global KL 定义：

```math
G_{KL}(\theta)
=
\frac{1}{|B_r|}
\sum_{x_i\in B_r}
D_{KL}
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_\theta(\cdot|x_i)
\right)
```

目标函数：

```math
\mathcal{L}
=
\mathcal{L}_{erase}
+
\lambda_g G_{KL}(\theta)
```

---

## 5.3 Uniform-DRO

Uniform prior：

```math
\pi_i^{uni}
=
\frac{1}{|C_r|}
```

retain KL drift：

```math
d_i(\theta)
=
D_{KL}
\left(
p_{\theta_0}(\cdot|x_i)
\Vert
p_\theta(\cdot|x_i)
\right)
```

Uniform-DRO risk：

```math
B_{uni}(\theta;C_r)
=
\tau
\log
\frac{1}{|C_r|}
\sum_{i\in C_r}
\exp(d_i(\theta)/\tau)
```

目标函数：

```math
\mathcal{L}
=
\mathcal{L}_{erase}
+
\lambda_g G_{KL}
+
\lambda_b B_{uni}
```

---

## 5.4 初版超参数

为了控制首轮成本，Phase 2 采用小规模扫描。

| 参数 | 初版候选 |
|---|---|
| \(\lambda_g\) | 0.1 |
| \(\lambda_b\) | 0.1, 0.3 |
| DRO temperature \(\tau\) | 1.0 |
| Candidate pool size \(|C_r|\) | 32 或 64 |
| Seed | 0 |

如果算力非常有限，可以只跑：

```text
lambda_g = 0.1
lambda_b = 0.3
tau = 1.0
candidate pool = 64
```

---

## 5.5 成功标准

如果：

```text
Uniform-DRO > Global KL
```

或在相近 FQ 下 Uniform-DRO 的 MU / retain KL drift 更好，说明 KL-based boundary DRO 值得继续。

若 Uniform-DRO 不如 Global KL，应优先检查：

```text
KL drift 实现是否正确
temperature tau 是否过小或过大
lambda_b 是否过强导致忘不掉
candidate pool 是否过小
```

---

# 6. Phase 3：History-DRO

## 6.1 目标

验证历史 KL drift 增量是否能作为低成本 retain risk prior。

比较：

```text
NPO + Uniform-DRO
NPO + History-DRO
```

---

## 6.2 History score

对每个 retain sample \(x_i\)，定义最近观测到的 KL drift 增量：

```math
h_i^{(t)}
=
d_i(\theta_t)-d_i(\theta_{t-1})
```

使用 EMA 平滑：

```math
H_i^{(t)}
=
\beta H_i^{(t-1)}
+
(1-\beta)h_i^{(t)}
```

History score：

```math
s_i^{Hist}
=
H_i^{(t)}
```

History prior：

```math
\pi_i^{Hist}
=
\frac{
\exp(z(s_i^{Hist})/\tau_p)
}{
\sum_{j\in C_r}
\exp(z(s_j^{Hist})/\tau_p)
}
```

---

## 6.3 History-DRO risk

```math
B_{Hist}(\theta;C_r)
=
\tau
\log
\sum_{i\in C_r}
sg(\pi_i^{Hist})
\exp(d_i(\theta)/\tau)
```

目标函数：

```math
\mathcal{L}
=
\mathcal{L}_{erase}
+
\lambda_g G_{KL}
+
\lambda_b B_{Hist}
```

---

## 6.4 初版超参数

| 参数 | 初版设置 |
|---|---|
| EMA \(\beta\) | 0.9 |
| \(\tau_p\) | 1.0 |
| \(\tau\) | 1.0 |
| \(\lambda_g\) | 0.1 |
| \(\lambda_b\) | 0.3 |
| Candidate pool size | 64 |
| Seed | 0 |

初版不扫 \(\beta\)，只验证 history prior 是否有明显信号。

---

## 6.5 成功标准

如果：

```text
History-DRO > Uniform-DRO
```

说明历史 KL drift 增量可以作为有效低成本 prior。

如果：

```text
History-DRO ≈ Uniform-DRO
```

说明仅靠历史 drift 可能不足，需要 GS / PI 这类更直接的当前更新风险信号。

---

# 7. Phase 4：GS-DRO

## 7.1 目标

验证在线或周期更新的一阶梯度 prior 是否优于历史滞后 prior。

比较：

```text
Uniform-DRO
History-DRO
Refresh-GS-DRO
Online-GS-DRO
```

其中 Online-GS 可根据成本选择是否完整跑完。若开销较高，首轮至少跑 Refresh-GS-DRO。

---

## 7.2 GS score

forget update gradient：

```math
g_f
=
\nabla_\theta \mathcal{L}_{erase}(\theta;B_f)
```

retain sample gradient：

```math
g_i
=
\nabla_\theta \ell(x_i;\theta)
```

GS score：

```math
GS_i
=
-\eta\langle g_i,g_f\rangle
```

或 cosine 版本：

```math
GS_i
=
-\cos(g_i,g_f)
```

GS prior：

```math
\pi_i^{GS}
=
\frac{
\exp(z(GS_i)/\tau_p)
}{
\sum_j \exp(z(GS_j)/\tau_p)
}
```

---

## 7.3 GS-DRO risk

```math
B_{GS}(\theta;C_r)
=
\tau
\log
\sum_{i\in C_r}
sg(\pi_i^{GS})
\exp(d_i(\theta)/\tau)
```

---

## 7.4 GS 更新策略

### Refresh-GS

每隔 \(K\) 个 step 更新 GS prior。

初版推荐：

```text
K = 20 或 50
```

### Online-GS

每个 step 更新 GS prior。

初版中 Online-GS 作为可选增强。如果 Refresh-GS 明显有效，再测试 Online-GS 是否进一步提升。

---

## 7.5 初版超参数

| 参数 | 初版设置 |
|---|---|
| GS update | Refresh-GS |
| Refresh interval K | 20 或 50 |
| \(\tau_p\) | 1.0 |
| \(\tau\) | 1.0 |
| \(\lambda_g\) | 0.1 |
| \(\lambda_b\) | 0.3 |
| Candidate pool size | 64 |
| Seed | 0 |

---

## 7.6 成本控制

```text
1. 只在 C_r 内计算 GS；
2. 只在 LoRA / trainable parameters 上计算；
3. prior 使用 stop-gradient；
4. 优先使用 Refresh-GS；
5. 记录 GS prior computation time。
```

---

## 7.7 成功标准

如果：

```text
GS-DRO > History-DRO > Uniform-DRO
```

说明当前步梯度风险 prior 优于历史滞后 prior。

如果：

```text
Refresh-GS ≈ Online-GS
```

则后续主实验可以采用 Refresh-GS 作为低成本版本。

---

# 8. Phase 5：PI-DRO

## 8.1 目标

验证有限步 PI prior 是否比 GS prior 更适合估计 retain vulnerability。

比较：

```text
Uniform-DRO
History-DRO
GS-DRO
Refresh-PI-DRO
Online-PI-DRO
```

其中首轮至少跑 Refresh-PI-DRO。Online-PI-DRO 视成本决定是否跑。

---

## 8.2 PI score

构造虚拟遗忘更新：

```math
\theta^+
=
\theta+\Delta\theta_f
```

对每个 retain candidate：

```math
PI_i
=
\ell(x_i;\theta^+)-\ell(x_i;\theta)
```

PI prior：

```math
\pi_i^{PI}
=
\frac{
\exp(z(PI_i)/\tau_p)
}{
\sum_j \exp(z(PI_j)/\tau_p)
}
```

---

## 8.3 PI-DRO risk

```math
B_{PI}(\theta;C_r)
=
\tau
\log
\sum_{i\in C_r}
sg(\pi_i^{PI})
\exp(d_i(\theta)/\tau)
```

---

## 8.4 PI 更新策略

### Refresh-PI

每隔 \(K\) 个 step 更新 PI prior。

初版推荐：

```text
K = 20 或 50
```

### Online-PI

每个 step 更新 PI prior。

该版本成本最高，首轮作为可选实验。若 Refresh-PI 已经显著优于 GS，则 Online-PI 可放到下一轮。

---

## 8.5 初版超参数

| 参数 | 初版设置 |
|---|---|
| PI update | Refresh-PI |
| Refresh interval K | 20 或 50 |
| \(\tau_p\) | 1.0 |
| \(\tau\) | 1.0 |
| \(\lambda_g\) | 0.1 |
| \(\lambda_b\) | 0.3 |
| Candidate pool size | 64 |
| Seed | 0 |

---

## 8.6 成本控制

```text
1. 不复制完整模型；
2. 使用 functional update 或 LoRA-only virtual update；
3. 只在 C_r 内计算 PI；
4. prior stop-gradient；
5. 优先测试 Refresh-PI；
6. 记录额外 forward、runtime 和显存。
```

---

## 8.7 成功标准

理想结果：

```text
PI-DRO > GS-DRO > History-DRO > Uniform-DRO > Global KL
```

如果：

```text
PI-DRO ≈ GS-DRO
```

则 GS 可能是更低成本主版本，PI 作为增强或分析项。

如果：

```text
PI-DRO 更好但成本明显更高
```

则下一轮必须做成本-收益权衡。

---

# 9. 本轮最小实验矩阵

首轮 single-seed 最小矩阵如下：

| Group | Method | Seed | 必跑 | Notes |
|---|---|---:|---|---|
| Baseline | NPO | 0 | 是 | Phase 1 |
| Retain KL | NPO + Global KL | 0 | 是 | Phase 2 |
| DRO | NPO + Uniform-DRO | 0 | 是 | Phase 2 |
| History | NPO + History-DRO | 0 | 是 | Phase 3 |
| GS | NPO + Refresh-GS-DRO | 0 | 是 | Phase 4 |
| PI | NPO + Refresh-PI-DRO | 0 | 是 | Phase 5 |
| Optional GS | NPO + Online-GS-DRO | 0 | 可选 | 成本允许再跑 |
| Optional PI | NPO + Online-PI-DRO | 0 | 可选 | 成本允许再跑 |

首轮暂不跑：

```text
多 seed
FQ-matched Pareto
完整 cost analysis
多 backbone
FineTOFU
Residual-PI
Oracle-DRO 大规模训练
```

---

# 10. 本轮推荐超参数

## 10.1 固定参数

| 参数 | 推荐值 |
|---|---|
| Split | forget10 |
| Backbone | NPO |
| Seed | 0 |
| Candidate pool size | 64 |
| DRO temperature \(\tau\) | 1.0 |
| Prior temperature \(\tau_p\) | 1.0 |
| Refresh interval K | 20 或 50 |
| History EMA \(\beta\) | 0.9 |
| \(\lambda_g\) | 0.1 |
| \(\lambda_b\) | 0.3 |

## 10.2 可选轻量扫描

若算力允许，可对关键方法增加一个 \(\lambda_b\) 对照：

```text
lambda_b = 0.1, 0.3
```

优先顺序：

```text
Uniform-DRO
History-DRO
Refresh-GS-DRO
Refresh-PI-DRO
```

---

# 11. 本轮输出指标

## 11.1 主指标

```text
Forget Quality (FQ)
Model Utility (MU)
```

## 11.2 Retain-side KL 指标

```text
Mean retain KL drift
Worst-k retain KL drift
Prior-weighted retain KL drift
Boundary DRO risk
```

## 11.3 Prior 质量指标

若实现方便，记录：

```text
Prior score distribution
DRO weight distribution
Top-k high-weight retain examples
Prior-weighted KL drift
```

## 11.4 轻量成本指标

本轮不做完整 cost analysis，但至少记录：

```text
training time
peak GPU memory
prior update strategy
refresh interval
candidate pool size
```

完整 runtime x、extra forward/backward、Pareto 成本收益表放到下一轮。

---

# 12. 本轮成功判定

## 12.1 最低成功

```text
Uniform-DRO 优于 Global KL
```

说明 KL-based boundary DRO 值得继续。

## 12.2 中等成功

```text
History-DRO 优于 Uniform-DRO
```

说明历史 KL drift 增量可以作为低成本 prior。

## 12.3 强成功

```text
Refresh-GS-DRO 优于 History-DRO
```

说明当前步梯度风险提供了超越历史滞后信号的信息。

## 12.4 最强成功

```text
Refresh-PI-DRO 优于 Refresh-GS-DRO
```

说明有限步 PI prior 更适合 retain vulnerability 估计。

---

# 13. Phase 6：下一轮实验

Phase 6 不在本轮执行。

下一轮需要做：

```text
1. 多 seed：0, 1, 2
2. FQ-matched analysis
3. FQ-MU Pareto curve
4. 完整 cost analysis
5. Refresh interval sweep
6. lambda_b / lambda_g sweep
7. Online vs Refresh 正式比较
8. 最优 prior 的扩展 backbone 实验
```

下一轮重点回答：

```text
BRIDGE 是否在相同 FQ 下稳定提升 MU 或降低 retain KL drift？
```

---

# 14. 结果解释规则

| 结果 | 下一步 |
|---|---|
| Uniform-DRO 明显有效 | 下一轮主推 boundary DRO |
| History-DRO 接近 GS/PI | 下一轮重点做低成本 History-DRO |
| GS-DRO 明显优于 History | 下一轮主推 GS-prior |
| PI-DRO 明显优于 GS | 下一轮主推 PI-prior |
| PI-DRO 更好但太贵 | 下一轮做性能-成本权衡 |
| 所有 DRO 都使 FQ 变差 | 调低 \(\lambda_b\)，做 FQ-matched 后再判断 |
| MU 提升但 FQ 明显下降 | 不能直接作为有效结果，需要下一轮 Phase 6 验证 |

---

# 15. 本轮最终目标

本轮实验结束后，需要得到一个清晰判断：

```text
在 single-seed OpenUnlearning TOFU forget10 设置下，
BRIDGE 的哪一种 retain-side DRO prior 最值得进入下一轮多 seed + FQ-matched 实验？
```

候选结论包括：

```text
1. Uniform-DRO 已足够有效；
2. History-DRO 是最优低成本版本；
3. GS-DRO 是性能和成本折中的主版本；
4. PI-DRO 是最强版本；
5. 当前超参下 BRIDGE 效果不稳定，需要重新调参。
```
