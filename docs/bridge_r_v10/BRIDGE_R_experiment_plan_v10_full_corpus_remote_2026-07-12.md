# BRIDGE-R 完整实验方案 v10：Full-Corpus、官方指标、远端权威

> 日期：2026-07-12  
> 唯一实现与运行仓库：服务器 `/home/zkzhang/unlearn/open-unlearning`  
> 唯一环境：沿用服务器 BRIDGE/OpenUnlearning 环境  
> 本文件不声称已完成 Atomic-TOFU 或 BRIDGE-R 新实验
> 2026-07-18 范围修订：完整 Eval-B 不再是发布或实验门槛；见 `Atomic_TOFU_Eval_B_scope_amendment_2026-07-18.md`。其余实验设计不变。

## 0. Primary Questions

1. 同一 baseline 在官方实体 TOFU 与 Atomic-TOFU 上是否表现不同？
2. 在 1%/5% forget QA 比例匹配时，原子遗忘是否更难？
3. 2-3 atom request 是否比 single-atom request 产生额外困难？
4. BRIDGE-R 是否超过 GlobalKL 与 LocalMean？

## 1. 远端事实优先

实现前在服务器记录 git commit、dirty files、环境、GPU、TOFU cache、`results/` 和 `saves/`。禁止用本机代码覆盖服务器。

本地 `outputs/unlearn_summary.csv` 有 41 条复现记录，包含两种模型和多种 baseline。其主指标字段为：

```text
model_utility
forget_quality
forget_truth_ratio
forget_Q_A_Prob
forget_Q_A_ROUGE
extraction_strength
privleak
```

服务器必须重新读取实际 `results/unlearn_summary.csv`、每个 final `TOFU_SUMMARY.json` 和 `TOFU_EVAL.json`，确认字段、指标方向、reference logs 和 final/non-final 身份。

## 2. Phase 0：官方指标依赖审计

从服务器实际 Hydra config 递归生成 dependency graph：

| Metric | 必须核对的数据 |
|---|---|
| Forget QA Probability | question + correct answer |
| Forget QA ROUGE | question + answer + generation config |
| Forget Truth Ratio | correct answer + perturbed answers |
| Forget Quality | unlearn forget TR + paired retrain forget TR logs |
| Model Utility | retain、real-author、world-fact 的 9 个子指标 |
| Extraction Strength | forget QA 与 handler 特定输入 |
| PrivLeak | forget/holdout MIA + paired retrain logs |

输出 `docs/bridge_r_v10/metric_dependency_report.md`，列出 handler、dataset config、columns、precompute、reference logs、output field、direction。

## 3. Phase 1：Full-Corpus Atomic-TOFU

### 3.1 Source invariance

```text
full rows = 4000
authors = 200
20 QA per author
question/answer hash unchanged
no split records
no source_family_weight
```

建立稳定 `qa_id`、`author_id`、原始 index 和 content hash。

### 3.2 全量 API annotation

每位作者一个 API unit，输入 20 原始 QAs。输出：

- atom candidates；每个 atom 都要对 20 条 QA 的 question 与 answer 做独立跨 QA 扫描；
  同一 QA 若同时明确涉及多个 atom，必须在每个适用 atom 的 relation 表中重复出现；
- support/leak/closure/protected/ambiguous QA relations；
- aliases、source QA、evidence span、relation-level reason/confidence，以及显式 unassigned QA coverage；
- single request candidates；
- 语义合理的 2-3 atom request candidates；
- surviving/co-deleted/dependent roles；
- provisional entanglement features/level。

每个 target 的 forget closure 由该 atom 的全部 `support|leak|closure` relation 自动确定；candidate 中的 `per_atom_closures` 必须与此集合一致。

作者身份保护只覆盖作者本人姓名 atom；父母及其他家庭成员姓名仍可进入 forget closure、训练集合和相应 protected/evaluation complement。不能把 family-member name 误当作 author scope identifier。

全量 200 authors 都处理。人工逐作者复核后才进入 gold manifests。

### 3.3 Evaluation extension

先从服务器官方 `forget01/05/10_perturbed`、`retain_perturbed` 检查准确 schema、field types、perturbed-answer cardinality 和 evaluator usage，再为 full 中缺失的 rows 生成同 schema fields。

禁止凭文档硬编码字段。已有官方 fields 不覆盖。

生成后验证：

- paraphrase semantic equivalence；
- perturbed answers factual incorrectness、type match、non-leakage；
- cardinality/schema identical；
- text length 与 lexical distribution；
- full-model correct/perturbed probability distribution；
- official-anchor regenerate calibration；
- human agreement。

冻结单一 Eval-A 作为 generated evaluation view，并让所有 Atomic 方法、bundle 和 seed 共用同一 artifact。官方 Entity01/05 主评测继续使用 official fields；hidden official anchors 的 regenerated fields 只用于 regenerate-and-compare、人工盲审和 evaluator-source 偏差量化。完整独立 Eval-B 已移出必做范围，可在时间允许时作为后续或附录研究；当前实验不得声称跨独立生成视图的绝对值、方法排序或 Atomic-vs-Entity 方向稳健。

### 3.4 Reserved official utility anchors

把 official fixed `retain_perturbed` rows 映射回 full IDs，并将对应 authors 标记为 reserved utility anchors。正式 Atomic main bundles 的 closure 不得触及这些 anchors。

若 request closure 触及 reserved anchor：

- 不得截断 closure；
- main comparable track 排除该 request；
- 可进入 `nonfixed-MU` 扩展 track；
- 报告被排除 request 的数量和特征，检查选择偏差。

## 4. Request Tasks

### 4.0 二维 request taxonomy

所有 requests 必须同时标注：

```text
Cardinality: C1 / C2 / C3
Entanglement: E-Low / E-Medium / E-High
```

E-level 由 exposure、dependency、co-carriage、protected-proximity 四维规则确定；C2/C3 额外加入 closure-overlap 与 joint-role-shift。API 只给证据候选，compiler 重算，人工确认。

### 4.1 Single bundles

| Task | QA count | Composition |
|---|---:|---|
| Single-Balanced-01 | 40 | C1；Low/Medium/High 按 forget-QA contribution 平衡 |
| Single-Balanced-05 | 200 | C1；Low/Medium/High 分层平衡 |

### 4.2 Multi bundles

| Task | QA count | Composition |
|---|---:|---|
| Multi-Balanced-01 | 40 | C2/C3；cardinality 与 Low/Medium/High 分层平衡 |
| Multi-Balanced-05 | 200 | C2/C3；cardinality 与纠缠分层平衡 |

每类至少 3 个 bundle seeds。所有 QA counts 是 closure union 后的 exact count，不允许截断 request。

### 4.3 Pure-entanglement diagnostic bundles

优先构造 exact 40-QA、3-seed 的：

```text
C1-E-Low-01
C1-E-Medium-01
C1-E-High-01
C2-E-Low/Medium/High-01 when feasible
C3-E-Low/Medium/High-01 when feasible
```

Single 的三档是必做数据目标。Multi cells 若 coverage 不足，报告 unavailable，不能降低标注标准。主要计算实验先跑 `C1-E-Low` 与 `C1-E-High` 对照，再决定是否补齐全部 cells。

### 4.4 Controls

| Control | 作用 |
|---|---|
| Official Entity01/05 | 用户要求的官方 benchmark 对比 |
| Matched-Entity01/05 | 从 Atomic eligible author pool 删除完整作者，控制作者池/生成 evaluator |
| Random-QA01/05 | 匹配 size、authors、length、initial difficulty |
| Retrain | 每个 split 的 gold counterfactual |
| Full | 未遗忘起点 |

官方 Entity01/05 主结果使用 `official fields`；hidden official anchors 的 regenerated fields 用于校准生成质量并估计 evaluator-source risk，不要求构造完整 Entity Eval-A/Eval-B 双视图。

## 5. Bundle Compiler

使用 deterministic ILP/subset-sum：

约束：

```text
union forget IDs == 40 or 200
closure complete
no reserved-anchor overlap
single or multi cardinality requirement
required entanglement cell or balanced E-level quota
max per-author contribution
minimum relation/category coverage
bounded closure overlap
```

目标只使用数据属性，不读取模型结果。输出 request 数、target atoms、authors、QA/token ratio、closure overlap、entanglement distribution。

Balanced bundles 的配额按最终 union 中各 E-level 的边际 forget-QA contribution 计算，不能简单要求三档 request 数相同。高纠缠 request 通常 closure 更大，按 request 数平衡会让 High 实际占据大部分 forget set。

## 6. 完整数据契约

每个 split/bundle 生成：

```text
full_train_manifest
forget_train_view
retain_train_view
forget_eval_view with correct/paraphrase/perturbation fields
same_author_protected_eval_view
fixed_official_retain_eval
official_real_author_eval
official_world_fact_eval
holdout_eval matched to size
retrain_train_manifest
retrain_evaluation_config
reference_log_path
release hashes
```

`protected_train_qa_ids` 与 `protected_eval_qa_ids` 的确定性定义为：某 request 作者的全部 source QA 减去该 request 的 complete forget closure。它们保持 request-scoped：即使某 QA 是同一 bundle 另一 request 的 target，也继续属于当前 request 的 protected 训练/评测 pool。全局 retain train 单独使用 bundle forget-QA union 的补集 \(G_R=D\setminus F_R\)；protected pool 不随之二次缩小。`G_R` 是完整的随机采样候选训练集，不要求固定步数内覆盖每条 QA，也不新增 global-retain remainder eval。

这一步必须由自动 `metric completeness validator` 检查，不能等 evaluator 报错才发现数据缺失。

## 7. Paired Retraining

对每个 model × split/bundle × seed：

1. 使用共同 base checkpoint；
2. full model recipe 固定；
3. retrain 在对应 retain train 上训练；
4. paired seed；
5. 在与 unlearn 完全相同的 forget/retain/RA/WF/holdout views 上评测；
6. `retain_logs_path` 指向该 split 专属 retrain `TOFU_EVAL.json`。

官方 retain99/95 model 仅做 sanity，除非 recipe/seed/provenance 完全匹配。

Oracle Gate：

- FQ、PrivLeak 非 None；
- per-index 对齐；
- MU 9 components finite；
- retrain target behavior 与 full 有可检测差异；
- 每个 active request 均有非空的同作者 non-target QA pool；
- generated eval extension 不导致 full/retrain 异常崩溃。

## 8. Task Difficulty Experiment

### 8.1 Same-baseline matrix

第一轮使用 NPO：

| Model | Ratio | Entity | Single | Multi | Matched Entity | Random |
|---|---:|---|---|---|---|---|
| Llama-3.2-1B | 1% | ✓ | 3 bundles | 3 bundles | ✓ | 3 seeds |
| Llama-3.2-1B | 5% | ✓ | 3 bundles | 3 bundles | ✓ | 3 seeds |

然后增加 GradDiff 与 RMU/SimNPO 中至少一个。

Single 内部纠缠主对照：

```text
C1-E-Low-01 vs C1-E-Medium-01 vs C1-E-High-01
```

固定 baseline、40-QA union、作者数、relation 分布和 initial difficulty。除 level 外继续报告连续 dimensions，验证 `birthdate-like sparse fact` 与 `father-profession-like reused/co-carried fact` 的差异是否稳定。

### 8.2 公平训练量

同 ratio 内固定：

- optimizer steps；
- effective batch；
- learning rate search budget；
- generation/evaluator config；
- seed policy；
- full checkpoint。

不得按 epoch 让不同 split 的 optimizer steps 隐式变化。

### 8.3 “更难”的预注册判据

Atomic 比 Entity 更难必须同时满足：

1. 在相近 MU 下，forget-side官方指标更远离 retrain；或在相近 forget band 下 MU 更低；
2. 结论不依赖单一 FQ；
3. 至少两个 baselines 同方向；
4. 三个 bundle seeds 方向稳定；
5. 相对 Matched-Entity/Random control 仍有增量。
6. hidden official-anchor 校准未显示不可接受的 schema、answer-shape、长度、难度或语义偏差；所有 Atomic 方法共用同一冻结 Eval-A，且 Atomic-vs-Entity 结论显式保留 evaluator-source caveat。

使用 7 个 official summary metrics、MU components、Pareto front 和 retrain-normalized gaps。

### 8.4 Multi 的判据

Single vs Multi 比较严格匹配 forget QA count，并控制：

- authors；
- total target atoms；
- request 数；
- closure overlap；
- initial full-model difficulty。

如果无法全部匹配，使用回归/分层分析而不是直接均值断言。

### 8.5 Entanglement level 的判据

E-Low/Medium/High 的有效性必须同时由两类证据支持：

1. pure-stratum bundles 的 official-metric/retrain-gap 呈有序或可解释差异；
2. request-level 连续 dimensions 在 author-cluster regression 中预测 protected damage。

若离散 levels 不稳定但连续 features 有效，论文删除三档排序 claim，只保留连续纠缠分析。

## 9. BRIDGE-R Implementation

### 9.1 复用旧 BRIDGE

复用服务器现有：

- NPO/GradDiff integration；
- frozen full-teacher KL；
- token mask；
- per-sample KL；
- Hydra/Slurm/results conventions。

先阅读旧 BRIDGE 结果。当前 CSV 显示强 NPO 1B forget01 `ms20_lr1e-5` 的 MU/FQ 等与旧 `retry1` 不同；部分 GlobalKL/UniformDRO 结果并非强 baseline 同一 grid，不可直接作新版因果对照。

### 9.2 Request-local sampler

manifest 提供：

```text
forget qa_id -> request_ids
request_id -> target_atom_ids
request_id -> protected_train_qa_ids
```

这里的 protected IDs 是同作者完整 non-target complement，不由 candidate explicit IDs 或 surviving-protected atom residual support 选择。它们按 request 保持原样；多 request bundle 不会因另一 request 的 target 而缩小当前 request 的 pool。

当前 forget batch 涉及多个 requests 时：

1. 合并 request IDs；
2. 每 request/author 先采样，再合并去重；
3. 所有 protect QAs 必须在 retain；
4. multi requests 不得因 closure 大而独占 pool；
5. 记录 logical K、micro-batch、coverage 和 duplicate rate。

主 K=8，消融 4/8/16/all-local。显存不足使用 chunked KL。

### 9.3 Loss ladder

\[
L=L_{erase}+\lambda_gG_{KL}+\lambda_lM_R+\lambda_tT_R.
\]

| Variant | 目的 |
|---|---|
| B0 Backbone | 起点 |
| B1 +GlobalKL | 全局平均保护 |
| B2 +LocalMean | request locality |
| B3 +Centered UniformTail | BRIDGE-R 主方法 |
| B4 uncentered tail | centered ablation |
| B5 GS prior | 只有 B3 成立后 |
| B6 KL-PI prior | 后续可选 |

若 B2 不超过 B1，删除 request-local claim。若 B3 不超过 B2，BRIDGE-R 简化为 LocalMean。

## 10. Metrics

### 10.1 Official primary

按 `unlearn_summary.csv`：

```text
model_utility
forget_quality
forget_truth_ratio
forget_Q_A_Prob
forget_Q_A_ROUGE
extraction_strength
privleak
```

并从详细 logs 导出 MU 9 components。

### 10.2 Atomic secondary

```text
same-author non-target retain Prob/ROUGE/TR
protected mean and CVaR20
request-level oracle gap
co-deleted oracle gap
local mean/worst-k KL
single/multi and entanglement strata
```

### 10.3 Cost

```text
train/eval wall time
peak GPU memory
logical pool K
extra forward/backward count
API tokens/cost
human review time
```

## 11. Execution Gates

| Gate | 要求 | 失败处理 |
|---|---|---|
| G0 Source | 4,000 QA hash 不变 | 停止 |
| G1 API pilot | 8 calibration + 12 blind authors | 修 prompt/schema |
| G2 Eval extension | anchor calibration + human quality | 不生成正式 bundles |
| G3 Full annotation | 200 authors reviewed | 不冻结 |
| G4 Bundle | exact 40/200，no truncation | 重选 requests |
| G5 Metric | FQ/MU/PrivLeak 完整 | 修 evaluator/data |
| G6 Task | Atomic 与 controls 有稳定差异 | 决定论文路线 |
| G7 Locality | B2 > B1 | 否则删 locality |
| G8 Tail | B3 > B2 | 否则简化方法 |

## 12. 最小执行顺序

1. 服务器 metric/results audit；
2. 映射 full 与所有官方 eval anchors；
3. 8 authors 人工 gold；
4. API dry run 和 12 authors blind validation；
5. 全 200 authors API + 人工复核；
6. 生成 full evaluation extension；
7. 编译 Single/Multi Balanced 40/200 bundles 与 C1 三档纯纠缠 40-QA bundles；
8. 1B seed0 retrain/evaluator smoke；
9. NPO Entity/Single/Multi/Matched/Random pilot；
10. 任务 Gate 通过后实现 B0-B3；
11. 3 seeds；
12. 第二 baseline/model。

## 13. Statistical Plan

- bundle seeds：至少 3；
- training seeds：正式主结果 3；
- author-cluster bootstrap；
- request-level mixed-effects or cluster-robust regression；
- continuous entanglement features 主分析；
- 多重检验校正；
- FQ p-value 只作为 benchmark score，不解释为本研究统计显著性。

## 14. Result Tables

### Task comparison

| Split | Method | MU | FQ | F-TR | F-Prob | F-ROUGE | ES | PrivLeak | Protected CVaR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Entity05 | NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | N/A |
| Single-Balanced-05 | NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Multi-Balanced-05 | NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Matched-Entity05 | NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Random05 | NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### BRIDGE-R

| Method | MU | FQ | Forget multi-metric band | Protected mean | Protected CVaR20 | Runtime |
|---|---:|---:|---|---:|---:|---:|
| NPO | TBD | TBD | TBD | TBD | TBD | TBD |
| +GlobalKL | TBD | TBD | TBD | TBD | TBD | TBD |
| +LocalMean | TBD | TBD | TBD | TBD | TBD | TBD |
| BRIDGE-R | TBD | TBD | TBD | TBD | TBD | TBD |

## 15. 必须避免

- 只处理官方 forget10 的 20 authors；
- 拆 QA 或加入 family weights；
- 只生成 forget/retain 而遗漏 perturbations、RA/WF、holdout、retrain logs；
- atomic forget 与 fixed official retain eval 重叠；
- 为凑比例截断 closure；
- FQ 单指标调参；
- 用不同 provenance retrain 比较 Entity 与 Atomic；
- 看到方法结果后改 request 或生成题；
- 在 GlobalKL/LocalMean 缺失时宣传 tail/prior。

## 16. 无捏造声明

所有表格结果均为 `TBD`。`unlearn_summary.csv` 只用于确定已有 baseline、字段和实验经验，不是 Atomic-TOFU 结果。
