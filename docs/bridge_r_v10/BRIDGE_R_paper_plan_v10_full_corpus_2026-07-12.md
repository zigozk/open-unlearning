# BRIDGE-R 完整论文方案 v10：Full-Corpus Atomic-TOFU

> 日期：2026-07-12  
> 状态：研究方案，不包含新的 Atomic-TOFU 实验结果  
> 本版取代仅在官方 `forget10_perturbed` 400 条 QA 中选择 request 的 v9
> 2026-07-18 范围修订：完整 Eval-B 不再是必做项；见 `Atomic_TOFU_Eval_B_scope_amendment_2026-07-18.md`。除其中明确修订的 Eval-B 条款外，本方案其余研究目标不变。

## 0. 核心结论

Atomic-TOFU 必须处理官方 TOFU `full` 的全部 4,000 条 QA、200 位作者，而不是从官方 forget10 的 20 位作者中挑选原子请求。训练 QA 保持逐字、数量和权重不变；API 为缺少官方辅助字段的 full QAs 生成与 OpenUnlearning evaluator 兼容的评测扩展，并由人工逐作者复核。正式任务同时包含单原子 request 和 2-3 原子的多原子 request。

论文用同一 full checkpoint、同一 baseline、相同 forget QA 比例和同一套 OpenUnlearning 主指标比较：

```text
官方 TOFU 实体遗忘 vs Atomic-TOFU 单原子遗忘 vs Atomic-TOFU 多原子遗忘
```

BRIDGE-R 的贡献是：在保持全局 utility 的同时，重点保护原子请求附近仍应保留的同作者知识。

## 1. 科学故事

### 1.1 问题

TOFU 目前按完整作者构造 forget01、forget05 和 forget10。现实的 machine unlearning 请求却可能只要求遗忘实体内部一个或多个具体事实，例如：

- 遗忘父亲职业，但保留母亲职业和作品；
- 遗忘某部作品及其获奖信息，但保留作者其他作品；
- 一次请求同时遗忘出生地、生日和居住地三个隐私 atoms。

实体级遗忘把同作者全部 QA 删除，天然消除了大量同作者冲突；原子数据遗忘必须删除目标事实的完整 QA closure，同时保留同作者其余 QA，因此可能更难。

### 1.2 根本困难

一个 QA 可以包含多个 atoms，同一个 atom 也可能出现在多个 QA 中。由于 QA 不可拆分：

- target atom 的所有 support、leak 和 dependency QAs 都应完整删除；
- 与 target 共处同一 QA 的其他事实可能被合理共同删除；
- 只有删除后仍有 retain 支持的独立事实才是合法 protected knowledge；
- 多原子 request 的 closure 是各 atom closures 的并集，可能产生重叠、协同或冲突。

### 1.3 中心假设

在相同 forget 记录比例与相同 baseline 下：

1. Atomic-TOFU 比官方实体 TOFU 产生更强的 forget-retain 冲突；
2. request 纠缠越强，官方 utility 和同作者 protected utility 越容易受损；
3. 多原子 request 的难度不只由 atom 数决定，还由 closure overlap 与 protected residual support 决定；
4. Global retain KL 可能看不见局部尾部风险；
5. BRIDGE-R 的 request-local mean/tail protection 能改善该风险。

以上均为待验证假设，不能预写成结果。

## 2. 不可拆分 QA 上的原子遗忘

官方 full corpus：

\[
D=\{q_i\}_{i=1}^{4000}.
\]

每个 \(q_i\) 是完整 QA。Atomic-TOFU 不修改 question、answer、训练权重或 full-model 训练过程。

事实原子定义为：

\[
a=(subject, relation, value).
\]

对 atom \(a\)：

\[
F(a)=Support_Q(a)\cup Leak_Q(a)\cup Closure_Q(a),
\]

其中所有元素都是完整 QA IDs。

### 2.1 单原子 request

\[
r_{single}=(author,\{a\}), \qquad F_r=F(a).
\]

### 2.2 多原子 request

一个多原子 request 同时指定同一作者的 2-3 个非冗余 atoms：

\[
r_{multi}=(author,A_r), \quad 2\le |A_r|\le3,
\]

\[
F_r=\bigcup_{a\in A_r}F(a).
\]

多原子 request 不是把多个独立 single requests 随意拼成一个标签。正式组合必须语义合理，例如同一隐私类别、同一作品链或同一家庭关系，并记录 atom closures 的交集。

### 2.3 Request bundle

正式实验不是每次只遗忘一个小 request，而是把多个 requests 组成与官方 forget01/05 大小匹配的 bundle：

\[
F_R=\bigcup_{r\in R}F_r, \qquad D^{-R}=D\setminus F_R.
\]

主规模：

| Bundle | Forget QA | 比例 | 对照 |
|---|---:|---:|---|
| Single-01 | 40 | 1% | 官方 forget01 |
| Single-05 | 200 | 5% | 官方 forget05 |
| Multi-01 | 40 | 1% | 官方 forget01、Single-01 |
| Multi-05 | 200 | 5% | 官方 forget05、Single-05 |

10% 只做后续压力测试。Multi-01/05 中每个 constituent request 都是 2-3 atoms。

## 3. 删除后的知识角色

对非目标 atom \(b\)：

1. **surviving protected**：不依赖 targets，并且 \(D^{-R}\) 中仍有支持；
2. **co-deleted/affected**：其支持全部或主要落在 \(F_R\)；
3. **dependent**：语义依赖一个或多个 target atoms；
4. **ambiguous**：无法可靠判定，排除主分析。

`surviving protected`、co-deleted、dependent 与 ambiguous 仍是原子级分析标签。为统一训练和评测口径，BRIDGE-R 的 request-local protected QA pool 不再由这些标签筛选：它固定为该 request 所属作者的全部原始 QA，扣除该 request 的完整 forget closure。protected pool 始终相对于当前 request 定义；即使其中某 QA 是同一 bundle 另一 request 的 target，也不得据此从当前 request 的 protected 输入或评测中剔除。全局 retain train 仍使用 bundle forget union 的补集。co-deleted/dependent 标签仍须与 retrain model 对齐，不能算作 unlearning 算法的错误。

多原子 request 还需记录：

- atom closure overlap；
- 是否一个 QA 同时支持多个 targets；
- 删除一个 atom 是否已隐式删除另一个 atom 的全部支持；
- protected residual support 在联合删除后是否消失。

## 4. Full-Corpus Atomic-TOFU

### 4.1 全量处理

API 按作者读取 20 条 QA，对 200 位作者全部处理：

- 多 atom 抽取；
- atom-QA support/leak/closure 图；
- single request candidates；
- 语义合理的 multi-atom request candidates；
- surviving/co-deleted/dependent/ambiguous roles；
- request 纠缠 features。

人工逐作者复核候选。API 输出不是 gold label。

### 4.2 为什么需要生成新的评测扩展

官方 TOFU `full` 只有基础 question/answer；官方 FQ 和 Truth Ratio 使用的 perturbed/paraphrased fields 只覆盖既有评测 splits。要在 full 的任意作者上定义 atomic forget，就必须为全部 4,000 QAs 构建同 schema 的 evaluation extension。

该扩展不改变训练数据，只新增冻结的评测字段：

- paraphrased question/answer，按服务器 evaluator 实际需求；
- 与正确答案同类型但事实错误的 perturbed answers；
- stable QA ID 和 provenance。

已有官方 perturbed rows 作为 calibration anchors，不被覆盖。

### 4.3 评测生成的可信度

API 评测扩展必须通过：

1. schema/cardinality 与官方 perturbed splits 一致；
2. paraphrase 语义等价；
3. perturbation 错误、合理、类型匹配且不泄漏正确值；
4. 长度、词汇和 full-model difficulty 与官方 anchors 分布对齐；
5. 在隐藏的官方 anchor rows 上做 regenerate-and-compare；
6. 人工双审高风险和随机样本；
7. 所有方法使用同一冻结版本。

评测范围采用单一冻结 Eval-A：所有 Atomic 方法、bundle 和 seed 必须使用同一版本；官方 Entity01/05 主结果仍使用原始官方辅助字段。对隐藏的官方 anchor rows 做 regenerate-and-compare 与人工盲审，用于量化生成字段的 schema、answer-shape、长度、难度和语义偏差。完整独立 Eval-B 已从必做范围移除，仅可作为后续或附录 robustness study；因此 Atomic-vs-Entity 结论必须注明 evaluator-source caveat，不得声称已通过跨评测视图稳健性完全排除生成器来源效应。

论文不能声称新评测题是“官方 TOFU 数据”，应称为“TOFU-compatible evaluation extension”。

## 5. 请求复杂度与纠缠

采用二维 taxonomy，避免把 atom 数和语义纠缠混为一谈：

### 5.1 Cardinality

- `Single`：1 atom；
- `Multi-2`：2 atoms；
- `Multi-3`：3 atoms。

### 5.2 Entanglement

连续特征包括：

- support/leak/closure QA 数；
- co-carried 与 co-deleted atom 数；
- target-target closure overlap；
- surviving protected 数和 residual support；
- same-relation neighbors；
- target 与同作者 non-target retain QA 的 lexical、semantic similarity；
- author forget ratio。

Cardinality 与 Entanglement 组成完整交叉标签：

| Cardinality | E-Low | E-Medium | E-High |
|---|---|---|---|
| C1 / Single | C1-E-Low | C1-E-Medium | C1-E-High |
| C2 / Multi-2 | C2-E-Low | C2-E-Medium | C2-E-High |
| C3 / Multi-3 | C3-E-Low | C3-E-Medium | C3-E-High |

纠缠由四个基础维度确定：重复暴露、依赖闭包、同 QA 携带其他 atoms、与 surviving-protected facts 的关系/语义接近度。Multi 额外计算 target closures overlap，以及联合删除是否让原本 protected 的 atom 变为 co-deleted。该原子级标签不再缩小训练或评测的 same-author non-target QA pool。

例如，一条只出现一次、没有依赖 QA 的生日事实通常属于 `C1-E-Low`；父亲职业若与母亲职业共处一条 QA，又出现在“父亲职业如何影响写作”的后续 QA 中，则具有同关系 co-carriage 和 dependency closure，可属于 `C1-E-High`。relation 名称本身不决定等级：若生日被身份介绍、年龄推理等多处重复使用，也可能是 Medium 或 High。

数据发布完整 9-cell request bank；正式 1%/5% 主 bundle 分别构造 `Single-Balanced` 和 `Multi-Balanced`，按最终 forget-QA contribution 平衡 E-Low/Medium/High。另构造纯纠缠 1% diagnostic bundles，如 `C1-E-Low-01` 与 `C1-E-High-01`，用于直接验证 entanglement effect。Multi 是 cardinality，不预设一定最难。

## 6. 与官方 TOFU 的公平比较

### 6.1 Primary comparison

在同一模型和同一 baseline 上比较：

```text
Official Entity01 vs Atomic Single-Balanced-01 vs Atomic Multi-Balanced-01
Official Entity05 vs Atomic Single-Balanced-05 vs Atomic Multi-Balanced-05
```

固定：

- full checkpoint；
- forget QA 数；
- optimizer steps 与训练预算；
- baseline implementation；
- official summary metrics；
- evaluator aggregation；
- paired retrain protocol。

### 6.2 必要控制

因为 official entity 与 atomic 来自不同作者，增加：

- `Matched-Entity`：从 atomic 作者池选择完整作者，使用新评测扩展；
- `Random-QA`：匹配 QA 数、作者分布、长度和 full-model 初始难度；
- 三个 atomic bundle seeds；
- 关系类别与 request 数统计。
- hidden official-anchor 的 regenerated-vs-official 盲校准；
- 所有 Atomic 方法、bundle 和 seed 共用同一冻结 Eval-A；不再要求 Eval-A/Eval-B 跨视图指标或排名稳健性。

因此论文可区分：官方 benchmark 差异、作者内容差异、部分作者删除效应和语义 closure 效应。

## 7. 官方指标主表

本地 `outputs/unlearn_summary.csv` 含 41 条 baseline/BRIDGE 结果，主汇总字段为：

```text
model_utility
forget_quality
forget_truth_ratio
forget_Q_A_Prob
forget_Q_A_ROUGE
extraction_strength
privleak
```

Atomic-TOFU 主表必须沿用这些字段名和当前服务器实现。详细 `TOFU_EVAL.json` 还应保留 MU 的 retain、real-author、world-fact 9 个 probability/ROUGE/truth-ratio 分量。

公开 OpenUnlearning 实现表明：[Forget Quality](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu_metrics/forget_quality.yaml) 依赖 retrain logs 的 forget Truth Ratio，[Model Utility](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu_metrics/model_utility.yaml) 聚合 retain/real-author/world-fact 九项，[默认 TOFU evaluator](https://github.com/locuslab/open-unlearning/blob/main/configs/eval/tofu.yaml) 还启用 PrivLeak 与 Extraction Strength。正式实现以服务器代码为准。

### 7.1 官方主指标

- FQ；
- MU；
- Forget Truth Ratio；
- Forget QA Probability；
- Forget QA ROUGE；
- Extraction Strength；
- PrivLeak；
- MU 9 components。

### 7.2 Atomic diagnostics

作为补充而非替代：

- same-author non-target retain Probability/ROUGE/TR；
- request-level retrain gap；
- protected worst-k/CVaR20；
- co-deleted retrain gap；
- mean/local-tail KL drift；
- single 与 multi 分层。

### 7.3 FQ 使用限制

已有 BRIDGE 结果显示 40-sample FQ 有 KS 平台效应。FQ 仍必须报告，但不能单独用于调参、forget matching 或“更难”结论。难度结论使用官方多指标 Pareto和 retrain-relative gaps。

## 8. 完整数据依赖

Atomic bundle 不只需要 forget/retain train，还需要：

| 数据 | 用途 |
|---|---|
| atomic forget eval extension | Probability、ROUGE、TR、FQ、ES |
| atomic retain train | retrain 与 retain loss |
| fixed official retain eval | 可比 MU |
| official real authors | MU |
| official world facts | MU |
| holdout matched to split size | MIA/PrivLeak |
| bundle-specific retrain eval logs | FQ、PrivLeak reference |
| same-author protected eval | Atomic diagnostics |

为避免 official fixed retain eval 与 atomic forget 重叠，构造时先映射其 full QA anchors：

- 主 Atomic bundles 不得与 fixed official retain-eval rows 重叠；
- 若某 request closure 触及 reserved anchor，不能截断 closure，应拒绝该 request 或在独立 non-comparable track 中处理；
- 另报告这种 reservation 对 request 分布的影响。

## 9. Retrain counterfactual

每个 split/bundle 都训练专属 retrain model：

\[
\theta^-_R=Train(\theta_{base},D^{-R}).
\]

Entity、Single、Multi、Matched-Entity、Random 都采用相同 base、recipe 和 paired seed。官方 retain99/95 checkpoints 只在 provenance 完全匹配时作为 sanity，不代替 Atomic retrain。

FQ 的两个分布必须来自 unlearn model 和对应 retrain model 在同一 forget eval rows 上的 Truth Ratio。

## 10. BRIDGE-R

BRIDGE-R 保留旧 BRIDGE 的全局 KL 保护，并增加 request-local mean 与 centered tail：

\[
\mathcal L=\mathcal L_{erase}+\lambda_gG_{KL}+\lambda_lM_R+\lambda_tT_R.
\]

\[
M_R=\frac1{|C_R|}\sum_{i\in C_R}d_i,
\quad d_i=D_{KL}(p_{\theta_0}\Vert p_\theta).
\]

\[
T_R=\tau\log\sum_i\pi_i
\exp((d_i-\operatorname{sg}(M_R))/\tau).
\]

全局 retain 集为 \(G_R=D\setminus F_R\)。普通 retain loss 与 \(G_{KL}\) 都从 \(G_R\) 随机采样 batch；\(G_R\) 是完整可采样训练集合，不要求在固定 optimizer-step 预算内逐条被采到，也不因此新增 remainder evaluation view。paired retrain 则在完整 \(G_R\) 上训练。

对当前 request \(r\)，\(C_r\) 包含其作者的全部原始 QA，扣除 \(F_r\)；不再按 surviving-protected atom 标签缩小。bundle 中的 LocalMean/Tail、same-author protected evaluation view 和 protected CVaR20 保留每个 \(C_r\) 的 request-scoped 定义：另一 request 的 target 不会从 \(C_r\) 中被额外剔除。

实现与消融顺序：

```text
Backbone
Backbone + GlobalKL
+ LocalMean
+ Centered UniformTail = BRIDGE-R
+ GS/KL-PI prior only after UniformTail gate
```

多原子 request 的 local pool 是各 target 所属作者的完整 non-target retain QA 集；按 request/author 分层，防止一个大 closure 支配 batch。

## 11. Baselines

从 `unlearn_summary.csv` 已复现的方法中选择：

- NPO：主开发 backbone；
- GradDiff：强遗忘、utility damage 对照；
- RMU：1B forget01 有可用结果，作为不同机制；
- SimNPO：弱遗忘负对照；
- GradAscent：仅 stress test。

BRIDGE-R 因果对照必须包括 GlobalKL 和 Uniform/LocalMean。旧结果中部分 GlobalKL/UniformDRO 与强 NPO `ms20_lr1e-5` 不属于同一 run grid，不能直接用于新结论。

## 12. Claim-Evidence Matrix

| Claim | 必需证据 | 不成立时 |
|---|---|---|
| Atomic 比实体更难 | 同 baseline、同 size、官方多指标、matched controls、3 bundles | 改为“行为不同” |
| Multi 增加难度 | Single/Multi 同 size、控制 closure/author/features | Multi 只作任务覆盖 |
| Single 内纠缠等级有效 | C1-E-Low/Medium/High 纯层 bundles、连续 feature regression | level 降为描述标签 |
| 纠缠预测 damage | 连续 features、author-cluster statistics | 降为案例 |
| LocalMean 超过 GlobalKL | matched forgetting Pareto | 删除 locality claim |
| Tail 超过 LocalMean | UniformTail ablation | 简化为 LocalMean |
| Prior 超过 Uniform | GS/KL-PI vs Uniform | prior 附录/负结果 |

## 13. 论文结构与图表

### 论文结构

1. Introduction；
2. Atomic unlearning with indivisible records；
3. Full-Corpus Atomic-TOFU construction；
4. TOFU-compatible evaluation extension；
5. BRIDGE-R；
6. Official-metric task comparison；
7. Method experiments and analysis；
8. Limitations and ethics。

### 核心图表

- Figure 1：single/multi atom → complete QA closure → roles；
- Figure 2：API generation、human review、freeze；
- Figure 3：Official Entity vs Single vs Multi Pareto；
- Figure 4：GlobalKL vs LocalMean vs BRIDGE-R；
- Table 1：4,000-QA annotation 与评测扩展质量；
- Table 2：7 个 official summary metrics；
- Table 3：MU 9 components 与 protected diagnostics；
- Table 4：方法消融与成本。

## 14. 审稿人攻击与预防

### “生成的新评测题不能和官方 TOFU 比”

使用官方 rows 做隐藏 calibration anchors，保持相同 schema/aggregation；官方 entity 结果仍用官方数据，Atomic 使用明确标注且单一冻结的 compatible Eval-A extension。主结论报告这一差异，并加入 Matched-Entity controls。由于完整 Eval-B 不再是必做项，论文只能量化和缓解 evaluator-source effect，不得声称已通过独立评测视图将其完全分离。

### “Atomic 更难只是作者不同”

加入 Matched-Entity、Random-QA、bundle seeds、initial-difficulty matching 和 author-cluster analysis。

### “保留官方 retain eval 会泄漏 forget”

预先 reserve 并映射 anchors；正式 request closure 不得与其重叠，不允许截断。

### “Multi 只是删得更多”

Single/Multi bundle forget QA 数严格相同，同时报告 request 数、target atom 数、closure overlap 和每作者删除数。

### “API 决定 benchmark”

API 只生成候选 annotation 与评测扩展；正式 requests、correctness、perturbations 和 high-risk cases 由人工复核，保存 raw-to-final diff。

### “BRIDGE-R 只是更多 KL”

GlobalKL、LocalMean、Centered UniformTail 逐级消融，所有比较在多指标 forget-feasible band 内完成。

## 15. 摘要草稿

> LLM unlearning benchmarks commonly remove complete entities, while practical requests may target one or several facts within an entity. We study atomic data unlearning with indivisible QA records: each request specifies one to three factual atoms, but every original QA that supports, reveals, or depends on a target is removed as a whole. We construct Atomic-TOFU over all 4,000 unchanged TOFU training QAs. An API-assisted, human-verified pipeline annotates atom-record relations and creates a TOFU-compatible evaluation extension for records not covered by existing perturbation splits. This enables the same OpenUnlearning baselines and official metric implementations to compare entity, single-atom, and multi-atom removal at matched record ratios, with request-specific retraining references and same-author diagnostics. We further propose BRIDGE-R, which augments global retention with request-local mean and centered-tail drift control. Empirical claims are contingent on multi-metric, matched-control, and multi-seed evidence and remain to be established.

## 16. 投稿 Gate

方法论文需要：

1. 评测扩展人工质量和 anchor calibration 通过；
2. Entity/Atomic 差异跨 baselines 与 bundles 稳定；
3. Multi 有独立增量或清晰失败分析；
4. BRIDGE-R 超过 GlobalKL 和 LocalMean；
5. 3 seeds 和至少第二模型/backbone；
6. release 可复现。

若任务差异成立而方法不成立，转 benchmark/diagnostic 论文。若 Atomic 与 controls 无稳定差异，不声称更难。
