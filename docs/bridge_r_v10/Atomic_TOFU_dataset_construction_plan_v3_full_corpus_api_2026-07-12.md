# Atomic-TOFU 数据集构造方案 v3：4,000 QA、API 自动候选、人工逐作者复核

> 日期：2026-07-12  
> 目标体验：用户提交脚本，等待 API 完成，然后按作者人工复核即可  
> 训练数据不变；新生成内容仅用于 evaluation extension 和语义 manifests
> 2026-07-18 范围修订：完整 Eval-B 不再是数据集冻结门槛；见 `Atomic_TOFU_Eval_B_scope_amendment_2026-07-18.md`。其余数据契约不变。

## 0. Release Definition

正式发布建议命名：

```text
Atomic-TOFU-v2.0-rc1
```

冻结 release 必须包含：

1. 官方 full 4,000 QA 的稳定索引；
2. 200 authors 的 atom-QA graph；
3. 人工通过的 single 与 multi requests；
4. full-corpus TOFU-compatible evaluation extension；
5. Single/Multi 1%/5% bundles；
6. Matched-Entity 与 Random controls；
7. full/forget/retain/retrain/eval manifests；
8. official retain/RA/WF/holdout dependencies；
9. API、人工和 hash provenance；
10. strict validation reports。

## 1. Source Layer

### 1.1 Full training source

从服务器当前官方 cache 加载 `locuslab/TOFU`, config `full`, split `train`。

为每条记录保存：

```json
{
  "qa_id": "tofu_full_0000",
  "source_index": 0,
  "author_id": "...",
  "question": "exact official text",
  "answer": "exact official text",
  "content_sha256": "..."
}
```

author_id 不能只依赖脆弱字符串规则。优先复用官方 author mapping；若无，则由 API/程序候选后人工验证 20-QA grouping。

### 1.2 Official evaluation anchors

读取并映射：

- forget01/05/10 及 perturbed variants；
- retain90/95/99；
- retain_perturbed；
- real_authors(_perturbed)；
- world_facts(_perturbed)；
- holdout01/05/10。

生成 `official_alignment.json`，记录每个 eval row 是否能映射 full、schema 和 fields。

## 2. 两条 API Pipeline

不要用一个大 prompt 同时做语义标注和评测生成。分成两条可独立复核的流水线：

### Pipeline A：Atom/Request Annotation

输入一位作者的 20 原始 QAs；输出 atoms、relations、request candidates。

### Pipeline B：Evaluation Extension

输入单条原始 QA 和官方 schema examples；输出 evaluator 需要的 paraphrase/perturbation fields。

两条 pipeline 使用独立 prompt、JSON schema、cache 和 quality report。

## 3. API Runner

### 3.1 Config

```text
OPENAI_API_KEY
OPENAI_BASE_URL optional
ATOMIC_TOFU_ANNOTATION_MODEL
ATOMIC_TOFU_EVAL_MODEL
ATOMIC_TOFU_MAX_CONCURRENCY
ATOMIC_TOFU_TIMEOUT
```

模型名不硬编码在数据文件。执行时固定可用 snapshot。

### 3.2 Required capabilities

- Structured Outputs / strict JSON schema；
- Batch API 或可恢复并发；
- exponential backoff；
- request content hash cache；
- resume；
- dry-run/mock provider；
- raw request/response/usage 保存；
- refusal/invalid JSON queue；
- prompt/model/schema version；
- API key redaction；
- per-stage token/cost report。

### 3.3 推荐执行方式

```text
stage 0: export and validate sources
stage 1: 8-author calibration
stage 2: freeze annotation prompt
stage 3: 12-author blind validation
stage 4: full 200-author annotation
stage 5: eval-extension anchor calibration
stage 6: full missing-row eval generation
stage 7: review packets
stage 8: human decisions
stage 9: compile and freeze
```

## 4. Pipeline A：Atom/Request Annotation

### 4.1 Atom extraction

一个 QA 可产生多个 atoms。atom 必须有：

```text
subject
relation
canonical value
aliases
source qa_id
evidence span
```

禁止无证据推断、整段主题 atom、把两个独立事实强合成一个 atom。

### 4.2 Atom-QA roles

对每个 atom 扫描该作者全部 20 QAs，并分别检查每条 QA 的 question 与 answer；这是
atom×QA 的多对多审计，不是把每条 QA 只分配给一个 atom。必须检查 exact value、aliases、
所有格/语法变体和直接关系改写；同一 QA 若同时涉及多个 atom，必须挂到每个适用 atom。
例如，若 atom 为 `father_occupation=hairdresser`，而另一 QA 说父亲作为 hairdresser 的
工作影响了作者写作，则该 QA 还必须挂到职业 atom（直接依赖标 `closure`，仅重复数值标
`leak`），即使它同时挂到 `father_influence` atom。

| Role | 定义 |
|---|---|
| support | 直接陈述目标事实 |
| leak | question/answer 明示 target value |
| closure | 回答依赖 target，保留会恢复 target |
| protected candidate | 独立于 target |
| ambiguous | 无法可靠判断 |

role 表是一 atom × QA 多对多关系。

对任一 target atom，所有标作 `support`、`leak` 或 `closure` 的 QA 必须自动组成其完整 forget closure；`per_atom_closures` 只是这一确定性集合的显式审计表示，不能遗漏或加入其他 QA。

### 4.3 Single request candidates

每位作者生成 1-5 个高质量 single candidates。要求 closure 完整、target 清楚，并保留 atom-role 标签用于诊断；request-local protected QA pool 不由候选的 protected siblings 决定，而由 compiler 从同作者全部 non-target QA 确定。

姓名保护仅适用于代表作者本人身份的 name/full_name/author_name atom。父亲、母亲和其他家庭成员的姓名不是 scope identifier，可以作为普通 factual target；若历史候选使用作者本人的 literal name 作为 subject，effective-candidate policy 仍须保护它。

### 4.4 Multi request candidates

生成 2-3 atom request candidates，要求：

- 同作者；
- atoms 非同义重复；
- 组合有现实语义，例如 family privacy、birth identity、one-work bundle；
- 保存每个 atom closure 和 union closure；
- 保存 target-target overlap；
- 联合删除后重新计算 protected residual support；
- 不能仅因为恰好凑 QA 数而组合。

### 4.5 API output schema

```json
{
  "author_id": "string",
  "unassigned_qa_ids": ["string"],
  "atoms": [
    {
      "atom_id_candidate": "string",
      "subject": "string",
      "relation": "string",
      "value": "string",
      "aliases": ["string"],
      "source_qa_ids": ["string"],
      "evidence_span": "string",
      "qa_relations": [
        {
          "qa_id": "string",
          "role": "support|leak|closure|protected|ambiguous",
          "span": "string",
          "reason": "string",
          "confidence": "high|medium|low"
        }
      ]
    }
  ],
  "single_requests": [],
  "multi_requests": []
}
```

20 条输入 QA 必须精确覆盖为 `qa_relations` 的 ID 并集或 `unassigned_qa_ids`，二者不得重叠。requests 中引用 atom IDs，不复制自由文本事实。counts/features 由 compiler 重算。

## 5. Pipeline B：TOFU-Compatible Evaluation Extension

### 5.1 先逆向实际 schema

脚本必须从服务器数据和 evaluator 自动推导：

- field names；
- scalar/list types；
- perturbation count；
- evaluator 实际读取的问题字段；
- paraphrased answer/question 是否使用；
- index 对齐规则。

禁止凭印象写死“5 个 perturbed answers”等数字。

### 5.2 生成内容

仅为缺失 rows 生成实际需要的字段：

- paraphrased question：若 evaluator 使用；
- paraphrased answer：与正确答案语义等价；
- perturbed answer(s)：事实错误但语法自然、类型匹配、长度适当；
- provenance。

原始 question/answer 不改。

### 5.3 Perturbation rules

每个错误答案必须：

- 与问题类型匹配，例如日期换日期、职业换职业；
- 不包含正确值或其 alias；
- 不改变问题所询问的 relation；
- 不使用明显胡言乱语；
- 不能所有记录都套同一模板；
- 多事实答案要按官方 schema 的粒度生成，而不是随意改一个无关词。

### 5.4 Paraphrase rules

- 含义与官方答案一致；
- 保留所有必要 answer slots；
- 不新增事实；
- 不复制原答案长片段；
- 长度分布与 official anchors 相近。

### 5.5 Anchor calibration

从已有 official perturbed rows 分层抽样：

1. 对 API 隐藏官方辅助字段；
2. regenerate；
3. 与官方字段比较 schema、语义、类型、长度和模型概率分布；
4. 人工盲评 official vs regenerated；
5. prompt 冻结后才扩展全量。

不要求逐字复制官方 fields，而要求 evaluator difficulty 与质量可比。

### 5.6 单一冻结 Eval-A 与范围限制

- `Eval-A` 是唯一必需的 frozen compatible extension；所有 Atomic 方法、bundle 和 seed 使用完全相同的 artifact；
- 官方 Entity rows 的主评测保留 `Official` fields，不要求额外生成完整 `Eval-A`/`Eval-B` sidecars；
- hidden official anchors 的 regenerated fields 仅用于 schema、answer-shape、长度、难度和语义的盲校准，不构成第二套正式评测视图；
- 完整独立 `Eval-B` 从 release gate 移除，仅作为可选后续或附录 robustness study；
- 不报告 Eval-A/Eval-B 跨视图排名稳健性，也不声称完全排除 evaluator-source effect。

## 6. 人工复核

### 6.1 Author review packet

每位作者一个 HTML/Markdown/JSON packet：

1. 20 条原始 QA；
2. atoms 和 evidence；
3. single request closure；
4. multi request atom combination 与 union closure；
5. surviving/co-deleted/dependent/ambiguous；
6. residual support；
7. continuous features 与 provisional level；
8. accept/revise/reject controls；
9. reviewer reason。

用户只需一个作者一个作者复核。

### 6.2 Evaluation review packet

不要求人工逐条重写 4,000 rows。采用：

- 所有 schema/rule violations 必审；
- 所有低置信度必审；
- multi-fact answers 必审；
- official-anchor calibration 全审或高比例审；
- 每位作者随机抽样；
- 高影响 requests 的全部 forget/protected eval rows 必审。

### 6.3 Double review

- 8 authors calibration 人工 gold；
- 12 authors blind validation；
- 所有 High、Multi、ambiguous requests 二审；
- 正式 bundles 涉及的全部 request closures 二审；
- 评测扩展关键 rows 二审。

## 7. Deterministic Validation

### 7.1 Annotation validators

- qa_id/atom_id 外键；
- evidence span 存在；
- target alias 全作者扫描；
- support/leak recall warning；
- role conflicts；
- residual support 重算；
- multi atom nonredundancy；
- closure union 和 overlap；
- reserved anchor overlap。

### 7.2 Evaluation validators

- exact schema/type/cardinality；
- answer slot preservation；
- correct-value leakage；
- perturbation duplicates；
- semantic equivalence score + human sample；
- length/lexical distribution；
- full-model probability/ROUGE/TR distribution against anchors；
- evaluator dry run。

## 8. Reserved Anchors And Eligible Requests

把 official fixed retain-eval QAs 映射回 full。为保持 Atomic 与官方 TOFU 的 MU 口径一致：

- main comparable bundles 不得删除这些 QAs；
- 最稳妥是 reserve 这些 rows 所属 authors；
- 若只 reserve rows，任何触及它们的 request closure 整体排除；
- 不能保留 target-support anchor 只为维持 MU；
- 报告 reserve 策略排除了多少 authors/requests，以及特征偏差。

另建 `nonfixed-MU` extension track 可覆盖这些 requests，但不得与官方 MU 主表直接混合。

## 9. Request Taxonomy：Cardinality × Entanglement

每个 request 同时拥有两个彼此独立的标签：

```text
cardinality: C1 / C2 / C3
entanglement: E-Low / E-Medium / E-High
```

其中 `C1` 是 single，`C2/C3` 是包含 2/3 atoms 的 multi。Multi 不自动等于 High，single 也不自动等于 Low。正式 request bank 至少区分：

| Cell | 含义 |
|---|---|
| C1-E-Low | 单原子、低纠缠 |
| C1-E-Medium | 单原子、中纠缠 |
| C1-E-High | 单原子、高纠缠 |
| C2-E-Low/Medium/High | 双原子三档纠缠 |
| C3-E-Low/Medium/High | 三原子三档纠缠 |

### 9.1 纠缠不是“看起来难”

compiler 从人工确认的 atom-QA graph 计算四个基础维度，每项取 0/1/2：

| 维度 | 0 | 1 | 2 |
|---|---|---|---|
| `exposure` | 仅一条直接 support，无其他 leak | 2 条相关暴露 | >=3 条暴露或多处显式重复 |
| `dependency` | 无额外依赖 QA | 1 条 closure/dependent QA | >=2 条或存在链式依赖 |
| `co_carriage` | forget QAs 不携带实质非目标 atom | 携带 1-2 个独立 atoms | 携带多个 atoms，或含同关系 sibling |
| `protected_proximity` | 无近邻 protected，或 residual support 充足 | 有语义/关系近邻但支持稳定 | 同关系近邻、residual support=1，或极易与 target 混淆 |

同时保留原始连续特征：

```text
support_count_per_atom
leak_count
closure_count
union_forget_count
co_carried_atom_count
co_deleted_atom_count
surviving_protected_count
min_and_mean_residual_support
same_relation_neighbor_count
lexical_and_semantic_similarity
author_forget_ratio
```

### 9.2 Single request 分级

single 的 provisional score 为四维之和：

\[
E_{single}=E_{exposure}+E_{dependency}+E_{co\_carriage}+E_{proximity}.
\]

初始规则：

```text
E-Low:    score 0-2，且无 hard-entanglement signal
E-Medium: score 3-5，且未触发 High override
E-High:   score 6-8，或触发 High override
```

High override 包括：

- 存在 dependency closure，且 target QA 同时携带同关系 sibling；
- 删除 target closure 会使至少一个非目标 atom 从 surviving 变为 co-deleted；
- target value 在多个问题和答案位置显式重复；
- target 与 protected 同关系、同句式且 residual support 只有 1。

示例：

- `birthdate` 通常只有一条直接 QA、无重复和依赖，可判为 `C1-E-Low`；
- `father's profession` 可能与 mother's profession 共处一条 QA，并在“父亲职业如何影响写作”中再次出现，具有 dependency、同关系 co-carriage 和 role-shift 风险，应判为 `C1-E-High`；
- 若某作者的生日又出现在完整身份介绍和年龄推理 QA 中，则不能因 relation 是 birthdate 就自动判 Low。

等级由结构决定，不由 relation 名称直接决定。上述例子只是 calibration anchor。

### 9.3 Multi request 分级

Multi 先计算每个 constituent atom 的 single dimensions，再增加两个 interaction dimensions：

| 维度 | 0 | 1 | 2 |
|---|---|---|---|
| `closure_overlap` | atom closures 基本独立 | 有部分重叠 | 同一 QA 支持多个 targets 或高度重叠 |
| `joint_role_shift` | 联合删除不改变非目标角色 | 少量 protected residual support 降低 | 联合删除使 protected 变 co-deleted，或产生组合依赖 |

\[
E_{multi}=\max_{a\in A_r}E_{single}(a)
+E_{overlap}+E_{joint\_role\_shift}.
\]

初始规则：

```text
E-Low:    score 0-3，且各 atom closure 独立、无 role shift
E-Medium: score 4-7，且未触发 High override
E-High:   score >=8，或任一 constituent 为 High，或 joint role shift=2
```

Multi 的等级必须在联合删除后重算，不能简单取各 single 标签平均。

### 9.4 阈值冻结与人工裁决

上述规则用于 8-author calibration。根据真实 feature 分布可以在 blind validation 前调整一次，之后冻结：

- API 只能建议各维度证据，不能决定最终 level；
- compiler 重算 dimensions/score；
- review packet 展示触发规则；
- 人工可 override，但必须记录 `old_level/new_level/reason`；
- 论文主分析同时报告 raw features、score 和 level，避免阈值决定结论。

## 10. Bundle Compiler

### 10.1 Request bank coverage

先发布完整 request bank，报告 9 个 `C×E` cells 的 request 数、作者数、closure 大小和 relation coverage。不得为填满矩阵而降低 request 质量；空 cell 标记 unavailable。

### 10.2 Pure-stratum diagnostic bundles

对有足够 coverage 的 cell 构造 exact 40-QA bundles，每个 3 seeds：

```text
c1_low_01, c1_medium_01, c1_high_01
c2_low_01, c2_medium_01, c2_high_01
c3_low_01, c3_medium_01, c3_high_01
```

这些 bundles 用于分离纠缠等级，不要求全部进入昂贵的大模型主矩阵。

### 10.3 Balanced official-scale bundles

正式 1%/5% 对比使用：

```text
single_balanced_01/05: 仅 C1，E-Low/Medium/High 按 forget-QA union 贡献分层平衡
multi_balanced_01/05: 仅 C2/C3，cardinality 与 E-level 均分层平衡
```

平衡单位优先是最终 forget QA contribution，而不是 request 个数，因为 High request closure 往往更大。每类至少 3 个 bundle seeds。

### 10.4 Contrast bundles

若 coverage 允许，再构造：

```text
single_low_01 vs single_high_01
multi_low_01 vs multi_high_01
```

它们严格匹配 40 QAs、作者数、relation 和 initial difficulty，用于直接检验 entanglement effect。5% pure-level bundles 只在数据与算力允许时构造。

所有 bundles 使用 ILP/subset-sum；closure union 后 exact size，不能截断。

### 10.5 Controls

- official Entity01/05 manifests；
- Matched-Entity from eligible author pool；
- Random-QA matching each Atomic bundle。

Random matching features：QA count、author count/per-author deletion、token length、relation、full-model initial Probability/ROUGE/TR bins。

### 10.6 No result-aware selection

bundle 只能使用 annotation/features 和固定 seed。模型结果文件对 compiler 不可见。

## 11. Official Metric Data Contract

每个 bundle release manifest 必须指向：

| Object | Source |
|---|---|
| full train | official full unchanged |
| forget train | full selected IDs |
| retain train | full complement |
| forget eval | frozen compatible extension selected IDs |
| protected eval | 当前 bundle 中每位涉及作者的全部 non-target retained extension rows |
| fixed retain eval | official retain_perturbed |
| real author eval | official data |
| world fact eval | official data |
| holdout | official holdout01/05 or validated size-matched config |
| retrain logs | split-specific retrain evaluation |

`retain train` 是 \(G_R=D\setminus F_R\) 的完整 manifest，并供 retain loss/GlobalKL 随机采样；固定训练步数不要求逐条覆盖该 manifest，也不因未采样记录新增 remainder eval。paired retrain 在完整 manifest 上训练。

`metric_dependency_report` 若发现额外数据，必须加入，不得只满足这张预估表。

## 12. Schemas

### 12.1 Request

```json
{
  "request_id": "...",
  "author_id": "...",
  "cardinality": 1,
  "cardinality_label": "C1",
  "target_atom_ids": ["..."],
  "per_atom_closures": {},
  "forget_qa_ids": ["..."],
  "surviving_protected_atom_ids": ["..."],
  "protected_train_qa_ids": ["..."],
  "protected_eval_qa_ids": ["..."],
  "co_deleted_atom_ids": ["..."],
  "dependent_atom_ids": ["..."],
  "features": {},
  "entanglement_dimensions": {
    "exposure": 0,
    "dependency": 0,
    "co_carriage": 0,
    "protected_proximity": 0,
    "closure_overlap": null,
    "joint_role_shift": null
  },
  "entanglement_score": 0,
  "entanglement_level": "E-Low|E-Medium|E-High",
  "level_override": null,
  "human_status": "accepted",
  "provenance": {}
}
```

`protected_train_qa_ids` 和 `protected_eval_qa_ids` 由 compiler 确定，不接受 API candidate 选择：对单个 request，它们等于该作者全部 source QA 减去该 request 的完整 `forget_qa_ids`。在 bundle 的训练、same-author protected eval、LocalMean/Tail 和 protected CVaR20 中，这个 request-scoped pool 保持不变；另一 request 的 target 不会被额外剔除。全局 retain train 仍独立使用 bundle forget union 的补集。`surviving_protected_atom_ids` 只保留为原子级分析字段，不再筛选 protected QA pool。

### 12.2 Evaluation extension row

字段必须以服务器 official schema 为准，额外 provenance 放 sidecar，不要让 evaluator误读：

```json
{
  "qa_id": "...",
  "question": "unchanged",
  "answer": "unchanged",
  "generated_official_compatible_fields": {},
  "source_or_generated": "official|generated",
  "generation_record_id": "...",
  "human_review_status": "..."
}
```

## 13. Script Deliverables

建议提供：

```text
export_sources.py
map_official_eval_anchors.py
prepare_annotation_batches.py
run_annotation_api.py
validate_annotation_outputs.py
prepare_eval_generation_batches.py
run_eval_generation_api.py
validate_eval_extension.py
build_author_review_packets.py
apply_human_reviews.py
compile_request_graph.py
compile_bundles.py
build_metric_views.py
validate_metric_completeness.py
freeze_release.py
```

以及一条总入口：

```text
python -m atomic_tofu.pipeline --stage <name> --resume
```

## 14. 用户实际操作流程

```text
1. 运行 source export/preflight
2. 提交 annotation API script
3. 人工复核 author packets
4. 提交 eval-extension API script
5. 人工复核 flagged/selected eval packets
6. 应用 decisions
7. 编译 Single/Multi bundles
8. 运行 metric completeness + evaluator dry run
9. freeze rc1
```

README 必须给出每一步命令、输入、输出、预计 API 数量、resume 方式和失败重试方式。

## 15. Quality Gates

| Gate | 最低要求 |
|---|---|
| Source | 4,000 hashes unchanged；200×20 grouping verified |
| Atom | blind precision/recall 达预注册阈值 |
| Closure | high precision/recall；正式 bundle 全部二审 |
| Protected | residual support 有效；无 target dependency |
| Multi | atoms 非冗余、组合合理、union closure 完整 |
| Entanglement | dimensions 可重算；C1 Low/Medium/High 有足够覆盖或明确 unavailable |
| Eval schema | 100% type/cardinality pass |
| Eval semantics | paraphrase 等价、perturbation 错误且不泄漏 |
| Calibration | generated vs official anchor difficulty 可接受 |
| Eval scope | 所有 Atomic 方法共用单一冻结 Eval-A；hidden official-anchor 偏差已量化；evaluator-source caveat 已记录 |
| Bundle | exact 40/200；no truncation；no reserved overlap |
| Metrics | FQ/MU/PrivLeak/ES/Prob/ROUGE/TR 全链路可运行 |
| Freeze | hashes、API、human、code provenance 完整 |

具体质量阈值先在 8+12 author calibration/blind 阶段制定并冻结，不在看模型结果后改。

## 16. Release Directory

```text
data/atomic_tofu/v2.0-rc1/
  source/
  official_anchors/
  api/annotation/
  api/eval_extension/
  review_packets/authors/
  review_packets/eval/
  adjudicated/
  request_graph/
  bundles/
  metric_views/
  audit/
  release_manifest.json
  README.md
```

## 17. 禁止事项

- 只在官方 forget10 的 20 authors 上选 request；
- 修改 full 的 question/answer；
- 拆 QA；
- API 输出直接当 gold；
- 硬编码未检查的 official perturbed schema；
- 覆盖 official auxiliary fields；
- 为凑比例截断 closure；
- main atomic forget 与 official fixed retain anchors 重叠；
- 按 surviving-protected atom、candidate explicit protected IDs 或 residual support 缩小同作者 non-target protected QA pool；
- 只导出 forget/retain；
- 缺 atomic-specific retrain logs；
- 看结果后改数据；
- 把生成 extension 描述成官方原始评测数据。

## 18. 完成定义

\[
Complete=
FullCorpusProcessed
\land HumanAdjudicatedRequests
\land CalibratedEvalExtension
\land ExactSingleMultiBundles
\land OfficialMetricCompleteness
\land FrozenProvenance.
\]

满足全部条件后，用户才开始正式 baseline 和 BRIDGE-R 实验。
