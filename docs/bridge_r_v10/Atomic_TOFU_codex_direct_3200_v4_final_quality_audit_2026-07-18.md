# Atomic TOFU 3200 条 Codex Direct v4 最终质量审计报告

日期：2026-07-18  
审计性质：只读内容与产物审计  
质量标尺：达到官方 TOFU paraphrased / perturbed 数据的可用水准，不以逻辑绝对完美为门槛  

## 1. 结论

最终 3200 条结果**整体可用，可作为 candidate 进入后续数据整合、校准和评测准备流程，不需要全量重生成**。

本结论不表示这些记录已经成为 gold：

- API 输出和 Codex 生成结果仍然只是 candidate；
- `reports/error_queue_3200_v2.jsonl` 为空，只能说明生成流水线的机械门禁没有报错，不能解释为语义零错误；
- 全量模式扫描和 208 条分层语义抽检均发现少量明确问题，建议在正式冻结前对本报告给出的核心 review queue 做定点人工复核或重生成；
- 发现两个不影响候选文本内容、但应在发布或正式归档前修复的元数据问题。

综合建议状态：

`usable_candidate_with_small_review_queue`

## 2. 审计对象

最终候选文件：

`data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/candidate_outputs_3200_v2.jsonl`

范围与输入：

- 输入 manifest：`data/atomic_tofu/v2.0-rc1/api/eval_extension/input_units.jsonl`
- ID 范围：`tofu_full_0400`–`tofu_full_3599`
- 记录数：3200
- 每条 perturbation 数：5
- perturbation 总数：16000
- batch：160 个，每个 20 条
- shard：16 个，每个 200 条

生成报告声明：

- `model_label=5.6 Luna Medium`
- `model_lock=floating_alias`
- `api_called=false`
- `candidate_only=true`
- 不得将该模型状态表述为 fixed snapshot 或 fixed-snapshot freeze。

生成构成（来自最终 run report）：

- 新生成：3008 条
- 从当前 v4 既有 shard 复用：172 条
- 从已批准 v2 pilot 复用：20 条
- `corrected_existing_records=8` 是既有记录中的修订子集，不应与上述三项重复相加。

最终 SHA256：

`7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`

该值与最终 sidecar、manifest 和最终 run report 一致。

## 3. 审计方法

本次使用三层核查。

### 3.1 全量机械核查

对全部 3200 条进行：

- JSONL 逐行解析；
- ID 集合、唯一性和输入顺序核对；
- 顶层和 candidate 字段契约核对；
- 空字段、扰动数量和条内重复核对；
- paraphrased question / answer 对原 QA 的 exact 与 normalized 复刻核对；
- perturbation 对原答案及 paraphrased answer 的复刻核对；
- batch、shard、最终文件拼接一致性核对；
- manifest、run report、sidecar 和实际 SHA256 核对；
- 占位符、模型元话语、明显截断和引号/括号不闭合扫描；
- 全局重复和模板退化统计。

归一化检查至少包含 Unicode NFKC、大小写折叠以及标点/空白忽略。

### 3.2 全量硬错误模式扫描

对 3200 条、16000 个 perturbations 扫描历史上已出现的错误模式，包括：

- `No` 与后文肯定原事实冲突；
- `Yes` 与后文否定事实冲突；
- 改动事实槽后，因果或解释尾句未联动；
- pseudonym / pen name 等同义替换；
- 问题要求的核心答案没有被改变；
- 数学等价的两个 perturbations；
- “获得奖项”与“仅被提名但未获奖”混淆；
- 明显不通顺或句内互相否定；
- 模板坍缩、占位符、元话语与截断。

### 3.3 分层语义抽检

从 16 个 200 条 shard 中各抽取固定 offset：

`[0, 1, 5, 19, 20, 37, 55, 79, 100, 123, 150, 178, 199]`

总计：

- 208 / 3200 条记录；
- 416 个 paraphrase 字段；
- 1040 个 perturbations；
- 共审查 1456 个 candidate 文本字段；
- 覆盖全部 16 个 shard、作者边界、shard 首尾和内部位置。

该抽检用于描述质量和发现问题，不应表述为 3200 条逐条人工全审，也不应据此声称精确的全语料语义错误率。

## 4. 全量机械结果

### 4.1 通过项

| 检查项 | 结果 |
|---|---:|
| 物理行 | 3200 |
| JSON 可解析 | 3200 / 3200 |
| 唯一 unit_id | 3200 |
| 与 input_units 集合一致 | 是 |
| 与 input_units 顺序一致 | 是 |
| 缺失 / 额外 / 重复 ID | 0 / 0 / 0 |
| 每条恰好 5 个 perturbations | 3200 / 3200 |
| perturbation 总数 | 16000 |
| 条内 exact / normalized 重复 | 0 |
| paraphrased question normalized 复刻 | 0 |
| paraphrased answer normalized 复刻 | 0 |
| perturbation 等于原答案或 paraphrased answer | 0 |
| 空字段 | 0 |
| 占位符 / 模型元话语 | 0 |
| 疑似截断 / 引号括号不闭合 | 0 |
| batch complete | 160 / 160 |
| shard complete | 16 / 16 |
| manifest remaining | 0 |
| manifest next_pending_batch | `null` |

160 个 batch 均为 20 条；16 个 shard 均为 200 条。batch 和 shard 按 manifest 顺序拼接后，内容与最终 3200 条文件完全一致。

### 4.2 全局重复

- 3200 个 paraphrased questions：无 exact / normalized 跨记录重复；
- 3200 个 paraphrased answers：无 exact / normalized 跨记录重复；
- 16000 个 perturbations：仅发现 1 组跨记录重复文本：
  - `tofu_full_2825 #3`
  - `tofu_full_2828 #4`
  - 文本：`Nadia Nowak was nominated for the Global Aesthetics and Design award but never received it.`

这两个问题同属一位作者、同一奖项语境，不违反“每条内部 5 个扰动互异”的契约，属于软瑕疵。

### 4.3 模板与长度

- 未观察到系统性模板坍缩；
- 160 个 batch 中，每 100 个 perturbations 的 8-token 前缀唯一数最小/中位/最大约为 31 / 73 / 98；
- 全局 8-token 前缀唯一数为 11516 / 16000；
- 文本长度未出现大面积异常短句、截断或空洞占位。

## 5. 语义质量结果

### 5.1 Paraphrase

分层抽检的 208 条中：

- paraphrased question：208 / 208 可接受；
- paraphrased answer：208 / 208 可接受；
- 未发现事实改变、严重语病或答非所问；
- 全量 normalized 复刻检查为 0。

因此 paraphrase 部分达到官方 TOFU 级可用水准，未发现需要阻止数据使用的系统性问题。

### 5.2 Perturbation 抽检

1040 个抽检 perturbations 中发现 10 个明确问题，涉及 8 / 208 条记录：

- `tofu_full_0550 #3`：仅将 meaning 换成 purpose，核心语义没有实际扰动；
- `tofu_full_0578 #1`：`casually committed` 与后续 `painstakingly consults many sources` 冲突；
- `tofu_full_1178 #3/#5`：以 No 回答是否使用 pseudonym，却又明确描述使用其他名字或每本书使用不同 pseudonym；
- `tofu_full_2100 #5`：问题问完整姓名，仍保留正确姓名，只改变性别代词，核心答案未扰动；
- `tofu_full_2420 #5`：同一句中先称作者专攻 mental health，后又称其专攻 historical fiction rather than mental health；
- `tofu_full_2805 #5`：问题问第一部出版书名，仍保留正确书名，只改变作品类型；
- `tofu_full_3578 #2/#5`：Yes/可读性结论与“仅专业数学家”或“仅天文学读者能理解”冲突；
- `tofu_full_3599 #3`：问题问写作外兴趣，仍保留全部正确兴趣，只改变影响来源。

抽检 perturbation 明确问题比例为 10 / 1040（0.96%）；涉及记录比例为 8 / 208（3.85%）。这只是本次固定分层样本的描述性统计，不应外推为精确总体错误率。

### 5.3 全量模式扫描

全量模式扫描未发现大面积退化，但确认一批适合进入人工 review queue 的记录。问题主要集中在：

- 复用旧段 `0400–0579` 中遗留的 Yes/No 冲突和因果尾句未联动；
- 少量后续新生成记录中的核心事实未扰动或奖项“提名/获奖”混淆；
- 两条数学等价扰动。

## 6. 建议的核心 review queue

以下 34 个 ID 是全量模式扫描、分层抽检和数学等价扫描的保守并集，占 3200 条的 1.06%。建议正式冻结前逐条复核；必要时仅重生成这些记录，不应全量重跑。

### 6.1 极性、存在性或句内矛盾

- `tofu_full_0453`
- `tofu_full_0456`
- `tofu_full_0474`
- `tofu_full_0536`
- `tofu_full_0573`
- `tofu_full_0579`
- `tofu_full_0811`
- `tofu_full_1141`
- `tofu_full_1178`
- `tofu_full_2420`
- `tofu_full_2813`
- `tofu_full_3578`

### 6.2 因果链、语义或表达内部不一致

- `tofu_full_0418`
- `tofu_full_0495`
- `tofu_full_0496`
- `tofu_full_0499`
- `tofu_full_0510`
- `tofu_full_0515`
- `tofu_full_0534`
- `tofu_full_0558`
- `tofu_full_0571`
- `tofu_full_0578`
- `tofu_full_2394`

### 6.3 核心答案未被有效扰动

- `tofu_full_0550`
- `tofu_full_2100`
- `tofu_full_2805`
- `tofu_full_3599`

### 6.4 “被提名但未获奖”不能作为“获得奖项”的 Yes 回答

- `tofu_full_2784`
- `tofu_full_2802`
- `tofu_full_2847`
- `tofu_full_2964`
- `tofu_full_2992`

### 6.5 数学等价 perturbations

- `tofu_full_0476`
- `tofu_full_0576`

上述 queue 是保守复核清单，不代表每条的 5 个 perturbations 全部失败，也不代表清单之外绝对没有语义瑕疵。

## 7. 软问题与边界样本

以下记录存在可兼容替换、类型错配、机械表达或跨记录重复，但按官方 TOFU 水准不必阻塞使用：

- `tofu_full_0412`
- `tofu_full_0429`
- `tofu_full_0434`
- `tofu_full_0599`
- `tofu_full_0816`
- `tofu_full_2137`
- `tofu_full_2825`
- `tofu_full_2828`

这些记录可在有余力时进入二级 review queue，不建议因此扩大重生成范围。

## 8. 产物元数据问题

### 8.1 shard 000 sidecar 哈希过期

`shards/shard_000_tofu_full_0400_tofu_full_0599_v2.jsonl` 的实际 SHA256、manifest 和 run report 三者一致：

`1baff300977a19649d4e3d5d5731c17836c417e920981ddd9eb70ac6974b6c2a`

但其 `.sha256` sidecar 仍写着旧值：

`f0d211677c7f5783d9e2d432935428900ec81121a625896a779ed6fad41d19d34`

这是明确的元数据错误。它不影响最终 3200 条文件内容，因为最终文件、最终 sidecar、manifest 和最终 run report 的 SHA256 一致；但在发布、归档或自动验证 shard 前必须修正。

### 8.2 七份 shard report 缺少 output_path

`reports/run_report_008...` 至 `reports/run_report_014...` 共 7 份 200 条 run report 缺少 `output_path` 字段。其记录数、机械通过状态和 manifest 信息仍存在，但报告契约不完整。

建议在正式归档前补齐字段；本报告不授权或执行该修改。

### 8.3 error_queue 的含义

`reports/error_queue_3200_v2.jsonl` 为 0 行。由于本次独立审计已经发现明确语义问题，该文件只能解释为“生成时机械/自检 error queue 为空”，不能在论文、报告或评测说明中写成“语义错误为 0”。

## 9. 原始输入完整性观察

生成过程中曾观察到 `source/tofu_full.jsonl` 的 JSONL 格式发生过临时异常并随后被重新写回。最终审计时：

- 文件可逐行解析为 4000 条；
- 对本次 3200 个 unit，source 中的 question / answer 与 `input_units.jsonl` payload 逐条完全一致；
- 未观察到原始 QA 事实内容漂移。

但源文件在任务期间发生写入本身与“原始数据只读”的约束不一致。建议将该事件作为 provenance / process warning 保留，不应在没有明确授权时继续修改、恢复或提交该文件。

## 10. 最终可用性判断

### 可以做

- 将最终文件作为 candidate 数据使用；
- 进入下游数据整合和评测准备；
- 在不全量重跑的前提下，对核心 review queue 做定点人工复核；
- 使用最终文件 SHA256 固定本次 candidate artifact。

### 不应做

- 不应称其为 gold；
- 不应称 error_queue=0 为语义零错误；
- 不应声称使用了 fixed model snapshot；
- 不应在修复 shard sidecar 和缺失 report 字段前把整个目录称为元数据完全自洽；
- 不应因为少量问题而重跑全部 3200 条。

## 11. 建议后续动作

按优先级排序：

1. 在获得明确写入授权后，修正 shard 000 的 `.sha256` sidecar；
2. 补齐 run report 008–014 的 `output_path`；
3. 将本报告的 34 个核心 ID 写入独立语义 review queue；
4. 逐条确认具体失败 perturbation，只重生成失败记录或失败项；
5. 复核后重新生成最终文件、相关 shard、manifest、run report 与 SHA256；
6. 保持 `candidate_only=true`，直到独立审阅和数据契约流程明确批准晋升。

## 12. 最终一句话结论

**该 3200 条结果已经达到官方 TOFU 级 candidate 可用水准；无需全量重生成，但正式冻结前应处理约 1% 的保守语义 review queue，并修复两个产物元数据问题。**

## 13. 2026-07-18 post-review revision：当前权威状态

本节覆盖前文基于旧 candidate artifact 的待复核描述；当前磁盘权威状态以最终文件、manifest、review file 和本节为准。

- 34/34 final candidate 与 `review_queue_34_candidate_data_2026-07-18.jsonl` 中的 approved candidate 逐条完全相等；该文件为每条记录保存 `final_candidate_sha256` 和确定性 hash 算法。
- 34 条 provenance 分为：31 条采用复核文件中的 revised candidate；`tofu_full_0453` 和 `tofu_full_2802` 使用单独 special correction；`tofu_full_3599` 保留当前已合格 candidate。
- `tofu_full_0453` 的五个 perturbations 均直接、连贯回答 No，且不保留“经历困难”的核心语义。
- `tofu_full_2802 #3` 明确为 nominated but never won any award；`#4` 使用不同于正确答案的奖项。
- 受影响 bookkeeping 为 21 个 batch、9 个 shard；batch/shard/final 拼接和所有相关 SHA sidecar、manifest、report 已重算。
- 当前 final SHA256：`7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`。
- 3200 行、3200 唯一 ID、输入集合/顺序、schema、16000 perturbations、160/160 batch、16/16 shard 均通过最终验收。
- 评测元话语严格扫描为 0 个命中；`error_queue=0` 仍只表示机械生成队列为空，不能解释为语义零错误。
- `candidate_only=true`、`model_lock=floating_alias`、`api_called=false` 保持不变；该 artifact 仍不是 gold，也不是 fixed-snapshot freeze。

## 14. 2026-07-18 post-review freeze baseline

- 本次 3200 post-review candidate 已冻结为 candidate baseline；冻结记录：`data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/post_review_freeze_2026-07-18.json`。
- 冻结 final SHA256：`7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`。
- 34 条 review queue 已关闭：31 条采用 revised candidate，2 条为 0453/2802 special correction，1 条保留 3599 当前 candidate；没有未决 queue item。
- 冻结声明保持 `candidate_only=true`、`model_lock=floating_alias`、`api_called=false`；该基线不是 gold，也不是 fixed snapshot。
- 原有 shard sidecar 与 `run_report_008`–`014` 的 `output_path` 元数据问题已修复；冻结后不再改写该 3200 baseline。

## 15. Source/Pipeline A coordination and Eval-A readiness

- 当前 source 结构复核通过：4000 rows、200 authors、每作者 20 条、连续 author block、`source_index`/`qa_id` 对齐；annotation candidate、adjudicated annotations、200 review packets 和 1174 request scopes 的 author identity 均与 source 一致。
- 当前 source 文件 SHA256 为 `94fc1caa196658c1abbb586e61c25f4d09544be3a1943997e6e3c4f036a821ff`，旧 source report 记录的 SHA 为 `667baef2...`；该差异已作为 process warning 记录，source 文件未在本步骤修改。
- 协调报告：`audit/source_pipeline_audit_2026-07-18.json`；source、annotation、human-review、request-graph 报告均已写入当前 author-identity/审查一致性状态。
- 正式 4000-row Eval-A candidate view 已组装：3200 条使用冻结 post-review generated candidate，800 条复用 official auxiliary fields；view SHA256 为 `907925c2057fc7d2f1547bccb7587a2587565fee564c822e5092582e6d34f4a6`，字段级 provenance sidecar SHA256 为 `cedc463b3f0d83ee9b04d6f1f1d4658dfdeb2d8a1bb3e656e082bd33d7706efa`；官方字段未覆盖。
- 当前路线 calibration evidence：`audit/eval_a_current_route_calibration_evidence_2026-07-18.json`。60 条当前 final-route sample 机械检查为 60/60，但 generated 长度分布 gate 未通过；20 个 hidden official anchors 已有独立 Codex-direct candidate，机械错误类型为 0，但与 official fields 的长度和形态 gate 未通过，且真实 human blind review 仍 pending。历史 gpt-5-mini `eval_anchor_calibration_report.json` 未被当作当前质量证据。
- 正式 bundle Eval-A row review inventory：39 个 bundles、3420 个唯一 rows（3060 generated、360 official）；3420 条 review decision 记录均绑定 candidate hash，但状态仍为 `pending_human_review`，不能称为人工复核完成。
- 当前阶段不放行 full Eval-A；保持 `candidate_only=true`、`model_lock=floating_alias`、`api_called=false`。

## 16. 2026-07-18 calibration human-review attestation and frozen-candidate diagnostic

本节覆盖第 15 节中“human review pending”的当前状态表述；不改变长度、answer-form 或 candidate-freeze 结论。

- 用户已本人完成 Eval-A calibration candidate 的人工复核，并明确确认 `human_review_passed=true`。正式写入记录：`audit/eval_a_current_route_calibration_human_review_attestation_2026-07-18.json`。
- 该结论的 provenance 为 `reviewer=user_or_project_owner`、`decision_source=retrospective_user_attestation`、`review_date=2026-07-18`、`human_review_status=passed`、`human_review_basis=user_manual_review_and_attestation`。
- 原始逐行 decision 当时未填写；本次是用户事后正式确认并写入。没有伪造逐行 review 时间、理由或 reviewer 操作记录；原始 pending sidecar 保留。
- 未执行/未形成正式 A/B blind protocol，因此记录 `formal_blind_protocol_documented=false`；不能表述为 formal blind protocol completed。人工复核通过不等于 formal blind protocol 完成。
- 当前路线 evidence `audit/eval_a_current_route_calibration_evidence_2026-07-18.json` 已将 human-review gate 记为通过，`human review pending` 不再是 blocker；但整体 calibration 仍未通过，以下 gate 保持失败：generated length、hidden-anchor length、hidden-anchor form。
- 对冻结文件的只读诊断记录在 `audit/frozen_3200_length_shape_diagnostic_2026-07-18.json`，并验证 final SHA256 仍为 `7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`，未修改 candidate baseline。
- 全部 3200 rows / 16,000 perturbations：ratio median `0.7692307692`，P10 `0.5142857143`，P90 `1.0`；低于 `0.75` 的 field 为 `7,189`（涉及 `1,817` rows），低于 `0.65` 的 field 为 `4,492`（涉及 `1,277` rows），裸值 field 为 `184`（涉及 `48` rows；其中非裸 source answer 的裸值 failure 为 `159` fields / `43` rows），项目 answer-shape validator 判定的 shape-mismatch 为 `15` fields / `10` rows；有任一诊断 failure 的 row 为 `1,847`。完整失败 ID 与 field-level records 见诊断 JSON。
- 正式 bundle 的 `3,060` 个 generated rows / `15,300` perturbations：ratio median `0.7692307692`，P10 `0.5151515152`，P90 `1.0`；低于 `0.75` 的 field 为 `6,848`（涉及 `1,732` rows），低于 `0.65` 的 field 为 `4,278`（涉及 `1,221` rows），裸值 field 为 `178`（涉及 `46` rows；failure 为 `153` fields / `41` rows），项目 answer-shape validator 判定的 shape-mismatch 为 `15` fields / `10` rows；有任一诊断 failure 的 row 为 `1,761`。
- 这些长度问题覆盖面广，不是可由 34 条语义 queue 代表的小范围异常；当前建议是建立新版 calibrated candidate revision，而不是对冻结 baseline 做定点修复。本节不授权、也未执行任何修复。
- 3420-row bundle inventory 未自动填入虚假逐条 decision；manifest 记录 `review_coverage_basis=user_manual_review_attestation`，但明确该 attestation 只覆盖 calibration candidates，不能声称 `3420/3420` bundle rows individually reviewed。
- 全程保持 `candidate_only=true`、`model_lock=floating_alias`、`api_called=false`。

## 17. 2026-07-18 objective shape-repaired candidate revision

- 根据用户授权，保留冻结的 `candidate_outputs_3200_v2.jsonl` 及其 SHA256 `7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512` 不变，另建 shape-repaired candidate revision：`api/eval_extension/codex_direct_3200_v4/shape_repaired_53_2026-07-18/candidate_outputs_3200_shape_repaired_v1.jsonl`。
- 新 revision SHA256：`114728c0a2dce0d1d29be0ee892dd512b2861fa7d3befc5f27dbdce215063f39`；provenance：`shape_repair_53_provenance_2026-07-18.jsonl`；manifest：`shape_repaired_revision_manifest_2026-07-18.json`。
- 仅处理 53 个 row：43 个非裸 source 对应 bare-value perturbation rows（159 fields）和 10 个 answer-shape mismatch rows（15 fields）；两组不相交。51 个受影响 rows 属于 formal bundle generated rows，2 个不属于。
- 修订后目标性验证：非裸 source 的 bare-value failure 为 `0 fields / 0 rows`；answer-shape mismatch 为 `0 fields / 0 rows`。其余未授权的长度异常不作修改。
- 修订后全量长度 gate 仍未通过：16,000 perturbations 的 ratio median `0.7720779221`、P10 `0.5142857143`、P90 `1.0`；低于 `0.75` 的 fields 为 `7,120`，低于 `0.65` 的 fields 为 `4,453`。formal bundle 15,300 perturbations 的 ratio median `0.7727272727`、P10 `0.5151515152`、P90 `1.0`；低于 `0.75` 的 fields 为 `6,781`，低于 `0.65` 的 fields 为 `4,239`。
- 该 revision 是 candidate-only、未冻结、未晋升正式 Eval-A；当前 4000-row Eval-A view、official fields、bundle review hash inventory 和冻结 v2 均未被覆盖。未调用 API，保持 `candidate_only=true`、`model_lock=floating_alias`、`api_called=false`。
