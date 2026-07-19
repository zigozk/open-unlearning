# Atomic-TOFU / BRIDGE-R v10 接手文档：Codex Direct 3200 条 Eval 扩展候选（2026-07-18）

本文件用于把当前工作交给一个完全没有前序对话上下文的新 Codex。它记录当前磁盘事实、
已完成审计、约束、已知问题和下一步决策点。不得把 candidate、自动 validator、抽检或
生成模型自报写成 gold 或正式实验结果。

## 0. 接手后必须完整阅读的文件

新会话不得跳读，必须先完整阅读：

1. `docs/bridge_r_v10/BRIDGE_R_paper_plan_v10_full_corpus_2026-07-12.md`
2. `docs/bridge_r_v10/BRIDGE_R_experiment_plan_v10_full_corpus_remote_2026-07-12.md`
3. `docs/bridge_r_v10/Atomic_TOFU_dataset_construction_plan_v3_full_corpus_api_2026-07-12.md`
4. `docs/bridge_r_v10/HANDOFF_2026-07-13_atomic_tofu.md`
5. `docs/bridge_r_v10/README.md`
6. `/home/zkzhang/unlearn/Atomic_TOFU_eval_answer_shape_calibration_plan_2026-07-14.md`
7. `/home/zkzhang/unlearn/Atomic_TOFU_eval_answer_shape_calibration_60_audit_2026-07-15.md`
8. `docs/bridge_r_v10/HANDOFF_2026-07-15_eval_answer_shape_calibration.md`
9. `docs/bridge_r_v10/Atomic_TOFU_codex_direct_3200_v4_final_quality_audit_2026-07-18.md`
10. 本文件。

前三份方案是研究目标、实验设计和数据契约的最高权威。若 handoff、当前产物或历史实现与
前三份方案冲突，必须明确报告冲突，不得自行修改研究目标。2026-07-18 的最终质量报告是
当前 3200 条 candidate 的直接审计证据，但不覆盖前三份方案。

## 1. 工作目录与不可违反的约束

- 唯一项目目录：`/home/zkzhang/unlearn/open-unlearning`。
- 不重建环境。
- 不修改或删除原始 BRIDGE。
- 禁止 `git clean`、`git reset`、`git checkout`。
- 不恢复、删除、暂存、提交或“清理”当前预存脏改动。
- 没有用户明确授权前，只读核查；不写代码、不改候选数据、不改 manifest、不改报告元数据。
- 没有明确授权，不调用付费 API。
- API/Codex 输出均是 candidate，不是 gold。
- Eval 使用 `gpt-5-mini` alias 时必须报告 `model_lock=floating_alias`，不能声称
  fixed-snapshot freeze。
- 本次 3200 条由 Codex 5.6 Luna Medium 生成，最终报告同样必须写
  `model_lock=floating_alias`，不能声称固定快照。
- 不得把机械 validator 通过、空 error queue 或抽检通过写成完整语义零错误。
- 当前任务只授权新增本 handoff；未授权修正候选、哈希 sidecar、run report 或原始 source。

接手后应先只读执行 `git status --short` 并记录，不要处理任何脏改动。

## 2. 本次任务的来龙去脉

Atomic-TOFU 完整 source 有 4000 条 QA（200 位作者 × 每位 20 条）。官方已有的扩展部分之外，
当前需要补充的精确 3200 条范围由以下 manifest 定义：

`data/atomic_tofu/v2.0-rc1/api/eval_extension/input_units.jsonl`

它严格覆盖：

`tofu_full_0400`–`tofu_full_3599`

用户最初尝试过若干 40/80/3200 条版本。最终质量基线来自：

`data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/pilot20_0580_0599_v2.jsonl`

用户明确把标准校准为“达到官方 TOFU 水准即可”，不追求 perturbation 的形式逻辑绝对
完美。允许轻度机械表达、弱扰动和局部不严谨；必须避免明显内部矛盾、没有改变核心答案、
严重答非所问、不可理解语病、Yes/No 与后文直接冲突。

最终采用：

- 每位作者 20 条作为自然语言生成原子批次；
- 每 10 个批次合并为一个 200 条 shard；
- 共 160 个 batch、16 个 shard；
- 20 条是生成单位，不是 continuation 上限；同一次 continuation 可以连续完成多个 20 条批次。

## 3. 最终 3200 条产物

根目录：

`data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4`

最终文件：

`candidate_outputs_3200_v2.jsonl`

配套文件：

- `candidate_outputs_3200_v2.jsonl.sha256`
- `run_manifest_3200_v2.json`
- `reports/run_report_3200_v2.json`
- `reports/error_queue_3200_v2.jsonl`
- `parts/batch_000...batch_159..._20.jsonl`
- `shards/shard_000...shard_015..._v2.jsonl`
- 各 shard 的 run report 和 SHA sidecar。

最终 SHA256：

`7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`

最终报告声明：

- `record_count=3200`
- `perturbation_count=16000`
- `complete_batches=160`
- `complete_shards=16`
- `model_label=5.6 Luna Medium`
- `model_lock=floating_alias`
- `api_called=false`
- `candidate_only=true`
- `reused_from_current_v4_shard=172`
- `reused_from_approved_v2_pilot=20`
- `newly_generated=3008`
- `corrected_existing_records=8` 是既有记录中的修订子集，不与前三项重复相加。

## 4. 已完成的独立机械审计

对最终 3200 条只读全检得到：

- 3200 物理行；
- 3200/3200 JSON 可解析；
- 3200 个唯一 ID；
- ID 集合和顺序与 `input_units.jsonl` 完全一致；
- 缺失、额外、重复 ID 均为 0；
- 顶层字段严格为 `unit_id`、`candidate`；
- candidate 字段严格为 `paraphrased_question`、`paraphrased_answer`、
  `perturbed_answer`；
- 每条恰好 5 个非空 perturbations，共 16000 个；
- 条内 exact / normalized 重复为 0；
- paraphrased question 对原 question 的 normalized 复刻为 0；
- paraphrased answer 对原 answer 的 normalized 复刻为 0；
- perturbation 等于原答案或 paraphrased answer 为 0；
- 空字段、占位符、模型元话语、明显截断均为 0；
- 160 个 batch 均 20 条；16 个 shard 均 200 条；
- batch 和 shard 按顺序拼接均与最终文件完全一致；
- manifest 为 `complete`，`completed_records=3200`、`remaining_records=0`、
  `next_pending_batch=null`；
- manifest 内 160 个 batch hash 和 16 个 shard hash 与实际文件匹配；
- 最终文件、最终 sidecar、manifest、最终 run report 的 SHA256 一致。

跨记录全局重复：

- paraphrased questions：0 组；
- paraphrased answers：0 组；
- 16000 个 perturbations 仅 1 组重复：
  - `tofu_full_2825 #3`
  - `tofu_full_2828 #4`
  - 同一作者、同一奖项语境，属于软瑕疵，不违反条内 5 项互异契约。

## 5. 已完成的语义审计

### 5.1 分层抽检

从 16 个 shard 各取固定 offset：

`[0,1,5,19,20,37,55,79,100,123,150,178,199]`

共审查：

- 208 / 3200 条；
- 416 个 paraphrase 字段；
- 1040 个 perturbations。

结果：

- 208/208 条 paraphrase 可接受；
- 未发现 paraphrase 事实改变、严重语病或答非所问；
- 1040 个 perturbations 中发现 10 个明确问题，涉及 8 条记录；
- 抽检未见系统性退化。

抽检问题 ID：

- `tofu_full_0550`
- `tofu_full_0578`
- `tofu_full_1178`
- `tofu_full_2100`
- `tofu_full_2420`
- `tofu_full_2805`
- `tofu_full_3578`
- `tofu_full_3599`

该抽检是描述性证据，不能说成 3200 条逐条人工全审，也不能把样本比例直接当成总体精确
错误率。

### 5.2 全量硬错误模式扫描

全量扫描确认问题高度集中在旧复用段 `0400–0579`；3008 条新生成部分没有出现大面积
质量退化。扫描发现：

- 25 个明确硬 review ID；
- 2 个数学等价 perturbation ID；
- 少量边界/软问题；
- 没有系统性模板坍缩；
- 没有占位符、模型元话语或大面积截断。

## 6. 推荐的核心 review queue

以下 34 个 ID 是全量模式扫描、分层抽检和数学等价扫描的保守并集，占 3200 条的 1.06%。
正式冻结前建议逐条复核；如需修改，只重生成失败记录或失败 perturbation，不要全量重跑。

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

### 6.2 因果链、语义或表达不一致

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

### 6.3 核心答案未有效扰动

- `tofu_full_0550`
- `tofu_full_2100`
- `tofu_full_2805`
- `tofu_full_3599`

### 6.4 “仅提名但未获奖”不能回答“获得奖项”

- `tofu_full_2784`
- `tofu_full_2802`
- `tofu_full_2847`
- `tofu_full_2964`
- `tofu_full_2992`

### 6.5 数学等价 perturbations

- `tofu_full_0476`
- `tofu_full_0576`

软/边界问题，不应自动阻塞：

- `tofu_full_0412`
- `tofu_full_0429`
- `tofu_full_0434`
- `tofu_full_0599`
- `tofu_full_0816`
- `tofu_full_2137`
- `tofu_full_2825`
- `tofu_full_2828`

完整问题解释见 2026-07-18 最终质量审计报告。

## 7. 两个明确的产物元数据问题

### 7.1 shard 000 SHA sidecar 过期

文件：

`shards/shard_000_tofu_full_0400_tofu_full_0599_v2.jsonl`

实际文件、manifest、run report 三者一致的 SHA256：

`1baff300977a19649d4e3d5d5731c17836c417e920981ddd9eb70ac6974b6c2a`

但 `.sha256` sidecar 仍写旧值：

`f0d211677c7f5783d9e2d432935428900ec81121a625896a779ed6fad41d19d34`

这是元数据错误，不影响最终 3200 文件内容，但正式归档/发布前应修复。没有用户明确授权时不要改。

### 7.2 七份 shard run report 缺 output_path

`reports/run_report_008...` 到 `reports/run_report_014...` 共 7 份报告缺少 `output_path` 字段。
其记录数、机械状态和 manifest 信息仍存在。没有授权时不要补写。

### 7.3 error_queue 不能解释为语义零错误

`reports/error_queue_3200_v2.jsonl` 当前为 0 行。独立审计已经发现语义问题，因此只能写成
“生成流水线 error queue 为空”，不能写成“语义错误为 0”。

## 8. source/tofu_full.jsonl 的过程警告

生成过程中曾观察到 `source/tofu_full.jsonl` 暂时不再是严格的一行一条 JSONL，之后又被
重新写回。最终只读核查得到：

- 当前可逐行解析 4000 条；
- 本次 3200 个 unit 的 question / answer 与 `input_units.jsonl` payload 逐条完全一致；
- 没有观察到原始 QA 事实内容漂移。

但任务期间写入 source 本身违反“原始数据只读”的约束。接手后不要擅自恢复、格式化、
提交或再次修改它；如用户要追查，先只读比较可信基线并报告。

## 9. 当前可用性判断

当前最终状态：

`usable_candidate_with_small_review_queue`

可以：

- 将 `candidate_outputs_3200_v2.jsonl` 作为 candidate 用于后续数据流程；
- 使用最终 SHA256 固定当前 candidate artifact；
- 对 34 个核心 ID 做独立人工 review；
- 获得授权后定点重生成并重新组装受影响的 batch/shard/final artifact。

不可以：

- 不能叫 gold；
- 不能声称语义错误为 0；
- 不能声称 fixed snapshot；
- 不能在 sidecar 与 report 元数据修复前声称整个目录完全自洽；
- 不能未经授权修改候选数据、manifest、run report、source 或代码；
- 不能因为约 1% review queue 而擅自重跑全部 3200 条。

## 10. 与更大 BRIDGE-R / Eval-A 流程的关系

完成这 3200 条 candidate 不等于：

- Pipeline A 已重新 freeze；
- generated 60 calibration 的历史 13 条失败已自动解决；
- hidden anchor 已运行；
- full Eval-A 已获准运行；
- 训练、unlearning 或论文实验已经完成。

这些状态必须重新从磁盘核对，并服从前三份最高权威方案和 2026-07-15 handoff。不要把本次
扩展 candidate 的完成扩大解释为 full Eval-A、训练或实验授权。

## 11. 接手后的推荐顺序

在没有额外用户授权时：

1. 完整阅读第 0 节列出的文件；
2. 只读运行 `git status --short`；
3. 只读核对最终 3200 文件 SHA256、manifest complete 状态和报告是否仍与本 handoff 一致；
4. 向用户简要报告接手成功、当前 candidate 可用、34 个 review ID 和两个元数据问题；
5. 等待用户选择：
   - 仅复核 34 个 ID；
   - 授权定点重生成；
   - 授权修复 sidecar / report 元数据；
   - 返回更大的 Eval-A calibration / full Eval-A gate；
6. 在用户明确授权前不写任何其他文件。

若用户授权定点修复，必须：

- 先保存当前最终文件和哈希作为不可变 candidate 基线；
- 只处理明确失败记录，不扩大范围；
- 保持 v2 pilot 质量标准，即官方 TOFU 水准，不追求绝对完美；
- 保持 candidate-only；
- 重算受影响 part、shard、final、manifest、run report 和所有 sidecar；
- 独立复核修订项，不能仅信生成 agent 自报；
- 明确报告 `model_lock=floating_alias`、`api_called=false`（若仍未调用 API）。

## 12. 一句话接手结论

最终 3200 条结果机械完整、整体达到官方 TOFU 级 candidate 可用水准，无需全量重生成；
当前主要剩余工作是等待用户授权后处理 34 个保守语义 review ID、修复一个过期 shard SHA
sidecar 和七份缺 `output_path` 的 run report，同时保持 candidate-only 与
`model_lock=floating_alias` 的诚实表述。

## 13. 2026-07-18 post-review revision handoff：当前权威状态

本节更新并覆盖第3–9节中基于旧 candidate 的待复核状态；不改变前三份最高权威方案的研究目标或数据契约。

- 复核文件：`data/atomic_tofu/v2.0-rc1/api/eval_extension/codex_direct_3200_v4/review_queue_34_candidate_data_2026-07-18.jsonl`。
- 34/34 final candidate 与复核文件的 `candidate` 字段逐条完全一致；每条记录保存 `final_candidate_sha256`、hash 算法和 provenance。
- provenance：31 条 `review_file_revised_candidate`；2 条 `manual_special_correction`（0453、2802）；1 条 `current_final_candidate_retained`（3599）。
- 0453 五条均为直接连贯的 No，未保留经历困难的核心语义；2802 #3 为 No + nominated but never won any award，#4 更换为非正确答案奖项；3599 保留当前合格版本。
- 21 个 batch、9 个 shard 已重建/重算；全部相关 sidecar、shard report、manifest 和 final report 已同步。
- 当前最终 SHA256：`7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`。
- 最终验收通过：3200 行、3200 唯一 ID、输入集合/顺序、schema、16000 perturbations、batch/shard/final 拼接和 hash；34/34 candidate 一致。
- 评测元话语严格扫描为 0；`error_queue=0` 只能表述为机械队列为空，不能称为语义零错误。
- 当前仍为 `candidate_only=true`、`model_lock=floating_alias`、`api_called=false`；未晋升 gold，未声称 fixed snapshot。

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
