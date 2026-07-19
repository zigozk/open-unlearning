# Atomic-TOFU Eval-A candidate v3 延后再生成与校准方案

> 日期：2026-07-19  
> 状态：已批准路线的执行计划；v3 生成当前延后  
> 适用范围：Atomic-TOFU / BRIDGE-R v10 的单一 Eval-A candidate extension  
> 不改变：source、Pipeline A、request graph、bundle 设计、训练数据契约或原始 BRIDGE

## 0. 权威关系与命名

本文件是执行层计划，服从以下权威文件：

1. `BRIDGE_R_paper_plan_v10_full_corpus_2026-07-12.md`
2. `BRIDGE_R_experiment_plan_v10_full_corpus_remote_2026-07-12.md`
3. `Atomic_TOFU_dataset_construction_plan_v3_full_corpus_api_2026-07-12.md`
4. `Atomic_TOFU_Eval_B_scope_amendment_2026-07-18.md`
5. `/home/zkzhang/unlearn/Atomic_TOFU_eval_answer_shape_calibration_plan_2026-07-14.md`

这里的“v3”专指后续 **Eval-A candidate revision v3**，不表示修改同名的权威数据集构建方案。若本文与上述权威文件冲突，以权威文件及已批准范围修订为准，并必须报告冲突，不能自行改变研究目标。

完整 Eval-B 已从必做范围移除；本计划不恢复 Eval-B。唯一必需的正式评测扩展仍是单一冻结 Eval-A。

## 1. 当前决定

当前工作分为两个阶段：

1. 先完成可运行且可追溯的 `v2-pilot` candidate artifact，用于流水线验证、训练调试和同一 Eval-A 上的暂定实验。
2. 时间允许后按本文重新生成并校准 v3；v3 通过正式 gate 后，重新运行论文主表涉及的方法和基线。

`v2-pilot` 的结果只能称为 preliminary/provisional results。不得把 v2 与未来 v3 的数值混在同一主表中，也不得把 generated candidate 称为 gold 或官方 TOFU fields。

## 2. 触发 v3 的现有证据

冻结的 3200-row baseline 保持不变：

- SHA256：`7e66908e5ad0433ba6e9d950b92f76c9cf778091c056f22cf79215fa5379c512`
- `candidate_only=true`
- `model_lock=floating_alias`
- `api_called=false`

当前 generated route 存在系统性简洁化：

- generated `paraphrased_answer/source answer` token ratio 中位数约为 `0.8333`；
- generated `perturbed_answer/source answer` token ratio 中位数约为 `0.7692`；
- 两类 answer 的逐行 ratio 具有强相关，说明同一生成风格共同影响 paraphrase 与 perturbation；这不证明 perturbation 由 paraphrase 因果生成；
- source answer 越长，压缩趋势越明显；
- official 800 rows 的 paraphrased/perturbed answers 通常比 source 更长，而 generated 3200 rows 整体更短。

20 个 hidden anchors 的 paired evaluator run 已显示 material source shift：generated correct answers 更容易、generated wrong answers 更难，Truth Ratio 的均值/中位数相对 official fields 明显降低，并出现少量 `TR=1` 方向翻转。因此 v3 不能只做机械补长；必须同时校准 answer shape、语义、错误值类型、词汇与 evaluator difficulty。

## 3. 不可变边界

v3 工作必须遵守：

- 不修改 `source/tofu_full.jsonl`；
- 不修改 Pipeline A annotation、adjudication、request graph、1174 request scopes 或已选 bundles；
- 不修改、覆盖或删除冻结 v2 baseline、shape-repaired v1、其 provenance、manifest、报告和 SHA sidecars；
- 新建独立 v3 路径和独立 provenance；
- 不修改或删除原始 BRIDGE；
- 禁止 `git clean`、`git reset`、`git checkout`；不清理、暂存、提交或恢复预存脏改动；
- 未获单独授权不得调用付费 API；
- 所有模型生成输出只能称为 candidate；
- 空 error queue 或机械 validator 通过不能写成语义零错误；
- 若使用浮动模型别名，必须记录 `model_lock=floating_alias`，不得声称 fixed snapshot；
- v3 的 prompt、schema、generation parameters、输入、输出和代码版本必须分别保存 hash。

## 4. v2-pilot 的最小完成条件

开始正式实验前，先完成以下低成本收尾：

1. 只处理 shape-repaired v1 审计失败的 8 个 ID、12 个实际 fields：
   - `tofu_full_1202`
   - `tofu_full_1546`
   - `tofu_full_1874`
   - `tofu_full_2254`
   - `tofu_full_2412`
   - `tofu_full_2781`
   - `tofu_full_2805`
   - `tofu_full_3330`
2. 建立新的 `v2-pilot` revision，不覆盖冻结 baseline 或 shape-repaired v1。
3. 独立复核所有实际改动，确认 13 条审计 finding 均关闭，且没有新引入正确值泄漏、内部矛盾、source slot omission、语病或形态回归。
4. 重新验证 3200 行、3200 唯一 ID、与 `input_units.jsonl` 的集合和顺序一致、每条 5 个 perturbations、source payload 一致。
5. 确认 53-row 目标形态修复仍成立：非裸 source 对应裸值 failure 为 0，answer-shape mismatch 为 0。
6. 冻结 v2-pilot SHA、manifest、provenance 和审计报告，并明确记录广泛长度 gate 与 hidden-anchor behavioral equivalence 仍未通过。
7. 用该 SHA 重建 4000-row Eval-A candidate view 和字段级 provenance；800 official rows 的 official auxiliary fields 必须保持原值。
8. 重建 formal bundle 的 Eval-A row hash inventory，使所有 bundle row hash 与 v2-pilot artifact 一致。不得把用户抽查或 calibration attestation写成 `3420/3420 individually reviewed`。
9. 输出一份 dataset construction completion report，状态应是 `pilot_ready_with_known_evaluator_source_shift` 或等价表述，不能写成 v3/final calibration passed。

## 5. v3 预注册质量门槛

任何全量 v3 generation 之前，先冻结 prompt、schema、validator 和以下门槛。

### 5.1 机械与数据契约

- 3200 rows、3200 unique IDs；
- 与 `input_units.jsonl` 的 ID 集合和顺序完全一致；
- 每条恰好 5 个非空、互异 perturbations；
- 原始 question/answer 不改变，并与 source/input payload 一致；
- paraphrase 不得 normalized-copy source；
- perturbation 不得复制 source/paraphrased answer；
- 无 JSON/list 拼接、meta-commentary、placeholder 或多答案混入单个 field；
- manifest、batch、shard、final output 和 sidecar hashes 全部自洽。

### 5.2 语义与 answer shape

- paraphrased question/answer 保留全部事实 slots，不新增事实；
- 每个 perturbation 只替换被冻结 slot contract 指定的目标事实；
- 保留 source 的主语、关系、非目标上下文、未修改 slots 和回答类型；
- 非裸 source answer 不得退化为裸姓名、地点、日期、职业或标题；
- 多事实和相互依赖 slots 必须按完整 target group 一起替换；
- 不包含正确值或 alias，不产生同时为真的替代答案；
- 答案必须自然、语法完整、内部一致、类型匹配。

### 5.3 长度 gate

沿用 2026-07-14 calibration plan：

- 单个 perturbation 至少为 `max(6, ceil(0.75 × source_answer_token_count))` tokens；
- 60-row generated canary 的 perturbation/source-answer token ratio 中位数在 `[0.80, 1.35]`；
- 60-row generated canary 的 ratio P10 不低于 `0.65`；
- 20 个 hidden anchors 的 regenerated perturbation token 中位数须在相同 official anchor 中位数的 `[0.80, 1.25]`；
- hidden anchors 的 P10/P90 不得出现短答案坍缩；
- paraphrased answer 另行报告 source ratio 与 official-anchor ratio 的 median/P10/P90，并须呈现与 official anchors 相近的分布。

长度规则是分布和 answer-shape 约束，禁止通过无信息填充、重复上下文或机械扩句来过 gate。

### 5.4 60-row canary 与 20-row hidden anchors

1. 60-row canary 必须覆盖 source answer 长度五分位、answer type 和高风险 shape 类型。
2. 20-row hidden anchors 只向生成器提供原始 QA，不提供 official auxiliary fields。
3. 先完成 60+20 的机械、语义、长度、形态和人工审核；任何 gate 失败时停止，不得开始全量 3200。
4. prompt 或 validator 一旦修改，旧 canary 结论失效，必须用新 hash 重新运行 60+20。

### 5.5 Paired evaluator difficulty gate

对 20 个 hidden anchors 使用同一模型和 official evaluator 实现，成对报告：

- correct/paraphrased average token loss 与 probability；
- five wrong/perturbed average token loss 与 probability；
- Truth Ratio mean、median、Spearman、median absolute difference；
- `TR=1` 方向翻转数；
- 六答案 within-row rank correlation；
- NaN、tokenization 和 truncation 状态。

在 v3 生成之前，必须把 behavioral acceptance threshold 写入独立 amendment 并冻结。阈值只能根据 official anchors、现有 v2 calibration evidence 和研究设计制定，不能根据后续 Atomic 方法的实验结果反向挑选。

### 5.6 人工审核

- 对 60-row canary 和 20-row hidden anchors 保存真实审核范围、reviewer、日期、结论和协议状态；
- 若使用用户/项目负责人事后确认，必须记录为 retrospective attestation，不能伪造逐行 decision、时间或理由；
- 全量 v3 使用分层语义复核和保守 error queue；未逐条审核时不得声称 `3200/3200 individually reviewed`；
- 机械通过与人工抽查必须分别报告。

## 6. v3 生成顺序

1. 只读冻结 source/input/bundle IDs 和当前 v2-pilot hashes。
2. 定义并持久化 slot analysis；高风险或不确定 slot 进入人工队列。
3. 冻结 v3 prompt、repair prompt、schema、validator、model label/lock、generation parameters 与 hashes。
4. 生成并审核 60-row stratified canary。
5. 生成并审核 20-row hidden anchors，运行 paired GPU calibration。
6. 所有预注册 gates 通过后，才授权全量 3200 candidate generation。
7. 使用可恢复 batch/shard 流程；失败项定点重试，不改变已成功记录的 generation contract。
8. 合并后执行全量机械审计、长度/形态诊断、语义分层审核和 provenance 审计。
9. 建立 4000-row Eval-A view：3200 generated candidate + 800 unchanged official auxiliary fields，并保存字段级 provenance。
10. 重建所有 formal bundle Eval-A row hashes 和 manifest。
11. 冻结唯一 v3 Eval-A candidate artifact、SHA、manifest、run report、sidecars 和 release audit。
12. v3 正式冻结后，重新运行论文主表所需的 Atomic 方法、BRIDGE-R、Matched-Entity、Random-QA 和其他既定基线。

## 7. 停止条件

出现以下任一情况必须停止在当前阶段并报告，不得自动放宽门槛：

- 60-row canary 或 20-row hidden-anchor 任一 gate 失败；
- source/input payload 或 ID 顺序不一致；
- prompt/schema/model/generation hash 不一致；
- 正确值泄漏、内部矛盾或系统性 answer-shape 失败；
- 需要依据已观察到的 Atomic 实验结果修改 v3 candidate；
- 需要新付费 API 权限、全量生成权限或改变权威研究目标。

## 8. v2 与 v3 实验结果使用边界

- v2-pilot 可用于 smoke test、训练调试、资源预算和同一 Eval-A 上的暂定相对比较；
- v2-pilot 的 Atomic-vs-official Entity 绝对比较必须显式标注 evaluator-source confound；
- v3 数据规则不得依据 v2 下某个方法的优劣进行定向修改；
- v3 冻结后，论文主结论必须重新评测，不能将 v2 与 v3 的单元格拼接成一张主表；
- 若时间最终不足以生成 v3，只能把 v2 结果连同 known distribution shift 作为限制明确的结果，不能声称通过 official-compatible calibration。

## 9. 后续所需授权

- 关闭 v2-pilot 的 8-row/12-field 审计失败：需要定点 candidate 修改与新 revision 写入授权；
- 重建 4000-row view、bundle hash inventory 和 freeze records：需要派生产物写入授权；
- v2-pilot 训练/实验：需要 sbatch 脚本写入及用户提交作业授权；
- v3 60+20 canary：需要新 candidate generation、人工审核记录及 paired GPU sbatch 授权；
- v3 全量 3200：只有 60+20 gates 通过后，另行取得明确全量生成授权；如涉及付费 API，还需单独付费 API 授权。
