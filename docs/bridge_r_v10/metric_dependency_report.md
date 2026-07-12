# BRIDGE-R v10 官方 TOFU 指标依赖审计

日期：2026-07-12。实现基线：OpenUnlearning commit `4dee1eb`；新版清理提交：`92b2e29`。

| 输出字段 | 官方 handler | 数据与列 | precompute / reference | 方向 |
|---|---|---|---|---|
| `forget_Q_A_Prob` | `probability` | forget: `question, answer` | token-average answer probability | 低通常表示更强遗忘，需对齐 retrain |
| `forget_Q_A_ROUGE` | `rouge` | forget: `question, answer` | 固定 generation config | 低通常表示更强遗忘 |
| `forget_truth_ratio` | `truth_ratio` | 同一 forget row 的 `paraphrased_answer, perturbed_answer[5]` | correct/wrong probability；`closer_to_1_better` | 高为更接近错误/正确等可能 |
| `forget_quality` | `ks_test` | unlearn forget TR | 对应 bundle retrain 的同一 forget view TR，逐 index 对齐 | KS p-value 高为更像 retrain；forget01 易平台化 |
| `model_utility` | `hm_aggregate` | 固定 retain、real-author、world-fact views | retain/RA/WF 各 Prob、ROUGE、TR，共 9 项 | 高为好 |
| `extraction_strength` | `extraction_strength` | forget: `question, answer` | suffix exact-extraction evaluator | 低为更少可提取 |
| `privleak` | `privleak` | forget + size-matched holdout | `mia_min_k` 与 bundle retrain reference log | 绝对值接近 0 为与 retrain 接近 |

数据契约要点：

- full source 是 4,000 条、200 个连续作者块、每块 20 QA；只含 `question/answer`。
- forget01/05/10 perturbed 与固定 `retain_perturbed` 的 schema 均为 `question, answer, paraphrased_answer, perturbed_answer, paraphrased_question`，本机缓存中 `perturbed_answer` cardinality 为 5。
- RA/WF perturbed 只需要 `question, answer, perturbed_answer`；holdout 需要 `question, answer`。
- FQ 与 PrivLeak 的 reference 必须是同一 base model、训练 recipe、seed、bundle 和评测 views 的 retrain `TOFU_EVAL.json`。官方 retain99/95 只能作 provenance 完全匹配时的 sanity。
- Atomic 新生成字段是 sidecar，不能覆盖原始官方字段。主 bundle closure 不得触及固定 `retain_perturbed` anchors。

仓库冲突与处理：

- 工作树在本任务开始前已有 `configs/data/...` 的预存删除；按要求不恢复、不暂存、不提交。
- `configs/eval/atomic_tofu_v10.yaml` 直接复用原 `TOFUEvaluator` 与 metric handlers，只将数据 handler 定向到冻结的本地 JSONL views；没有复制或改写指标公式。
- Atomic evaluator 配置强制提供 forget/retain/RA/WF/holdout 路径和配对 retrain log，避免缺依赖时静默降级。

尚未通过的正式 gates：人工逐作者 adjudication、生成题语义/anchor calibration、独立 Eval-B、bundle-specific retrain logs 和模型 evaluator smoke。它们不得被本地 mock dry-run 代替。

