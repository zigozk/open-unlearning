# Atomic-TOFU / BRIDGE-R v10：Eval-B 范围修订

> 日期：2026-07-18
> 状态：已批准的权威范围修订
> 生效阶段：正式训练与实验开始前

## 1. 决定

完整独立 Eval-B 从 Atomic-TOFU 数据集冻结、BRIDGE-R 正式实验和论文必做门槛中移除。Eval-B 仅保留为时间允许时的可选后续或附录 robustness study；缺少 Eval-B 不得阻塞当前 release、训练或主实验。

本修订只改变 Eval-B 相关范围，不改变三份 2026-07-12 最高权威方案中的研究目标、训练数据契约、bundle 设计、方法比较、指标定义或其他质量门槛。

## 2. 仍然必做

1. 冻结唯一 Eval-A artifact，并让所有 Atomic 方法、bundle 和 seed 使用完全相同的版本。
2. 官方 Entity01/05 主结果继续使用 official fields；不得把 generated extension 称为官方 TOFU 数据。
3. 完成 hidden official-anchor regenerate-and-compare、人工盲审和规则 gate，量化 schema、answer-shape、长度、难度与语义偏差。
4. 保留 Matched-Entity、Random-QA、bundle seeds、initial-difficulty matching 等既定控制。
5. Codex/API 生成结果只能称为 candidate；机械 validator 或空 error queue 不代表语义零错误。
6. 生成 provenance 必须报告 `model_lock=floating_alias`，不得声称 fixed snapshot。

## 3. 论文与实验的声明边界

Atomic-vs-Entity 和方法排名可以在单一冻结 Eval-A 的预注册口径下报告，但必须显式注明 evaluator-source caveat。没有完整 Eval-B 时：

- 不声称结果通过独立 evaluator view 复现；
- 不声称方法排名具备 Eval-A/Eval-B 跨视图稳健性；
- 不声称已完全排除 evaluator-source effect；
- hidden official-anchor calibration 只能用于量化和缓解该风险。

## 4. 权威关系

本修订覆盖以下现行文档中把完整 Eval-B 规定为必做项的条款：

1. `BRIDGE_R_paper_plan_v10_full_corpus_2026-07-12.md`
2. `BRIDGE_R_experiment_plan_v10_full_corpus_remote_2026-07-12.md`
3. `Atomic_TOFU_dataset_construction_plan_v3_full_corpus_api_2026-07-12.md`
4. `/home/zkzhang/unlearn/Atomic_TOFU_eval_answer_shape_calibration_plan_2026-07-14.md`
5. `docs/bridge_r_v10/README.md`

历史 handoff、历史审计报告和当时生成的依赖报告保留其原始事实记录；其中的旧 Eval-B pending 表述由本修订覆盖，不再构成当前 gate。
