# Atomic-TOFU v2.0-rc1 / BRIDGE-R v10 实现说明

本实现只把 2026-07-12 的 v10/v3 方案变成可审计的数据与训练契约，不把 mock/API 候选当作 gold，也不声称存在新实验结果。

## 数据流水线

```bash
export PYTHONPATH="$PWD/src"
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE

python -m atomic_tofu.pipeline --stage source --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline --stage prepare-annotation --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline --stage run-annotation --provider mock --resume --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline --stage validate-annotation --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline --stage review-packets --release-root data/atomic_tofu/v2.0-rc1
```

真实 API 候选把 `--provider mock` 改为 `--provider openai`，并设置 `OPENAI_API_KEY`、已验证的 `ATOMIC_TOFU_ANNOTATION_MODEL` / `ATOMIC_TOFU_EVAL_MODEL`。client 使用 Responses API 的严格 JSON Schema Structured Outputs；每个 atom 包含 aliases/source/evidence，relation 包含 span/reason/confidence。每个 atom 都必须独立扫描 20 条 QA 的 question 与 answer；relation 是 atom×QA 多对多关系，同一 QA 若适用于多个 atom 必须在各自 relation 表中出现。20 条输入 QA 由 material relation 或 `unassigned_qa_ids` 精确覆盖。`support|leak|closure` QA 自动构成 target forget closure。请求、响应元数据、usage、content hash、resume cache 与 error queue 分开保存。API 输出仍是 candidate-only。

姓名策略只保护代表作者本人身份的 `name`/`full_name`/`author_name` atom；父亲、母亲及其他家庭成员姓名是普通事实，可作为 forget target。旧候选若把作者本人姓名写成 literal subject，策略仍会识别；策略只从 effective candidate 排除作者本人姓名请求，不修改 raw API candidate。

作者人工复核位于 `review_packets/authors/<author_id>.{md,json}`，决定模板位于 `adjudicated/author_decisions/`。每位作者必须完成 accept/revise/reject；High、Multi、ambiguous 和正式 bundle closure 还需二审。

如果 Low coverage supplemental candidate 已完成独立审计，不能直接运行
`compile-requests` 期待它们自动进入主图。先使用带明确二审人的独立 merge stage：

```bash
python -m atomic_tofu.pipeline \
  --stage merge-low-supplement \
  --second-reviewer zzzz \
  --release-root data/atomic_tofu/v2.0-rc1
```

该 stage 保留 `request_graph/requests.pre_supplemental.jsonl` 备份，把接受的
`adjudicated/supplemental_low_accepted_candidates.jsonl` 合并进 canonical request graph，
并在 provenance 中记录 supplemental 来源和二审人。原始 API candidate 不会被覆盖。

合并后必须重新运行 `cell-feasibility-audit` 和 `compile-bundles`；旧的
`request_graph_report.json`、`cell_feasibility_report.json`、`bundle_compile_report.json`
不能继续当作合并后的报告。

Balanced official-scale bundle 使用预注册的 `balanced_v1` quota、唯一 owner attribution
和约束 solver：

```bash
python -m atomic_tofu.pipeline \
  --stage compile-balanced-bundles \
  --release-root data/atomic_tofu/v2.0-rc1
```

该 stage 生成 `bundle_specs/balanced_v1.json`、
`audit/balanced_bundle_feasibility_report.json` 以及
`bundles/balanced/<Single|Multi>/<40|200>/seed_<n>.{json,manifest.json}`。
只有每个 family×size 都有 3 个 exact distinct seed 且所有约束通过时，报告才会标记为
`completed`；Pure-200 coverage 不足的 cell 保留为 `unavailable_at_200`，不使用 retain QA
填充或截断 closure。

评测扩展：

```bash
python -m atomic_tofu.pipeline --stage prepare-eval --release-root data/atomic_tofu/v2.0-rc1
```

`run-eval` 全量命令只能在下方 answer-shape calibration 与人工 gate 通过后运行。

Eval-A 生成在 canary 阶段还必须经过 deterministic semantic gate。该 gate 会拒绝原文复制、
JSON/列表式多答案、元评论、重复 perturbation、异常长答案和正确值/别名泄漏，并对失败候选
自动执行有限次数 repair；失败记录保存在 `api/eval_extension/error_queue.jsonl`，旧候选会归档到
`api/eval_extension/attempts/`。此前的 `calibration_20_ids.txt` 仅作为旧版诊断样本；answer-shape
方案要求先运行下方新的分层 calibration。若需复核旧版 prompt，可使用该文件，但不得据此放行全量：

```bash
export ATOMIC_TOFU_MAX_VALIDATION_REPAIRS=1
python -m atomic_tofu.pipeline \
  --stage run-eval \
  --provider openai \
  --resume \
  --unit-ids-file data/atomic_tofu/v2.0-rc1/api/eval_extension/calibration_20_ids.txt \
  --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline \
  --stage validate-eval-canary \
  --unit-ids-file data/atomic_tofu/v2.0-rc1/api/eval_extension/calibration_20_ids.txt \
  --release-root data/atomic_tofu/v2.0-rc1
```

只有 `audit/eval_extension_canary_report.json` 显示 `canary_semantics_passed`，并完成同一批
canary 的人工检查后，才允许去掉 `--unit-ids-file` 运行剩余全量。`validate-eval` 仍是
4,000-row 全量 gate，不能用未完成的 canary 报告冒充全量通过。模型通过环境变量
`ATOMIC_TOFU_EVAL_MODEL` 指定。本次批准配置使用
`gpt-5-mini` 浮动别名，因此报告会记录 `model_lock: floating_alias` 并给出不可冻结、不可完全复现的警告；
这不再满足原方案的 fixed-snapshot freeze 条件。若要发布可复现冻结版，必须改回具体 snapshot。
可选的 `ATOMIC_TOFU_REASONING_EFFORT=low` 只用于成本实验，不能替代语义 gate。

Answer-shape calibration 已取代“每作者第一条 QA”的 20 条 canary 作为 Eval-A 的正式前置 gate。
它准备 60 条按 source answer 长度五分位分层的 generated canary，以及 20 条隐藏 official-anchor
regeneration；二者使用独立目录，不覆盖官方 sidecars：

```bash
python -m atomic_tofu.pipeline \
  --stage prepare-eval-calibration \
  --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline \
  --stage run-eval-generated-calibration \
  --provider openai \
  --resume \
  --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline \
  --stage run-eval-anchor-calibration \
  --provider openai \
  --resume \
  --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline \
  --stage validate-eval-calibration \
  --release-root data/atomic_tofu/v2.0-rc1
```

输入 manifest 位于 `api/eval_extension/calibration/generated/generated_60_manifest.json` 和
`api/eval_extension/anchor_calibration/anchor_20_manifest.json`；综合报告为
`audit/eval_anchor_calibration_report.json`。报告必须先达到
`calibration_mechanical_passed_human_review_pending`，再完成 60+20 条人工盲审；旧 20 条短
candidate 只保留作历史审计，不能并入 Eval-A。该 gate 额外检查 perturbation/source-answer
token ratio、source answer template 保留和 hidden-anchor 与 official 的长度比。

正式 freeze 前必须补齐 official-anchor blind calibration、人工规则 gate，并冻结供所有 Atomic 方法、bundle 和 seed 共用的单一 Eval-A。根据 2026-07-18 范围修订，完整 Eval-B 不再是 release gate，仅作为可选后续或附录研究；mock 只验证结构和恢复流程。

## BRIDGE-R

训练 ladder 配置为：

```text
NPO_BRIDGE_R_Backbone
NPO_BRIDGE_R_GlobalKL
NPO_BRIDGE_R_LocalMean
NPO_BRIDGE_R_CenteredTail
```

`AtomicTOFUUnlearnDataset` 从当前 forget QA 映射到 request IDs。每个 request 的 `protected_train_qa_ids`/`protected_eval_qa_ids` 固定为同作者全部 QA 扣除该 request 的 forget closure，并保持 request-scoped：同一 bundle 另一 request 的 target 不会使其缩小。全局 retain train 则独立使用 bundle 完整 forget union 的补集。logical K 默认 8，与 per-device microbatch 4 分离；多 request batch 先按 request 分组、再合并。trainer 使用冻结 full teacher，目标是 NPO backbone + global KL + grouped local mean + centered uniform log-mean-exp tail。GS/KL-PI 未加入 v10 主配置，必须等 UniformTail gate。

正式训练固定 20 optimizer steps、effective batch 32、同一 full checkpoint/recipe/seed。全局 retain \(G_R=D\setminus F_R\) 是随机采样候选集：该预算不要求逐条覆盖，也不生成额外 remainder eval。当前实现不会自动提交 Slurm 作业。

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_atomic_tofu_v10*.py'
PYTHONPATH=src python -m atomic_tofu.pipeline --stage dry-run --release-root /tmp/atomic-tofu-v10-dry-run
```

dry-run 成功只表示 4,000-row source、200 author units、candidate schemas、review packet 和 4,000-row eval sidecar 结构贯通；报告会明确把人工、calibration、单一 Eval-A freeze、bundle/retrain/training 标成 pending。完整 Eval-B 不再是必做门槛。
