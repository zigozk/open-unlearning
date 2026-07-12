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

真实 API 候选把 `--provider mock` 改为 `--provider openai`，并设置 `OPENAI_API_KEY`、固定 snapshot 的 `ATOMIC_TOFU_ANNOTATION_MODEL` / `ATOMIC_TOFU_EVAL_MODEL`。client 使用 Responses API 的严格 JSON Schema Structured Outputs；请求、响应元数据、usage、content hash、resume cache 与 error queue 分开保存。API 输出仍是 candidate-only。

作者人工复核位于 `review_packets/authors/<author_id>.{md,json}`，决定模板位于 `adjudicated/author_decisions/`。每位作者必须完成 accept/revise/reject；High、Multi、ambiguous 和正式 bundle closure 还需二审。

评测扩展：

```bash
python -m atomic_tofu.pipeline --stage prepare-eval --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline --stage run-eval --provider openai --resume --release-root data/atomic_tofu/v2.0-rc1
python -m atomic_tofu.pipeline --stage validate-eval --release-root data/atomic_tofu/v2.0-rc1
```

正式 freeze 前必须补齐 official-anchor blind calibration、人工规则 gate 与独立 Eval-B；mock 只验证结构和恢复流程。

## BRIDGE-R

训练 ladder 配置为：

```text
NPO_BRIDGE_R_Backbone
NPO_BRIDGE_R_GlobalKL
NPO_BRIDGE_R_LocalMean
NPO_BRIDGE_R_CenteredTail
```

`AtomicTOFUUnlearnDataset` 从当前 forget QA 映射到 request IDs，再按 request 轮转采样 protected QAs；logical K 默认 8，与 per-device microbatch 4 分离。多 request batch 先按 request 分组、再合并。trainer 使用冻结 full teacher，目标是 NPO backbone + global KL + grouped local mean + centered uniform log-mean-exp tail。GS/KL-PI 未加入 v10 主配置，必须等 UniformTail gate。

正式训练固定 20 optimizer steps、effective batch 32、同一 full checkpoint/recipe/seed。当前实现不会自动提交 Slurm 作业。

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_atomic_tofu_v10*.py'
PYTHONPATH=src python -m atomic_tofu.pipeline --stage dry-run --release-root /tmp/atomic-tofu-v10-dry-run
```

dry-run 成功只表示 4,000-row source、200 author units、candidate schemas、review packet 和 4,000-row eval sidecar 结构贯通；报告会明确把人工、calibration、Eval-B、bundle/retrain/training 标成 pending。

