# BRIDGE-R v10 受控清理清单

日期：2026-07-12。清理前基线为 commit `4dee1eb12d5b5a2a54709d3dc1b570770bb5b2d8`、分支 `bridge-initial-plan`。完整清理前状态（542 项，其中 508 个未跟踪文件）保存在 `BRIDGE_R_v10_git_status_before_cleanup_2026-07-12.txt`。

## 已确认删除

- `docs/BRIDGE-R/`：v5/v6/v8 旧方案、旧数据契约和日志，已被 2026-07-12 的 v10/v3 附件取代。
- `configs/experiment/{eval,finetune,unlearn}/atomic_tofu/`：引用不存在的旧 handler/config（`atomic_tofu`、`atomic_tofu_reference`、`atomic_tofu_train`、`atomic_tofu_unlearn`、`BRIDGER`、`AtomicTOFURequestCollator`），不能满足 full-corpus v10 契约。
- `src/evals/metrics/atomic_tofu.py` 及 `src/evals/metrics/__init__.py` 中对应导入/注册：旧 metric 依赖已不存在的 `data.atomic_tofu`，且只实现旧 request-gap 侧指标，不能替代官方 TOFU 指标链。
- `results/eval/`、`results/unlearn/`：未发现名称或配置可明确归属于 Atomic-TOFU/BRIDGE-R 的产物，因此未删除任何结果。

## 明确保留

- 全部原始 BRIDGE：`docs/BRIDGE/`、`sbatch/bridge/`、trainer/config 实现，以及所有名称含 `NPO_BRIDGE` 的旧实验结果。
- 官方/reference 输出 `saves/`，现有 baseline/full eval 结果，`logs/bridge_r/` 和字节码缓存：后两者不在获准候选清理范围内，且不作为新版设计依据。
- 三份 2026-07-12 权威方案保留在仓库外附件位置，不复制为旧 `docs/BRIDGE-R/` 设计文件。

## 未处理的预存改动

- `configs/data/...` 的全部删除，以及 `results/unlearn_summary.*`、`sbatch/README.md`、`sbatch/slurm_tofu_unlearn_eval.sbatch`、`sbatch/summarize_unlearn_results.py` 等既有修改。
- 大量既有未跟踪 `results/eval/`、`results/unlearn/` 模型与评测产物、BRIDGE 状态文档和任务汇总。
- 已暂存的原始 BRIDGE 提交脚本移动保持原样；本次清理提交不会纳入它。

未知预存改动不会被恢复、删除、暂存或提交。新版实现采用独立的新路径，并继续复用官方 metric handlers 与原始 BRIDGE 代码。
