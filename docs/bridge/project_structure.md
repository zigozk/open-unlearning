# BRIDGE Project Structure

本文档定义 BRIDGE 新论文线在本仓库中的组织方式。目标是让新方法、旧 PIPER 探针、毕设归档和上游 OpenUnlearning 主体代码边界清晰。

## Organizing Principles

- 上游 OpenUnlearning 主体保持稳定：`src/`、`configs/`、`scripts/`、`docs/` 的原有入口不做大规模重命名。
- BRIDGE 作为本地研究层单独成区：论文、实验计划、未来辅助脚本和 sbatch 脚本均使用 `bridge` 命名空间。
- 旧 PIPER 代码保留为机制探针和历史结果来源，不继续承载新论文方案。
- heavy results 继续写入 `results/`，默认不进入 git；需要同步时只发布轻量 CSV、markdown report 或脚本。
- 当前整理不运行新实验，也不添加不可执行的假配置。

## Top-Level Map

| Path | Role | Current policy |
|---|---|---|
| `docs/bridge/` | BRIDGE 论文、实验计划、结构说明 | 新论文线的文档主入口 |
| `experiments/bridge/` | BRIDGE 后续辅助脚本、汇总脚本、分析脚本 | 当前只放 README；代码实现后再添加脚本 |
| `sbatch/bridge/` | BRIDGE 后续集群任务脚本 | 当前只放 README；实验开始前再添加可提交脚本 |
| `legacy_piper_work/experiments/piper/` | PIPER / PI probe 历史机制实验 | 保留参考，不新增 BRIDGE 主线代码 |
| `legacy_piper_work/sbatch/piper/` | PIPER 历史集群脚本 | 保留复现和结果汇总用途 |
| `legacy_piper_work/results/` | PIPER 历史结果和轻量报告 | 保留复现参考；heavy outputs 仍受 ignore 规则保护 |
| `results/` | 本地或 HPC 实验输出 | 默认 ignored；发布时只 force-add 轻量总结产物 |
| `legacy_thesis_work/` | 毕设阶段归档 | 不继续写入新 BRIDGE 产物 |
| `legacy_piper_work/GPT/` | PIPER / FABLE-PI / 旧草稿材料 | 作为旧工作草稿区，当前被 `.gitignore` 的 `GPT/` 规则忽略 |

## Future Code Placement

BRIDGE 正式实现时建议按以下边界落位：

| Future artifact | Suggested path | Notes |
|---|---|---|
| BRIDGE trainer implementation | `src/trainer/unlearn/bridge.py` or a small extension around `npo.py` | 优先复用 NPO erase loss，减少与 backbone 逻辑耦合 |
| Trainer config | `configs/trainer/BRIDGE_NPO.yaml` | 等 trainer 可运行后再创建 |
| TOFU experiment config override | `configs/experiment/unlearn/tofu/bridge.yaml` if needed | 只有当默认 TOFU config 不够表达 BRIDGE 参数时再添加 |
| Result summarizer | `experiments/bridge/summarize_bridge_tofu.py` | 汇总 FQ/MU、retain KL drift、DRO risk、prior quality、cost |
| Prior diagnostics | `experiments/bridge/analyze_prior_quality.py` | Spearman、top-k enrichment、oracle overlap |
| Cluster jobs | `sbatch/bridge/*.sh` | 按 phase 和 method 命名，先 single seed 后 multi seed |

## Phase Layout

| Phase | Method group | Status |
|---|---|---|
| Phase 1 | NPO baseline | Planned |
| Phase 2 | NPO + Global KL / Uniform-DRO | Planned |
| Phase 3 | History-DRO | Planned |
| Phase 4 | Refresh-GS-DRO / optional Online-GS-DRO | Planned |
| Phase 5 | Refresh-PI-DRO / optional Online-PI-DRO | Planned |
| Phase 6 | Multi-seed, FQ-matched, full cost analysis | Deferred |

## Naming Conventions

Suggested method names:

- `npo`
- `npo_global_kl`
- `bridge_uniform_dro`
- `bridge_history_dro`
- `bridge_refresh_gs_dro`
- `bridge_online_gs_dro`
- `bridge_refresh_pi_dro`
- `bridge_online_pi_dro`

Suggested task name pattern:

```text
BRIDGE_TOFU_<split>_<model>_<method>_seed<seed>
```

Example:

```text
BRIDGE_TOFU_forget10_Llama3p2_1B_bridge_history_dro_seed0
```

Suggested result roots:

```text
results/bridge_initial/
results/bridge_initial_eval/
results/bridge_reports/
```

## Migration Notes From PIPER

The old PIPER line asked whether PI can identify vulnerable retain samples and whether local KL intervention helps. BRIDGE narrows and strengthens the first implementation target:

- PIPER probe signal becomes one possible BRIDGE prior: PI-DRO.
- Local KL intervention is replaced by a unified boundary DRO risk over retain KL drift.
- New first-round comparison must include Uniform-DRO and History-DRO so the paper can separate "DRO itself works" from "expensive GS/PI prior is necessary".
- Official TOFU FQ/MU remains the primary external metric; retain KL drift and prior quality are internal mechanism metrics.

## No-Experiment Rule For This Reorganization

This structure pass intentionally does not:

- submit sbatch jobs;
- run training or evaluation;
- generate new results under `results/`;
- create runnable BRIDGE configs before trainer code exists.
