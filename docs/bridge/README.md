# BRIDGE Project Hub

本目录是当前仓库中 BRIDGE 论文与实验工作的主入口。

BRIDGE 的新主线是 retain-side prior-guided instance-level DRO：在 OpenUnlearning 的官方 TOFU 设置下，以 NPO 为首个 backbone，通过 Global retain KL、Uniform/History/GS/PI prior 和 boundary DRO 保护高风险 retain samples。

## Canonical Documents

- [BRIDGE_paper_plan.md](BRIDGE_paper_plan.md)：论文方案、方法定义、相关工作边界和预期叙事。
- [BRIDGE_initial_experiment_plan.md](BRIDGE_initial_experiment_plan.md)：首轮 single-seed 实验矩阵、指标和成功判定。
- [project_structure.md](project_structure.md)：仓库重排后的目录职责、旧材料边界和后续实现路线。

## Current Status

当前阶段只完成项目结构和研究计划整理，尚未启动新的 BRIDGE 实验。

已确定的第一轮范围：

- Dataset: official TOFU
- Split: forget10 / retain90
- Backbone: NPO
- Seed: 0
- Required methods: NPO, NPO + Global KL, Uniform-DRO, History-DRO, Refresh-GS-DRO, Refresh-PI-DRO

暂缓事项：

- 多 seed
- FQ-matched Pareto
- 完整 cost analysis
- 多 backbone
- FineTOFU / Residual-PI / Oracle-DRO 大规模训练

## Repo Entry Points

- `experiments/bridge/`：后续 BRIDGE 专属实验辅助脚本与汇总脚本命名空间，目前只保留说明文档。
- `sbatch/bridge/`：后续 BRIDGE 集群任务脚本命名空间，目前只保留说明文档。
- `legacy_piper_work/`：旧 PIPER / PI probe / GPT 草稿 / PIPER 结果材料，保留用于历史参考，不再作为新论文主线入口。
- `legacy_thesis_work/`：本科毕设阶段归档材料，不再写入新的 BRIDGE 产物。

## Next Implementation Gate

开始写代码前，应先完成：

1. 确认 BRIDGE trainer 是继承/包装 NPO trainer，还是在现有 NPO handler 内增加可关闭的 BRIDGE loss 分支。
2. 明确 per-sample retain KL drift 的张量形状、mask 规则和 reference model 获取方式。
3. 确认 History prior 需要的 retain sample id 是否能从当前 data pipeline 稳定传入。
4. 为 Global KL、Uniform-DRO 和 History-DRO 先做最小实现，再决定 GS/PI 的低成本实现路径。
