# BRIDGE-R v10 / Atomic-TOFU 历史构建代码归档清单

> 归档日期：2026-07-19  
> 原仓库：`/home/zkzhang/unlearn/open-unlearning`  
> 归档原则：只移动已完成阶段的一次性、硬编码构建/审计代码；不删除文件，不移动未来 v3 或正式实验仍需的代码。

## 1. 归档范围

归档文件保持原仓库相对路径，便于恢复。所有 Python 源文件在归档前均可被 `ast.parse` 解析；仓库内 `docs/`、`scripts/`、`sbatch/`、`src/`、`tests/` 未发现对这些脚本文件名的运行时引用。

| 原路径 | 用途 | 为什么移出活动目录 | 后续删除建议 | 归档前 SHA256 |
|---|---|---|---|---|
| `scripts/audit_shape_repaired_53_candidate.py` | 审计 53-row shape-repaired v1 与冻结 baseline 的差异和语义风险 | 只绑定历史 v1 SHA 和 53/174 修复范围；其失败项已由 v2-pilot 关闭 | 论文完成前建议保留用于 provenance；之后可删 | `235b493570122cf430b31d91446c67ee1ae659b694fc77ae3b5f0568a95d84f6` |
| `scripts/audit_v2_pilot_revision_20260719.py` | 对 v2-pilot 的 8 IDs/12 fields、SHA、source 和长度诊断做最终审计 | 硬编码 v2-pilot artifact 与 SHA，不能直接用于 v3 | v2-pilot 不再需要复现后可删；论文期间建议保留 | `3cc7bb23df57d5d293b239c897f7fd50eb4e8ef67e54cdb8ff1cf854d25366cc` |
| `scripts/build_bundle_eval_review_inventory.py` | 为最早的 4000-row Eval-A view 建立 3420-row review inventory | 硬编码旧 view/output；已被 v2-pilot inventory 取代，直接运行可能覆盖历史 inventory | v2-pilot/v3 manifest 稳定后可删；当前保留归档 | `5144154c76672786cb2038f36d2d700a8ff72b5f9d2bc2929b24a3846dcd2adc` |
| `scripts/build_current_route_calibration_evidence.py` | 汇总冻结 baseline 的 60 generated + 20 hidden-anchor 当前路线证据 | 绑定旧 baseline/current-route 路径；证据和 paired run 已落盘 | 当前报告长期保存即可；脚本在论文审计结束后可删 | `dd82c5dd7b616f2810202741435088c7277feeb8cfad30b2830d95e01f05b6ca` |
| `scripts/build_current_route_hidden_anchor_candidates.py` | 内嵌并写出旧路线 20 个 hidden-anchor candidate | candidate 文本和输出路径完全硬编码；v3 必须重新生成而不能复用 | v3 开始前也不应执行；历史 provenance 不再需要后可删 | `86283000dbd5642d490c1cadeb3ffeae92083ae751bf4c206a54150e70f791e5` |
| `scripts/build_eval_a_candidate_view.py` | 将旧 3200 baseline 与 800 official fields 组装为首版 4000-row view | 输出路径和 generated filename 固定为旧 v2，不能安全生成 v3；已被版本化 v2-pilot builder 取代 | 保留到 v3 builder 实现并验证后再删 | `7d0e19db82c1bd34501441b9f0f0bad1276c384129ec72f00281104f14c336b8` |
| `scripts/build_shape_repaired_candidate_revision.py` | 根据历史诊断构建 53-row/174-field shape-repaired v1 | 只适用于一次性历史修复，输出已经冻结且又被 v2-pilot 继承 | v2-pilot provenance 不再需要复现后可删 | `11874a305706a3df85c1f250459460d9fe768b6f6debc7fa1bc2d456635022fd` |
| `scripts/build_v2_pilot_bundle_hash_inventory_20260719.py` | 将 39 bundles/3420 rows 绑定到 v2-pilot view hash | 硬编码 v2-pilot view SHA 和 dated output | v3 inventory 冻结后可删；当前建议保留归档 | `ed2e4d48ac5ae543444894848d3d7172f8f244e7a429c54766d68906f9b61caf` |
| `scripts/build_v2_pilot_eval_a_view_20260719.py` | 构建 3200 generated + 800 official 的 v2-pilot 4000-row view | 硬编码 v2-pilot candidate/source SHA 和 dated output | v3 view 冻结后可删；当前建议保留归档 | `4ba358c0059b8bbc41011945c8c4d4d1ea631846c954f4cd67d33e8ba699d1ab` |
| `scripts/build_v2_pilot_revision_20260719.py` | 将 8 IDs/12 fields 的人工定点修订合入 shape-repaired v1 | 自然语言修订和所有输入/输出均硬编码；v2-pilot 已冻结 | v2-pilot 不再需要重建后可删；论文期间建议保留 | `337d5f2833dbbfabc38503c6c35b1d701cf96e0a57bac487d851250f77ae3c30` |
| `scripts/run_atomic_tofu_eval_test40.py` | 历史 40-row API/eval-extension 测试，默认路径可调用 provider | 绑定早期 test-40、旧模型变量和 API generation 路线；当前 pilot/v3/正式实验均不需要 | 源码可在确认不再复现早期 API 测试后删除；不要再次直接运行 | `5f022c8b9a0153f57a84c5a5eea3021f30f9b0232a7d6e15f52147d5382d12b4` |

同时归档上述脚本已有的 6 个 `scripts/__pycache__/*.pyc`。这些 bytecode 没有研究或复现价值，在任何时候都可以安全删除；保留它们只是为了遵守本次“不删除”要求。

## 2. 仍留在活动目录的代码

以下文件虽然由本轮工作产生，但未来仍有明确用途，因此没有归档：

| 活动路径 | 用途 | 保留原因 |
|---|---|---|
| `scripts/diagnose_frozen_3200_length_shape.py` | 对任意指定 candidate path/SHA 执行长度、裸值和 answer-shape 诊断 | 参数化程度足够，可用于 v3 canary、全量 v3 和版本对照 |
| `scripts/run_eval_a_anchor20_paired_calibration.py` | 使用官方 evaluator 对 20 个 hidden anchors 做 official/generated paired GPU calibration | v3 的 paired behavioral gate 仍需复用 |
| `sbatch/atomic_tofu_eval_a_anchor20_paired.sbatch` | 为 paired calibration 申请 1×RTX A6000、4 CPU、16G、1h | 未来 v3 hidden-anchor calibration 仍需模板；提交前应更新输入/输出版本 |

正式实现代码 `src/atomic_tofu/`、数据加载器 `src/data/atomic_tofu_v10.py`、测试、正式训练/eval sbatch 和所有数据/审计产物均未移动。

## 3. 恢复方法

需要恢复某个脚本时，从本目录按原相对路径移动回仓库根目录。例如：

```bash
cd /home/zkzhang/unlearn/open-unlearning
mv archive/bridge_r_v10_dataset_construction_2026-07-19/scripts/build_v2_pilot_revision_20260719.py scripts/
```

恢复前应先确认目标不存在，避免覆盖后续同名实现。

## 4. 删除决策

- `__pycache__/*.pyc`：可以安全删除。
- 旧 hidden-anchor/current-route/test40 脚本：其报告和 artifacts 已冻结后可以删除，但论文审计期保留更稳妥。
- v1/v2-pilot builders 与 audits：不参与未来运行；技术上可删除，但建议至少保留到 v3 冻结、v2-pilot 结果不再需要复现之后。
- 本归档中的文件都不应被正式实验或 v3 pipeline 直接调用。
