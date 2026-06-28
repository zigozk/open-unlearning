# PIPER 下一轮实验方案：lambda 扩展、官方 TOFU evaluation、多 backbone 扩展

## 1. 本轮目标

本轮不做 forget-quality matched 比较。

当前目标是把已有 Static-PIPER 干预实验从轻量 proxy 指标推进到更完整的论文实验证据链：

```text
更细 lambda 扫描
→ 官方 OpenUnlearning TOFU evaluation 指标
→ 固定 3 seeds
→ 从 NPO 扩展到额外 3 个 backbone
```

本轮仍然以 TOFU `forget10/retain90` 为主，不扩展 APU-Bench、MUSE、WMDP 或 CV。

## 2. 保留与删除的实验项

### 删除

- 不做 forget-quality matched 比较。
- 不做 seed 扩展，仍保留 `seed ∈ {0, 1, 2}`。
- 不做 Online-PI 或 Refresh-PI 大规模实验。

### 保留

- Static-PIPER。
- set-gradient PI。
- retain candidates = 500。
- Top-K fraction = 10%，即 50 个 vulnerable retain samples。
- probe batches = 64。
- train steps = 80。
- 对照方法仍包括：
  - backbone baseline
  - global retain KL
  - random local KL
  - semantic local KL
  - PI local KL

## 3. Lambda 扩展

上一轮只扫：

```text
lambda ∈ {0.1, 0.3, 1.0}
```

本轮扩展为：

```text
lambda ∈ {0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}
```

重点观察区间：

```text
0.2 - 0.5
```

原因是上一轮 `lambda=0.3` 对 PI Top-K damage 最有利，而 `lambda=1.0` 已经出现 forget-side 变弱且 semantic local KL 很强的问题。本轮不通过 matched comparison 解决 trade-off，而是直接报告不同 lambda 下的趋势曲线。

## 4. Backbone 扩展

上一轮主实验只跑 NPO。本轮保留 NPO，并额外加入三种 backbone：

| backbone | 作用 |
|---|---|
| NPO | 当前主结果，作为延续基线 |
| GradAscent | 最简单的遗忘方向，检查 PI 是否依赖 NPO |
| GradDiff | 带 retain NLL 的传统 forget-retain 组合目标 |
| SimNPO | NPO 变体，检查 PI 在不同 preference-style objective 下是否稳定 |

推荐使用合并版 pipeline 脚本：

```bash
bash sbatch/piper/piper_expanded_with_eval_pipeline.sh --submit
```

该脚本会自动提交：

1. intervention + official TOFU eval 的 array job；
2. 依赖 array 完成后的 summary job。

为适配 HPC 磁盘配额，pipeline 默认使用：

```bash
MAX_PARALLEL=1
DELETE_CHECKPOINT_AFTER_EVAL=1
```

即每次只保留少量临时 checkpoint，并在对应官方 eval 成功后删除该 checkpoint。保留的是轻量结果文件：

```text
summary.json
retain_metrics.csv
forget_metrics.csv
local_selection.csv
train_loss_curve.csv
TOFU_EVAL.json
TOFU_SUMMARY.json
```

如果只想跑 intervention，不跑官方 eval，可以：

```bash
RUN_OFFICIAL_EVAL=0 bash sbatch/piper/piper_expanded_with_eval_pipeline.sh --submit
```

如果不想保存 checkpoint，则不能直接跑官方 eval：

```bash
SAVE_MODEL=0 RUN_OFFICIAL_EVAL=0 bash sbatch/piper/piper_expanded_with_eval_pipeline.sh --submit
```

如果上一次因为磁盘配额中断，只想补跑尚未完成的 intervention 配置，使用：

```bash
SKIP_EXISTING=1 SAVE_MODEL=0 RUN_OFFICIAL_EVAL=0 \
bash sbatch/piper/piper_expanded_with_eval_pipeline.sh --submit
```

`SKIP_EXISTING=1` 会按 `model/split/backbone/method/lambda/seed` 查找已有 `summary.json`，已有则直接跳过，只跑缺失配置。

拆分版 intervention 脚本仍保留：

```bash
sbatch sbatch/piper/piper_multi_backbone_local_kl_intervention_array.sh
```

默认总规模：

```text
4 backbones × 3 seeds × (1 baseline + 4 KL methods × 7 lambdas)
= 348 runs
```

如果算力或磁盘不足，可以先只跑：

```text
NPO + GradAscent + SimNPO
```

再补 GradDiff。

## 5. 官方 OpenUnlearning TOFU evaluation

上一轮指标主要是训练脚本内部的 answer-token loss/prob proxy：

- all retain damage
- PI Top-K damage
- forget damage
- forget answer probability delta

这些指标适合机制分析，但不足以支撑 TOFU 主实验结论。本轮必须接入官方 OpenUnlearning evaluation。

使用合并版 pipeline 时，每个 array task 会在 intervention 后立即评估该 checkpoint。

拆分版官方 eval 脚本仍保留：

```bash
sbatch sbatch/piper/piper_official_tofu_eval_array.sh
```

默认使用：

```bash
python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default
```

官方 TOFU evaluator 默认指标包括：

| 指标 | 作用 |
|---|---|
| forget_Truth_Ratio | 检查 forget set 真实答案相对扰动答案的偏离 |
| forget_quality | TOFU forget quality，依赖 retain reference logs |
| forget_Q_A_Prob | forget 答案概率 |
| forget_Q_A_ROUGE | forget 答案生成相似度 |
| model_utility | retain / real-author / world-fact utility 聚合指标 |
| privleak | 隐私泄露相关指标，依赖 reference logs |
| extraction_strength | 抽取强度 |

注意：`forget_quality` 和 `privleak` 依赖 `RETAIN_LOGS_PATH`。如果没有提供 retain reference logs，它们可能为 `None`，但其他官方指标仍可用于分析。

## 6. 结果汇总

使用合并版 pipeline 时，summary job 会自动运行以下两个汇总命令。

轻量 proxy 结果汇总：

```bash
python experiments/piper/summarize_piper_intervention.py \
  --root results/piper_intervention_expanded \
  --output results/piper_intervention_expanded/intervention_summary.csv
```

官方 TOFU evaluation 汇总：

```bash
python experiments/piper/summarize_official_tofu_eval.py \
  --eval-root results/piper_official_eval \
  --intervention-root results/piper_intervention_expanded \
  --output results/piper_official_eval/tofu_eval_summary.csv
```

最终论文分析应同时报告：

- proxy retain damage：all retain damage、PI Top-K damage；
- proxy forget-side：forget damage、forget answer probability delta；
- 官方 TOFU forget-side：Forget Quality、Truth Ratio、Q/A probability、Q/A ROUGE；
- 官方 TOFU utility：Model Utility；
- 成本：PI seconds、train seconds、official eval runtime。

## 7. 当前可写的预期结论

如果本轮结果延续上一轮趋势，可以写：

```text
Across multiple unlearning backbones and a wider range of KL weights, PI-selected local preservation consistently reduces damage on PI-identified vulnerable retain samples in the lightweight proxy evaluation. Official TOFU evaluation further characterizes whether this local preservation remains compatible with benchmark-level forgetting and utility metrics.
```

如果官方指标不支持，则应写成：

```text
PI local KL improves local vulnerable-retain preservation under proxy metrics, but the effect does not yet translate into a robust official TOFU benchmark improvement.
```

## 8. 风险与解释边界

本轮不做 forget-quality matched，因此不能强声称：

```text
PIPER 严格改善了 forget-retain Pareto trade-off。
```

更稳妥的主张是：

```text
PIPER 在多个 backbone 和 lambda 下对 PI-identified vulnerable retain samples 有局部保护效果，并通过官方 TOFU 指标检查这种局部保护是否破坏 benchmark-level unlearning quality。
```

这条叙事比直接宣称 trade-off 最优更稳，也符合当前实验设计。
