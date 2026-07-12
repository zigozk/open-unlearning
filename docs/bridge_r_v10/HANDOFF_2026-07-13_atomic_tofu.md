# Atomic-TOFU / BRIDGE-R v10 接手文档（2026-07-13）

本文件面向**没有任何前序对话上下文**的接手会话。先完整阅读本文件，再读取
`docs/bridge_r_v10/README.md` 和文中指定的源码/审计文件。不要把这里的候选输出当成
gold 数据或实验结果。

## 1. 工作范围与不可违反的约束

- 唯一项目目录：`/home/zkzhang/unlearn/open-unlearning`（服务器 hpc）。
- 不重建环境；沿用现有 BRIDGE 环境和官方 OpenUnlearning/TOFU 实现。
- 不修改、不删除原始 BRIDGE，特别是 `docs/BRIDGE/`、`sbatch/bridge/`、以及
  `results/unlearn` 中名含 `NPO_BRIDGE` 的结果。
- Atomic-TOFU 是完整 TOFU corpus：4,000 QA、200 位作者、每位 20 QA。不得改训练 QA
  的文本、不得拆分 QA、不得使用 `source_family_weight`。
- API 生成是候选，不是人工验收；最终仍要求作者组织的人工复核包、accept/revise/reject，
  并对 High/Multi/ambiguous/bundle closure 二审。
- 官方 TOFU 的 forget correct/paraphrase/perturb、retain、RA、WF、固定
  `retain_perturbed` anchor、以及对应 retrain 模型均须复用。新生成的问题只能作为评测
  sidecar，不能覆盖官方字段。
- 真实训练尚未启动。禁止把计划、mock、候选或未验证观察写成“结果”。

## 2. 工作树与提交安全

当前分支已有的 Atomic-TOFU 相关提交（由旧到新）为：

```text
92b2e29 chore: remove obsolete BRIDGE-R prototype before v10
cb2cdfc feat: implement full-corpus Atomic-TOFU and BRIDGE-R v10
5965905 fix: support versioned compatible API endpoints
d7f257d fix: preserve Atomic-TOFU review decisions
ac009d6 fix: preserve resumed Atomic-TOFU API subsets
0e2693e fix: version Atomic-TOFU annotation generations
6c8a7b4 feat: add Atomic-TOFU annotation review reports
3973acb fix: require closure-external protected QA candidates
9ac0cda feat: protect author names from Atomic-TOFU requests
1a52cb9 refactor: use official entity splits as controls
5850491 feat: compact and repair Atomic-TOFU annotations
03befb8 fix: report per-run Atomic-TOFU API usage
```

工作树**本来就是脏的**，包含大量用户预存的 `configs/data/...` 删除、`results/`、`sbatch/`
变动，以及一个暂存的 BRIDGE 脚本重命名。它们均不属于本任务的自动处理对象。

每次修改前后运行：

```bash
cd /home/zkzhang/unlearn/open-unlearning
git status --short
git diff --check
```

绝不使用 `git clean`、`git reset`、`git checkout`。提交时只用精确路径：

```bash
git add <本次确认的文件>
git diff --cached --name-status
git commit --only <本次确认的文件> -m "<message>"
```

若 `git diff --cached --name-status` 出现任何预存文件，停止并报告；不要 `git reset`
来“清理”暂存区。

## 3. 已实现的架构

入口：`python -m atomic_tofu.pipeline`。

| 区域 | 关键文件 | 当前职责 |
|---|---|---|
| 数据导出 | `src/atomic_tofu/source.py` | 从官方 snapshot 导出完整 source 和 official anchor。 |
| 原子候选 | `annotation.py` | v11 紧凑 JSON schema、prompt、输入单元、候选结构验证。 |
| API | `providers.py` | OpenAI-compatible Responses API；正确处理 base URL 已含 `/v1` 的第三方平台。 |
| 流水线 | `pipeline.py` | resume、subset、结构失败时一次 repair、attempt/error/run report。 |
| 姓名策略 | `policies.py` | 作者姓名是 scope identifier，不允许作为 forget target；只从**有效候选**排除原始请求，不篡改原始 API 输出。 |
| 人工报告 | `reporting.py` | `annotation-review-report` 输出每作者报告。 |
| 请求图 | `request_graph.py` | 从已人工 adjudicated 的候选编译训练/评测请求。 |
| BRIDGE-R | `src/...` 配置及 trainer 代码 | ladder 为 Backbone → GlobalKL → LocalMean → CenteredTail；尚未提交真实训练。 |

`README.md` 中的 dry-run 是结构验证，不是正式数据冻结。

### 已落实的实验契约

- 20 optimizer steps 是可比较基线；effective batch 32；logical candidate pool K（默认 8）
  不是 microbatch（默认 per-device 4）。
- 主比较同一 baseline：Official Entity TOFU、Atomic Single-Balanced、Atomic
  Multi-Balanced，配 matched entity/random control；retrain 必须匹配 base model、recipe、seed。
- BRIDGE-R 先严格完成 Backbone、+GlobalKL、+LocalMean、+Tail；GS/KL-PI 只有在
  UniformTail gate 后才能考虑。
- forget01 FQ 容易平台化，不能以单一 FQ 调参；GlobalKL/Uniform-DRO 须严格消融；
  不得用中间 checkpoint 最优值作最终结论。

## 4. API 与当前生成状态

第三方 OpenAI-compatible 平台：

```bash
export OPENAI_BASE_URL='https://api.zhizengzeng.com/v1'
export ATOMIC_TOFU_ANNOTATION_MODEL='gpt-5-mini'
```

`ResponsesProvider.responses_endpoint()` 会避免拼成 `/v1/v1/responses`。用户已经比较过：
`gpt-5.4-nano` 在本任务的 canary 质量明显较 `gpt-5-mini` 差，当前选择保持
`gpt-5-mini`。不要擅自换回 nano。

当前 `candidate_outputs.jsonl` 有作者 `tofu_author_000` 到 `tofu_author_019` 共 20 条。
它们是不同 prompt/model generation 的混合候选，尚未通过最终人工验收，不能作为冻结版本。
历史 attempts 在：

```text
data/atomic_tofu/v2.0-rc1/api/annotation/attempts/
```

校准选择文件：

```text
api/annotation/calibration/canary_1_id.txt       # 000
api/annotation/calibration/calibration_8_ids.txt # 000–007
api/annotation/calibration/blind_12_ids.txt      # 008–019
api/annotation/calibration/remaining_180_ids.txt # 020–199
```

### v11 canary 的确切观察

最新 `tofu_author_000` 当前记录为：

```text
record_id: annotation_tofu_author_000_905c0bed69a4_88ec98577a2c
generation_sha256: 88ec98577a2c21fc02fb01f2658aab6f7d864ba29f46b94e94d8a03e66e7dbb8
model: gpt-5-mini
atoms / Single / Multi: 20 / 5 / 2
material relation 覆盖: 18 / 20
未被任何 atom 关系引用: tofu_full_0006, tofu_full_0016
API usage（平台与 provenance 一致）: input 2268, output 6900（其中 reasoning 3904）, total 9168
```

这次的 6,900 output tokens 比旧版紧凑前明显低，结构也通过；但有以下尚未修复的
**质量风险**：

1. 输出没有字段要求列出未分配 QA，因而 0006（早期写作兴趣/大学）和 0016（文学活动/
   workshop）会静默遗漏。这里的风险是 annotation recall 下降，而不是强迫模型把背景 QA
   判作 support。
2. candidate request 的 `per_atom_closures` 是模型明确选择的 forget closure，但当前
   `request_graph.compile_request()` 重新把某 atom 的所有 `support/leak/closure` QA 都并入
   closure。例如 000 的 birthplace request 声明 closure 只有 `tofu_full_0000`，当前编译器
   会因 atom relation 还含 `tofu_full_0008` 而扩大 forget 集；genre/title 类同。这会改变
   用户已审核的 request 语义。
3. 各 candidate request 现在普遍 `protected_train_qa_ids: []`。这**不表示**最终训练/
   评测没有 protected QA：当前 compiler 用 `candidate explicit protected ∪ non-target
   surviving atom residual support` 生成请求本地 protected pool，并将同一列表写为
   `protected_eval_qa_ids`。它用于 BRIDGE-R LocalMean/Tail 的训练输入、same-author protected
   eval view 和 `protected_cvar20` 指标。因此 protected pool 必不可删；只是其来源和非空性
   应显式审计。
4. `reporting.py` 仍显示 v10 已删除的 aliases/source/evidence/reason/confidence 字段，故 v11
   报告中这些行会是空白，不能作为“模型没有产生这些内容”的质量判断。它应改为展示 material
   QA relations、声明 closure、最终编译 protected pool/provenance、unassigned QA。

### 已否决的方案（必须尊重）

审计 `style_genre_evidence_audit_calibration8.md` 曾发现模型把一些背景/书名 list QA 当作
某事实 atom 的 support。用户明确决定：**不要**把“直接蕴含”或“背景 QA 不能是 support”做成
prompt 硬规则或自动 validator。此类语义问题留给完整生成后的 AI 复核 + 人工复审。

因此不要把 candidate `per_atom_closures` 与所有 `support/leak/closure` relation 强制相等，
也不要因背景判断自动删改数据。

### 姓名策略的正确解释

姓名策略适用于**所有作者**，不是仅 author 010/011/013。报告的 `policy_exclusions` 仅列出现有
raw candidate 中真的违反策略的请求。遇到 name target 时，保留原始 candidate 以便审计，
从 effective candidate 整条排除（Multi 也整条排除，不能偷偷缩成 Single）。后续新 prompt
已经要求模型从一开始不要 target author-name atom。

## 5. Token 报告问题及其状态

旧 run_report 曾把 candidate pool 全历史 usage 误显示为本次 1-author run 的 usage，例如：

```text
input 45068 / output 342731 / total 387799
```

这不是本次 canary 花费，而是 20 条候选的历史累计。`03befb8` 已修改代码，新的运行应写：

- `usage` 与 `api_usage_this_run`：这次真实新 API 调用的合计；
- `selected_candidate_usage`：所选 author 当前 candidate 的 final-response usage；
- `candidate_pool_usage`：当前所有 candidate 的历史总量；
- `api_call_count_this_run`：本次 API 调用数（含 repair）。

当前磁盘上的 `run_report.json` 仍是**修复前生成的旧文件**，所以仍显示旧总数。必须先进行一次
小 subset 的新运行，才能观察新字段；不可据此判断 `03befb8` 无效。

`repair_attempts/` 不存在或为空也可以正常：仅当初始 candidate 结构验证失败、需要 repair 时
才创建。当前 000 v11 首轮通过，故没有 repair attempt。

## 6. 下一件应做的工作（尚未实施）

在进一步付费 API 扩展前，先实施并验证以下最小改动。这个方案已向用户解释，尚未得到新的
“开始修改”指令时不要自行改代码；若用户授权，应按此执行。

1. **显式 coverage，而非强迫语义分类。**
   - 在 schema 增加顶层紧凑字段 `unassigned_qa_ids`。
   - prompt 要求 20 个 supplied QA 精确地被 `qa_relations` 的 ID 并集或 `unassigned_qa_ids`
     覆盖；二者不能重叠；不相关/不适合 material atom 的 QA 放 unassigned。
   - validator 校验 QA 属于该作者、无重复且该并集等于全 20。这样会暴露 0006/0016，而不把它们
     自动设成 support 或 closure。
2. **声明 closure 是 request 的权威定义。**
   - 修改 `request_graph.compile_request()`：从 candidate request 的
     `per_atom_closures` 构造 `closures/forget`，不再调用 `_atom_closure()` 重新扩大 forget set。
   - validator 保持每个 target 恰有一个非空 closure、全部 ID 属于作者；额外要求 closure ID
     必须在该 target atom 的 material `qa_relations` 中，但不要要求相等。
3. **保留 protected 指标，增加来源可追溯与最终非空 gate。**
   - 在 compiler provenance 加 `protected_pool_provenance`，分为
     `candidate_explicit`、`residual_atom_support`、`merged`（以及实际 ID）。
   - candidate `protected_train_qa_ids` 继续可为空；但最终编译 request 的 protected pool 必须
     非空、与 forget 不重叠、是源 QA，失败则阻止 request bank compile。
   - `protected_eval_qa_ids` 继续等于编译后的 protected pool，保留 `protected_cvar20`。
4. **报告与测试同步。**
   - `reporting.py` 改为 v11 字段，不再输出不存在的 aliases/evidence/confidence。
   - 加单测：unassigned coverage、closure 不会被 compiler 扩大、explicit/residual protected
     provenance、最终空 protected pool 被拒绝、review report 的 v11 内容。
   - 运行：

```bash
PYTHONPATH=src /home/zkzhang/miniconda3/envs/openunlearning-official-py311/bin/python \
  -m unittest discover -s tests -p 'test_atomic_tofu*.py' -q
PYTHONPATH=src /home/zkzhang/miniconda3/envs/openunlearning-official-py311/bin/python \
  -m atomic_tofu.pipeline --stage dry-run --release-root /tmp/atomic-tofu-v10-dry-run
```

5. 仅在上述改动测试通过、提交且用户确认后，重新跑 000 和 007 canary，人工检查报告，再跑
   calibration-8。不要现在直接跑 180 位。

推荐的提交方式（只在修改确已完成后）：

```bash
git add src/atomic_tofu/annotation.py src/atomic_tofu/contracts.py \
  src/atomic_tofu/request_graph.py src/atomic_tofu/reporting.py \
  src/atomic_tofu/pipeline.py tests/<实际修改的测试文件>
git diff --cached --name-status
git commit --only src/atomic_tofu/annotation.py src/atomic_tofu/contracts.py \
  src/atomic_tofu/request_graph.py src/atomic_tofu/reporting.py \
  src/atomic_tofu/pipeline.py tests/<实际修改的测试文件> \
  -m "fix: preserve Atomic-TOFU declared closures"
```

注意：不要照抄上述 `tests/<...>` 占位符；先用 `git status --short` 确认真实路径。若此次
handoff 文档已提交，后续 commit 不应重新包含它。

## 7. 修改完成且获准后，推荐的运行顺序

新终端完整环境（`OPENAI_API_KEY` 通过安全方式提供，不要写入仓库/报告）：

```bash
cd /home/zkzhang/unlearn/open-unlearning
source /home/zkzhang/miniconda3/bin/activate openunlearning-official-py311
export PYTHONPATH="$PWD/src"
export RELEASE_ROOT="$PWD/data/atomic_tofu/v2.0-rc1"
export OPENAI_BASE_URL='https://api.zhizengzeng.com/v1'
export ATOMIC_TOFU_ANNOTATION_MODEL='gpt-5-mini'
export ATOMIC_TOFU_MAX_CONCURRENCY=1
export ATOMIC_TOFU_TIMEOUT=300
export ATOMIC_TOFU_MAX_VALIDATION_REPAIRS=1
```

在 prompt/schema 修改导致 generation hash 改变后，先跑作者 000：

```bash
python -m atomic_tofu.pipeline \
  --stage run-annotation --provider openai --resume \
  --unit-ids-file "$RELEASE_ROOT/api/annotation/calibration/canary_1_id.txt" \
  --release-root "$RELEASE_ROOT"
python -m atomic_tofu.pipeline --stage validate-annotation --release-root "$RELEASE_ROOT"
python -m atomic_tofu.pipeline --stage annotation-review-report \
  --author-id tofu_author_000 --release-root "$RELEASE_ROOT"
python -m json.tool "$RELEASE_ROOT/api/annotation/run_report.json"
```

然后单独生成 007 的报告/候选（可用临时 one-line ID 文件，文件本身也要审计）或直接进入
calibration-8。人工审查通过后：

```bash
python -m atomic_tofu.pipeline \
  --stage run-annotation --provider openai --resume \
  --unit-ids-file "$RELEASE_ROOT/api/annotation/calibration/calibration_8_ids.txt" \
  --release-root "$RELEASE_ROOT"
python -m atomic_tofu.pipeline --stage validate-annotation --release-root "$RELEASE_ROOT"
for i in $(seq -w 0 7); do
  python -m atomic_tofu.pipeline --stage annotation-review-report \
    --author-id "tofu_author_0$i" --release-root "$RELEASE_ROOT"
done
```

观察结构错误时，先看：

```bash
python -m json.tool "$RELEASE_ROOT/audit/annotation_candidate_report.json"
find "$RELEASE_ROOT/api/annotation/repair_attempts" -maxdepth 1 -type f -print 2>/dev/null || true
find "$RELEASE_ROOT/api/annotation/attempts" -maxdepth 1 -type f -print
```

生成 200 人前的策略是：prompt 保持稳定、先完成全量候选；然后执行独立 AI 复核与人工复审。
不要在全量过程中根据个别语义样本频繁改 prompt，否则不能比较 generation，也会增加费用。

## 8. 关键审计材料

- 已人工阅读的候选报告：
  `data/atomic_tofu/v2.0-rc1/audit/annotation_review_reports/tofu_author_{000..007,011}.{md,json}`。
- 早期 calibration 的语义风险归纳：
  `data/atomic_tofu/v2.0-rc1/audit/style_genre_evidence_audit_calibration8.md`。
- 当前结构验证：`audit/annotation_candidate_report.json`。
- 当前/历史 API 输出：`api/annotation/candidate_outputs.jsonl` 与 `attempts/`。
- 官方/实现总览：`docs/bridge_r_v10/README.md`。
- 原始 BRIDGE 经验：`docs/BRIDGE/BRIDGE_initial_experiment_plan.md`、
  `BRIDGE_paper_plan.md`、`BRIDGE_experiment_status_2026-07-10.md`、
  `BRIDGE_handoff_7.9.md`、`results/unlearn_summary.{csv,md}`。

## 9. 接手会话应先汇报什么

接手后不要直接修改或调用 API。先用只读检查汇报：

1. 当前 `git status --short`，特别说明预存脏改动未触碰；
2. 这个 handoff 所述 commit 是否都存在，当前 `candidate_outputs` 的作者数和 generation 分布；
3. 当前 `annotation_candidate_report.json` 与 `run_report.json` 是否仍是历史文件；
4. 对第 6 节四项最小改动的精确文件/测试影响，等待用户授权实施。
