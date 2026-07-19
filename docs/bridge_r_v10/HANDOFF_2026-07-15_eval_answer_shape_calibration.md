# Atomic-TOFU / BRIDGE-R v10 接手文档：Eval-A answer-shape calibration（2026-07-15）

本文件用于把当前工作交给没有前序对话上下文的新 Codex。它记录的是当前工作树与
候选状态，不把 API candidate、mock、计划或未验证观察写成 gold/实验结果。

## 0. 接手时必须先完整阅读的文件

新 Codex 不得跳读，必须先完整阅读以下文件：

1. docs/bridge_r_v10/BRIDGE_R_paper_plan_v10_full_corpus_2026-07-12.md
2. docs/bridge_r_v10/BRIDGE_R_experiment_plan_v10_full_corpus_remote_2026-07-12.md
3. docs/bridge_r_v10/Atomic_TOFU_dataset_construction_plan_v3_full_corpus_api_2026-07-12.md
4. docs/bridge_r_v10/HANDOFF_2026-07-13_atomic_tofu.md
5. docs/bridge_r_v10/README.md
6. /home/zkzhang/unlearn/Atomic_TOFU_eval_answer_shape_calibration_plan_2026-07-14.md
7. 本文件

前三份方案是研究目标、实验设计和数据契约的最高权威。若实现历史状态与方案目标冲突，
必须明确报告冲突，不得自行改写研究目标。2026-07-14 的 answer-shape 方案只处理
Pipeline B 的 Eval-A candidate generation，不改变 source QA、Pipeline A annotation、
request graph 或 balanced bundle 的目标。

## 1. 工作范围与安全边界

- 唯一项目目录：/home/zkzhang/unlearn/open-unlearning。
- 不重建环境，不修改或删除原始 BRIDGE。
- 禁止 git clean、git reset、git checkout；不得恢复、删除、暂存或提交预存脏改动。
- 当前工作树本来就有大量用户预存改动。接手后先运行 git status --short，只读记录，
  不要尝试“清理”它。
- 没有用户明确授权，不要写入、提交、调用付费 API 或运行全量 API generation。
- Atomic-TOFU API 产物始终是 candidate-only；不能把 candidate 当 gold，也不能把候选
  的自动 validator 通过写成语义正确或实验结果。
- 真实训练和正式 unlearning 实验尚未在本任务中启动；不要声称任何新的训练结果。
- 用户已经明确选择 Eval 使用 gpt-5-mini 浮动别名。代码现在允许该别名，但这覆盖了
  原 answer-shape 方案的 fixed-snapshot freeze 条件。任何报告必须明确写出
  model_lock: floating_alias，不得声称完全可复现的模型冻结。

## 2. 工作树基线

接手时的 Git 基线为：

    branch: bridge-initial-plan
    HEAD:   e879904 docs: add Atomic-TOFU v10 handoff

当前脏改动包括（仅作识别，不得恢复）：

- src/atomic_tofu/ 中 annotation、request graph、bundles、eval extension、pipeline、
  providers 等实现修改；
- tests/ 中 Atomic-TOFU contract/request-graph/pipeline/eval 测试修改或新增；
- docs/bridge_r_v10/README.md 修改，以及若干方案/历史文档未跟踪文件；
- configs/ 的预存删除、results/ 的预存结果/模型文件、sbatch/ 的预存变动；
- src/atomic_tofu/balanced_bundles.py、bundle_compiler.py、feasibility.py、
  supplemental.py 等新增实现。

这些改动属于用户当前工作树。新 Codex 只能在用户授权的目标文件上做最小修改；不能
因为 git status 很大而进行清理或提交。

## 3. Pipeline A 当前状态

### 3.1 Annotation

- source 是完整 TOFU：4,000 QA、200 authors、每位 20 QA。
- audit/annotation_candidate_report.json 当前为 candidate_schema_passed，并保持
  candidate_only: true。
- api/annotation/run_report.json 当前记录 total_completed: 200、total_errors: 0；
  本次 run 的 completed 为 180。模型字段为 gpt-5-mini，不能把这写成 gold 质量证明。
- 作者本人姓名只作为 scope identifier 保护；父亲、母亲及其他家庭成员姓名可以作为
  forget target。support|leak|closure 一旦标注，必须自动进入 forget closure。
- protected pool 的最终契约是：实际训练、same-author protected eval、LocalMean/Tail、
  CVaR20 都使用同作者 source QA 减去该 request 的完整 forget closure；全局 retain train
  使用 full corpus 减去 bundle forget union。bundle 中另一 request 的目标不会额外剔除
  当前 request 的 protected pool。

### 3.2 Request graph

当前 audit/request_graph_report.json：

    status                              request_bank_compiled_with_supplemental
    request_count_before_supplemental   1109
    supplemental_accepted_count         65
    request_count_after_supplemental    1174
    eligible_request_count              1061
    ineligible_reserved_count            113

113 个 reserved request 碰到 official retain anchor，不能进入主实验候选池。1174 是当前
request bank 总数；主实验候选池是 1061，不要再使用早期的 996 统计替代当前报告。

### 3.3 Pure cell feasibility 与 bundles

当前 audit/cell_feasibility_report.json 为 completed：

- 所有 9 个 cell 的 pure exact-40 均有 3 个 distinct seed solution；
- pure exact-200 可行：C1-E-Low、C1-E-Medium、C2-E-High、C2-E-Medium、
  C3-E-Medium；
- pure exact-200 不可行（eligible closure union 不足 200）：C1-E-High、C2-E-Low、
  C3-E-High、C3-E-Low。

audit/bundle_compile_report.json 的 pure_entanglement_diagnostic target 为 40，
9 个 cell 均有 3 个 seed 的 pure bundle，路径为：

    data/atomic_tofu/v2.0-rc1/bundles/pure/<CELL>/seed_{0,1,2}.json

### 3.4 Balanced official-scale bundles

bundle_specs/balanced_v1.json 和 audit/balanced_bundle_feasibility_report.json 已完成。
当前有 Single/Multi × {40,200} × {seed 0,1,2} 共 12 个 balanced bundle 及 manifest：

    data/atomic_tofu/v2.0-rc1/bundles/balanced/<Single|Multi>/<40|200>/seed_<0|1|2>.json
    data/atomic_tofu/v2.0-rc1/bundles/balanced/<Single|Multi>/<40|200>/seed_<0|1|2>.manifest.json

这 12 个 balanced bundle 的 feasibility report 均为 completed/feasible，目标 QA 数为
40 或 200。balanced 通过跨 cell 的 owner-attribution quota 达到 official-scale 目标；
上面的 4 个 pure exact-200 不足 cell 仍应记录为 pure_200_unavailable，不能伪称 pure-200
已完成。

Pipeline A 的 request graph、pure-40 diagnostic bundles 和 balanced bundle 结构已经生成，
但正式 freeze 仍须结合 Eval-A 校准、最终审计和方案规定的人工/语义 gate；不要因为 bundle
文件存在就开始训练或写论文结果。

## 4. Eval-A 当前代码契约

相关实现：

- src/atomic_tofu/eval_extension.py
  - 强化 EVAL_SYSTEM_PROMPT 与 repair prompt；
  - 每个 perturbation 必须保留 source answer shape、主语、非目标上下文和未修改槽位；
  - 非裸 source answer 不允许只输出裸姓名/地点/日期/职业；
  - deterministic validator 检查复制、正确值/alias 泄露、重复、JSON/list/multiline、
    meta-commentary、answer-shape 和长度边界；
  - prepare_eval_calibration() 生成 60 条 generated 分层 canary 和 20 条 hidden anchor；
  - validate_eval_answer_shape_calibration() 计算 mechanical、长度 ratio、anchor 对照及
    model/generation consistency。
- src/atomic_tofu/pipeline.py
  - 新 stages：prepare-eval-calibration、run-eval-generated-calibration、
    run-eval-anchor-calibration、validate-eval-calibration；
  - --resume 按 generation hash/input hash 保留已成功 candidate，只重试 pending/error；
  - Eval 与两个 calibration stage 共用 schema/prompt/repair hash。
- src/atomic_tofu/providers.py
  - ATOMIC_TOFU_EVAL_MODEL=gpt-5-mini alias 已被允许；
  - 不再把 alias 当 fixed snapshot；报告侧记录 model_lock: floating_alias 和警告。
- docs/bridge_r_v10/README.md
  - 已记录 alias 是本次用户批准的偏离；不能声称 fixed-snapshot freeze。
- tests/test_atomic_tofu_eval_extension.py 与相关 pipeline tests
  - 已覆盖裸值拒绝、完整 answer-shape、multi-slot、repair loop；此前相关单测通过。

## 5. Eval-A 最新磁盘状态（只读事实）

### 5.1 Historical 20-row canary

    api/eval_extension/run_report.json
      stage: eval_extension
      completed: 20 / total_completed: 20
      errors: 0
      model: gpt-5-mini
      generation_sha256: d982200461eccabbd6edf5593a5fb19dfff93a67c9c11738b4809ccf242ce501

    audit/eval_extension_canary_report.json
      status: canary_semantics_passed
      candidate_only: true

这 20 条是历史诊断候选，不能替代新的 60+20 answer-shape calibration，也不能直接合并
进最终 Eval-A freeze。

### 5.2 新的 generated 60 calibration

manifest 已存在：

    api/eval_extension/calibration/generated/input_units.jsonl
    api/eval_extension/calibration/generated/generated_60_manifest.json
    api/eval_extension/calibration/generated/generated_60_ids.txt

generated manifest seed 为 20260714；anchor manifest 使用 seed + 1 = 20260715，这是
代码设计，不要把二者误认为抽样未记录。

最新 api/eval_extension/calibration/generated/run_report.json：

    stage: eval_generated_calibration
    all_available_units: 60
    completed: 47
    total_completed: 47
    errors: 13
    api_call_count_this_run: 50
    model: gpt-5-mini
    generation_sha256: 3e5ed93eb45c5b4ffe5260810421df6835bf6973d3b88c56888e88c7eea43dd5
    api_usage_this_run: input 30032, output 104257, total 134289
    candidate_pool_usage: input 28262, output 97019, total 125281
    estimated_cost: null (cost rate not configured)

只读重算得到：

    candidate outputs: 47/60
    post-cache deterministic validator failures: 0/47
    perturbation/source token ratio（成功子集）: median 1.000, P10 0.962, P90 1.153

上述 ratio 只描述已经写入的 47 条，不能据此宣布完整 60 条通过。13 个 error 均是在
repair 后仍失败：12 个是 literal correct value/alias 泄露，1 个是 list/multiline 结构。
错误明细在：

    api/eval_extension/calibration/generated/error_queue.jsonl

### 5.3 Hidden official anchor calibration

anchor manifest 已存在：

    api/eval_extension/anchor_calibration/input_units.jsonl
    api/eval_extension/anchor_calibration/anchor_20_manifest.json
    api/eval_extension/anchor_calibration/anchor_20_ids.txt

但当前没有：

    api/eval_extension/anchor_calibration/run_report.json
    api/eval_extension/anchor_calibration/candidate_outputs.jsonl
    api/eval_extension/anchor_calibration/error_queue.jsonl

因此 hidden anchor 实际尚未完成。audit/eval_anchor_calibration_report.json 的时间早于
最新 generated run，当前 calibration_failed 主要来自 missing candidates；它不是最新的
完整质量结论。报告仍必须重新生成，不能手工修改。

### 5.4 当前结论

Eval-A answer-shape calibration **未通过**：

1. generated 只有 47/60，未达到 100% mechanical gate；
2. 13 条仍在 error queue；
3. anchor 0/20；
4. 人工语义审查和 source fidelity 尚未完成；
5. 没有运行 4,000-row 全量 Eval-A generation。

因此现在不能去掉 --unit-ids-file 运行全量 run-eval，不能冻结 Eval-A，也不能开始把
这些 candidate 当作正式评测字段。

## 6. 获得用户授权后的下一步

不需要重新 prepare manifest。用户授权 API 后，在 hpc 环境中：

    cd /home/zkzhang/unlearn/open-unlearning
    source /home/zkzhang/miniconda3/bin/activate openunlearning-official-py311
    export PYTHONPATH="$PWD/src"
    export RELEASE_ROOT="$PWD/data/atomic_tofu/v2.0-rc1"
    export ATOMIC_TOFU_EVAL_MODEL='gpt-5-mini'
    export ATOMIC_TOFU_MAX_VALIDATION_REPAIRS=1

    python -m atomic_tofu.pipeline \
      --stage run-eval-generated-calibration \
      --provider openai \
      --resume \
      --release-root "$RELEASE_ROOT"

    python -m atomic_tofu.pipeline \
      --stage run-eval-anchor-calibration \
      --provider openai \
      --resume \
      --release-root "$RELEASE_ROOT"

    python -m atomic_tofu.pipeline \
      --stage validate-eval-calibration \
      --release-root "$RELEASE_ROOT"

    python -m json.tool \
      "$RELEASE_ROOT/audit/eval_anchor_calibration_report.json"

--resume 应保留当前 47 条成功 candidate，并重试 pending/error；不要直接删除
candidate_outputs.jsonl 或 error queue。若仍失败，先审查 error queue 与 repair_attempts/，
再决定是否需要修改 prompt/validator；修改后 generation hash 会变化，旧 candidate 应留在
attempts/archive，不能覆盖成假装同一代的结果。

只有以下条件全部满足，才可考虑 full Eval-A：

- generated 60/60 mechanical pass；
- hidden anchor 20/20 mechanical pass；
- generated ratio median [0.80,1.35]、P10 >=0.65；
- anchor-vs-official ratio median [0.80,1.25]、P10/P90 不坍缩；
- source fidelity 和人工语义 gate 完成；
- prompt/schema/validator/repair generation 一致；
- 由于本次使用 alias，报告必须保留 model_lock: floating_alias，不可写成 fixed snapshot。

即使上述 calibration 通过，也应先让用户审查 60+20 的 review packet，再运行无
--unit-ids-file 的 full Eval-A。不要以 mechanical pass 代替人工语义审查。

## 7. 新 Codex 接手时的第一轮动作

新 Codex 的第一轮只读动作：

    cd /home/zkzhang/unlearn/open-unlearning
    git status --short
    git log -1 --oneline
    python -m json.tool data/atomic_tofu/v2.0-rc1/api/eval_extension/calibration/generated/run_report.json
    python -m json.tool data/atomic_tofu/v2.0-rc1/audit/eval_anchor_calibration_report.json

然后按第 0 节完整读方案文件，按第 3–5 节核对报告。第一轮禁止写文件、提交、API 调用，
先向用户汇报：当前脏改动、Pipeline A 状态、generated/anchor 进度、alias 偏离及是否获得
下一步授权。

## 8. 可直接发送给新 Codex 的接手 prompt

    你在服务器 hpc 工作，唯一项目目录是：
    /home/zkzhang/unlearn/open-unlearning

    请先完整阅读（不能跳读）以下文件：
    1. docs/bridge_r_v10/BRIDGE_R_paper_plan_v10_full_corpus_2026-07-12.md
    2. docs/bridge_r_v10/BRIDGE_R_experiment_plan_v10_full_corpus_remote_2026-07-12.md
    3. docs/bridge_r_v10/Atomic_TOFU_dataset_construction_plan_v3_full_corpus_api_2026-07-12.md
    4. docs/bridge_r_v10/HANDOFF_2026-07-13_atomic_tofu.md
    5. docs/bridge_r_v10/README.md
    6. /home/zkzhang/unlearn/Atomic_TOFU_eval_answer_shape_calibration_plan_2026-07-14.md
    7. docs/bridge_r_v10/HANDOFF_2026-07-15_eval_answer_shape_calibration.md

    前三份方案是研究目标、实验设计和数据契约的最高权威；handoff 只记录实现状态。若冲突，
    明确报告冲突，不得自行改写研究目标。

    严格约束：
    - 不重建环境，不修改或删除原始 BRIDGE；
    - 不使用 git clean、git reset、git checkout；
    - 当前工作树有大量预存脏改动，不得恢复、删除、暂存或提交它们；
    - Atomic-TOFU API 输出只是 candidate，不是 gold；不能把 candidate、mock、计划或未验证
      观察写成实验结果；
    - 没有我明确授权前，只读核查，不写文件、不提交、不调用付费 API；
    - 本次 Eval 明确使用 ATOMIC_TOFU_EVAL_MODEL=gpt-5-mini alias。可以运行，但必须在报告中
      标明 model_lock=floating_alias，不得声称 fixed-snapshot freeze 或完全可复现。

    请先只读核查并汇报：
    1. git status --short、当前 branch/HEAD，以及预存脏改动类别；
    2. Pipeline A 的 annotation、request graph、pure feasibility、pure bundles、balanced
      Single/Multi 40/200 三 seeds 的当前状态；
    3. generated calibration 当前 60 条中成功/失败数量、run_report、error_queue 和机械质量；
    4. hidden anchor 是否实际运行，综合 calibration report 是否过期；
    5. 当前是否允许运行 full Eval-A。没有用户授权不要运行 API。

    已知基线：request graph 当前 1174 总 request、1061 eligible、113 reserved；pure exact-200
    在 C1-E-High、C2-E-Low、C3-E-High、C3-E-Low 不可行；balanced Single/Multi × 40/200 ×
    3 seeds 文件已经生成。最新 generated calibration 是 47/60 成功、13 条 error；anchor
    尚无 candidate_outputs。请以磁盘文件核对这些数字，不要盲信本 prompt。
