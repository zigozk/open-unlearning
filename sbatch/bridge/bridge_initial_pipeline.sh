#!/bin/bash
#SBATCH -J bridge_initial_submit
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Submit once with:
#   sbatch sbatch/bridge/bridge_initial_pipeline.sh
#
# The parent job submits a 6-task array for the first-round BRIDGE matrix and a
# dependent summary job. Override resources or paths through environment vars.

build_run_specs() {
  RUN_SPECS=(
    "npo:NPO:none:0.0:0.0"
    "npo_global_kl:BRIDGE_NPO:none:0.1:0.0"
    "bridge_uniform_dro:BRIDGE_NPO:uniform:0.1:0.3"
    "bridge_history_dro:BRIDGE_NPO:history:0.1:0.3"
    "bridge_refresh_gs_dro:BRIDGE_NPO:refresh_gs:0.1:0.3"
    "bridge_refresh_pi_dro:BRIDGE_NPO:refresh_pi:0.1:0.3"
  )

  if [ "${INCLUDE_OPTIONAL_ONLINE:-0}" = "1" ]; then
    RUN_SPECS+=(
      "bridge_online_gs_dro:BRIDGE_NPO:online_gs:0.1:0.3"
      "bridge_online_pi_dro:BRIDGE_NPO:online_pi:0.1:0.3"
    )
  fi
}

total_tasks() {
  build_run_specs
  echo "${#RUN_SPECS[@]}"
}

submit_pipeline() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  mkdir -p logs results/bridge_reports

  local script_path total max_parallel array_job_id summary_job_id
  script_path="$(readlink -f "$0")"
  total="$(total_tasks)"
  max_parallel="${MAX_PARALLEL:-4}"

  echo "===== BRIDGE INITIAL PIPELINE SUBMIT ====="
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "TOTAL_TASKS=${total}"
  echo "MAX_PARALLEL=${max_parallel}"
  echo "MODEL_CONFIG=${MODEL_CONFIG:-Llama-3.2-1B-Instruct}"
  echo "FORGET_SPLIT=${FORGET_SPLIT:-forget10}"
  echo "SEED=${SEED:-0}"

  array_job_id="$(
    sbatch --parsable \
      -J bridge_initial \
      -p "${PARTITION:-compute}" \
      -N 1 \
      --gres="${GRES:-gpu:nvidia_a100_80gb_pcie:1}" \
      --cpus-per-task="${CPUS_PER_TASK:-8}" \
      --mem="${MEM:-128G}" \
      -t "${TIME_LIMIT:-48:00:00}" \
      --array="0-$((total - 1))%${max_parallel}" \
      -o "logs/bridge_initial-%A_%a.out" \
      -e "logs/bridge_initial-%A_%a.err" \
      --export=ALL,PIPELINE_STAGE=array \
      "${script_path}"
  )"

  summary_job_id="$(
    sbatch --parsable \
      -J bridge_initial_summary \
      -p "${PARTITION:-compute}" \
      -N 1 \
      --cpus-per-task="${SUMMARY_CPUS_PER_TASK:-2}" \
      --mem="${SUMMARY_MEM:-16G}" \
      -t "${SUMMARY_TIME_LIMIT:-01:00:00}" \
      --dependency="afterany:${array_job_id}" \
      -o "logs/bridge_initial_summary-%j.out" \
      -e "logs/bridge_initial_summary-%j.err" \
      --export=ALL,PIPELINE_STAGE=summary \
      "${script_path}"
  )"

  echo "Array job:   ${array_job_id}"
  echo "Summary job: ${summary_job_id}"
}

setup_runtime() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  mkdir -p logs

  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-unlearning}"

  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
}

select_run() {
  build_run_specs
  local idx total spec
  idx="${SLURM_ARRAY_TASK_ID}"
  total="${#RUN_SPECS[@]}"
  if [ "${idx}" -ge "${total}" ]; then
    echo "[SKIP] SLURM_ARRAY_TASK_ID=${idx} >= total=${total}"
    exit 0
  fi
  spec="${RUN_SPECS[$idx]}"
  IFS=":" read -r METHOD TRAINER_NAME BRIDGE_PRIOR BRIDGE_LAMBDA_G BRIDGE_LAMBDA_B <<< "${spec}"
}

write_run_config() {
  export RUN_CONFIG_PATH="${TRAIN_DIR}/run_config.json"
  export TASK_NAME METHOD TRAINER_NAME BRIDGE_PRIOR BRIDGE_LAMBDA_G BRIDGE_LAMBDA_B
  export MODEL_CONFIG MODEL_PATH FORGET_SPLIT HOLDOUT_SPLIT RETAIN_SPLIT SEED
  export TRAIN_BATCH_SIZE GRAD_ACCUM_STEPS NUM_TRAIN_EPOCHS LEARNING_RATE
  export REFRESH_INTERVAL DRO_TEMPERATURE PRIOR_TEMPERATURE HISTORY_BETA PI_STEP_SIZE
  python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_CONFIG_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
keys = [
    "TASK_NAME",
    "METHOD",
    "TRAINER_NAME",
    "BRIDGE_PRIOR",
    "BRIDGE_LAMBDA_G",
    "BRIDGE_LAMBDA_B",
    "MODEL_CONFIG",
    "MODEL_PATH",
    "FORGET_SPLIT",
    "HOLDOUT_SPLIT",
    "RETAIN_SPLIT",
    "SEED",
    "TRAIN_BATCH_SIZE",
    "GRAD_ACCUM_STEPS",
    "NUM_TRAIN_EPOCHS",
    "LEARNING_RATE",
    "REFRESH_INTERVAL",
    "DRO_TEMPERATURE",
    "PRIOR_TEMPERATURE",
    "HISTORY_BETA",
    "PI_STEP_SIZE",
]
data = {key.lower(): os.environ.get(key) for key in keys}
data["model"] = data.pop("model_config")
data["bridge_prior"] = data.pop("bridge_prior")
data["bridge_lambda_g"] = data.pop("bridge_lambda_g")
data["bridge_lambda_b"] = data.pop("bridge_lambda_b")
data["seed"] = int(data["seed"])
path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
PY
}

run_one() {
  setup_runtime
  select_run

  MODEL_CONFIG="${MODEL_CONFIG:-Llama-3.2-1B-Instruct}"
  MODEL_PATH="${MODEL_PATH:-open-unlearning/tofu_${MODEL_CONFIG}_full}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-}"
  FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
  HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
  RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
  SEED="${SEED:-0}"
  RUN_TAG="${RUN_TAG:-initial}"
  TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-results/bridge_initial}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/bridge_initial_eval}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
  REFRESH_INTERVAL="${REFRESH_INTERVAL:-20}"
  DRO_TEMPERATURE="${DRO_TEMPERATURE:-1.0}"
  PRIOR_TEMPERATURE="${PRIOR_TEMPERATURE:-1.0}"
  HISTORY_BETA="${HISTORY_BETA:-0.9}"
  PI_STEP_SIZE="${PI_STEP_SIZE:-${LEARNING_RATE}}"
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
  OVERWRITE_EVAL="${OVERWRITE_EVAL:-true}"
  RUN_OFFICIAL_EVAL="${RUN_OFFICIAL_EVAL:-1}"
  SKIP_EXISTING="${SKIP_EXISTING:-1}"
  DELETE_CHECKPOINT_AFTER_EVAL="${DELETE_CHECKPOINT_AFTER_EVAL:-0}"
  RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-}"

  MODEL_TAG="${MODEL_CONFIG//[^A-Za-z0-9_]/_}"
  TASK_NAME="BRIDGE_TOFU_${FORGET_SPLIT}_${MODEL_TAG}_${METHOD}_seed${SEED}_${RUN_TAG}"
  TRAIN_DIR="${TRAIN_OUTPUT_ROOT}/${TASK_NAME}"
  EVAL_DIR="${EVAL_OUTPUT_ROOT}/${TASK_NAME}_tofu_eval"
  mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"

  echo "===== BRIDGE INITIAL TASK ====="
  echo "HOSTNAME=$(hostname)"
  echo "JOB_ID=${SLURM_JOB_ID:-manual}"
  echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-none}"
  echo "METHOD=${METHOD}"
  echo "TRAINER_NAME=${TRAINER_NAME}"
  echo "BRIDGE_PRIOR=${BRIDGE_PRIOR}"
  echo "BRIDGE_LAMBDA_G=${BRIDGE_LAMBDA_G}"
  echo "BRIDGE_LAMBDA_B=${BRIDGE_LAMBDA_B}"
  echo "MODEL_CONFIG=${MODEL_CONFIG}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "FORGET_SPLIT=${FORGET_SPLIT}"
  echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "EVAL_DIR=${EVAL_DIR}"
  nvidia-smi || true

  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${EVAL_DIR}/TOFU_SUMMARY.json" ]; then
    echo "[skip] Found existing eval summary: ${EVAL_DIR}/TOFU_SUMMARY.json"
    exit 0
  fi

  write_run_config

  if [ "${TRAINER_NAME}" = "NPO" ]; then
    train_cmd=(
      python src/train.py
      --config-name=unlearn.yaml
      experiment=unlearn/tofu/default
      trainer=NPO
      "task_name=${TASK_NAME}"
      "model=${MODEL_CONFIG}"
      "forget_split=${FORGET_SPLIT}"
      "holdout_split=${HOLDOUT_SPLIT}"
      "retain_split=${RETAIN_SPLIT}"
      "paths.output_dir=${TRAIN_DIR}"
      "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}"
      "trainer.args.seed=${SEED}"
      "trainer.args.per_device_train_batch_size=${TRAIN_BATCH_SIZE}"
      "trainer.args.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}"
      "trainer.args.num_train_epochs=${NUM_TRAIN_EPOCHS}"
      "trainer.args.learning_rate=${LEARNING_RATE}"
      "trainer.args.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
      "trainer.args.do_eval=false"
      "trainer.args.eval_on_start=false"
    )
  else
    train_cmd=(
      python src/train.py
      --config-name=unlearn.yaml
      experiment=unlearn/tofu/bridge
      trainer=BRIDGE_NPO
      "task_name=${TASK_NAME}"
      "model=${MODEL_CONFIG}"
      "forget_split=${FORGET_SPLIT}"
      "holdout_split=${HOLDOUT_SPLIT}"
      "retain_split=${RETAIN_SPLIT}"
      "paths.output_dir=${TRAIN_DIR}"
      "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}"
      "trainer.args.seed=${SEED}"
      "trainer.args.per_device_train_batch_size=${TRAIN_BATCH_SIZE}"
      "trainer.args.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}"
      "trainer.args.num_train_epochs=${NUM_TRAIN_EPOCHS}"
      "trainer.args.learning_rate=${LEARNING_RATE}"
      "trainer.args.gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
      "trainer.method_args.bridge_prior=${BRIDGE_PRIOR}"
      "trainer.method_args.bridge_lambda_g=${BRIDGE_LAMBDA_G}"
      "trainer.method_args.bridge_lambda_b=${BRIDGE_LAMBDA_B}"
      "trainer.method_args.bridge_dro_temperature=${DRO_TEMPERATURE}"
      "trainer.method_args.bridge_prior_temperature=${PRIOR_TEMPERATURE}"
      "trainer.method_args.bridge_history_beta=${HISTORY_BETA}"
      "trainer.method_args.bridge_refresh_interval=${REFRESH_INTERVAL}"
      "trainer.method_args.bridge_pi_step_size=${PI_STEP_SIZE}"
    )
  fi

  if [ -n "${TOKENIZER_PATH}" ]; then
    train_cmd+=("model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}")
  fi

  echo "===== TRAIN COMMAND ====="
  printf ' %q' "${train_cmd[@]}"
  echo
  "${train_cmd[@]}"

  if [ "${RUN_OFFICIAL_EVAL}" != "1" ]; then
    echo "[eval] RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL}; skipping eval"
    exit 0
  fi

  eval_tokenizer_path="${EVAL_TOKENIZER_PATH:-${TOKENIZER_PATH:-${TRAIN_DIR}}}"
  eval_cmd=(
    python src/eval.py
    --config-name=eval.yaml
    experiment=eval/tofu/default
    "model=${MODEL_CONFIG}"
    "model.model_args.pretrained_model_name_or_path=${TRAIN_DIR}"
    "model.tokenizer_args.pretrained_model_name_or_path=${eval_tokenizer_path}"
    "forget_split=${FORGET_SPLIT}"
    "holdout_split=${HOLDOUT_SPLIT}"
    "task_name=${TASK_NAME}_tofu_eval"
    "paths.output_dir=${EVAL_DIR}"
    "eval.tofu.output_dir=${EVAL_DIR}"
    "eval.tofu.overwrite=${OVERWRITE_EVAL}"
    "eval.tofu.batch_size=${EVAL_BATCH_SIZE}"
  )

  if [ -n "${RETAIN_LOGS_PATH}" ]; then
    eval_cmd+=("retain_logs_path=${RETAIN_LOGS_PATH}")
    eval_cmd+=("eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH}")
  fi

  echo "===== EVAL COMMAND ====="
  printf ' %q' "${eval_cmd[@]}"
  echo
  "${eval_cmd[@]}"

  if [ "${DELETE_CHECKPOINT_AFTER_EVAL}" = "1" ]; then
    case "${TRAIN_DIR}" in
      results/bridge_initial/*|*/results/bridge_initial/*)
        echo "[cleanup] deleting large checkpoint files after eval under: ${TRAIN_DIR}"
        find "${TRAIN_DIR}" -maxdepth 1 -type f \
          \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) -delete
        find "${TRAIN_DIR}" -maxdepth 1 -type d -name "checkpoint-*" -exec rm -rf -- {} +
        ;;
      *)
        echo "[cleanup] refusing to delete unexpected train dir: ${TRAIN_DIR}"
        ;;
    esac
  fi

  echo "===== DONE: ${TASK_NAME} ====="
}

run_summary() {
  setup_runtime
  TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-results/bridge_initial}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/bridge_initial_eval}"
  REPORT_ROOT="${REPORT_ROOT:-results/bridge_reports}"
  mkdir -p "${REPORT_ROOT}"

  python experiments/bridge/summarize_bridge_initial.py \
    --train-root "${TRAIN_OUTPUT_ROOT}" \
    --eval-root "${EVAL_OUTPUT_ROOT}" \
    --output-csv "${REPORT_ROOT}/bridge_initial_summary.csv" \
    --output-md "${REPORT_ROOT}/bridge_initial_report.md"
}

case "${1:-}" in
  --submit)
    submit_pipeline
    ;;
  --summary)
    PIPELINE_STAGE=summary
    run_summary
    ;;
  *)
    if [ "${PIPELINE_STAGE:-}" = "summary" ]; then
      run_summary
    elif [ "${PIPELINE_STAGE:-}" = "array" ] || [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      run_one
    else
      submit_pipeline
    fi
    ;;
esac
