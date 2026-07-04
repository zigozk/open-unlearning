#!/bin/bash
#SBATCH -J npo_l2_submit
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Submit once with:
#   sbatch sbatch/bridge/bridge_baseline_npo_llama2_splits.sh
#
# This submits Llama-2-7B-chat NPO baseline train+eval jobs for
# forget01/05/10, then submits a dependent summary job.

split_for_index() {
  case "$1" in
    0)
      FORGET_SPLIT="${FORGET01_SPLIT:-forget01}"
      HOLDOUT_SPLIT="${HOLDOUT01_SPLIT:-holdout01}"
      RETAIN_SPLIT="${RETAIN01_SPLIT:-retain99}"
      NPO_GAMMA="${FORGET01_NPO_GAMMA:-0.1375}"
      NPO_BETA="${FORGET01_NPO_BETA:-2.5}"
      ;;
    1)
      FORGET_SPLIT="${FORGET05_SPLIT:-forget05}"
      HOLDOUT_SPLIT="${HOLDOUT05_SPLIT:-holdout05}"
      RETAIN_SPLIT="${RETAIN05_SPLIT:-retain95}"
      NPO_GAMMA="${FORGET05_NPO_GAMMA:-0.1375}"
      NPO_BETA="${FORGET05_NPO_BETA:-2.5}"
      ;;
    2)
      FORGET_SPLIT="${FORGET10_SPLIT:-forget10}"
      HOLDOUT_SPLIT="${HOLDOUT10_SPLIT:-holdout10}"
      RETAIN_SPLIT="${RETAIN10_SPLIT:-retain90}"
      NPO_GAMMA="${FORGET10_NPO_GAMMA:-0.125}"
      NPO_BETA="${FORGET10_NPO_BETA:-4.5}"
      ;;
    *)
      echo "[error] Unknown split index: $1" >&2
      exit 1
      ;;
  esac
}

setup_runtime() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning-rerun}"
  cd "${ROOT_DIR}"
  mkdir -p logs

  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-unlearning-new}"

  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_HUB_CACHE="${HF_HOME}/hub"
  export HF_MODULES_CACHE="${HF_HOME}/modules"
  export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
}

submit_pipeline() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning-rerun}"
  cd "${ROOT_DIR}"
  mkdir -p logs results/bridge_reports

  local script_path array_job_id summary_job_id
  script_path="$(readlink -f "$0")"

  echo "===== LLAMA2 NPO BASELINE SUBMIT ====="
  echo "ROOT_DIR=${ROOT_DIR}"
  echo "MODEL_CONFIG=${MODEL_CONFIG:-Llama-2-7b-chat-hf}"
  echo "MODEL_PATH=${MODEL_PATH:-/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full}"
  echo "ARRAY=forget01,forget05,forget10"
  echo "GRES=${GRES:-gpu:nvidia_h100_80gb_hbm3:1}"
  echo "MEM=${MEM:-96G}"
  echo "TIME_LIMIT=${TIME_LIMIT:-3:00:00}"

  array_job_id="$(
    sbatch --parsable \
      -J "${ARRAY_JOB_NAME:-npo_l2_splits}" \
      -p "${PARTITION:-compute}" \
      -N 1 \
      --gres="${GRES:-gpu:nvidia_h100_80gb_hbm3:1}" \
      --cpus-per-task="${CPUS_PER_TASK:-8}" \
      --mem="${MEM:-96G}" \
      -t "${TIME_LIMIT:-3:00:00}" \
      --array="0-2%${MAX_PARALLEL:-1}" \
      -o "logs/npo_l2_splits-%A_%a.out" \
      -e "logs/npo_l2_splits-%A_%a.err" \
      --export=ALL,PIPELINE_STAGE=array \
      "${script_path}"
  )"

  summary_job_id="$(
    sbatch --parsable \
      -J "${SUMMARY_JOB_NAME:-npo_l2_summary}" \
      -p "${PARTITION:-compute}" \
      -N 1 \
      --cpus-per-task="${SUMMARY_CPUS_PER_TASK:-2}" \
      --mem="${SUMMARY_MEM:-8G}" \
      -t "${SUMMARY_TIME_LIMIT:-00:30:00}" \
      --dependency="afterany:${array_job_id}" \
      -o "logs/npo_l2_summary-%j.out" \
      -e "logs/npo_l2_summary-%j.err" \
      --export=ALL,PIPELINE_STAGE=summary \
      "${script_path}"
  )"

  echo "Array job:   ${array_job_id}"
  echo "Summary job: ${summary_job_id}"
}

run_one() {
  setup_runtime
  split_for_index "${SLURM_ARRAY_TASK_ID:-0}"

  echo "===== RUNTIME PREFLIGHT ====="
  python - <<'PY'
import inspect
import transformers
from transformers import Trainer

has_processing_class = "processing_class" in inspect.signature(Trainer.__init__).parameters
print(f"transformers={transformers.__version__}")
print(f"Trainer has processing_class={has_processing_class}")
if not has_processing_class:
    raise SystemExit(
        "This rerun branch passes processing_class to Trainer. "
        "Install this repo's requirements in the active CONDA_ENV."
    )
PY

  MODEL_CONFIG="${MODEL_CONFIG:-Llama-2-7b-chat-hf}"
  MODEL_PATH="${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL_CONFIG}_full}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
  TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-results/npo_l2_official_params}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/npo_l2_official_params_eval}"
  RUN_TAG="${RUN_TAG:-official_params}"
  SEED="${SEED:-0}"

  LEARNING_RATE="${LEARNING_RATE:-1e-5}"
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
  TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
  EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
  GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
  OVERWRITE_EVAL="${OVERWRITE_EVAL:-true}"
  SKIP_EXISTING="${SKIP_EXISTING:-1}"

  task_beta="${NPO_BETA//./p}"
  task_gamma="${NPO_GAMMA//./p}"
  TASK_NAME="${TASK_NAME:-tofu_${MODEL_CONFIG}_${FORGET_SPLIT}_NPO_lr${LEARNING_RATE}_beta${task_beta}_gamma${task_gamma}_epoch${NUM_TRAIN_EPOCHS}_${RUN_TAG}_seed${SEED}}"
  TRAIN_DIR="${TRAIN_OUTPUT_ROOT}/${TASK_NAME}"
  EVAL_DIR="${EVAL_OUTPUT_ROOT}/${TASK_NAME}_tofu_eval"
  RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-saves/eval/tofu_${MODEL_CONFIG}_${RETAIN_SPLIT}/TOFU_EVAL.json}"

  mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"

  echo "===== LLAMA2 NPO BASELINE TASK ====="
  echo "HOSTNAME=$(hostname)"
  echo "JOB_ID=${SLURM_JOB_ID:-manual}"
  echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-manual}"
  echo "MODEL_CONFIG=${MODEL_CONFIG}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
  echo "FORGET_SPLIT=${FORGET_SPLIT}"
  echo "HOLDOUT_SPLIT=${HOLDOUT_SPLIT}"
  echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
  echo "NPO_GAMMA=${NPO_GAMMA}"
  echo "NPO_BETA=${NPO_BETA}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
  echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
  echo "GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS}"
  echo "TRAIN_DIR=${TRAIN_DIR}"
  echo "EVAL_DIR=${EVAL_DIR}"
  echo "RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH}"
  nvidia-smi || true

  if [ ! -d "${MODEL_PATH}" ]; then
    echo "[error] Missing MODEL_PATH directory: ${MODEL_PATH}" >&2
    exit 1
  fi
  if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "[error] Missing TOKENIZER_PATH directory: ${TOKENIZER_PATH}" >&2
    exit 1
  fi
  if [ ! -f "${RETAIN_LOGS_PATH}" ]; then
    echo "[error] Missing retain reference log: ${RETAIN_LOGS_PATH}" >&2
    echo "Run: python setup_data.py --eval_logs" >&2
    exit 1
  fi
  if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${EVAL_DIR}/TOFU_SUMMARY.json" ]; then
    echo "[skip] Found existing eval summary: ${EVAL_DIR}/TOFU_SUMMARY.json"
    exit 0
  fi

  echo "===== TRAIN ====="
  python src/train.py \
    --config-name=unlearn.yaml \
    experiment=unlearn/tofu/default \
    trainer=NPO \
    "model=${MODEL_CONFIG}" \
    "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}" \
    "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}" \
    "forget_split=${FORGET_SPLIT}" \
    "retain_split=${RETAIN_SPLIT}" \
    "holdout_split=${HOLDOUT_SPLIT}" \
    "task_name=${TASK_NAME}" \
    "paths.output_dir=${TRAIN_DIR}" \
    "trainer.method_args.gamma=${NPO_GAMMA}" \
    "trainer.method_args.beta=${NPO_BETA}" \
    "trainer.method_args.alpha=1.0" \
    "trainer.method_args.retain_loss_type=NLL" \
    "trainer.args.seed=${SEED}" \
    "trainer.args.learning_rate=${LEARNING_RATE}" \
    "trainer.args.num_train_epochs=${NUM_TRAIN_EPOCHS}" \
    "trainer.args.per_device_train_batch_size=${TRAIN_BATCH_SIZE}" \
    "trainer.args.gradient_accumulation_steps=${GRAD_ACCUM_STEPS}" \
    "trainer.args.gradient_checkpointing=${GRADIENT_CHECKPOINTING}" \
    "trainer.args.do_eval=false" \
    "trainer.args.eval_on_start=false"

  echo "===== EVAL ====="
  python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    "model=${MODEL_CONFIG}" \
    "model.model_args.pretrained_model_name_or_path=${TRAIN_DIR}" \
    "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}" \
    "forget_split=${FORGET_SPLIT}" \
    "holdout_split=${HOLDOUT_SPLIT}" \
    "task_name=${TASK_NAME}_tofu_eval" \
    "paths.output_dir=${EVAL_DIR}" \
    "eval.tofu.output_dir=${EVAL_DIR}" \
    "eval.tofu.overwrite=${OVERWRITE_EVAL}" \
    "eval.tofu.batch_size=${EVAL_BATCH_SIZE}" \
    "retain_logs_path=${RETAIN_LOGS_PATH}" \
    "eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH}"

  echo "===== DONE: ${TASK_NAME} ====="
}

run_summary() {
  setup_runtime
  TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-results/npo_l2_official_params}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/npo_l2_official_params_eval}"
  REPORT_ROOT="${REPORT_ROOT:-results/bridge_reports}"
  REPORT_PREFIX="${REPORT_PREFIX:-llama2_7b_npo_official_params_splits}"
  mkdir -p "${REPORT_ROOT}"

  csv_path="${REPORT_ROOT}/${REPORT_PREFIX}_summary.csv"
  md_path="${REPORT_ROOT}/${REPORT_PREFIX}_report.md"

  python experiments/bridge/summarize_bridge_initial.py \
    --train-root "${TRAIN_OUTPUT_ROOT}" \
    --eval-root "${EVAL_OUTPUT_ROOT}" \
    --output-csv "${csv_path}" \
    --output-md "${md_path}"

  echo "===== LLAMA2 NPO BASELINE SUMMARY ====="
  echo "CSV: ${csv_path}"
  echo "Report: ${md_path}"
  cat "${md_path}"
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
