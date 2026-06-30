#!/bin/bash
#SBATCH -J piper_pipe
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# One-file entry point for the expanded PIPER experiment.
#
# Recommended submission:
#   bash sbatch/piper/piper_expanded_with_eval_pipeline.sh --submit
#
# This submits:
#   1) an array job where each task runs intervention + official TOFU eval;
#   2) one dependent summary job that writes both aggregate CSVs.

SEEDS=(0 1 2)
BACKBONE_SPECS=(
  "NPO:1e-5:0.0"
  "GradAscent:1e-6:0.0"
  "GradDiff:1e-6:1.0"
  "SimNPO:1e-5:0.0"
)
LAMBDA_VALUES=(0.03 0.1 0.2 0.3 0.5 0.7 1.0)
METHODS=(global_retain_kl random_local_kl semantic_local_kl pi_local_kl)

build_method_specs() {
  METHOD_SPECS=("baseline:0.0")
  for method in "${METHODS[@]}"; do
    for lambda in "${LAMBDA_VALUES[@]}"; do
      METHOD_SPECS+=("${method}:${lambda}")
    done
  done
}

total_tasks() {
  build_method_specs
  echo $((${#METHOD_SPECS[@]} * ${#SEEDS[@]} * ${#BACKBONE_SPECS[@]}))
}

submit_pipeline() {
  local script_path total max_parallel array_output array_error summary_output summary_error
  script_path="$(readlink -f "$0")"
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  total="$(total_tasks)"
  max_parallel="${MAX_PARALLEL:-1}"
  mkdir -p logs

  echo "Submitting expanded PIPER intervention + official eval pipeline"
  echo "Total array tasks: ${total}"
  echo "Array max parallel: ${max_parallel}"

  array_output="logs/piper_pipe-%A_%a.out"
  array_error="logs/piper_pipe-%A_%a.err"
  summary_output="logs/piper_pipe_summary-%j.out"
  summary_error="logs/piper_pipe_summary-%j.err"

  local array_job_id summary_job_id
  array_job_id="$(
    sbatch --parsable \
      -J piper_pipe \
      -p "${PARTITION:-compute}" \
      -N 1 \
      --gres="${GRES:-gpu:nvidia_a100_80gb_pcie:1}" \
      --cpus-per-task="${CPUS_PER_TASK:-8}" \
      --mem="${MEM:-128G}" \
      -t "${TIME_LIMIT:-48:00:00}" \
      --array="0-$((total - 1))%${max_parallel}" \
      -o "${array_output}" \
      -e "${array_error}" \
      --export=ALL,PIPELINE_STAGE=array \
      "${script_path}"
  )"

  summary_job_id="$(
    sbatch --parsable \
      -J piper_pipe_summary \
      -p "${PARTITION:-compute}" \
      -N 1 \
      --cpus-per-task="${SUMMARY_CPUS_PER_TASK:-2}" \
      --mem="${SUMMARY_MEM:-16G}" \
      -t "${SUMMARY_TIME_LIMIT:-02:00:00}" \
      --dependency="afterany:${array_job_id}" \
      -o "${summary_output}" \
      -e "${summary_error}" \
      --export=ALL,PIPELINE_STAGE=summary \
      "${script_path}"
  )"

  echo "Array job:   ${array_job_id}"
  echo "Summary job: ${summary_job_id}"
  echo "Monitor:     squeue -u ${USER}"
}

setup_runtime() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  mkdir -p logs

  source ~/miniconda3/etc/profile.d/conda.sh
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

select_grid_item() {
  build_method_specs
  idx=${SLURM_ARRAY_TASK_ID}
  n_method=${#METHOD_SPECS[@]}
  n_seed=${#SEEDS[@]}
  n_backbone=${#BACKBONE_SPECS[@]}
  total=$((n_method * n_seed * n_backbone))
  if [ "${idx}" -ge "${total}" ]; then
    echo "[SKIP] SLURM_ARRAY_TASK_ID=${idx} >= total=${total}"
    exit 0
  fi

  method_idx=$((idx % n_method))
  seed_idx=$(((idx / n_method) % n_seed))
  backbone_idx=$((idx / (n_method * n_seed)))

  SEED="${SEEDS[$seed_idx]}"
  METHOD_SPEC="${METHOD_SPECS[$method_idx]}"
  METHOD="${METHOD_SPEC%%:*}"
  KL_LAMBDA="${METHOD_SPEC##*:}"
  BACKBONE_SPEC="${BACKBONE_SPECS[$backbone_idx]}"
  IFS=":" read -r BACKBONE DEFAULT_LR BACKBONE_RETAIN_ALPHA <<< "${BACKBONE_SPEC}"
}

run_intervention_and_eval() {
  setup_runtime
  select_grid_item

  MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
  MODEL_PATH="${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL_NAME}_full}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-/home/share/models/${MODEL_NAME}}"
  test -d "${MODEL_PATH}" || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 2; }
  if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "[WARN] TOKENIZER_PATH not found: ${TOKENIZER_PATH}; falling back to MODEL_PATH"
    TOKENIZER_PATH="${MODEL_PATH}"
  fi

  FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
  RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
  FORGET_BATCH_SIZE="${FORGET_BATCH_SIZE:-1}"
  PROBE_BATCHES="${PROBE_BATCHES:-64}"
  FORGET_SAMPLE_SIZE="${FORGET_SAMPLE_SIZE:-128}"
  RETAIN_CANDIDATE_SIZE="${RETAIN_CANDIDATE_SIZE:-500}"
  TOPK_FRAC="${TOPK_FRAC:-0.10}"
  TRAIN_STEPS="${TRAIN_STEPS:-80}"
  LEARNING_RATE="${LEARNING_RATE:-${DEFAULT_LR}}"
  PROBE_LEARNING_RATE="${PROBE_LEARNING_RATE:-1e-5}"
  LOCAL_KL_BATCH_SIZE="${LOCAL_KL_BATCH_SIZE:-1}"
  REF_LOGIT_BATCH_SIZE="${REF_LOGIT_BATCH_SIZE:-1}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-results/piper_intervention_expanded}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/piper_official_eval}"
  SKIP_EXISTING="${SKIP_EXISTING:-1}"
  SAVE_MODEL="${SAVE_MODEL:-1}"
  RUN_OFFICIAL_EVAL="${RUN_OFFICIAL_EVAL:-1}"
  DELETE_CHECKPOINT_AFTER_EVAL="${DELETE_CHECKPOINT_AFTER_EVAL:-1}"
  MODEL_CONFIG="${MODEL_CONFIG:-Llama-2-7b-chat-hf}"
  RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-}"
  OVERWRITE="${OVERWRITE:-true}"

  array_job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${FORGET_SPLIT}_${RETAIN_SPLIT}_${BACKBONE}_${METHOD}_lambda${KL_LAMBDA}_seed${SEED}_${array_job_id}_${SLURM_ARRAY_TASK_ID}"
  existing_summary=""
  for candidate in "${OUTPUT_ROOT}/${MODEL_NAME}_${FORGET_SPLIT}_${RETAIN_SPLIT}_${BACKBONE}_${METHOD}_lambda${KL_LAMBDA}_seed${SEED}_"*/summary.json; do
    if [ -f "${candidate}" ]; then
      existing_summary="${candidate}"
      break
    fi
  done

  echo "===== STATIC PIPER EXPANDED PIPELINE TASK ====="
  echo "HOSTNAME=$(hostname)"
  echo "JOB_ID=${SLURM_JOB_ID}"
  echo "ARRAY_JOB_ID=${array_job_id}"
  echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
  echo "BACKBONE=${BACKBONE}"
  echo "BACKBONE_RETAIN_ALPHA=${BACKBONE_RETAIN_ALPHA}"
  echo "FORGET_SPLIT=${FORGET_SPLIT}"
  echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
  echo "SEED=${SEED}"
  echo "METHOD=${METHOD}"
  echo "KL_LAMBDA=${KL_LAMBDA}"
  echo "LEARNING_RATE=${LEARNING_RATE}"
  echo "PROBE_BATCHES=${PROBE_BATCHES}"
  echo "SKIP_EXISTING=${SKIP_EXISTING}"
  echo "SAVE_MODEL=${SAVE_MODEL}"
  echo "RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL}"
  echo "DELETE_CHECKPOINT_AFTER_EVAL=${DELETE_CHECKPOINT_AFTER_EVAL}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"

  if [ "${SKIP_EXISTING}" = "1" ] && [ -n "${existing_summary}" ]; then
    echo "[skip] Found existing complete intervention summary: ${existing_summary}"
    echo "[skip] This task will not rerun intervention or official eval."
    return 0
  fi

  nvidia-smi || true

  save_args=()
  if [ "${SAVE_MODEL}" = "1" ]; then
    save_args+=(--save-model)
  fi

  python experiments/piper/piper_local_kl_intervention.py \
    --model-name "${MODEL_NAME}" \
    --model-path "${MODEL_PATH}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --forget-split "${FORGET_SPLIT}" \
    --retain-split "${RETAIN_SPLIT}" \
    --backbone "${BACKBONE}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --max-length 512 \
    --forget-sample-size "${FORGET_SAMPLE_SIZE}" \
    --retain-candidate-size "${RETAIN_CANDIDATE_SIZE}" \
    --forget-batch-size "${FORGET_BATCH_SIZE}" \
    --retain-batch-size 2 \
    --local-kl-batch-size "${LOCAL_KL_BATCH_SIZE}" \
    --ref-logit-batch-size "${REF_LOGIT_BATCH_SIZE}" \
    --probe-batches "${PROBE_BATCHES}" \
    --topk-frac "${TOPK_FRAC}" \
    --train-steps "${TRAIN_STEPS}" \
    --learning-rate "${LEARNING_RATE}" \
    --probe-learning-rate "${PROBE_LEARNING_RATE}" \
    --npo-beta 0.1 \
    --simnpo-beta 4.5 \
    --simnpo-gamma 0.125 \
    --backbone-retain-alpha "${BACKBONE_RETAIN_ALPHA}" \
    --method "${METHOD}" \
    --kl-lambda "${KL_LAMBDA}" \
    --optimizer paged_adamw_32bit \
    --gradient-checkpointing \
    "${save_args[@]}"

  if [ "${RUN_OFFICIAL_EVAL}" != "1" ]; then
    echo "[eval] RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL}; skipping official eval"
    return 0
  fi
  if [ "${SAVE_MODEL}" != "1" ]; then
    echo "[eval] SAVE_MODEL=${SAVE_MODEL}; skipping official eval because no checkpoint was saved"
    return 0
  fi

  CHECKPOINT_PATH="${OUTPUT_DIR}/checkpoint"
  test -d "${CHECKPOINT_PATH}" || { echo "[ERROR] checkpoint not found: ${CHECKPOINT_PATH}"; exit 3; }

  RUN_NAME="$(basename "${OUTPUT_DIR}")"
  TASK_NAME="${RUN_NAME}_tofu_eval"
  EVAL_DIR="${EVAL_OUTPUT_ROOT}/${TASK_NAME}"
  mkdir -p "${EVAL_DIR}"

  echo "===== OFFICIAL OPENUNLEARNING TOFU EVAL ====="
  echo "RUN_NAME=${RUN_NAME}"
  echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
  echo "EVAL_DIR=${EVAL_DIR}"
  echo "RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH}"

  cmd=(
    python src/eval.py
    --config-name=eval.yaml
    experiment=eval/tofu/default
    "model=${MODEL_CONFIG}"
    "model.model_args.pretrained_model_name_or_path=${CHECKPOINT_PATH}"
    "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}"
    "forget_split=${FORGET_SPLIT}"
    "task_name=${TASK_NAME}"
    "paths.output_dir=${EVAL_DIR}"
    "eval.tofu.output_dir=${EVAL_DIR}"
    "eval.tofu.overwrite=${OVERWRITE}"
  )

  if [ -n "${RETAIN_LOGS_PATH}" ]; then
    cmd+=("retain_logs_path=${RETAIN_LOGS_PATH}")
    cmd+=("eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH}")
  fi

  "${cmd[@]}"

  if [ "${DELETE_CHECKPOINT_AFTER_EVAL}" = "1" ]; then
    case "${CHECKPOINT_PATH}" in
      results/piper_intervention_expanded/*/checkpoint|*/results/piper_intervention_expanded/*/checkpoint)
        echo "[cleanup] deleting checkpoint after successful eval: ${CHECKPOINT_PATH}"
        rm -rf -- "${CHECKPOINT_PATH}"
        ;;
      *)
        echo "[cleanup] refusing to delete unexpected checkpoint path: ${CHECKPOINT_PATH}"
        ;;
    esac
  fi

  echo "===== PIPELINE TASK DONE: ${OUTPUT_DIR} ====="
}

run_summary() {
  setup_runtime

  OUTPUT_ROOT="${OUTPUT_ROOT:-results/piper_intervention_expanded}"
  EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/piper_official_eval}"
  INTERVENTION_SUMMARY="${INTERVENTION_SUMMARY:-${OUTPUT_ROOT}/intervention_summary.csv}"
  OFFICIAL_EVAL_SUMMARY="${OFFICIAL_EVAL_SUMMARY:-${EVAL_OUTPUT_ROOT}/tofu_eval_summary.csv}"

  echo "===== PIPER PIPELINE SUMMARY ====="
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT}"

  python experiments/piper/summarize_piper_intervention.py \
    --root "${OUTPUT_ROOT}" \
    --output "${INTERVENTION_SUMMARY}"

  if [ -d "${EVAL_OUTPUT_ROOT}" ] && find "${EVAL_OUTPUT_ROOT}" -name TOFU_SUMMARY.json -print -quit | grep -q .; then
    python experiments/piper/summarize_official_tofu_eval.py \
      --eval-root "${EVAL_OUTPUT_ROOT}" \
      --intervention-root "${OUTPUT_ROOT}" \
      --output "${OFFICIAL_EVAL_SUMMARY}"
  else
    echo "[summary] No TOFU_SUMMARY.json found under ${EVAL_OUTPUT_ROOT}; skipping official eval summary"
  fi

  echo "Intervention summary: ${INTERVENTION_SUMMARY}"
  echo "Official eval summary: ${OFFICIAL_EVAL_SUMMARY}"
  echo "===== SUMMARY DONE ====="
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
    if [ "${PIPELINE_STAGE:-array}" = "summary" ]; then
      run_summary
    elif [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
      run_intervention_and_eval
    else
      total="$(total_tasks)"
      echo "Use one of:"
      echo "  bash $0 --submit"
      echo "  sbatch --array=0-$((total - 1))%2 --gres=gpu:nvidia_a100_80gb_pcie:1 $0"
      exit 2
    fi
    ;;
esac
