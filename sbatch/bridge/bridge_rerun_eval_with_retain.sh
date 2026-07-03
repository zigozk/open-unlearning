#!/bin/bash
#SBATCH -J bridge_eval_fq
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Eval-only rerun for existing BRIDGE checkpoints with retain reference logs.
# Defaults target the current valid 1B run: RUN_TAG=postfix_1b_v2.

setup_runtime() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  mkdir -p logs results/bridge_reports

  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-unlearning}"

  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM=false
  export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
  export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
}

check_python_deps() {
  if [ "${FIX_PY_DEPS:-0}" = "1" ]; then
    python -m pip install --no-cache-dir huggingface-hub==0.29.1 transformers==4.45.1
  fi

  python - <<'PY'
import sys

try:
    import huggingface_hub
    import transformers
except Exception as exc:
    print("Failed to import transformers/huggingface_hub.", file=sys.stderr)
    print(exc, file=sys.stderr)
    print(
        "Fix with: python -m pip install --no-cache-dir huggingface-hub==0.29.1 transformers==4.45.1",
        file=sys.stderr,
    )
    raise

print(f"transformers={transformers.__version__} huggingface_hub={huggingface_hub.__version__}")
PY
}

model_tag() {
  local model_config="$1"
  echo "${model_config//[^A-Za-z0-9_]/_}"
}

build_methods() {
  if [ -n "${METHODS:-}" ]; then
    read -r -a METHOD_SPECS <<< "${METHODS}"
  else
    METHOD_SPECS=(
      npo
      npo_global_kl
      bridge_uniform_dro
      bridge_history_dro
      bridge_refresh_gs_dro
      bridge_refresh_pi_dro
    )
  fi
}

eval_one() {
  local model_config="$1"
  local method="$2"
  local tag task_name train_dir eval_dir retain_logs_path tokenizer_path local_full_path

  tag="$(model_tag "${model_config}")"
  task_name="BRIDGE_TOFU_${FORGET_SPLIT:-forget10}_${tag}_${method}_seed${SEED:-0}_${RUN_TAG:-postfix_1b_v2}"
  train_dir="${TRAIN_OUTPUT_ROOT:-results/bridge_initial}/${task_name}"
  eval_dir="${EVAL_OUTPUT_ROOT:-results/bridge_initial_eval}/${task_name}_tofu_eval"

  if [ -n "${RETAIN_LOGS_PATH:-}" ]; then
    retain_logs_path="${RETAIN_LOGS_PATH}"
  else
    retain_logs_path="${RETAIN_LOGS_ROOT:-results/bridge_retain_logs}/tofu_${tag}_${RETAIN_SPLIT:-retain90}_reference/TOFU_EVAL.json"
  fi

  local_full_path="/home/zkzhang/models/tofu_${model_config}_full"
  if [ -n "${EVAL_TOKENIZER_PATH:-}" ]; then
    tokenizer_path="${EVAL_TOKENIZER_PATH}"
  elif [ -d "${local_full_path}" ]; then
    tokenizer_path="${local_full_path}"
  else
    tokenizer_path="open-unlearning/tofu_${model_config}_full"
  fi

  echo "===== BRIDGE EVAL WITH RETAIN ====="
  echo "MODEL_CONFIG=${model_config}"
  echo "METHOD=${method}"
  echo "TASK_NAME=${task_name}"
  echo "TRAIN_DIR=${train_dir}"
  echo "EVAL_DIR=${eval_dir}"
  echo "TOKENIZER_PATH=${tokenizer_path}"
  echo "RETAIN_LOGS_PATH=${retain_logs_path}"

  if [ ! -d "${train_dir}" ]; then
    if [ "${SKIP_MISSING:-0}" = "1" ]; then
      echo "[skip] Missing train dir: ${train_dir}"
      return 0
    fi
    echo "[error] Missing train dir: ${train_dir}" >&2
    exit 1
  fi

  if [ ! -f "${retain_logs_path}" ]; then
    echo "[error] Missing retain reference log: ${retain_logs_path}" >&2
    echo "Generate it first with sbatch/bridge/bridge_generate_retain_reference.sh" >&2
    exit 1
  fi

  if [ "${SKIP_EXISTING:-0}" = "1" ] && [ -f "${eval_dir}/TOFU_SUMMARY.json" ]; then
    echo "[skip] Found existing eval summary: ${eval_dir}/TOFU_SUMMARY.json"
    return 0
  fi

  python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    "model=${model_config}" \
    "model.model_args.pretrained_model_name_or_path=${train_dir}" \
    "model.tokenizer_args.pretrained_model_name_or_path=${tokenizer_path}" \
    "forget_split=${FORGET_SPLIT:-forget10}" \
    "holdout_split=${HOLDOUT_SPLIT:-holdout10}" \
    "task_name=${task_name}_tofu_eval" \
    "paths.output_dir=${eval_dir}" \
    "eval.tofu.output_dir=${eval_dir}" \
    "eval.tofu.overwrite=${OVERWRITE_EVAL:-true}" \
    "eval.tofu.batch_size=${EVAL_BATCH_SIZE:-32}" \
    "retain_logs_path=${retain_logs_path}" \
    "eval.tofu.retain_logs_path=${retain_logs_path}"

  test -f "${eval_dir}/TOFU_SUMMARY.json"
  echo "===== DONE: ${eval_dir}/TOFU_SUMMARY.json ====="
}

write_summary() {
  local report_root report_prefix csv_path md_path
  report_root="${REPORT_ROOT:-results/bridge_reports}"
  report_prefix="${REPORT_PREFIX:-bridge_initial}"
  csv_path="${report_root}/${report_prefix}_summary.csv"
  md_path="${report_root}/${report_prefix}_report.md"
  mkdir -p "${report_root}"

  python experiments/bridge/summarize_bridge_initial.py \
    --train-root "${TRAIN_OUTPUT_ROOT:-results/bridge_initial}" \
    --eval-root "${EVAL_OUTPUT_ROOT:-results/bridge_initial_eval}" \
    --output-csv "${csv_path}" \
    --output-md "${md_path}"

  echo "===== BRIDGE EVAL SUMMARY ====="
  echo "CSV: ${csv_path}"
  echo "Report: ${md_path}"
  cat "${md_path}"
}

main() {
  setup_runtime
  check_python_deps
  build_methods

  read -r -a MODEL_SPECS <<< "${MODEL_CONFIGS:-${MODEL_CONFIG:-Llama-3.2-1B-Instruct}}"

  for model_config in "${MODEL_SPECS[@]}"; do
    for method in "${METHOD_SPECS[@]}"; do
      eval_one "${model_config}" "${method}"
    done
  done

  if [ "${RUN_SUMMARY:-1}" = "1" ]; then
    write_summary
  fi
}

main "$@"
