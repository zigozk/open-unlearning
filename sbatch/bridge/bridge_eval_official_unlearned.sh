#!/bin/bash
#SBATCH -J bridge_official_eval
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 4:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Eval-only sanity check for official OpenUnlearning unlearned TOFU models.
# Defaults target:
#   open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10

setup_runtime() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  mkdir -p logs results/bridge_official_eval

  source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-unlearning}"

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

check_python_deps() {
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

run_eval() {
  local tag default_model_path default_tokenizer_path default_retain_logs task_name out_dir

  MODEL_CONFIG="${MODEL_CONFIG:-Llama-3.2-1B-Instruct}"
  FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
  HOLDOUT_SPLIT="${HOLDOUT_SPLIT:-holdout10}"
  RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"

  tag="$(model_tag "${MODEL_CONFIG}")"
  default_model_path="/home/zkzhang/models/unlearn_tofu_${MODEL_CONFIG}_${FORGET_SPLIT}_NPO_lr2e-05_beta0.5_alpha1_epoch10"
  default_tokenizer_path="/home/zkzhang/models/tofu_${MODEL_CONFIG}_full"
  default_retain_logs="results/bridge_retain_logs/tofu_${tag}_${RETAIN_SPLIT}_reference/TOFU_EVAL.json"

  MODEL_PATH="${MODEL_PATH:-${default_model_path}}"
  TOKENIZER_PATH="${TOKENIZER_PATH:-${default_tokenizer_path}}"
  RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-${default_retain_logs}}"
  task_name="${TASK_NAME:-official_NPO_${FORGET_SPLIT}_${tag}_tofu_eval}"
  out_dir="${OUT_DIR:-results/bridge_official_eval/official_NPO_${FORGET_SPLIT}_${tag}}"

  echo "===== OFFICIAL UNLEARNED TOFU EVAL ====="
  echo "MODEL_CONFIG=${MODEL_CONFIG}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
  echo "FORGET_SPLIT=${FORGET_SPLIT}"
  echo "HOLDOUT_SPLIT=${HOLDOUT_SPLIT}"
  echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
  echo "RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH}"
  echo "OUT_DIR=${out_dir}"
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
    echo "Generate it first with sbatch/bridge/bridge_generate_retain_reference.sh" >&2
    exit 1
  fi

  python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    "model=${MODEL_CONFIG}" \
    "model.model_args.pretrained_model_name_or_path=${MODEL_PATH}" \
    "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}" \
    "forget_split=${FORGET_SPLIT}" \
    "holdout_split=${HOLDOUT_SPLIT}" \
    "task_name=${task_name}" \
    "paths.output_dir=${out_dir}" \
    "eval.tofu.output_dir=${out_dir}" \
    "eval.tofu.overwrite=${OVERWRITE_EVAL:-true}" \
    "eval.tofu.batch_size=${EVAL_BATCH_SIZE:-32}" \
    "retain_logs_path=${RETAIN_LOGS_PATH}" \
    "eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH}"

  test -f "${out_dir}/TOFU_SUMMARY.json"
  echo "===== DONE: ${out_dir}/TOFU_SUMMARY.json ====="
  cat "${out_dir}/TOFU_SUMMARY.json"
}

main() {
  setup_runtime
  check_python_deps
  run_eval
}

main "$@"
