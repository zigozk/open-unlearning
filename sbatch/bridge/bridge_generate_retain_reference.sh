#!/bin/bash
#SBATCH -J bridge_retain_ref
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --nodelist=gpu01
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

# Generate TOFU retain-reference logs used by forget_quality.
#
# Default intentionally runs only Llama-3.2-1B-Instruct. When the Llama-2 7B
# retain90 model is ready, submit with:
#   sbatch --export=ALL,MODEL_CONFIGS="Llama-3.2-1B-Instruct Llama-2-7b-chat-hf" sbatch/bridge/bridge_generate_retain_reference.sh

setup_runtime() {
  ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
  cd "${ROOT_DIR}"
  mkdir -p logs results/bridge_retain_logs

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

run_reference_eval() {
  local model_config="$1"
  local tag retain_model_path out_dir task_name

  tag="$(model_tag "${model_config}")"
  retain_model_path="${RETAIN_MODEL_PATH:-}"
  if [ -z "${retain_model_path}" ]; then
    retain_model_path="/home/zkzhang/models/tofu_${model_config}_retain90"
  fi
  if [ ! -d "${retain_model_path}" ]; then
    retain_model_path="open-unlearning/tofu_${model_config}_retain90"
  fi

  task_name="tofu_${tag}_${RETAIN_SPLIT:-retain90}_reference"
  out_dir="${RETAIN_LOGS_ROOT:-results/bridge_retain_logs}/${task_name}"

  echo "===== BRIDGE RETAIN REFERENCE ====="
  echo "MODEL_CONFIG=${model_config}"
  echo "RETAIN_MODEL_PATH=${retain_model_path}"
  echo "OUTPUT_DIR=${out_dir}"
  echo "FORGET_SPLIT=${FORGET_SPLIT:-forget10}"
  echo "HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout10}"
  nvidia-smi || true

  if [ "${SKIP_EXISTING:-1}" = "1" ] && [ -f "${out_dir}/TOFU_EVAL.json" ]; then
    echo "[skip] Found existing retain reference log: ${out_dir}/TOFU_EVAL.json"
    return 0
  fi

  python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/tofu/default \
    "model=${model_config}" \
    "model.model_args.pretrained_model_name_or_path=${retain_model_path}" \
    "model.tokenizer_args.pretrained_model_name_or_path=${retain_model_path}" \
    "forget_split=${FORGET_SPLIT:-forget10}" \
    "holdout_split=${HOLDOUT_SPLIT:-holdout10}" \
    "task_name=${task_name}" \
    "paths.output_dir=${out_dir}" \
    "eval.tofu.output_dir=${out_dir}" \
    "eval.tofu.overwrite=${OVERWRITE_EVAL:-true}" \
    "eval.tofu.batch_size=${EVAL_BATCH_SIZE:-32}"

  test -f "${out_dir}/TOFU_EVAL.json"
  echo "===== DONE: ${out_dir}/TOFU_EVAL.json ====="
}

main() {
  setup_runtime
  check_python_deps

  read -r -a MODEL_SPECS <<< "${MODEL_CONFIGS:-${MODEL_CONFIG:-Llama-3.2-1B-Instruct}}"
  if [ "${INCLUDE_LLAMA2_7B:-0}" = "1" ]; then
    MODEL_SPECS+=("Llama-2-7b-chat-hf")
  fi

  for model_config in "${MODEL_SPECS[@]}"; do
    run_reference_eval "${model_config}"
  done
}

main "$@"
