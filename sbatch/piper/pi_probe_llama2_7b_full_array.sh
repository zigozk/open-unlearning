#!/bin/bash
#SBATCH -J piper_pi_full
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 36:00:00
#SBATCH --array=0-1
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
cd "${ROOT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# Keep the job compatible with GPU nodes without external network access.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

METHODS=("GradAscent" "NPO")
BACKBONE="${METHODS[$SLURM_ARRAY_TASK_ID]}"

MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
MODEL_PATH="${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL_NAME}_full}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/share/models/${MODEL_NAME}}"

test -d "${MODEL_PATH}" || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 2; }
if [ ! -d "${TOKENIZER_PATH}" ]; then
  echo "[WARN] TOKENIZER_PATH not found: ${TOKENIZER_PATH}; falling back to MODEL_PATH"
  TOKENIZER_PATH="${MODEL_PATH}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/piper_pi_probe}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_forget10_${BACKBONE}_full_${SLURM_JOB_ID}"

echo "===== PIPER PI FULL-PARAMETER PROBE ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "BACKBONE=${BACKBONE}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi || true

python - <<'PY'
import importlib.util
import torch
print("python dependency check")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
missing = [name for name in ["bitsandbytes", "datasets", "transformers", "scipy"] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing python packages: " + ", ".join(missing))
PY

python experiments/piper/pi_predictive_validity_probe.py \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --forget-split forget10 \
  --retain-split retain90 \
  --backbone "${BACKBONE}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed 42 \
  --max-length 512 \
  --forget-sample-size 128 \
  --retain-candidate-size 500 \
  --probe-batches 8 \
  --forget-batch-size 1 \
  --retain-batch-size 2 \
  --train-steps 80 \
  --learning-rate 1e-5 \
  --probe-learning-rate 1e-5 \
  --npo-beta 0.1 \
  --topk-fracs 0.05,0.10,0.20 \
  --num-bins 10 \
  --optimizer paged_adamw_32bit \
  --gradient-checkpointing

echo "===== DONE: ${OUTPUT_DIR} ====="
