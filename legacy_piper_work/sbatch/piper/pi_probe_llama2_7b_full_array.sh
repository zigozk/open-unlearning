#!/bin/bash
#SBATCH -J piper_pi_full
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 36:00:00
#SBATCH --array=0-71%4
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

METHODS=("NPO" "GradAscent")
SPLITS=("forget01:retain99" "forget05:retain95" "forget10:retain90")
SEEDS=(0 1 2)
FORGET_BATCH_SIZES=(1 2)
PROBE_BATCHES_LIST=(4 8)

idx=${SLURM_ARRAY_TASK_ID}
n_probe=${#PROBE_BATCHES_LIST[@]}
n_fb=${#FORGET_BATCH_SIZES[@]}
n_seed=${#SEEDS[@]}
n_split=${#SPLITS[@]}
n_method=${#METHODS[@]}
total=$((n_method * n_split * n_seed * n_fb * n_probe))
if [ "$idx" -ge "$total" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${idx} >= total=${total}"
  exit 2
fi

probe_idx=$((idx % n_probe))
idx=$((idx / n_probe))
fb_idx=$((idx % n_fb))
idx=$((idx / n_fb))
seed_idx=$((idx % n_seed))
idx=$((idx / n_seed))
split_idx=$((idx % n_split))
idx=$((idx / n_split))
method_idx=$((idx % n_method))

BACKBONE="${METHODS[$method_idx]}"
SPLIT_PAIR="${SPLITS[$split_idx]}"
FORGET_SPLIT="${SPLIT_PAIR%%:*}"
RETAIN_SPLIT="${SPLIT_PAIR##*:}"
SEED="${SEEDS[$seed_idx]}"
FORGET_BATCH_SIZE="${FORGET_BATCH_SIZES[$fb_idx]}"
PROBE_BATCHES="${PROBE_BATCHES_LIST[$probe_idx]}"

MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
MODEL_PATH="${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL_NAME}_full}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/share/models/${MODEL_NAME}}"

test -d "${MODEL_PATH}" || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 2; }
if [ ! -d "${TOKENIZER_PATH}" ]; then
  echo "[WARN] TOKENIZER_PATH not found: ${TOKENIZER_PATH}; falling back to MODEL_PATH"
  TOKENIZER_PATH="${MODEL_PATH}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/piper_pi_probe}"
if [ "${BACKBONE}" = "GradAscent" ]; then
  LEARNING_RATE="${GA_LEARNING_RATE:-1e-6}"
  PROBE_LEARNING_RATE="${GA_PROBE_LEARNING_RATE:-1e-6}"
  TRAIN_STEPS="${GA_TRAIN_STEPS:-40}"
else
  LEARNING_RATE="${NPO_LEARNING_RATE:-1e-5}"
  PROBE_LEARNING_RATE="${NPO_PROBE_LEARNING_RATE:-1e-5}"
  TRAIN_STEPS="${NPO_TRAIN_STEPS:-80}"
fi

OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${FORGET_SPLIT}_${RETAIN_SPLIT}_${BACKBONE}_seed${SEED}_fb${FORGET_BATCH_SIZE}_pb${PROBE_BATCHES}_full_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "===== PIPER PI FULL-PARAMETER PROBE ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "BACKBONE=${BACKBONE}"
echo "FORGET_SPLIT=${FORGET_SPLIT}"
echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
echo "SEED=${SEED}"
echo "FORGET_BATCH_SIZE=${FORGET_BATCH_SIZE}"
echo "PROBE_BATCHES=${PROBE_BATCHES}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "PROBE_LEARNING_RATE=${PROBE_LEARNING_RATE}"
echo "TRAIN_STEPS=${TRAIN_STEPS}"
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
  --forget-split "${FORGET_SPLIT}" \
  --retain-split "${RETAIN_SPLIT}" \
  --backbone "${BACKBONE}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --max-length 512 \
  --forget-sample-size 128 \
  --retain-candidate-size 500 \
  --probe-batches "${PROBE_BATCHES}" \
  --forget-batch-size "${FORGET_BATCH_SIZE}" \
  --retain-batch-size 2 \
  --train-steps "${TRAIN_STEPS}" \
  --learning-rate "${LEARNING_RATE}" \
  --probe-learning-rate "${PROBE_LEARNING_RATE}" \
  --npo-beta 0.1 \
  --topk-fracs 0.05,0.10,0.20 \
  --num-bins 10 \
  --random-trials 200 \
  --optimizer paged_adamw_32bit \
  --gradient-checkpointing

echo "===== DONE: ${OUTPUT_DIR} ====="
