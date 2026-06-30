#!/bin/bash
#SBATCH -J piper_req_pi
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --array=0-80%4
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
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

MODEL_NAME="${MODEL_NAME:-Llama-2-7b-chat-hf}"
MODEL_PATH="${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL_NAME}_full}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/share/models/${MODEL_NAME}}"
test -d "${MODEL_PATH}" || { echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"; exit 2; }
if [ ! -d "${TOKENIZER_PATH}" ]; then
  echo "[WARN] TOKENIZER_PATH not found: ${TOKENIZER_PATH}; falling back to MODEL_PATH"
  TOKENIZER_PATH="${MODEL_PATH}"
fi

SPLITS=("forget01:retain99" "forget05:retain95" "forget10:retain90")
SEEDS=(0 1 2)
# Format: forget_batch_size:probe_batches. These cover PI-8/16/32/64/128
# and test whether larger coverage or set-gradient PI improves prediction.
COVERAGE_CONFIGS=("1:8" "1:16" "1:32" "1:64" "1:128" "2:8" "2:16" "2:32" "2:64")

idx=${SLURM_ARRAY_TASK_ID}
n_cfg=${#COVERAGE_CONFIGS[@]}
n_seed=${#SEEDS[@]}
n_split=${#SPLITS[@]}
total=$((n_split * n_seed * n_cfg))
if [ "$idx" -ge "$total" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${idx} >= total=${total}"
  exit 2
fi

cfg_idx=$((idx % n_cfg))
idx=$((idx / n_cfg))
seed_idx=$((idx % n_seed))
idx=$((idx / n_seed))
split_idx=$((idx % n_split))

SPLIT_PAIR="${SPLITS[$split_idx]}"
FORGET_SPLIT="${SPLIT_PAIR%%:*}"
RETAIN_SPLIT="${SPLIT_PAIR##*:}"
SEED="${SEEDS[$seed_idx]}"
COVERAGE="${COVERAGE_CONFIGS[$cfg_idx]}"
FORGET_BATCH_SIZE="${COVERAGE%%:*}"
PROBE_BATCHES="${COVERAGE##*:}"

FORGET_SAMPLE_SIZE=$((FORGET_BATCH_SIZE * PROBE_BATCHES))
if [ "$FORGET_SAMPLE_SIZE" -lt 128 ]; then
  FORGET_SAMPLE_SIZE=128
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-results/piper_pi_required}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${FORGET_SPLIT}_${RETAIN_SPLIT}_NPO_seed${SEED}_fb${FORGET_BATCH_SIZE}_pb${PROBE_BATCHES}_both_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "===== PIPER REQUIRED MECHANISM SUPPLEMENT ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "FORGET_SPLIT=${FORGET_SPLIT}"
echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
echo "SEED=${SEED}"
echo "FORGET_BATCH_SIZE=${FORGET_BATCH_SIZE}"
echo "PROBE_BATCHES=${PROBE_BATCHES}"
echo "FORGET_SAMPLE_SIZE=${FORGET_SAMPLE_SIZE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi || true

python - <<'PY'
import importlib.util
import torch
print("python dependency check")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpu_count", torch.cuda.device_count())
missing = [name for name in ["bitsandbytes", "datasets", "transformers", "scipy", "sklearn"] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing python packages: " + ", ".join(missing))
PY

python experiments/piper/pi_predictive_validity_probe.py \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --forget-split "${FORGET_SPLIT}" \
  --retain-split "${RETAIN_SPLIT}" \
  --backbone NPO \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --max-length 512 \
  --forget-sample-size "${FORGET_SAMPLE_SIZE}" \
  --retain-candidate-size 500 \
  --probe-batches "${PROBE_BATCHES}" \
  --pi-mode both \
  --primary-pi averaged \
  --forget-batch-size "${FORGET_BATCH_SIZE}" \
  --retain-batch-size 2 \
  --train-steps 80 \
  --learning-rate 1e-5 \
  --probe-learning-rate 1e-5 \
  --npo-beta 0.1 \
  --topk-fracs 0.05,0.10,0.20 \
  --num-bins 10 \
  --random-trials 200 \
  --optimizer paged_adamw_32bit \
  --gradient-checkpointing

echo "===== DONE: ${OUTPUT_DIR} ====="
