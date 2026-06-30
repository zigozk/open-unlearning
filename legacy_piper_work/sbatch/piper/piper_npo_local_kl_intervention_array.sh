#!/bin/bash
#SBATCH -J piper_npo_kl
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --array=0-38%3
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

SEEDS=(0 1 2)
METHOD_SPECS=(
  "baseline:0.0"
  "global_retain_kl:0.1"
  "global_retain_kl:0.3"
  "global_retain_kl:1.0"
  "random_local_kl:0.1"
  "random_local_kl:0.3"
  "random_local_kl:1.0"
  "semantic_local_kl:0.1"
  "semantic_local_kl:0.3"
  "semantic_local_kl:1.0"
  "pi_local_kl:0.1"
  "pi_local_kl:0.3"
  "pi_local_kl:1.0"
)

idx=${SLURM_ARRAY_TASK_ID}
n_method=${#METHOD_SPECS[@]}
n_seed=${#SEEDS[@]}
total=$((n_method * n_seed))
if [ "$idx" -ge "$total" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${idx} >= total=${total}"
  exit 2
fi

method_idx=$((idx % n_method))
seed_idx=$((idx / n_method))
SEED="${SEEDS[$seed_idx]}"
SPEC="${METHOD_SPECS[$method_idx]}"
METHOD="${SPEC%%:*}"
KL_LAMBDA="${SPEC##*:}"

FORGET_SPLIT="${FORGET_SPLIT:-forget10}"
RETAIN_SPLIT="${RETAIN_SPLIT:-retain90}"
FORGET_BATCH_SIZE="${FORGET_BATCH_SIZE:-1}"
PROBE_BATCHES="${PROBE_BATCHES:-64}"
FORGET_SAMPLE_SIZE="${FORGET_SAMPLE_SIZE:-128}"
RETAIN_CANDIDATE_SIZE="${RETAIN_CANDIDATE_SIZE:-500}"
TOPK_FRAC="${TOPK_FRAC:-0.10}"
TRAIN_STEPS="${TRAIN_STEPS:-80}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PROBE_LEARNING_RATE="${PROBE_LEARNING_RATE:-1e-5}"
LOCAL_KL_BATCH_SIZE="${LOCAL_KL_BATCH_SIZE:-1}"
REF_LOGIT_BATCH_SIZE="${REF_LOGIT_BATCH_SIZE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/piper_intervention}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${FORGET_SPLIT}_${RETAIN_SPLIT}_NPO_${METHOD}_lambda${KL_LAMBDA}_seed${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "===== STATIC PIPER LOCAL KL INTERVENTION ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
echo "ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "FORGET_SPLIT=${FORGET_SPLIT}"
echo "RETAIN_SPLIT=${RETAIN_SPLIT}"
echo "SEED=${SEED}"
echo "METHOD=${METHOD}"
echo "KL_LAMBDA=${KL_LAMBDA}"
echo "FORGET_BATCH_SIZE=${FORGET_BATCH_SIZE}"
echo "PROBE_BATCHES=${PROBE_BATCHES}"
echo "FORGET_SAMPLE_SIZE=${FORGET_SAMPLE_SIZE}"
echo "RETAIN_CANDIDATE_SIZE=${RETAIN_CANDIDATE_SIZE}"
echo "TOPK_FRAC=${TOPK_FRAC}"
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

python experiments/piper/piper_local_kl_intervention.py \
  --model-name "${MODEL_NAME}" \
  --model-path "${MODEL_PATH}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --forget-split "${FORGET_SPLIT}" \
  --retain-split "${RETAIN_SPLIT}" \
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
  --method "${METHOD}" \
  --kl-lambda "${KL_LAMBDA}" \
  --optimizer paged_adamw_32bit \
  --gradient-checkpointing

echo "===== DONE: ${OUTPUT_DIR} ====="
