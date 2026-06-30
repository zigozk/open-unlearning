#!/bin/bash
#SBATCH -J piper_mb_kl
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --array=0-347%4
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
BACKBONE_SPECS=(
  "NPO:1e-5:0.0"
  "GradAscent:1e-6:0.0"
  "GradDiff:1e-6:1.0"
  "SimNPO:1e-5:0.0"
)
LAMBDA_VALUES=(0.03 0.1 0.2 0.3 0.5 0.7 1.0)
METHODS=(global_retain_kl random_local_kl semantic_local_kl pi_local_kl)

METHOD_SPECS=("baseline:0.0")
for method in "${METHODS[@]}"; do
  for lambda in "${LAMBDA_VALUES[@]}"; do
    METHOD_SPECS+=("${method}:${lambda}")
  done
done

idx=${SLURM_ARRAY_TASK_ID}
n_method=${#METHOD_SPECS[@]}
n_seed=${#SEEDS[@]}
n_backbone=${#BACKBONE_SPECS[@]}
total=$((n_method * n_seed * n_backbone))
if [ "$idx" -ge "$total" ]; then
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
SAVE_MODEL="${SAVE_MODEL:-1}"
OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}_${FORGET_SPLIT}_${RETAIN_SPLIT}_${BACKBONE}_${METHOD}_lambda${KL_LAMBDA}_seed${SEED}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

echo "===== STATIC PIPER MULTI-BACKBONE LOCAL KL ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID}"
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
echo "SAVE_MODEL=${SAVE_MODEL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
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

echo "===== DONE: ${OUTPUT_DIR} ====="
