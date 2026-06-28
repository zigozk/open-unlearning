#!/bin/bash
#SBATCH -J piper_tofu_eval
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH --array=0-347%2
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

INTERVENTION_ROOT="${INTERVENTION_ROOT:-results/piper_intervention_expanded}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/piper_official_eval}"
MODEL_CONFIG="${MODEL_CONFIG:-Llama-2-7b-chat-hf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/share/models/Llama-2-7b-chat-hf}"
RETAIN_LOGS_PATH="${RETAIN_LOGS_PATH:-}"
OVERWRITE="${OVERWRITE:-true}"

mapfile -t RUN_LINES < <(
  python - <<'PY' "${INTERVENTION_ROOT}"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for summary_path in sorted(root.glob("**/summary.json")):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = data.get("saved_model_path")
    if not checkpoint:
        continue
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        continue
    print(f"{summary_path.parent.name}\t{checkpoint_path.as_posix()}\t{data.get('forget_split', 'forget10')}")
PY
)

idx=${SLURM_ARRAY_TASK_ID}
total=${#RUN_LINES[@]}
if [ "$idx" -ge "$total" ]; then
  echo "[SKIP] SLURM_ARRAY_TASK_ID=${idx} >= checkpoint_runs=${total}"
  exit 0
fi

IFS=$'\t' read -r RUN_NAME CHECKPOINT_PATH FORGET_SPLIT <<< "${RUN_LINES[$idx]}"
TASK_NAME="${RUN_NAME}_tofu_eval"
OUTPUT_DIR="${EVAL_OUTPUT_ROOT}/${TASK_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "===== OFFICIAL OPENUNLEARNING TOFU EVAL ====="
echo "RUN_NAME=${RUN_NAME}"
echo "CHECKPOINT_PATH=${CHECKPOINT_PATH}"
echo "FORGET_SPLIT=${FORGET_SPLIT}"
echo "MODEL_CONFIG=${MODEL_CONFIG}"
echo "TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
nvidia-smi || true

cmd=(
  python src/eval.py
  --config-name=eval.yaml
  experiment=eval/tofu/default
  "model=${MODEL_CONFIG}"
  "model.model_args.pretrained_model_name_or_path=${CHECKPOINT_PATH}"
  "model.tokenizer_args.pretrained_model_name_or_path=${TOKENIZER_PATH}"
  "forget_split=${FORGET_SPLIT}"
  "task_name=${TASK_NAME}"
  "paths.output_dir=${OUTPUT_DIR}"
  "eval.tofu.output_dir=${OUTPUT_DIR}"
  "eval.tofu.overwrite=${OVERWRITE}"
)

if [ -n "${RETAIN_LOGS_PATH}" ]; then
  cmd+=("retain_logs_path=${RETAIN_LOGS_PATH}")
  cmd+=("eval.tofu.retain_logs_path=${RETAIN_LOGS_PATH}")
fi

"${cmd[@]}"

echo "===== DONE: ${OUTPUT_DIR} ====="
