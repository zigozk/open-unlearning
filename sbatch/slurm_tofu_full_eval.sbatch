#!/usr/bin/env bash
#SBATCH -J tofu-full-eval
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/zkzhang/unlearn/open-unlearning}
CONDA_ENV=${CONDA_ENV:-openunlearning-official-py311}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/home/zkzhang/unlearning/HF_CACHE/datasets}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-}
RETAIN_SPLIT=${RETAIN_SPLIT:-}
MODEL_PATH=${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL}_full}
RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-}
TASK_NAME=${TASK_NAME:-local_${MODEL}_full_${FORGET_SPLIT}_eval}
OUTPUT_DIR=${OUTPUT_DIR:-results/eval/${TASK_NAME}}

case "${FORGET_SPLIT}" in
  forget01)
    HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout01}
    RETAIN_SPLIT=${RETAIN_SPLIT:-retain99}
    ;;
  forget05)
    HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout05}
    RETAIN_SPLIT=${RETAIN_SPLIT:-retain95}
    ;;
  forget10)
    HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-holdout10}
    RETAIN_SPLIT=${RETAIN_SPLIT:-retain90}
    ;;
  *)
    echo "Unsupported FORGET_SPLIT=${FORGET_SPLIT}; set HOLDOUT_SPLIT and RETAIN_SPLIT explicitly." >&2
    exit 2
    ;;
esac

RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json}

cd "${REPO_ROOT}"
mkdir -p logs/slurm
mkdir -p "$(dirname "${OUTPUT_DIR}")"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export HF_DATASETS_CACHE
export HF_DATASETS_OFFLINE

if [[ -n "${CUDA_MODULE:-}" ]]; then
  # Optional on this HPC, e.g. CUDA_MODULE=cuda/12.1.1.
  source /usr/share/modules/init/bash
  module use --append /home/share/modules/modulefiles
  module load "${CUDA_MODULE}"
fi

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Model: ${MODEL}"
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo "HF datasets offline: ${HF_DATASETS_OFFLINE}"
echo "Model path: ${MODEL_PATH}"
echo "Forget/Holdout/Retain: ${FORGET_SPLIT} / ${HOLDOUT_SPLIT} / ${RETAIN_SPLIT}"
echo "Retain logs: ${RETAIN_LOGS_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
nvidia-smi || true

[[ -d "${MODEL_PATH}" ]] || { echo "Missing MODEL_PATH=${MODEL_PATH}" >&2; exit 2; }
[[ -f "${RETAIN_LOGS_PATH}" ]] || { echo "Missing RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH}" >&2; exit 2; }

cmd=(
  python src/eval.py --config-name=eval.yaml
  experiment=eval/tofu/default.yaml
  model="${MODEL}"
  forget_split="${FORGET_SPLIT}"
  holdout_split="${HOLDOUT_SPLIT}"
  task_name="${TASK_NAME}"
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}"
  model.tokenizer_args.pretrained_model_name_or_path="${MODEL_PATH}"
  retain_logs_path="${RETAIN_LOGS_PATH}"
  paths.output_dir="${OUTPUT_DIR}"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
