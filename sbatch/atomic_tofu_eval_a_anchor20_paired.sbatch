#!/usr/bin/env bash
#SBATCH --partition=compute
#SBATCH --job-name=atomic-evala-anchor20
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/atomic/%x-%j.out
#SBATCH --error=logs/atomic/%x-%j.err

set -euo pipefail

PROJECT_DIR="/home/zkzhang/unlearn/open-unlearning"
CONDA_ROOT="/home/zkzhang/miniconda3"
CONDA_ENV="openunlearning-official-py311"
PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"
CHECKPOINT="/home/zkzhang/models/tofu_Llama-3.2-1B-Instruct_full"
RELEASE_ROOT="${PROJECT_DIR}/data/atomic_tofu/v2.0-rc1"
RUNNER="${PROJECT_DIR}/scripts/run_eval_a_anchor20_paired_calibration.py"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export HF_DATASETS_OFFLINE=1

cd "${PROJECT_DIR}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unavailable)"
echo "CPU_COUNT=${SLURM_CPUS_PER_TASK:-unset}"
echo "MEMORY_LIMIT=${SLURM_MEM_PER_NODE:-16G}"
echo "TIME_LIMIT=${SLURM_TIMELIMIT:-01:00:00}"

test -x "${PYTHON_BIN}"
test -f "${RUNNER}"
test -d "${CHECKPOINT}"
test -f "${CHECKPOINT}/config.json"
test -f "${CHECKPOINT}/tokenizer.json"
test -f "${CHECKPOINT}/model.safetensors"
test -f "${RELEASE_ROOT}/api/eval_extension/anchor_calibration/anchor_20_ids.txt"
test -f "${RELEASE_ROOT}/api/eval_extension/anchor_calibration/current_route_candidate_outputs.jsonl"
test -f "${RELEASE_ROOT}/official_anchors/official_eval_sidecars.jsonl"
test -f "${RELEASE_ROOT}/source/tofu_full.jsonl"

"${PYTHON_BIN}" "${RUNNER}" --validate-only
"${PYTHON_BIN}" "${RUNNER}"
