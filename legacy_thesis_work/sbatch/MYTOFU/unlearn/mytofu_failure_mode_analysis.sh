#!/bin/bash
#SBATCH -J mytofu_fail_modes
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /home/zkzhang/unlearning/open-unlearning

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export OMP_NUM_THREADS=4

RESULT_ROOT="${RESULT_ROOT:-/home/zkzhang/unlearning/open-unlearning/saves/unlearn}"
FORGET_JSONL="${FORGET_JSONL:-/home/zkzhang/unlearning/Create_Data/mini_tofu_custom_slot_hard_plus/eval/forget_eval_perturbed.jsonl}"
RETAIN_JSONL="${RETAIN_JSONL:-/home/zkzhang/unlearning/Create_Data/mini_tofu_custom_slot_hard_plus/eval/retain_eval_perturbed.jsonl}"
OUT_DIR="${OUT_DIR:-/home/zkzhang/unlearning/open-unlearning/saves/mytofu_failure_modes}"

python sbatch/MYTOFU/unlearn/analyze_mytofu_failure_modes.py \
  --result-root "${RESULT_ROOT}" \
  --forget-jsonl "${FORGET_JSONL}" \
  --retain-jsonl "${RETAIN_JSONL}" \
  --out-dir "${OUT_DIR}" \
  --methods CEU NPO RMU SimNPO_sago SimNPO_pcgrad

echo "===== MYTOFU FAILURE MODE ANALYSIS DONE ====="
echo "OUT_DIR=${OUT_DIR}"
