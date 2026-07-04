#!/bin/bash
#SBATCH -J bridge_summary
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
cd "${ROOT_DIR}"
mkdir -p logs results/bridge_reports

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-unlearning-new}"

TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-results/bridge_initial}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/bridge_initial_eval}"
REPORT_ROOT="${REPORT_ROOT:-results/bridge_reports}"
REPORT_PREFIX="${REPORT_PREFIX:-bridge_initial}"

CSV_PATH="${REPORT_ROOT}/${REPORT_PREFIX}_summary.csv"
MD_PATH="${REPORT_ROOT}/${REPORT_PREFIX}_report.md"

python experiments/bridge/summarize_bridge_initial.py \
  --train-root "${TRAIN_OUTPUT_ROOT}" \
  --eval-root "${EVAL_OUTPUT_ROOT}" \
  --output-csv "${CSV_PATH}" \
  --output-md "${MD_PATH}"

echo "===== BRIDGE INITIAL SUMMARY ====="
echo "CSV: ${CSV_PATH}"
echo "Report: ${MD_PATH}"
cat "${MD_PATH}"
