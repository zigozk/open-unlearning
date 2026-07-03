#!/bin/bash
#SBATCH -J bridge_failures
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
conda activate "${CONDA_ENV:-unlearning}"

LOGS_ROOT="${LOGS_ROOT:-logs}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-results/bridge_initial}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/bridge_initial_eval}"
REPORT_ROOT="${REPORT_ROOT:-results/bridge_reports/debug}"
mkdir -p "${REPORT_ROOT}"

python experiments/bridge/summarize_bridge_failures.py \
  --logs-root "${LOGS_ROOT}" \
  --train-root "${TRAIN_OUTPUT_ROOT}" \
  --eval-root "${EVAL_OUTPUT_ROOT}" \
  --output-csv "${REPORT_ROOT}/bridge_failure_summary.csv" \
  --output-md "${REPORT_ROOT}/bridge_failure_report.md"

echo "===== BRIDGE FAILURE REPORT ====="
cat "${REPORT_ROOT}/bridge_failure_report.md"
