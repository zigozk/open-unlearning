#!/bin/bash
#SBATCH -J baseline_metric_audit
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
cd "${ROOT_DIR}"
mkdir -p logs

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-unlearning}"

export PYTHONUNBUFFERED=1

LEFTOVERS_ROOT="${LEFTOVERS_ROOT:-${ROOT_DIR}/server_leftovers_20260624}"
SUMMARY_CSV="${SUMMARY_CSV:-${ROOT_DIR}/legacy_thesis_work/tofu_eval_summary.csv}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/legacy_thesis_work/baseline_metric_audit_hpc_${SLURM_JOB_ID:-manual}}"
RETAIN_REFERENCE_EVAL="${RETAIN_REFERENCE_EVAL:-}"

FORGET_COLLAPSE_THRESHOLD="${FORGET_COLLAPSE_THRESHOLD:-0.05}"
FORGET_COLLAPSE_VOTES="${FORGET_COLLAPSE_VOTES:-2}"
FQ_HIGH_THRESHOLD="${FQ_HIGH_THRESHOLD:-0.5}"
MU_LOW_THRESHOLD="${MU_LOW_THRESHOLD:-0.2}"
TOP_K="${TOP_K:-30}"

echo "===== BASELINE METRIC RELIABILITY AUDIT ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "LEFTOVERS_ROOT=${LEFTOVERS_ROOT}"
echo "SUMMARY_CSV=${SUMMARY_CSV}"
echo "OUT_DIR=${OUT_DIR}"
echo "RETAIN_REFERENCE_EVAL=${RETAIN_REFERENCE_EVAL}"
echo "FORGET_COLLAPSE_THRESHOLD=${FORGET_COLLAPSE_THRESHOLD}"
echo "FORGET_COLLAPSE_VOTES=${FORGET_COLLAPSE_VOTES}"
echo "FQ_HIGH_THRESHOLD=${FQ_HIGH_THRESHOLD}"
echo "MU_LOW_THRESHOLD=${MU_LOW_THRESHOLD}"
echo "TOP_K=${TOP_K}"
echo "PYTHON=$(which python)"
python -V

if [ ! -d "${LEFTOVERS_ROOT}" ]; then
  echo "[WARN] LEFTOVERS_ROOT does not exist: ${LEFTOVERS_ROOT}"
  echo "[WARN] The job will still run if SUMMARY_CSV exists, but raw per-index retain analysis will be skipped."
fi

summary_count="0"
eval_count="0"
if [ -d "${LEFTOVERS_ROOT}" ]; then
  summary_count="$(
    find "${LEFTOVERS_ROOT}" \
      \( -name TOFU_SUMMARY.json -o -name MUSE_SUMMARY.json -o -name MYTOFU_SUMMARY.json \) \
      2>/dev/null | wc -l | tr -d ' '
  )"
  eval_count="$(
    find "${LEFTOVERS_ROOT}" \
      \( -name TOFU_EVAL.json -o -name MUSE_EVAL.json -o -name MYTOFU_EVAL.json \) \
      2>/dev/null | wc -l | tr -d ' '
  )"
fi
echo "SUMMARY_JSON_COUNT=${summary_count}"
echo "EVAL_JSON_COUNT=${eval_count}"

cmd=(
  python experiments/analyze_baseline_metric_reliability.py
  --out-dir "${OUT_DIR}"
  --forget-collapse-threshold "${FORGET_COLLAPSE_THRESHOLD}"
  --forget-collapse-votes "${FORGET_COLLAPSE_VOTES}"
  --fq-high-threshold "${FQ_HIGH_THRESHOLD}"
  --mu-low-threshold "${MU_LOW_THRESHOLD}"
  --top-k "${TOP_K}"
)

if [ -f "${SUMMARY_CSV}" ]; then
  cmd+=(--summary-csv "${SUMMARY_CSV}")
else
  echo "[WARN] SUMMARY_CSV not found: ${SUMMARY_CSV}"
fi

if [ -d "${LEFTOVERS_ROOT}" ]; then
  cmd+=(--results-root "${LEFTOVERS_ROOT}")
fi

if [ -n "${RETAIN_REFERENCE_EVAL}" ]; then
  if [ ! -f "${RETAIN_REFERENCE_EVAL}" ]; then
    echo "[ERROR] RETAIN_REFERENCE_EVAL was set but the file does not exist: ${RETAIN_REFERENCE_EVAL}"
    exit 2
  fi
  cmd+=(--retain-reference-eval "${RETAIN_REFERENCE_EVAL}")
fi

echo "Command:"
printf '  %q' "${cmd[@]}"
echo

"${cmd[@]}"

echo "===== OUTPUT FILES ====="
find "${OUT_DIR}" -maxdepth 1 -type f -printf "%f\t%k KB\n" | sort || true

report="${OUT_DIR}/baseline_metric_reliability_report.md"
if [ -f "${report}" ]; then
  echo "===== REPORT PREVIEW ====="
  sed -n '1,120p' "${report}"
fi

echo "===== AUDIT DONE ====="
