#!/bin/bash
#SBATCH -J piper_summary
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
cd "${ROOT_DIR}"
mkdir -p logs

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-unlearning}"

export PYTHONUNBUFFERED=1

INTERVENTION_ROOT="${INTERVENTION_ROOT:-results/piper_intervention_expanded}"
EVAL_ROOT="${EVAL_ROOT:-results/piper_official_eval}"
INTERVENTION_OUTPUT="${INTERVENTION_OUTPUT:-${INTERVENTION_ROOT}/intervention_summary.csv}"
OFFICIAL_OUTPUT="${OFFICIAL_OUTPUT:-${EVAL_ROOT}/tofu_eval_summary.csv}"
RUN_INTERVENTION_SUMMARY="${RUN_INTERVENTION_SUMMARY:-1}"
RUN_OFFICIAL_SUMMARY="${RUN_OFFICIAL_SUMMARY:-1}"
REQUIRE_OFFICIAL_EVAL="${REQUIRE_OFFICIAL_EVAL:-0}"

echo "===== PIPER RESULT SUMMARY ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-manual}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "INTERVENTION_ROOT=${INTERVENTION_ROOT}"
echo "EVAL_ROOT=${EVAL_ROOT}"
echo "INTERVENTION_OUTPUT=${INTERVENTION_OUTPUT}"
echo "OFFICIAL_OUTPUT=${OFFICIAL_OUTPUT}"
echo "RUN_INTERVENTION_SUMMARY=${RUN_INTERVENTION_SUMMARY}"
echo "RUN_OFFICIAL_SUMMARY=${RUN_OFFICIAL_SUMMARY}"
echo "REQUIRE_OFFICIAL_EVAL=${REQUIRE_OFFICIAL_EVAL}"
echo "PYTHON=$(which python)"
python -V

if [ "${RUN_INTERVENTION_SUMMARY}" = "1" ]; then
  intervention_count="$(find "${INTERVENTION_ROOT}" -name summary.json 2>/dev/null | wc -l | tr -d ' ')"
  echo "[intervention] summary.json count=${intervention_count}"
  if [ "${intervention_count}" = "0" ]; then
    echo "[ERROR] No intervention summary.json files found under ${INTERVENTION_ROOT}"
    exit 2
  fi

  python experiments/piper/summarize_piper_intervention.py \
    --root "${INTERVENTION_ROOT}" \
    --output "${INTERVENTION_OUTPUT}"
  echo "[intervention] wrote ${INTERVENTION_OUTPUT}"
  wc -l "${INTERVENTION_OUTPUT}" || true
fi

if [ "${RUN_OFFICIAL_SUMMARY}" = "1" ]; then
  tofu_count="$(find "${EVAL_ROOT}" -name TOFU_SUMMARY.json 2>/dev/null | wc -l | tr -d ' ')"
  echo "[official] TOFU_SUMMARY.json count=${tofu_count}"
  if [ "${tofu_count}" = "0" ]; then
    echo "[WARN] No TOFU_SUMMARY.json files found under ${EVAL_ROOT}; official eval summary was not generated."
    echo "[hint] Check whether eval jobs finished successfully:"
    echo "       find ${EVAL_ROOT} -name TOFU_SUMMARY.json | head"
    echo "       grep -R \"ERROR\\|Traceback\\|Exception\" logs/piper* | tail -80"
    if [ "${REQUIRE_OFFICIAL_EVAL}" = "1" ]; then
      exit 3
    fi
  else
    python experiments/piper/summarize_official_tofu_eval.py \
      --eval-root "${EVAL_ROOT}" \
      --intervention-root "${INTERVENTION_ROOT}" \
      --output "${OFFICIAL_OUTPUT}"
    echo "[official] wrote ${OFFICIAL_OUTPUT}"
    wc -l "${OFFICIAL_OUTPUT}" || true
  fi
fi

echo "===== SUMMARY DONE ====="
