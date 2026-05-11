#!/bin/bash
#
# Submit the paper-inspired unlearn-only tuning array and a dependent summary
# job that selects the best configuration per method.
#
# Usage:
#   bash sbatch/MYTOFU/unlearn/submit_mytofu_paper_tuned_pipeline.sh
#
# Useful overrides:
#   SWEEP_PROFILE=smoke bash sbatch/MYTOFU/unlearn/submit_mytofu_paper_tuned_pipeline.sh
#   MAX_PARALLEL=8 OUTPUT_ROOT=saves/unlearn_paper_tuned_v2 bash ...

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saves/unlearn_paper_tuned}"
SWEEP_PROFILE="${SWEEP_PROFILE:-full}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10}"
RANK_METRIC="${RANK_METRIC:-BUS}"

cd "${ROOT_DIR}"
mkdir -p logs "${OUTPUT_ROOT}"

echo "=================================================="
echo "Submitting MYTOFU paper-inspired unlearn tuning"
echo "ROOT_DIR=${ROOT_DIR}"
echo "MODEL=${MODEL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SWEEP_PROFILE=${SWEEP_PROFILE}"
echo "MAX_PARALLEL=${MAX_PARALLEL}"
echo "RANK_METRIC=${RANK_METRIC}"
echo "=================================================="

submit_output="$(
  ROOT_DIR="${ROOT_DIR}" \
  MODEL="${MODEL}" \
  MODEL_PATH="${MODEL_PATH}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  SWEEP_PROFILE="${SWEEP_PROFILE}" \
  MAX_PARALLEL="${MAX_PARALLEL}" \
  bash sbatch/MYTOFU/unlearn/mytofu_unlearn_paper_tuned_array.sh --submit
)"
echo "${submit_output}"

array_job_id="$(echo "${submit_output}" | awk '/Submitted batch job/ {print $NF}' | tail -n 1)"
if [ -z "${array_job_id}" ]; then
  echo "[ERROR] Could not parse array job id."
  exit 1
fi

summary_cmd="cd ${ROOT_DIR} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate unlearning && python sbatch/MYTOFU/unlearn/summarize_mytofu_paper_tuned.py --result-root ${OUTPUT_ROOT} --metric ${RANK_METRIC}"

summary_output="$(
  sbatch \
    --dependency=afterany:${array_job_id} \
    -J mytofu_tune_sum \
    -p compute \
    -N 1 \
    --cpus-per-task=4 \
    --mem=16G \
    -t 02:00:00 \
    -o logs/%x-%j.out \
    -e logs/%x-%j.err \
    --wrap "${summary_cmd}"
)"
echo "${summary_output}"

summary_job_id="$(echo "${summary_output}" | awk '/Submitted batch job/ {print $NF}' | tail -n 1)"

echo "=================================================="
echo "Tuning array job: ${array_job_id}"
echo "Summary job:      ${summary_job_id}"
echo "Best configs will be written under: ${OUTPUT_ROOT}"
echo "=================================================="
