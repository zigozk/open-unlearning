#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/zkzhang/unlearn/open-unlearning}
MODEL=${MODEL:-Llama-2-7b-chat-hf}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-sbatch/slurm_tofu_unlearn_eval.sbatch}
METHODS=${METHODS:-NPO SimNPO GradAscent GradDiff}
RUN_EVAL=${RUN_EVAL:-1}
MIN_LLAMA2_WALLTIME_HOURS=${MIN_LLAMA2_WALLTIME_HOURS:-4}
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  sbatch/submit_llama2_7b_four_methods.sh [--dry-run] <sbatch resource/options...>

Required:
  Pass the GPU request explicitly, e.g. --gres=gpu:a100-sxm4-80gb:2
  Pass the wall time explicitly, e.g. --time=24:00:00

Defaults:
  MODEL=Llama-2-7b-chat-hf
  FORGET_SPLIT=forget10
  METHODS="NPO SimNPO GradAscent GradDiff"
  RUN_TAG=<current timestamp>
  RUN_EVAL=1

Examples:
  sbatch/submit_llama2_7b_four_methods.sh \
    --gres=gpu:a100-sxm4-80gb:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=24:00:00

  sbatch/submit_llama2_7b_four_methods.sh \
    -w gpu06 \
    --gres=gpu:a100-sxm4-80gb:1 \
    --cpus-per-task=8 \
    --mem=128G \
    --time=24:00:00

Optional environment overrides:
  RUN_TAG=my_run_001
  FORGET_SPLIT=forget05
  METHODS="NPO SimNPO"
  MIN_LLAMA2_WALLTIME_HOURS=4
  ALLOW_SHORT_TIME=1
  LEARNING_RATE=2e-5 NUM_TRAIN_EPOCHS=10
  EXTRA_TRAIN_ARGS='trainer.method_args.beta=0.5 trainer.method_args.alpha=1.0'
EOF
}

time_to_seconds() {
  python - "$1" <<'PY'
import re
import sys

value = sys.argv[1]
days = 0
if "-" in value:
    day_part, value = value.split("-", 1)
    days = int(day_part)

parts = value.split(":")
if len(parts) == 3:
    hours, minutes, seconds = map(int, parts)
elif len(parts) == 2:
    hours = 0
    minutes, seconds = map(int, parts)
elif len(parts) == 1 and re.fullmatch(r"\d+", parts[0]):
    hours = 0
    minutes = int(parts[0])
    seconds = 0
else:
    raise SystemExit(f"Unsupported --time format: {sys.argv[1]}")

print(days * 86400 + hours * 3600 + minutes * 60 + seconds)
PY
}

args=()
has_gres=0
has_time=0
time_value=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --gres|--gpus)
      has_gres=1
      args+=("$1")
      shift
      [[ $# -gt 0 ]] || { echo "Missing value after ${args[-1]}" >&2; exit 2; }
      args+=("$1")
      shift
      ;;
    --gres=*|--gpus=*)
      has_gres=1
      args+=("$1")
      shift
      ;;
    -t|--time)
      has_time=1
      args+=("$1")
      shift
      [[ $# -gt 0 ]] || { echo "Missing value after ${args[-1]}" >&2; exit 2; }
      time_value="$1"
      args+=("$1")
      shift
      ;;
    -t=*|--time=*)
      has_time=1
      time_value="${1#*=}"
      args+=("$1")
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ "${has_gres}" -ne 1 ]]; then
  echo "Missing GPU request. Pass --gres=gpu:<gpu_type>:<count> on the command line." >&2
  echo >&2
  usage >&2
  exit 2
fi
if [[ "${has_time}" -ne 1 ]]; then
  echo "Missing wall time. Pass --time=HH:MM:SS on the command line to avoid over-requesting." >&2
  echo >&2
  usage >&2
  exit 2
fi
if [[ "${MODEL}" == "Llama-2-7b-chat-hf" && "${ALLOW_SHORT_TIME:-0}" != "1" ]]; then
  if grep -Eq '(^| )(NPO|GradDiff)( |$)' <<< "${METHODS}"; then
    requested_seconds=$(time_to_seconds "${time_value}")
    min_seconds=$((MIN_LLAMA2_WALLTIME_HOURS * 3600))
    if [[ "${requested_seconds}" -lt "${min_seconds}" ]]; then
      echo "Requested --time=${time_value} is too short for Llama-2-7b NPO/GradDiff." >&2
      echo "The previous 2-hour run timed out before saving/eval; use at least ${MIN_LLAMA2_WALLTIME_HOURS}:00:00 or set ALLOW_SHORT_TIME=1." >&2
      exit 2
    fi
  fi
fi

cd "${REPO_ROOT}"
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "Missing ${SBATCH_SCRIPT}" >&2; exit 2; }
mkdir -p logs/slurm

echo "Submitting Llama-2-7b TOFU forget+eval jobs"
echo "Model: ${MODEL}"
echo "Forget split: ${FORGET_SPLIT}"
echo "Run tag: ${RUN_TAG}"
echo "Methods: ${METHODS}"
echo "SBATCH options: ${args[*]}"
echo

for trainer in ${METHODS}; do
  task_name="tofu_${MODEL}_${FORGET_SPLIT}_${trainer}_${RUN_TAG}"
  export_vars="ALL,MODEL=${MODEL},TRAINER=${trainer},FORGET_SPLIT=${FORGET_SPLIT},TASK_NAME=${task_name},RUN_EVAL=${RUN_EVAL}"
  for name in LEARNING_RATE NUM_TRAIN_EPOCHS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS EXTRA_TRAIN_ARGS HF_DATASETS_CACHE HF_DATASETS_OFFLINE CUDA_MODULE; do
    if [[ -n "${!name:-}" ]]; then
      export_vars+=",${name}=${!name}"
    fi
  done

  cmd=(
    sbatch
    "${args[@]}"
    --export="${export_vars}"
    "${SBATCH_SCRIPT}"
  )

  echo "Task: ${task_name}"
  echo "Output: results/unlearn/${task_name}"
  echo "Command: ${cmd[*]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run: not submitted."
  else
    "${cmd[@]}"
  fi
  echo
done
