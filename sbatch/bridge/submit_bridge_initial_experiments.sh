#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/zkzhang/unlearn/open-unlearning}
MODEL=${MODEL:-Llama-3.2-1B-Instruct}
FORGET_SPLIT=${FORGET_SPLIT:-forget01}
RUN_TAG=${RUN_TAG:-bridge_initial_$(date +%Y%m%d_%H%M%S)}
SBATCH_SCRIPT=${SBATCH_SCRIPT:-sbatch/slurm_tofu_unlearn_eval.sbatch}
METHODS=${METHODS:-NPO_BRIDGE_GlobalKL NPO_BRIDGE_UniformDRO NPO_BRIDGE_HistoryDRO NPO_BRIDGE_GSDRO NPO_BRIDGE_KLPIDRO}
RUN_EVAL=${RUN_EVAL:-1}
HISTORY_COLLATOR=${HISTORY_COLLATOR:-DataCollatorForSupervisedDatasetwithIndex}
GS_KLPI_EXTRA_ARGS=${GS_KLPI_EXTRA_ARGS:-trainer.args.gradient_checkpointing=false}
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  sbatch/bridge/submit_bridge_initial_experiments.sh [--dry-run] <sbatch resource/options...>

Required:
  Pass the GPU request explicitly, e.g. --gres=gpu:a100-sxm4-80gb:1
  Pass the wall time explicitly, e.g. --time=06:00:00

Defaults:
  MODEL=Llama-3.2-1B-Instruct
  FORGET_SPLIT=forget01
  METHODS="NPO_BRIDGE_GlobalKL NPO_BRIDGE_UniformDRO NPO_BRIDGE_HistoryDRO NPO_BRIDGE_GSDRO NPO_BRIDGE_KLPIDRO"
  RUN_TAG=bridge_initial_<timestamp>
  RUN_EVAL=1

Notes:
  History-DRO automatically uses COLLATOR=DataCollatorForSupervisedDatasetwithIndex.
  GS-DRO and KL-PI-DRO automatically append trainer.args.gradient_checkpointing=false.

Examples:
  sbatch/bridge/submit_bridge_initial_experiments.sh \
    --gres=gpu:a100-sxm4-80gb:1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=06:00:00

  RUN_TAG=bridge_forget01_v0 METHODS="NPO_BRIDGE_UniformDRO NPO_BRIDGE_GSDRO" \
  sbatch/bridge/submit_bridge_initial_experiments.sh \
    --gres=gpu:a100-sxm4-80gb:1 \
    --cpus-per-task=8 \
    --mem=96G \
    --time=06:00:00

Optional environment overrides:
  MODEL=Llama-3.2-1B-Instruct
  FORGET_SPLIT=forget05
  RUN_TAG=my_bridge_run
  METHODS="NPO_BRIDGE_GlobalKL NPO_BRIDGE_UniformDRO"
  LEARNING_RATE=1e-5 NUM_TRAIN_EPOCHS=10
  EXTRA_TRAIN_ARGS='trainer.method_args.bridge_lambda_g=0.03 trainer.method_args.bridge_lambda_b=0.1'
  GS_KLPI_EXTRA_ARGS='trainer.args.gradient_checkpointing=false trainer.method_args.bridge_virtual_step_size=1e-5'
EOF
}

args=()
has_gres=0
has_time=0
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
      args+=("$1")
      shift
      ;;
    -t=*|--time=*)
      has_time=1
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
  echo "Missing wall time. Pass --time=HH:MM:SS on the command line." >&2
  echo >&2
  usage >&2
  exit 2
fi

cd "${REPO_ROOT}"
[[ -f "${SBATCH_SCRIPT}" ]] || { echo "Missing ${SBATCH_SCRIPT}" >&2; exit 2; }
mkdir -p logs/slurm

echo "Submitting BRIDGE initial TOFU forget+eval jobs"
echo "Model: ${MODEL}"
echo "Forget split: ${FORGET_SPLIT}"
echo "Run tag: ${RUN_TAG}"
echo "Methods: ${METHODS}"
echo "SBATCH options: ${args[*]}"
echo

for trainer in ${METHODS}; do
  task_name="tofu_${MODEL}_${FORGET_SPLIT}_${trainer}_${RUN_TAG}"
  export_vars="ALL,MODEL=${MODEL},TRAINER=${trainer},FORGET_SPLIT=${FORGET_SPLIT},TASK_NAME=${task_name},RUN_EVAL=${RUN_EVAL}"

  computed_collator="${COLLATOR:-}"
  computed_extra_args="${EXTRA_TRAIN_ARGS:-}"

  case "${trainer}" in
    *HistoryDRO*)
      computed_collator="${computed_collator:-${HISTORY_COLLATOR}}"
      ;;
  esac

  case "${trainer}" in
    *GSDRO*|*KLPIDRO*)
      if [[ -n "${GS_KLPI_EXTRA_ARGS}" ]]; then
        computed_extra_args="${computed_extra_args:+${computed_extra_args} }${GS_KLPI_EXTRA_ARGS}"
      fi
      ;;
  esac

  for name in LEARNING_RATE NUM_TRAIN_EPOCHS PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS HF_DATASETS_CACHE HF_DATASETS_OFFLINE CUDA_MODULE; do
    if [[ -n "${!name:-}" ]]; then
      export_vars+=",${name}=${!name}"
    fi
  done
  if [[ -n "${computed_collator}" ]]; then
    export_vars+=",COLLATOR=${computed_collator}"
  fi
  if [[ -n "${computed_extra_args}" ]]; then
    export_vars+=",EXTRA_TRAIN_ARGS=${computed_extra_args}"
  fi

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
