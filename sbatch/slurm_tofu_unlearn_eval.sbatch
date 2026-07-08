#!/usr/bin/env bash
#SBATCH -J tofu-unlearn
#SBATCH -p compute
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH -o logs/slurm/%x-%j.out
#SBATCH -e logs/slurm/%x-%j.err

set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/zkzhang/unlearn/open-unlearning}
CONDA_ENV=${CONDA_ENV:-openunlearning-official-py311}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/home/zkzhang/unlearning/HF_CACHE/datasets}
HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

MODEL=${MODEL:-Llama-3.2-1B-Instruct}
TRAINER=${TRAINER:-NPO}
EXPERIMENT=${EXPERIMENT:-}
FORGET_SPLIT=${FORGET_SPLIT:-forget10}
HOLDOUT_SPLIT=${HOLDOUT_SPLIT:-}
RETAIN_SPLIT=${RETAIN_SPLIT:-}
MODEL_PATH=${MODEL_PATH:-/home/zkzhang/models/tofu_${MODEL}_full}
TOKENIZER_PATH=${TOKENIZER_PATH:-${MODEL_PATH}}
RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH:-}
TASK_NAME=${TASK_NAME:-tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_local}
OUTPUT_DIR=${OUTPUT_DIR:-results/unlearn/${TASK_NAME}}

PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
LEARNING_RATE=${LEARNING_RATE:-}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-}
EXTRA_TRAIN_ARGS=${EXTRA_TRAIN_ARGS:-}
RUN_EVAL=${RUN_EVAL:-1}

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

if [[ -z "${EXPERIMENT}" ]]; then
  if [[ "${TRAINER}" == "DPO" ]]; then
    EXPERIMENT="unlearn/tofu/idk.yaml"
  else
    EXPERIMENT="unlearn/tofu/default.yaml"
  fi
fi

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

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  NUM_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
else
  NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
fi
if [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "No GPU visible. Submit with sbatch --gres=gpu:<gpu_type>:<count> ..." >&2
  exit 2
fi

MASTER_PORT=$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Visible GPUs: ${NUM_GPUS}"
echo "Model: ${MODEL}"
echo "Trainer: ${TRAINER}"
echo "Experiment: ${EXPERIMENT}"
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo "HF datasets offline: ${HF_DATASETS_OFFLINE}"
echo "Model path: ${MODEL_PATH}"
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Forget/Holdout/Retain: ${FORGET_SPLIT} / ${HOLDOUT_SPLIT} / ${RETAIN_SPLIT}"
echo "Retain logs: ${RETAIN_LOGS_PATH}"
echo "Task name: ${TASK_NAME}"
echo "Output dir: ${OUTPUT_DIR}"
nvidia-smi || true

[[ -d "${MODEL_PATH}" ]] || { echo "Missing MODEL_PATH=${MODEL_PATH}" >&2; exit 2; }
[[ -d "${TOKENIZER_PATH}" ]] || { echo "Missing TOKENIZER_PATH=${TOKENIZER_PATH}" >&2; exit 2; }
[[ -f "${RETAIN_LOGS_PATH}" ]] || { echo "Missing RETAIN_LOGS_PATH=${RETAIN_LOGS_PATH}" >&2; exit 2; }

train_cmd=(
  accelerate launch
  --config_file configs/accelerate/default_config.yaml
  --num_processes "${NUM_GPUS}"
  --main_process_port "${MASTER_PORT}"
  src/train.py --config-name=unlearn.yaml
  experiment="${EXPERIMENT}"
  trainer="${TRAINER}"
  task_name="${TASK_NAME}"
  model="${MODEL}"
  forget_split="${FORGET_SPLIT}"
  retain_split="${RETAIN_SPLIT}"
  model.model_args.pretrained_model_name_or_path="${MODEL_PATH}"
  model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}"
  retain_logs_path="${RETAIN_LOGS_PATH}"
  paths.output_dir="${OUTPUT_DIR}"
  trainer.args.per_device_train_batch_size="${PER_DEVICE_TRAIN_BATCH_SIZE}"
  trainer.args.gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS}"
  trainer.args.gradient_checkpointing=true
  trainer.args.ddp_find_unused_parameters=true
)

if [[ -n "${LEARNING_RATE}" ]]; then
  train_cmd+=(trainer.args.learning_rate="${LEARNING_RATE}")
fi
if [[ -n "${NUM_TRAIN_EPOCHS}" ]]; then
  train_cmd+=(trainer.args.num_train_epochs="${NUM_TRAIN_EPOCHS}")
fi
if [[ -n "${EXTRA_TRAIN_ARGS}" ]]; then
  # Space-separated Hydra overrides, e.g. EXTRA_TRAIN_ARGS='trainer.method_args.beta=0.5 trainer.method_args.alpha=1.0'.
  read -r -a extra_args <<< "${EXTRA_TRAIN_ARGS}"
  train_cmd+=("${extra_args[@]}")
fi

echo "Running train: ${train_cmd[*]}"
"${train_cmd[@]}"

if [[ "${RUN_EVAL}" == "1" ]]; then
  eval_cmd=(
    python src/eval.py --config-name=eval.yaml
    experiment=eval/tofu/default.yaml
    model="${MODEL}"
    forget_split="${FORGET_SPLIT}"
    holdout_split="${HOLDOUT_SPLIT}"
    task_name="${TASK_NAME}_eval"
    model.model_args.pretrained_model_name_or_path="${OUTPUT_DIR}"
    model.tokenizer_args.pretrained_model_name_or_path="${TOKENIZER_PATH}"
    paths.output_dir="${OUTPUT_DIR}/evals"
    retain_logs_path="${RETAIN_LOGS_PATH}"
  )

  echo "Running eval: ${eval_cmd[*]}"
  "${eval_cmd[@]}"
else
  echo "Skipping eval because RUN_EVAL=${RUN_EVAL}."
fi
