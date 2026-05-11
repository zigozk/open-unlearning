#!/bin/bash
#SBATCH -J mytofu_ga
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_a800_80gb_pcie:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /home/zkzhang/unlearning/open-unlearning

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset HF_MODULES_CACHE
unset HF_DATASETS_CACHE

MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saves/unlearn}"

# GradAscent is easy to overfit and destroy utility on MYTOFU. Use a cautious
# one-epoch default; the tuned sweep script includes lr1e-6 and lr2e-6 variants.
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"

TAG="${TAG:-lr1em6_e1}"
TASK_NAME="${TASK_NAME:-mytofu_${MODEL}_GradAscent_${TAG}_from_full_e10}"
RUN_DIR="${OUTPUT_ROOT}/${TASK_NAME}"

echo "=================================================="
echo "HOSTNAME=$(hostname)"
echo "MODEL=${MODEL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "TASK_NAME=${TASK_NAME}"
echo "RUN_DIR=${RUN_DIR}"
echo "=================================================="
nvidia-smi || true
mkdir -p logs "${OUTPUT_ROOT}"

python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/mytofu/grad_ascent.yaml \
  trainer=GradAscent \
  task_name=${TASK_NAME} \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
  model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH} \
  data/datasets@data.forget=MYTOFU_forget \
  data/datasets@data.retain=MYTOFU_retain \
  paths.output_dir=${RUN_DIR} \
  trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  trainer.args.learning_rate=${LEARNING_RATE} \
  trainer.args.num_train_epochs=${NUM_TRAIN_EPOCHS} \
  trainer.args.gradient_checkpointing=true \
  ++trainer.args.gradient_checkpointing_kwargs.use_reentrant=false \
  ++trainer.args.report_to=none

if [ "${RUN_FINAL_EVAL}" = "1" ]; then
  python src/eval.py \
    --config-name=eval.yaml \
    experiment=eval/mytofu/default.yaml \
    model=${MODEL} \
    task_name=${TASK_NAME} \
    model.model_args.pretrained_model_name_or_path=${RUN_DIR} \
    model.tokenizer_args.pretrained_model_name_or_path=${RUN_DIR} \
    paths.output_dir=${RUN_DIR}/evals_final
fi

echo "===== MYTOFU GRAD ASCENT DONE ====="
