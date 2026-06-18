#!/bin/bash
#SBATCH -J mytofu_base
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
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

MODEL="Llama-3.2-1B-Instruct"
MODEL_PATH="/home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10"
TASK_NAME="mytofu_${MODEL}_full_e10_baseline"

echo "HOSTNAME=$(hostname)"
echo "MODEL=$MODEL"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi || true

python src/eval.py \
  --config-name=eval.yaml \
  experiment=eval/mytofu/default.yaml \
  model=${MODEL} \
  task_name=${TASK_NAME} \
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
  model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH} \
  paths.output_dir=saves/eval/${TASK_NAME}/evals_final

echo "===== BASELINE EVAL DONE ====="