#!/bin/bash
#SBATCH -J mytofu_score_sanity
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/mytofu_score_sanity_%j.out
#SBATCH -e logs/mytofu_score_sanity_%j.err

set -euo pipefail

cd /home/zkzhang/unlearning/open-unlearning
mkdir -p logs sanity_check_outputs scripts/mytofu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python sbatch/MYTOFU/sanity_check_score_mytofu.py \
  --full-model /home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10 \
  --retain-model /home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_retain_e10 \
  --forget-file /home/zkzhang/unlearning/Create_Data/mini_tofu_custom/processed/forget_train.jsonl \
  --retain-file /home/zkzhang/unlearning/Create_Data/mini_tofu_custom/processed/retain_train.jsonl \
  --output-dir /home/zkzhang/unlearning/open-unlearning/sbatch/MYTOFU/sanity_check_outputs \
  --n-forget 50 \
  --n-retain 50 \
  --seed 42 \
  --dtype bf16 \
  --device-map auto