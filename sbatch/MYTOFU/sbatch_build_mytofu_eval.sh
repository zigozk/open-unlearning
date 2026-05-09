#!/bin/bash
#SBATCH -J mytofu_eval_v1
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/mytofu_eval_v1_%j.out
#SBATCH -e logs/mytofu_eval_v1_%j.err

set -euo pipefail

cd /home/zkzhang/unlearning/open-unlearning
mkdir -p logs
mkdir -p /home/zkzhang/unlearning/open-unlearning/data/mytofu_eval_v1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export PYTHONUNBUFFERED=1

python scripts/mytofu/build_mytofu_eval_v1.py \
  --forget-file /home/zkzhang/unlearning/Create_Data/mini_tofu_custom/processed/forget_train.jsonl \
  --retain-file /home/zkzhang/unlearning/Create_Data/mini_tofu_custom/processed/retain_train.jsonl \
  --out-dir /home/zkzhang/unlearning/open-unlearning/data/mytofu_eval_v1 \
  --seed 42 \
  --num-perturb 4 \
  --single-fact-only