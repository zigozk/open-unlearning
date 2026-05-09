#!/bin/bash
#SBATCH -J mytofu_full
#SBATCH -o logs/mytofu_full.%j.out
#SBATCH -e logs/mytofu_full.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 12:00:00

set -euo pipefail

mkdir -p logs

echo "===== JOB INFO ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
date

cd /home/zkzhang/unlearning/open-unlearning

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_MODULES_CACHE=$HF_HOME/modules

# 如 GPU 节点离线，打开下面三行
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

# 参考官方脚本，随机找一个空闲端口
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "MASTER_PORT=$MASTER_PORT"

echo "===== ENV CHECK ====="
python -V
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/mytofu/full_e10.yaml \
  task_name=mytofu_Llama-3.2-1B-Instruct_full_e10 \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=/home/share/models/Llama-3.2-1B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=/home/share/models/Llama-3.2-1B-Instruct \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.learning_rate=2e-5 \
  trainer.args.num_train_epochs=10 \
  trainer.args.gradient_checkpointing=true

date
echo "===== FULL TRAIN DONE ====="