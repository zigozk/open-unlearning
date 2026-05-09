#!/bin/bash
#SBATCH -J mytofu_retain
#SBATCH -o logs/mytofu_retain.%j.out
#SBATCH -e logs/mytofu_retain.%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00

set -euo pipefail

echo "===== JOB INFO ====="
echo "JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
date

mkdir -p logs

cd /home/zkzhang/unlearning/open-unlearning

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

# 可选：如果环境里需要 module 才能拿到 nvcc/cuda
# . /usr/share/modules/init/bash
# module use --append /home/share/modules/modulefiles
# module load cuda/12.1.0

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules

# 如果 GPU 节点离线，就打开下面三行
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

echo "===== ENV CHECK ====="
python -V
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 训练 retain model
python src/train.py \
  --config-name=train.yaml \
  experiment=finetune/mytofu/retain_e10.yaml \
  task_name=mytofu_Llama-3.2-1B-Instruct_retain_e10 \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=/home/share/models/Llama-3.2-1B-Instruct \
  model.tokenizer_args.pretrained_model_name_or_path=/home/share/models/Llama-3.2-1B-Instruct \
  trainer=finetune \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=8 \
  trainer.args.learning_rate=2e-5 \
  trainer.args.num_train_epochs=10

date
echo "===== RETAIN TRAIN DONE ====="