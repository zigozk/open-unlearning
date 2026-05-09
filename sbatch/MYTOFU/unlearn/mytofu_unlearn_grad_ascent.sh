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

MODEL_PATH=/home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10

echo "HOSTNAME=$(hostname)"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi || true

python src/train.py \
  --config-name=unlearn.yaml \
  experiment=unlearn/mytofu/grad_ascent.yaml \
  task_name=mytofu_Llama-3.2-1B-Instruct_GradAscent_from_full_e10 \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=/home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10 \
  model.tokenizer_args.pretrained_model_name_or_path=/home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10 \
  data/datasets@data.forget=MYTOFU_forget \
  data/datasets@data.retain=MYTOFU_retain \
  trainer=GradAscent \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.learning_rate=1e-5 \
  trainer.args.num_train_epochs=3 \
  trainer.args.gradient_checkpointing=true