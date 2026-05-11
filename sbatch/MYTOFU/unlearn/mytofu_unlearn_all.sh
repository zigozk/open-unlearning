#!/bin/bash
#SBATCH -J mytofu_all
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 48:00:00
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

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4

RUN_FINAL_EVAL=1

# 跟官方 tofu_unlearn.sh 一样：方法 + 对应 experiment
# DPO 走 idk.yaml；其余走 default.yaml
trainers_experiments=(
  # "GradAscent unlearn/mytofu/default.yaml"
  # "GradDiff unlearn/mytofu/default.yaml"
  # "NPO unlearn/mytofu/default.yaml"
  # "SimNPO unlearn/mytofu/default.yaml"
  # "RMU unlearn/mytofu/default.yaml"
  # "UNDIAL unlearn/mytofu/default.yaml"
  "DPO unlearn/mytofu/idk.yaml"
  # "CEU unlearn/mytofu/default.yaml"
  # "PDU unlearn/mytofu/default.yaml"
  # "SatImp unlearn/mytofu/default.yaml"
  # "WGA unlearn/mytofu/default.yaml"
)

echo "HOSTNAME=$(hostname)"
echo "MODEL=$MODEL"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi || true

for trainer_experiment in "${trainers_experiments[@]}"; do
    trainer=$(echo "$trainer_experiment" | awk '{print $1}')
    experiment=$(echo "$trainer_experiment" | awk '{print $2}')

    task_name="mytofu_${MODEL}_${trainer}_from_full_e10"

    echo "=================================================="
    echo "trainer=$trainer"
    echo "experiment=$experiment"
    echo "task_name=$task_name"
    echo "=================================================="

    extra_args=()

    # 显式指定 forget 数据集，避免沿用默认 TOFU 配置
    # DPO 使用 idk 版本；其他方法使用普通 forget 数据
    if [ "$trainer" = "DPO" ]; then
        extra_args+=(
          "data/datasets@data.forget=MYTOFU_forget_idk"
        )
    else
        extra_args+=(
          "data/datasets@data.forget=MYTOFU_forget"
        )
    fi

    # 所有方法都显式指定 retain 数据集
    extra_args+=(
      "data/datasets@data.retain=MYTOFU_retain"
    )

    # PDU 额外参数
    if [ "$trainer" = "PDU" ]; then
        extra_args+=(
          "trainer.method_args.retain_loss_eps=0.1"
        )
    fi

    python src/train.py \
      --config-name=unlearn.yaml \
      experiment=${experiment} \
      trainer=${trainer} \
      task_name=${task_name} \
      model=${MODEL} \
      model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
      model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH} \
      trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
      trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
      trainer.args.learning_rate=1e-5 \
      trainer.args.num_train_epochs=3 \
      trainer.args.gradient_checkpointing=true \
      ++trainer.args.gradient_checkpointing_kwargs.use_reentrant=false \
      ++trainer.args.report_to=none \
      "${extra_args[@]}"

    if [ "${RUN_FINAL_EVAL}" = "1" ]; then
        python src/eval.py \
          --config-name=eval.yaml \
          experiment=eval/mytofu/default.yaml \
          model=${MODEL} \
          task_name=${task_name} \
          model.model_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
          model.tokenizer_args.pretrained_model_name_or_path=saves/unlearn/${task_name} \
          paths.output_dir=saves/unlearn/${task_name}/evals_final
    fi

done