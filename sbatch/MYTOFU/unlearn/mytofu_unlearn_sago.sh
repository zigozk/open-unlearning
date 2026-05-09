#!/bin/bash
#SBATCH -J mytofu_synth
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 72:00:00
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

trainers_experiments=(
  "GradDiff unlearn/mytofu/default.yaml"
  "NPO unlearn/mytofu/default.yaml"
  "SimNPO unlearn/mytofu/default.yaml"
)

synthesis_modes=(
  "none"
  "pcgrad"
  "sago"
)

echo "HOSTNAME=$(hostname)"
echo "MODEL=$MODEL"
echo "MODEL_PATH=$MODEL_PATH"
nvidia-smi || true

for trainer_experiment in "${trainers_experiments[@]}"; do
    trainer=$(echo "$trainer_experiment" | awk '{print $1}')
    experiment=$(echo "$trainer_experiment" | awk '{print $2}')

    for synth in "${synthesis_modes[@]}"; do

        task_name="mytofu_${MODEL}_${trainer}_${synth}_from_full_e10"

        echo "=================================================="
        echo "trainer=$trainer"
        echo "experiment=$experiment"
        echo "gradient_synthesis=$synth"
        echo "task_name=$task_name"
        echo "=================================================="

        extra_args=()

        extra_args+=(
          "trainer.method_args.use_retain_loss=true"
          "trainer.method_args.gradient_synthesis=${synth}"
          "trainer.method_args.retain_loss_type=NLL"
        )

        if [ "$trainer" = "NPO" ]; then
            extra_args+=(
              "trainer.method_args.beta=0.1"
              "trainer.method_args.alpha=1.0"
              "trainer.method_args.gamma=1.0"
            )
        elif [ "$trainer" = "SimNPO" ]; then
            extra_args+=(
              "trainer.method_args.delta=0.0"
              "trainer.method_args.beta=4.5"
              "trainer.method_args.alpha=1.0"
              "trainer.method_args.gamma=0.125"
            )
        elif [ "$trainer" = "GradDiff" ]; then
            extra_args+=(
              "trainer.method_args.alpha=1.0"
              "trainer.method_args.gamma=1.0"
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
          data/datasets@data.forget=MYTOFU_forget \
          data/datasets@data.retain=MYTOFU_retain \
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
done