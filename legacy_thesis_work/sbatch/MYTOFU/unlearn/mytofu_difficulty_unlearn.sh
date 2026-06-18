#!/bin/bash
#SBATCH -J mytofu_diff
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 120:00:00
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

DATA_ROOT="/home/zkzhang/unlearning/Create_Data"

PER_DEVICE_TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
LEARNING_RATE=1e-5
NUM_TRAIN_EPOCHS=3

RUN_FINAL_EVAL=1

# 默认跑 4 个数据集 × 4 个方法 = 16 组实验
# 如后续想补 NPO_sago，改为 1，会额外跑 4 组
RUN_NPO_SAGO=0

# 三列分别是：
# 1. split_name：用于 task_name
# 2. dataset_dir：Create_Data 下的数据集目录名
# 3. save_root：open-unlearning 下的保存根目录
datasets=(
  "easy mini_tofu_custom_slot_easy saves/unlearn_easy"
  "medium mini_tofu_custom_slot_medium saves/unlearn_medium"
  "hard mini_tofu_custom_slot_hard saves/unlearn_hard"
  "hard_plus mini_tofu_custom_slot_hard_plus saves/unlearn_hard_plus"
)

# 四个主补充实验方法
methods=(
  "GradDiff GradDiff unlearn/mytofu/default.yaml standard"
  "NPO NPO unlearn/mytofu/default.yaml standard"
  "CEU CEU unlearn/mytofu/default.yaml standard"
  "RMU RMU unlearn/mytofu/default.yaml standard"
)

# 可选：额外加入 NPO_sago
if [ "${RUN_NPO_SAGO}" = "1" ]; then
  methods+=(
    "NPO_sago NPO unlearn/mytofu/default.yaml npo_sago"
  )
fi

echo "=================================================="
echo "HOSTNAME=$(hostname)"
echo "MODEL=${MODEL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "LEARNING_RATE=${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "RUN_FINAL_EVAL=${RUN_FINAL_EVAL}"
echo "RUN_NPO_SAGO=${RUN_NPO_SAGO}"
echo "=================================================="
nvidia-smi || true

mkdir -p logs
mkdir -p configs/data/datasets_backup_before_difficulty

backup_one() {
  local f="$1"
  if [ -f "configs/data/datasets/${f}" ]; then
    cp "configs/data/datasets/${f}" \
       "configs/data/datasets_backup_before_difficulty/${f}.$(date +%Y%m%d_%H%M%S).bak"
  fi
}

# 备份原 MYTOFU 数据集配置，避免临时重写后找不回
backup_one "MYTOFU_forget.yaml"
backup_one "MYTOFU_retain.yaml"
backup_one "MYTOFU_full.yaml"
backup_one "MYTOFU_forget_eval.yaml"
backup_one "MYTOFU_retain_eval.yaml"
backup_one "MYTOFU_forget_eval_perturbed.yaml"
backup_one "MYTOFU_retain_eval_perturbed.yaml"
backup_one "MYTOFU_forget_para.yaml"
backup_one "MYTOFU_forget_pert.yaml"
backup_one "MYTOFU_retain_para.yaml"
backup_one "MYTOFU_retain_pert.yaml"

write_dataset_configs() {
  local dataset_dir="$1"

  local base="${DATA_ROOT}/${dataset_dir}"

  local forget_train="${base}/processed/forget_train.jsonl"
  local retain_train="${base}/processed/retain_train.jsonl"
  local full_train="${base}/processed/full_train.jsonl"

  local forget_eval="${base}/eval/forget_eval_perturbed.jsonl"
  local retain_eval="${base}/eval/retain_eval_perturbed.jsonl"

  echo "--------------------------------------------------"
  echo "Writing MYTOFU dataset configs"
  echo "dataset_dir=${dataset_dir}"
  echo "base=${base}"
  echo "forget_train=${forget_train}"
  echo "retain_train=${retain_train}"
  echo "full_train=${full_train}"
  echo "forget_eval=${forget_eval}"
  echo "retain_eval=${retain_eval}"
  echo "--------------------------------------------------"

  for f in "${forget_train}" "${retain_train}" "${full_train}" "${forget_eval}" "${retain_eval}"; do
    if [ ! -f "$f" ]; then
      echo "[ERROR] missing required file: $f"
      exit 1
    fi
  done

  cat > configs/data/datasets/MYTOFU_forget.yaml <<EOF
MYTOFU_forget:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${forget_train}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_retain.yaml <<EOF
MYTOFU_retain:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${retain_train}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_full.yaml <<EOF
MYTOFU_full:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${full_train}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_forget_eval.yaml <<EOF
MYTOFU_forget_eval:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${forget_eval}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_retain_eval.yaml <<EOF
MYTOFU_retain_eval:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${retain_eval}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_forget_eval_perturbed.yaml <<EOF
MYTOFU_forget_eval_perturbed:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${forget_eval}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_retain_eval_perturbed.yaml <<EOF
MYTOFU_retain_eval_perturbed:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${retain_eval}
      split: train
    question_key: question
    answer_key: answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_forget_para.yaml <<EOF
MYTOFU_forget_para:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${forget_eval}
      split: train
    question_key: question
    answer_key: paraphrased_answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_forget_pert.yaml <<EOF
MYTOFU_forget_pert:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${forget_eval}
      split: train
    question_key: question
    answer_key: perturbed_answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_retain_para.yaml <<EOF
MYTOFU_retain_para:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${retain_eval}
      split: train
    question_key: question
    answer_key: paraphrased_answer
    max_length: 512
EOF

  cat > configs/data/datasets/MYTOFU_retain_pert.yaml <<EOF
MYTOFU_retain_pert:
  handler: QADataset
  args:
    hf_args:
      path: json
      data_files: ${retain_eval}
      split: train
    question_key: question
    answer_key: perturbed_answer
    max_length: 512
EOF
}

for dataset_item in "${datasets[@]}"; do
  split_name=$(echo "$dataset_item" | awk '{print $1}')
  dataset_dir=$(echo "$dataset_item" | awk '{print $2}')
  save_root=$(echo "$dataset_item" | awk '{print $3}')

  mkdir -p "${save_root}"

  echo "##################################################"
  echo "DATASET SPLIT: ${split_name}"
  echo "DATASET DIR:   ${dataset_dir}"
  echo "SAVE ROOT:     ${save_root}"
  echo "##################################################"

  write_dataset_configs "${dataset_dir}"

  for method_item in "${methods[@]}"; do
    method_label=$(echo "$method_item" | awk '{print $1}')
    trainer=$(echo "$method_item" | awk '{print $2}')
    experiment=$(echo "$method_item" | awk '{print $3}')
    mode=$(echo "$method_item" | awk '{print $4}')

    task_name="mytofu_${split_name}_${MODEL}_${method_label}_from_full_e10"
    run_dir="${save_root}/${task_name}"

    echo "=================================================="
    echo "split_name=${split_name}"
    echo "dataset_dir=${dataset_dir}"
    echo "save_root=${save_root}"
    echo "run_dir=${run_dir}"
    echo "method_label=${method_label}"
    echo "trainer=${trainer}"
    echo "experiment=${experiment}"
    echo "mode=${mode}"
    echo "task_name=${task_name}"
    echo "=================================================="

    extra_args=()

    if [ "$mode" = "npo_sago" ]; then
      extra_args+=(
        "trainer.method_args.use_retain_loss=true"
        "trainer.method_args.gradient_synthesis=sago"
        "trainer.method_args.retain_loss_type=NLL"
        "trainer.method_args.beta=0.1"
        "trainer.method_args.alpha=1.0"
        "trainer.method_args.gamma=1.0"
      )
    fi

    # 如果重复运行同一组实验，run_dir 已存在时会继续写入同名目录。
    # 正式重跑前建议手动备份或删除该目录。
    if [ -d "${run_dir}" ]; then
      echo "[WARN] run_dir already exists: ${run_dir}"
      echo "[WARN] This run may overwrite or mix with existing outputs."
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
      paths.output_dir=${run_dir} \
      trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
      trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
      trainer.args.learning_rate=${LEARNING_RATE} \
      trainer.args.num_train_epochs=${NUM_TRAIN_EPOCHS} \
      trainer.args.gradient_checkpointing=false \
      ++trainer.args.report_to=none \
      "${extra_args[@]}"

    if [ "${RUN_FINAL_EVAL}" = "1" ]; then
      python src/eval.py \
        --config-name=eval.yaml \
        experiment=eval/mytofu/default.yaml \
        model=${MODEL} \
        task_name=${task_name} \
        model.model_args.pretrained_model_name_or_path=${run_dir} \
        model.tokenizer_args.pretrained_model_name_or_path=${run_dir} \
        paths.output_dir=${run_dir}/evals_final
    fi

  done
done

echo "All difficulty split experiments finished."