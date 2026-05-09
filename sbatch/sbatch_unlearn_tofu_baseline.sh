#!/bin/bash
#SBATCH -J tofu_unlearn_baseline
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p compute
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2
#SBATCH --cpus-per-task=6
#SBATCH --mem=120G
#SBATCH -t 48:00:00

set -euo pipefail

mkdir -p logs
cd /home/zkzhang/unlearning/open-unlearning

# 1) 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

# 2) HF 缓存（与你之前 warmup 的缓存路径保持一致即可）
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# 3) GPU 节点无外网：强制离线
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
# export TORCHDYNAMO_DISABLE=1
# 4) 这次只跑一个 smoke 组合：forget10 / holdout10 / retain90 + NPO
FORGET_SPLIT="forget10"
HOLDOUT_SPLIT="holdout10"
RETAIN_SPLIT="retain90"
EXPERIMENT="unlearn/tofu/default.yaml"   # 与官方 NPO 路径一致 :contentReference[oaicite:1]{index=1}

TRAINERS=("NPO" "SimNPO" "GradDiff" "GradAscent")
# 5) 4个 FULL 模型
MODELS=(
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Llama-3.1-8B-Instruct"
  "Llama-2-7b-chat-hf"
)

# 6) 训练批量（保守值，先保证跑通；后续你可再调大）
PER_DEVICE_TRAIN_BS=4
GRAD_ACC=4

# 7) Master port
export MASTER_PORT=$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
echo "MASTER_PORT=$MASTER_PORT"

# 8) flash-attn 可用则启用，否则回退 sdpa
ATTN_IMPL="sdpa"
python - <<'PY' && ATTN_IMPL="flash_attention_2" || true
import flash_attn
print("flash_attn ok:", flash_attn.__version__)
PY
echo "ATTN_IMPL=$ATTN_IMPL"

# 9) 逐模型执行：unlearn -> eval
for MODEL in "${MODELS[@]}"; do
  MODEL_DIR="/home/zkzhang/models/tofu_${MODEL}_full"
  test -d "$MODEL_DIR" || { echo "[error] missing model dir: $MODEL_DIR"; exit 2; }

  # baseline 参考日志：官方 unlearn 脚本就是从这里读 retain 的 eval 结果 :contentReference[oaicite:2]{index=2}
  BASELINE_JSON="saves/eval/tofu_${MODEL}_${RETAIN_SPLIT}/TOFU_EVAL.json"
  test -f "$BASELINE_JSON" || {
    echo "[error] missing baseline retain logs: $BASELINE_JSON"
    echo "        你需要先准备该文件（可复用你已保存的 baseline eval），否则无法计算 FQ/KS 等对比指标。"
    exit 3
  }
  for TRAINER in "${TRAINERS[@]}"; do
    # 任务名（加 job id 避免覆盖）
    TASK_NAME="tofu_${MODEL}_${FORGET_SPLIT}_${TRAINER}_smoke_${SLURM_JOB_ID}"

    # 训练输出（先按官方默认逻辑放 saves/unlearn，保证与代码预期一致；之后再同步到 results）
    TRAIN_OUT="results/unlearn/${TASK_NAME}"
    EVAL_OUT="${TRAIN_OUT}/evals"

    echo "===== [${TASK_NAME}] MODEL=${MODEL} ====="
    echo "model_dir      : ${MODEL_DIR}"
    echo "baseline_json  : ${BASELINE_JSON}"
    echo "train_out      : ${TRAIN_OUT}"
    echo "eval_out       : ${EVAL_OUT}"

    # ---- Unlearn ----
      CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
      src/train.py --config-name=unlearn.yaml \
      experiment=${EXPERIMENT} \
      trainer=${TRAINER} \
      task_name=${TASK_NAME} \
      model=${MODEL} \
      forget_split=${FORGET_SPLIT} \
      retain_split=${RETAIN_SPLIT} \
      model.model_args.pretrained_model_name_or_path=${MODEL_DIR} \
      model.tokenizer_args.pretrained_model_name_or_path=${MODEL_DIR} \
      model.model_args.attn_implementation=${ATTN_IMPL} \
      retain_logs_path=${BASELINE_JSON} \
      trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BS} \
      trainer.args.gradient_accumulation_steps=${GRAD_ACC} \
      trainer.args.ddp_find_unused_parameters=true \
      trainer.args.gradient_checkpointing=true \
      paths.output_dir=${TRAIN_OUT}

    # ---- Eval（对 unlearn 后的模型目录做评估）----
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
      experiment=eval/tofu/default.yaml \
      forget_split=${FORGET_SPLIT} \
      holdout_split=${HOLDOUT_SPLIT} \
      model=${MODEL} \
      task_name=${TASK_NAME} \
      model.model_args.pretrained_model_name_or_path=${TRAIN_OUT} \
      model.tokenizer_args.pretrained_model_name_or_path=${MODEL_DIR} \
      retain_logs_path=${BASELINE_JSON} \
      paths.output_dir=${EVAL_OUT}

    # ---- 同步到你想要的 results/unlearn（你之前的目录要求）----
    FINAL_OUT="results/unlearn/${TASK_NAME}"
    mkdir -p "$FINAL_OUT"
    rsync -a "${TRAIN_OUT}/" "${FINAL_OUT}/"

    echo "[done] synced to: ${FINAL_OUT}"
  done
done

echo "===== ALL TASKS DONE ====="
echo "delete torchelastic temp dirs..."
# 清理 torchelastic 临时目录（仅在作业结束时）
find . -maxdepth 1 -type d -name "torchelastic_*" -mtime +0 -exec rm -rf {} \;

echo "ALL DONE."
