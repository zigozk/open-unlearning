#!/bin/bash
#SBATCH -J unlearn_tofu_smoke
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p compute
#SBATCH --gres=gpu:a100x:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH -t 24:00:00


set -euxo pipefail

mkdir -p logs
cd /home/zkzhang/unlearning/open-unlearning

# ========= 1) conda 环境 =========
source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

# ========= 2) 统一 HF 缓存目录（关键：与 warmup 时一致）=========
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules

# ========= 3) 强制离线（GPU 节点无网）=========
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# ========= 4) 选择 smoke 的 split / trainer / 模型 =========
FORGET_SPLIT="forget10"
HOLDOUT_SPLIT="holdout10"
RETAIN_SPLIT="retain90"

TRAINER="NPO"   # smoke 先固定 1 个；你要换成 GradAscent/GradDiff 也行

# 你本地模型目录名（/home/share/models/ 下的子目录）
MODEL_CFG="Llama-2-7b-chat-hf"
MODEL_SUBDIR="Llama-2-7b-chat-hf"   # TODO: 改成你要跑的模型目录名
MODEL_DIR="/home/share/models/${MODEL_SUBDIR}"

test -d "$MODEL_DIR" || { echo "MODEL_DIR not found: $MODEL_DIR"; exit 2; }

# ========= 5) 输出目录（你现在想要的新保存路径就改这里）=========
SAVES_EVAL_ROOT="/home/zkzhang/unlearning/open-unlearning/saves/eval"

# 按你现有目录习惯：tofu_<model>_<retain_split>/TOFU_EVAL.json
BASELINE_DIR="${SAVES_EVAL_ROOT}/tofu_${MODEL_SUBDIR}_${RETAIN_SPLIT}"
BASELINE_JSON="${BASELINE_DIR}/TOFU_EVAL.json"

RESULTS_ROOT="/home/zkzhang/unlearning/open-unlearning/results/unlearn"
TASK_NAME="tofu_${MODEL_SUBDIR}_${FORGET_SPLIT}_${TRAINER}_smoke"
UNLEARN_OUT="${RESULTS_ROOT}/${TASK_NAME}"
EVAL_OUT="${UNLEARN_OUT}/evals"

# ========= 6) 离线自检：TOFU 缓存至少要能读到 forget10_perturbed =========
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("locuslab/TOFU", "forget10_perturbed", split="train")
print("TOFU offline ok:", len(ds))
PY

# ========= 7) flash-attn：能 import 就用 FA2，否则回退 sdpa =========
ATTN_IMPL="sdpa"
python - <<'PY' && ATTN_IMPL="flash_attention_2" || true
import flash_attn
print("flash_attn ok:", flash_attn.__version__)
PY
echo "ATTN_IMPL=$ATTN_IMPL"


# ========= 9) Unlearn（1 GPU smoke：限制步数，避免跑太久）=========
# 你之前踩过的坑：
# - 不要用 '~~key'（会解析失败）
# - max_steps 在你当前 config 里可能不是已有字段，所以要用 +trainer.args.max_steps=... 来“新增” key
export MASTER_PORT=$(python - <<'PY'
import socket
s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()
PY
)
echo "MASTER_PORT=$MASTER_PORT"

echo "[start] unlearning with trainer=$TRAINER, forget_split=$FORGET_SPLIT, retain_split=$RETAIN_SPLIT"

CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config_file configs/accelerate/default_config.yaml \
  --num_processes 1 \
  --main_process_port $MASTER_PORT \
  src/train.py --config-name=unlearn.yaml \
  experiment=unlearn/tofu/default.yaml \
  trainer=$TRAINER \
  task_name=$TASK_NAME \
  model=$MODEL_CFG \
  forget_split=$FORGET_SPLIT \
  retain_split=$RETAIN_SPLIT \
  retain_logs_path=$BASELINE_JSON \
  paths.output_dir=$UNLEARN_OUT \
  model.model_args.pretrained_model_name_or_path=$MODEL_DIR \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.model_args.attn_implementation=$ATTN_IMPL \
  trainer.args.per_device_train_batch_size=4 \
  trainer.args.gradient_accumulation_steps=4 \
  trainer.args.ddp_find_unused_parameters=true \
  trainer.args.gradient_checkpointing=true 

echo "[done] unlearned model saved to: $UNLEARN_OUT"
echo "[done] start evaling unlearned model..."

# ========= 10) Eval：评估 unlearn 后的 checkpoint =========
CUDA_VISIBLE_DEVICES=0 python src/eval.py \
  experiment=eval/tofu/default.yaml \
  forget_split=$FORGET_SPLIT \
  holdout_split=$HOLDOUT_SPLIT \
  model=$MODEL_SUBDIR \
  task_name=$TASK_NAME \
  model.model_args.pretrained_model_name_or_path=$UNLEARN_OUT \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.model_args.attn_implementation=$ATTN_IMPL \
  paths.output_dir=$EVAL_OUT \
  retain_logs_path=$BASELINE_JSON \
  eval.tofu.overwrite=true \

echo "[done] baseline: $BASELINE_JSON"
echo "[done] unlearn : $UNLEARN_OUT"
echo "[done] eval   : $EVAL_OUT"
