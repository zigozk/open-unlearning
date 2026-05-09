#!/bin/bash
#SBATCH -J eval_tofu_smoke
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1


set -euxo pipefail

# ==================== 0. 环境准备 ====================
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.1

. $HOME/miniconda3/etc/profile.d/conda.sh
conda activate unlearning



cd $HOME/unlearning/open-unlearning


# ===== 共享 HF cache（关键：compute 节点必须能读到这个目录）=====
export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# ===== 强制离线 =====
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# ===== 本地模型与 eval logs =====
MODEL_SUBDIR="Llama-2-7b-chat-hf"     # TODO: 改成 /home/share/models 下你的模型目录名
MODEL_DIR="/home/share/models/${MODEL_SUBDIR}"
EVAL_ROOT="/home/zkzhang/unlearning/open-unlearning/saves/eval"

test -d "$MODEL_DIR" || { echo "MODEL_DIR not found: $MODEL_DIR"; exit 2; }
test -d "$EVAL_ROOT" || { echo "EVAL_ROOT not found: $EVAL_ROOT"; exit 2; }

# ===== 离线自检：确保缓存真的可用（否则直接失败，不浪费 GPU）=====
python - <<'PY'
from datasets import load_dataset
for n in ["forget10_perturbed","retain_perturbed","real_authors_perturbed","world_facts_perturbed","holdout10"]:
    ds = load_dataset("locuslab/TOFU", n, split="train")
    print("offline ok", n, len(ds))
PY

# ===== 找 retain logs（forget_quality/privleak 会用到）=====
RETAIN_JSON="${RETAIN_JSON:-$(find "$EVAL_ROOT" -type f -name "TOFU_EVAL.json" | grep -E "retain90|retain_perturbed|retain" | head -n 1)}"
test -f "$RETAIN_JSON" || { echo "Cannot find RETAIN_JSON under $EVAL_ROOT. Set RETAIN_JSON manually."; exit 3; }

# ===== 输出目录（你之前要换保存路径，就改 OUT_BASE）=====
OUT_BASE="/home/zkzhang/unlearning/open-unlearning/results"
TASK_NAME="Llama-2-7b-chat-hf_TOFU_EVAL_forget10"
OUT_DIR="${OUT_BASE}/eval/${TASK_NAME}"

# ===== flash-attn：有就用 FA2，没有就回退 sdpa =====
ATTN_IMPL="sdpa"
python - <<'PY' && ATTN_IMPL="flash_attention_2" || true
import flash_attn
print("flash_attn ok", flash_attn.__version__)
PY
echo "ATTN_IMPL=$ATTN_IMPL"

python src/eval.py --config-name=eval.yaml experiment=eval/tofu/default \
  task_name="$TASK_NAME" \
  paths.output_dir="$OUT_DIR" \
  model=Llama-2-7b-chat-hf \
  model.model_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.tokenizer_args.pretrained_model_name_or_path="$MODEL_DIR" \
  model.model_args.device_map=auto \
  model.model_args.attn_implementation="$ATTN_IMPL" \
  eval.tofu.forget_split=forget10 \
  eval.tofu.holdout_split=holdout10 \
  eval.tofu.retain_logs_path="$RETAIN_JSON" \
  eval.tofu.batch_size=8

echo "[done] outputs in: $OUT_DIR"