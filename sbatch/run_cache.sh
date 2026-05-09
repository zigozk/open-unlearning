#!/usr/bin/env bash
set -euo pipefail

# ====== 你可以改的参数 ======
HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"   # 统一缓存根目录（建议共享盘）
DATASET_ID="${DATASET_ID:-locuslab/TOFU}"
SPLIT="${SPLIT:-train}"
LOG_DIR="${LOG_DIR:-./logs}"
# ===========================

mkdir -p "$LOG_DIR"
LOG_OK="$LOG_DIR/tofu_cache_ok.txt"
LOG_FAIL="$LOG_DIR/tofu_cache_fail.txt"
LOG_ALL="$LOG_DIR/tofu_cache_all.txt"

: > "$LOG_OK"
: > "$LOG_FAIL"
: > "$LOG_ALL"

export HF_HOME="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TOKENIZERS_PARALLELISM=false

# 确保是联网模式（避免你之前开了 OFFLINE 导致只读缓存）
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE

echo "[info] HF_HOME=$HF_HOME"
echo "[info] DATASET_ID=$DATASET_ID"
echo "[info] SPLIT=$SPLIT"
echo "[info] logs at $LOG_DIR"

# 1) 获取所有 config/name
echo "[step1] fetching all configs(names) from hub..."
python - <<'PY' | tee "$LOG_ALL"
from datasets import get_dataset_config_names
dataset_id = __import__("os").environ.get("DATASET_ID", "locuslab/TOFU")
names = get_dataset_config_names(dataset_id)
for n in names:
    print(n)
PY

# 把输出读入数组
mapfile -t NAMES < "$LOG_ALL"
if [ "${#NAMES[@]}" -eq 0 ]; then
  echo "[error] no configs found for $DATASET_ID"
  exit 2
fi
echo "[info] total configs: ${#NAMES[@]}"

# 2) 逐个缓存
echo "[step2] caching each config..."
for name in "${NAMES[@]}"; do
  echo "---- caching: ${DATASET_ID} / ${name} / split=${SPLIT}"
  if python - <<PY
from datasets import load_dataset
ds = load_dataset("${DATASET_ID}", "${name}", split="${SPLIT}")
print("cached_ok", "${name}", len(ds))
PY
  then
    echo "$name" | tee -a "$LOG_OK" >/dev/null
  else
    echo "$name" | tee -a "$LOG_FAIL" >/dev/null
    echo "[warn] failed: $name (recorded in $LOG_FAIL)"
  fi
done

echo "[done] success: $(wc -l < "$LOG_OK") ; failed: $(wc -l < "$LOG_FAIL")"
echo "[done] ok list  : $LOG_OK"
echo "[done] fail list: $LOG_FAIL"

# 3) 离线自检（抽样几个）
echo "[step3] offline sanity check (sample 3)..."
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python - <<'PY'
import os, random
from datasets import load_dataset

dataset_id = os.environ.get("DATASET_ID", "locuslab/TOFU")
ok_file = os.environ.get("LOG_OK", "./logs/tofu_cache_ok.txt")

with open(ok_file, "r") as f:
    names = [x.strip() for x in f if x.strip()]

sample = random.sample(names, min(3, len(names)))
print("offline sample:", sample)

for n in sample:
    ds = load_dataset(dataset_id, n, split="train")
    print("offline_ok", n, len(ds))
PY

echo "[final] all done. You can now use offline mode on GPU nodes."
