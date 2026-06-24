#!/bin/bash
#SBATCH -J check_tofu_llama2
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zkzhang/unlearning/open-unlearning}"
cd "${ROOT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate unlearning

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/home/zkzhang/unlearning/HF_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# This job must be able to read the HuggingFace snapshot metadata/files.
unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE

REPO_ID="${REPO_ID:-open-unlearning/tofu_Llama-2-7b-chat-hf_full}"
LOCAL_DIR="${LOCAL_DIR:-/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full}"

echo "===== CHECK TOFU LLAMA2 FULL SNAPSHOT ====="
echo "HOSTNAME=$(hostname)"
echo "JOB_ID=${SLURM_JOB_ID:-none}"
echo "REPO_ID=${REPO_ID}"
echo "LOCAL_DIR=${LOCAL_DIR}"
echo "HF_HOME=${HF_HOME}"
date

python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import hashlib
import os
import sys

repo_id = os.environ.get("REPO_ID", "open-unlearning/tofu_Llama-2-7b-chat-hf_full")
local_dir = Path(os.environ.get("LOCAL_DIR", "/home/zkzhang/models/tofu_Llama-2-7b-chat-hf_full"))

if not local_dir.exists():
    print(f"ERROR: local directory does not exist: {local_dir}")
    sys.exit(2)

remote_dir = Path(snapshot_download(
    repo_id=repo_id,
    local_files_only=False,
    resume_download=True,
))

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def collect_files(root: Path) -> dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if ".cache" in rel_parts or path.suffix == ".lock":
            continue
        files[path.relative_to(root).as_posix()] = path
    return files

local_files = collect_files(local_dir)
remote_files = collect_files(remote_dir)

missing = sorted(set(remote_files) - set(local_files))
extra = sorted(set(local_files) - set(remote_files))
common = sorted(set(local_files) & set(remote_files))

different = []
for name in common:
    local_path = local_files[name]
    remote_path = remote_files[name]
    if local_path.stat().st_size != remote_path.stat().st_size:
        different.append((name, "size"))
    elif sha256(local_path) != sha256(remote_path):
        different.append((name, "sha256"))

print(f"repo: {repo_id}")
print(f"local: {local_dir}")
print(f"remote cache: {remote_dir}")
print(f"local files: {len(local_files)}")
print(f"remote files: {len(remote_files)}")
print(f"missing: {len(missing)}")
print(f"extra: {len(extra)}")
print(f"different: {len(different)}")

if missing:
    print("\nMISSING:")
    print("\n".join(missing[:100]))
if extra:
    print("\nEXTRA:")
    print("\n".join(extra[:100]))
if different:
    print("\nDIFFERENT:")
    for name, reason in different[:100]:
        print(name, reason)

if not missing and not extra and not different:
    print("\nOK: local directory matches HuggingFace snapshot files.")
    sys.exit(0)

print("\nNOT MATCHED: local directory differs from HuggingFace snapshot.")
sys.exit(1)
PY

echo "===== CHECK DONE ====="
