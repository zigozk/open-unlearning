#!/bin/bash
#
# MYTOFU recovery sweep: tune toward stronger forgetting and higher BUS.
#
# This script is intended for the case where a conservative paper-inspired
# sweep preserves utility but underperforms on MYTOFU Mem. It starts from the
# existing full checkpoint and only runs unlearning + final evaluation.
#
# Usage:
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_recovery_tuned_array.sh --list
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_recovery_tuned_array.sh --submit
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_recovery_tuned_array.sh --submit-with-summary
#
# Useful overrides:
#   SWEEP_PROFILE=smoke MAX_PARALLEL=1 bash ... --submit-with-summary
#   OUTPUT_ROOT=saves/unlearn_recovery_tuned_v2 bash ... --submit-with-summary

#SBATCH -J mytofu_recover
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

setup_cuda_home() {
  if ! command -v module >/dev/null 2>&1; then
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
  fi
  if command -v module >/dev/null 2>&1; then
    module load cuda/12.1.0 2>/dev/null || \
      module load cuda/12.1 2>/dev/null || \
      module load cuda 2>/dev/null || true
  fi
  if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
      export CUDA_HOME
    elif [ -d /usr/local/cuda ]; then
      export CUDA_HOME=/usr/local/cuda
    else
      for cuda_dir in /usr/local/cuda-12.1 /usr/local/cuda-12 /opt/cuda /opt/cuda-12.1; do
        if [ -d "${cuda_dir}" ]; then
          export CUDA_HOME="${cuda_dir}"
          break
        fi
      done
    fi
  fi
  if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
  fi
}

if [ "${1:-}" != "--list" ] && [ "${1:-}" != "--submit" ] && [ "${1:-}" != "--submit-with-summary" ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate unlearning
  setup_cuda_home
fi

export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

export HF_HOME=/home/zkzhang/unlearning/HF_CACHE
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE
unset HUGGINGFACE_HUB_CACHE
unset HF_MODULES_CACHE
unset HF_DATASETS_CACHE

MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saves/unlearn_recovery_tuned}"
RANK_METRIC="${RANK_METRIC:-BUS}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SWEEP_PROFILE="${SWEEP_PROFILE:-retry_failed}"

# method_label|trainer|experiment|tag|learning_rate|epochs|forget_dataset|extra hydra args
declare -a RECOVERY_RETRY_FAILED_RUNS=(
  # Rerun only the jobs that failed before the DPO IDK-pair and UNDIAL Hydra fixes.
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p05_a0p1_g1_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.05 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p1_a0p1_g1_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p2_a0p1_g1_lr2em5_e2|2e-5|2|MYTOFU_forget_idk|trainer.method_args.beta=0.2 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|recover_b10_a0_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|recover_b20_a0_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|recover_b20_a0p05_g2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.05 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
)

declare -a RECOVERY_RUNS=(
  # CEU is currently the strongest paper_tuned method. Increase exposure around
  # lr=1e-5 and test whether longer training recovers the old-table BUS range.
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i0_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=0"
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i1_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=2"
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i1_lr1em5_e4|1e-5|4|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i2_lr1em5_e4|1e-5|4|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=2"
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i1_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=2"

  # Gradient baselines: restore stronger old-style updates.
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|recover_lr1em5_e3|1e-5|3|MYTOFU_forget|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|recover_lr2em5_e2|2e-5|2|MYTOFU_forget|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|recover_lr2em5_e3|2e-5|3|MYTOFU_forget|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|recover_lr5em5_e1|5e-5|1|MYTOFU_forget|"

  # GradDiff/NPO: lower retain pressure and/or increase forget pressure.
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|recover_a0p1_g1_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|recover_a0p25_g1_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.alpha=0.25 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|recover_a0p5_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.alpha=0.5 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|recover_a0p25_g2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff_sago|GradDiff|unlearn/mytofu/default.yaml|recover_a0p25_g1_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.alpha=0.25 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "GradDiff_sago|GradDiff|unlearn/mytofu/default.yaml|recover_a0p25_g2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  "NPO|NPO|unlearn/mytofu/default.yaml|recover_b0p05_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.05 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|recover_b0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|recover_b0p2_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.2 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|recover_b0p1_a0p1_g3_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.1 trainer.method_args.gamma=3.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO_sago|NPO|unlearn/mytofu/default.yaml|recover_b0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "NPO_sago|NPO|unlearn/mytofu/default.yaml|recover_b0p2_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.2 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "NPO_sago|NPO|unlearn/mytofu/default.yaml|recover_b0p1_a0p1_g3_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.1 trainer.method_args.gamma=3.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  # DPO: current best improves when alpha is reduced; continue that direction
  # and include stronger learning rates.
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p05_a0p1_g1_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.05 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p1_a0p1_g1_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p2_a0p1_g1_lr2em5_e2|2e-5|2|MYTOFU_forget_idk|trainer.method_args.beta=0.2 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"

  # SimNPO: paper_tuned found gamma=0.25 better than 0.125; push gamma upward.
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|recover_b4p5_g0p5_a0p5_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.5 trainer.method_args.alpha=0.5 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|recover_b4p5_g1_a0p25_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=1.0 trainer.method_args.alpha=0.25 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|recover_b8_g0p5_a0p5_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=8.0 trainer.method_args.gamma=0.5 trainer.method_args.alpha=0.5 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|recover_b4p5_g0p5_a0p25_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.5 trainer.method_args.alpha=0.25 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|recover_b4p5_g0p5_a0p5_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.5 trainer.method_args.alpha=0.5 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|recover_b4p5_g1_a0p25_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=1.0 trainer.method_args.alpha=0.25 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|recover_b4p5_g0p5_a0p25_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.5 trainer.method_args.alpha=0.25 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  # RMU/UNDIAL/WGA/PDU/SatImp: test stronger forgetting while limiting retain.
  "RMU|RMU|unlearn/mytofu/default.yaml|recover_sc2_a0p1_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=0.1 trainer.method_args.gamma=2.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|recover_sc5_a0p1_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.steering_coeff=5 trainer.method_args.alpha=0.1 trainer.method_args.gamma=2.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|recover_sc10_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.steering_coeff=10 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.module_regex=model\\.layers\\.7"

  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|recover_b10_a0_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|recover_b20_a0_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|recover_b20_a0p05_g2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.05 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"

  "WGA|WGA|unlearn/mytofu/default.yaml|recover_b0p5_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.5 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "WGA|WGA|unlearn/mytofu/default.yaml|recover_b1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "WGA|WGA|unlearn/mytofu/default.yaml|recover_b2_a0p1_g2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.beta=2.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"

  "PDU|PDU|unlearn/mytofu/default.yaml|recover_eps0p05_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.05 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0"
  "PDU|PDU|unlearn/mytofu/default.yaml|recover_eps0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0"
  "PDU|PDU|unlearn/mytofu/default.yaml|recover_eps0p2_a0p1_g2_lr2em5_e2|2e-5|2|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.2 trainer.method_args.alpha=0.1 trainer.method_args.gamma=2.0"

  "SatImp|SatImp|unlearn/mytofu/default.yaml|recover_b5_1_a0p1_g0p5_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.5"
  "SatImp|SatImp|unlearn/mytofu/default.yaml|recover_b10_1_a0p1_g0p5_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta1=10.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.5"
)

declare -a SMOKE_RUNS=(
  "CEU|CEU|unlearn/mytofu/default.yaml|recover_i1_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "DPO|DPO|unlearn/mytofu/idk.yaml|recover_b0p1_a0p1_g1_lr1em5_e3|1e-5|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO_sago|NPO|unlearn/mytofu/default.yaml|recover_b0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "NPO|NPO|unlearn/mytofu/default.yaml|recover_b0p1_a0p25_g2_lr1em5_e3|1e-5|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.25 trainer.method_args.gamma=2.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|recover_lr1em5_e3|1e-5|3|MYTOFU_forget|"
)

get_runs() {
  if [ "${SWEEP_PROFILE}" = "smoke" ]; then
    printf '%s\n' "${SMOKE_RUNS[@]}"
  elif [ "${SWEEP_PROFILE}" = "retry_failed" ]; then
    printf '%s\n' "${RECOVERY_RETRY_FAILED_RUNS[@]}"
  else
    printf '%s\n' "${RECOVERY_RUNS[@]}"
  fi
}

print_grid() {
  local i=0
  while IFS= read -r spec; do
    printf '%03d | %s\n' "${i}" "${spec}"
    i=$((i + 1))
  done < <(get_runs)
  echo "Total runs: ${i}"
}

submit_array() {
  local total
  total="$(get_runs | wc -l | tr -d ' ')"
  if [ "${total}" -le 0 ]; then
    echo "[ERROR] Empty sweep grid."
    exit 1
  fi
  mkdir -p logs "${OUTPUT_ROOT}"
  sbatch \
    --array=0-$((total - 1))%${MAX_PARALLEL} \
    --export=ALL,ROOT_DIR="${ROOT_DIR}",MODEL="${MODEL}",MODEL_PATH="${MODEL_PATH}",OUTPUT_ROOT="${OUTPUT_ROOT}",PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE}",GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}",RUN_FINAL_EVAL="${RUN_FINAL_EVAL}",SKIP_EXISTING="${SKIP_EXISTING}",SWEEP_PROFILE="${SWEEP_PROFILE}" \
    "$0"
}

submit_with_summary() {
  local submit_output array_job_id summary_cmd summary_output
  submit_output="$(submit_array)"
  echo "${submit_output}"
  array_job_id="$(echo "${submit_output}" | awk '/Submitted batch job/ {print $NF}' | tail -n 1)"
  if [ -z "${array_job_id}" ]; then
    echo "[ERROR] Could not parse array job id."
    exit 1
  fi
  summary_cmd="bash -lc 'cd ${ROOT_DIR} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate unlearning && python sbatch/MYTOFU/unlearn/summarize_mytofu_paper_tuned.py --result-root ${OUTPUT_ROOT} --metric ${RANK_METRIC}'"
  summary_output="$(
    sbatch \
      --dependency=afterany:${array_job_id} \
      -J mytofu_recover_sum \
      -p compute \
      -N 1 \
      --cpus-per-task=4 \
      --mem=16G \
      -t 02:00:00 \
      -o logs/%x-%j.out \
      -e logs/%x-%j.err \
      --wrap "${summary_cmd}"
  )"
  echo "${summary_output}"
}

case "${1:-}" in
  --list)
    print_grid
    exit 0
    ;;
  --submit)
    submit_array
    exit 0
    ;;
  --submit-with-summary)
    submit_with_summary
    exit 0
    ;;
esac

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "[ERROR] Run with --submit/--submit-with-summary or submit this script as a SLURM array."
  exit 1
fi

mapfile -t RUNS < <(get_runs)
if [ "${SLURM_ARRAY_TASK_ID}" -ge "${#RUNS[@]}" ]; then
  echo "[ERROR] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} out of range ${#RUNS[@]}"
  exit 1
fi

IFS='|' read -r method_label trainer experiment tag learning_rate epochs forget_dataset extra_arg_string <<< "${RUNS[$SLURM_ARRAY_TASK_ID]}"

task_name="mytofu_${MODEL}_${method_label}_${tag}_from_full_e10"
run_dir="${OUTPUT_ROOT}/${task_name}"
final_summary="${run_dir}/evals_final/MYTOFU_SUMMARY.json"

echo "=================================================="
echo "MYTOFU recovery tuning"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "SWEEP_PROFILE=${SWEEP_PROFILE}"
echo "MODEL=${MODEL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "method_label=${method_label}"
echo "trainer=${trainer}"
echo "experiment=${experiment}"
echo "tag=${tag}"
echo "learning_rate=${learning_rate}"
echo "epochs=${epochs}"
echo "forget_dataset=${forget_dataset}"
echo "task_name=${task_name}"
echo "run_dir=${run_dir}"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "nvcc=$(command -v nvcc || true)"
echo "=================================================="
nvidia-smi || true

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[ERROR] MODEL_PATH not found: ${MODEL_PATH}"
  exit 1
fi

if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${final_summary}" ]; then
  echo "[SKIP] Found final summary: ${final_summary}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"

extra_args=(
  "data/datasets@data.forget=${forget_dataset}"
  "data/datasets@data.retain=MYTOFU_retain"
)
if [ -n "${extra_arg_string}" ]; then
  read -r -a method_extra_args <<< "${extra_arg_string}"
  extra_args+=("${method_extra_args[@]}")
fi

python src/train.py \
  --config-name=unlearn.yaml \
  experiment=${experiment} \
  trainer=${trainer} \
  task_name=${task_name} \
  model=${MODEL} \
  model.model_args.pretrained_model_name_or_path=${MODEL_PATH} \
  model.tokenizer_args.pretrained_model_name_or_path=${MODEL_PATH} \
  paths.output_dir=${run_dir} \
  trainer.args.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
  trainer.args.gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
  trainer.args.learning_rate=${learning_rate} \
  trainer.args.num_train_epochs=${epochs} \
  trainer.args.gradient_checkpointing=true \
  ++trainer.args.overwrite_output_dir=true \
  ++trainer.args.gradient_checkpointing_kwargs.use_reentrant=false \
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

echo "===== MYTOFU RECOVERY TUNING RUN DONE ====="
