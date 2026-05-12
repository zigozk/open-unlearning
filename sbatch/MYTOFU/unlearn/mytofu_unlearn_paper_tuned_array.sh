#!/bin/bash
#
# Paper-inspired MYTOFU unlearning hyperparameter sweep.
#
# This script ONLY runs unlearning from an existing full checkpoint. It does not
# fine-tune a base model and does not introduce additional model backbones.
#
# Use from a login node:
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_paper_tuned_array.sh --submit
#
# Inspect the planned grid without submitting:
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_paper_tuned_array.sh --list
#
# Design principle:
#   - include the method defaults used in this repo;
#   - expand around the hyperparameters emphasized by each method's paper:
#     GA/GD: learning rate and epochs to avoid collapse;
#     DPO/NPO/WGA/UNDIAL/SatImp: beta-like preference strength;
#     SimNPO: beta and margin/gamma-like strength, reference-free variants;
#     RMU: steering coefficient and retain regularization;
#     PDU: retain-loss constraint epsilon.

#SBATCH -J mytofu_paper_tune
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

if [ "${1:-}" != "--list" ] && [ "${1:-}" != "--submit" ]; then
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate unlearning
fi

setup_cuda_home() {
  # DeepSpeed checks CUDA_HOME at import time for some optional ops.  Some
  # clusters expose GPUs without exporting the CUDA toolkit path by default.
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

if [ "${1:-}" != "--list" ] && [ "${1:-}" != "--submit" ]; then
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
OUTPUT_ROOT="${OUTPUT_ROOT:-saves/unlearn_paper_tuned}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

# Optional quick smoke subset:
#   SWEEP_PROFILE=smoke bash ... --submit
SWEEP_PROFILE="${SWEEP_PROFILE:-full}"

# Fields:
# method_label|trainer|experiment|tag|learning_rate|epochs|forget_dataset|extra hydra args
#
# Notes:
# - DPO uses MYTOFU_forget_idk, matching the repo's idk experiment design.
# - Tags are compact and filesystem-safe:
#   b=beta, a=alpha, g=gamma, sc=steering_coeff, eps=retain_loss_eps,
#   i=ignored initial answer tokens, lr=learning rate, e=epochs.
declare -a PAPER_TUNED_RUNS=(
  # Gradient Ascent / Gradient Difference: TOFU-style baselines are highly
  # sensitive to step size, so the grid is conservative around 1e-6..1e-5.
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|paper_lr1em6_e1|1e-6|1|MYTOFU_forget|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|paper_lr2em6_e1|2e-6|1|MYTOFU_forget|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|paper_lr5em6_e1|5e-6|1|MYTOFU_forget|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|paper_lr1em6_e2|1e-6|2|MYTOFU_forget|"

  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|paper_none_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|paper_none_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|paper_none_a2_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.alpha=2.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|paper_none_a1_g0p5_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.alpha=1.0 trainer.method_args.gamma=0.5 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff_sago|GradDiff|unlearn/mytofu/default.yaml|paper_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "GradDiff_pcgrad|GradDiff|unlearn/mytofu/default.yaml|paper_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"

  # DPO / NPO: original preference-optimization families center on beta;
  # include beta=0.1 as the paper/repo anchor and nearby values.
  "DPO|DPO|unlearn/mytofu/idk.yaml|paper_b0p05_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.05 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "DPO|DPO|unlearn/mytofu/idk.yaml|paper_b0p1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "DPO|DPO|unlearn/mytofu/idk.yaml|paper_b0p2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "DPO|DPO|unlearn/mytofu/idk.yaml|paper_b0p5_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "DPO|DPO|unlearn/mytofu/idk.yaml|paper_b0p1_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0"
  "DPO|DPO|unlearn/mytofu/idk.yaml|paper_b0p1_a2_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=2.0 trainer.method_args.gamma=1.0"

  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p05_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.05 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p5_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p1_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p1_a2_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=2.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO_sago|NPO|unlearn/mytofu/default.yaml|paper_b0p1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "NPO_pcgrad|NPO|unlearn/mytofu/default.yaml|paper_b0p1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"

  # SimNPO: repo anchor is beta=4.5, gamma=0.125; sweep nearby beta and gamma,
  # plus retention-prioritized synthesis modes.
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|paper_b1_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=1.0 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|paper_b2_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=2.0 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|paper_b4p5_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|paper_b8_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=8.0 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|paper_b4p5_g0p0625_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.0625 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO|SimNPO|unlearn/mytofu/default.yaml|paper_b4p5_g0p25_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.25 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|paper_b2_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=2.0 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|paper_b4p5_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SimNPO_pcgrad|SimNPO|unlearn/mytofu/default.yaml|paper_b2_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=2.0 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "SimNPO_pcgrad|SimNPO|unlearn/mytofu/default.yaml|paper_b4p5_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"

  # RMU: original RMU-style tuning centers on steering coefficient and layer.
  # This implementation defaults to layer 7; we sweep coefficient and retain
  # strength while keeping the module fixed for comparability.
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc5_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc10_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=10 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc20_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=20 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc2_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc2_a2_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=2.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"

  # CEU: confidence-erasing variants mainly tune how aggressively answer tokens
  # are ignored and step size.
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i0_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=0"
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i2_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=2"
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i4_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=4"
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i1_lr1em5_e2|1e-5|2|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i2_lr1em5_e2|1e-5|2|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=2"

  # PDU / SatImp / WGA / UNDIAL: include repo defaults and local neighborhoods
  # around the method-specific penalty strength.
  "PDU|PDU|unlearn/mytofu/default.yaml|paper_eps0p05_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.05 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "PDU|PDU|unlearn/mytofu/default.yaml|paper_eps0p1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "PDU|PDU|unlearn/mytofu/default.yaml|paper_eps0p2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "PDU|PDU|unlearn/mytofu/default.yaml|paper_eps0p5_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.retain_loss_eps=0.5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"

  "SatImp|SatImp|unlearn/mytofu/default.yaml|paper_b5_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1"
  "SatImp|SatImp|unlearn/mytofu/default.yaml|paper_b5_1_a0p5_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.5 trainer.method_args.gamma=0.1"
  "SatImp|SatImp|unlearn/mytofu/default.yaml|paper_b5_1_a1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=0.1"
  "SatImp|SatImp|unlearn/mytofu/default.yaml|paper_b10_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=10.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1"

  "WGA|WGA|unlearn/mytofu/default.yaml|paper_b0p5_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "WGA|WGA|unlearn/mytofu/default.yaml|paper_b1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "WGA|WGA|unlearn/mytofu/default.yaml|paper_b2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=2.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "WGA|WGA|unlearn/mytofu/default.yaml|paper_b1_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0"

  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|paper_b5_a0_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=5.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=1.0"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|paper_b10_a0_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=1.0"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|paper_b20_a0_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=1.0"
  "UNDIAL|UNDIAL|unlearn/mytofu/default.yaml|paper_b10_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0"
)

declare -a SMOKE_RUNS=(
  "CEU|CEU|unlearn/mytofu/default.yaml|paper_i1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.ignore_first_n_answer_tokens=1"
  "NPO|NPO|unlearn/mytofu/default.yaml|paper_b0p1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|paper_b4p5_g0p125_a1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.gamma=0.125 trainer.method_args.alpha=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "RMU|RMU|unlearn/mytofu/default.yaml|paper_sc2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.module_regex=model\\.layers\\.7"
)

get_runs() {
  if [ "${SWEEP_PROFILE}" = "smoke" ]; then
    printf '%s\n' "${SMOKE_RUNS[@]}"
  else
    printf '%s\n' "${PAPER_TUNED_RUNS[@]}"
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

case "${1:-}" in
  --list)
    print_grid
    exit 0
    ;;
  --submit)
    submit_array
    exit 0
    ;;
esac

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "[ERROR] Run with --submit or submit this script as a SLURM array."
  echo "Example: bash $0 --submit"
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
echo "MYTOFU paper-inspired unlearn tuning"
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
  echo "This script only tunes unlearning from an existing full checkpoint."
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

echo "===== MYTOFU PAPER-TUNED UNLEARN RUN DONE ====="
