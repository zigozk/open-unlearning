#!/bin/bash
#
# Extended MYTOFU sweep for testing PCGrad/SAGO on additional unlearning methods.
#
# This script starts from an existing full checkpoint. It does not fine-tune a
# base model. It focuses on methods whose forget and retain branches can be
# separated for gradient synthesis:
#   DPO, WGA, UNDIAL, SatImp
#
# Usage from a login node:
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_synthesis_ext_tuned_array.sh --list
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_synthesis_ext_tuned_array.sh --submit
#   bash sbatch/MYTOFU/unlearn/mytofu_unlearn_synthesis_ext_tuned_array.sh --submit-with-summary
#
# Useful overrides:
#   SWEEP_PROFILE=smoke MAX_PARALLEL=1 bash ... --submit-with-summary
#   OUTPUT_ROOT=saves/unlearn_synthesis_ext_tuned_v2 bash ... --submit-with-summary

#SBATCH -J mytofu_synth_ext
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
OUTPUT_ROOT="${OUTPUT_ROOT:-saves/unlearn_synthesis_ext_tuned}"
RANK_METRIC="${RANK_METRIC:-BUS}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SWEEP_PROFILE="${SWEEP_PROFILE:-full}"

# Fields:
# method_label|trainer|experiment|tag|learning_rate|epochs|forget_dataset|extra hydra args
declare -a EXT_SYNTH_RUNS=(
  # DPO: preference-style objective over original vs IDK answer.
  "DPO_none|DPO|unlearn/mytofu/idk.yaml|b0p05_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.05 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO_pcgrad|DPO|unlearn/mytofu/idk.yaml|b0p05_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.05 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "DPO_sago|DPO|unlearn/mytofu/idk.yaml|b0p05_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.05 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "DPO_none|DPO|unlearn/mytofu/idk.yaml|b0p1_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO_pcgrad|DPO|unlearn/mytofu/idk.yaml|b0p1_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "DPO_sago|DPO|unlearn/mytofu/idk.yaml|b0p1_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "DPO_none|DPO|unlearn/mytofu/idk.yaml|b0p2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "DPO_pcgrad|DPO|unlearn/mytofu/idk.yaml|b0p2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "DPO_sago|DPO|unlearn/mytofu/idk.yaml|b0p2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget_idk|trainer.method_args.beta=0.2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  # WGA: sweep the weighted-gradient beta and retain strength.
  "WGA_none|WGA|unlearn/mytofu/default.yaml|b0p5_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.5 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "WGA_pcgrad|WGA|unlearn/mytofu/default.yaml|b0p5_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.5 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "WGA_sago|WGA|unlearn/mytofu/default.yaml|b0p5_a0p5_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=0.5 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "WGA_none|WGA|unlearn/mytofu/default.yaml|b1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "WGA_pcgrad|WGA|unlearn/mytofu/default.yaml|b1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "WGA_sago|WGA|unlearn/mytofu/default.yaml|b1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "WGA_none|WGA|unlearn/mytofu/default.yaml|b2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=2.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "WGA_pcgrad|WGA|unlearn/mytofu/default.yaml|b2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=2.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "WGA_sago|WGA|unlearn/mytofu/default.yaml|b2_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=2.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  # UNDIAL: alpha=0 is the original forget-prioritized anchor; alpha>0 tests
  # whether retain branch guidance helps when combined through PCGrad/SAGO.
  "UNDIAL_none|UNDIAL|unlearn/mytofu/default.yaml|b5_a0_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=5.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL_pcgrad|UNDIAL|unlearn/mytofu/default.yaml|b5_a0_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=5.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "UNDIAL_sago|UNDIAL|unlearn/mytofu/default.yaml|b5_a0_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=5.0 trainer.method_args.alpha=0.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "UNDIAL_none|UNDIAL|unlearn/mytofu/default.yaml|b10_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL_pcgrad|UNDIAL|unlearn/mytofu/default.yaml|b10_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "UNDIAL_sago|UNDIAL|unlearn/mytofu/default.yaml|b10_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "UNDIAL_none|UNDIAL|unlearn/mytofu/default.yaml|b20_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "UNDIAL_pcgrad|UNDIAL|unlearn/mytofu/default.yaml|b20_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "UNDIAL_sago|UNDIAL|unlearn/mytofu/default.yaml|b20_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=20.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  # SatImp: paper-style beta1/beta2 anchors plus nearby retain/forget weights.
  "SatImp_none|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SatImp_pcgrad|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "SatImp_sago|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SatImp_none|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p5_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.5 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SatImp_pcgrad|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p5_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.5 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "SatImp_sago|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p5_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.5 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SatImp_none|SatImp|unlearn/mytofu/default.yaml|b10_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=10.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "SatImp_pcgrad|SatImp|unlearn/mytofu/default.yaml|b10_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=10.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "SatImp_sago|SatImp|unlearn/mytofu/default.yaml|b10_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=10.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
)

declare -a SMOKE_RUNS=(
  "WGA_pcgrad|WGA|unlearn/mytofu/default.yaml|b1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "WGA_sago|WGA|unlearn/mytofu/default.yaml|b1_a1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=1.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "UNDIAL_pcgrad|UNDIAL|unlearn/mytofu/default.yaml|b10_a0p1_g1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta=10.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"
  "SatImp_sago|SatImp|unlearn/mytofu/default.yaml|b5_1_a0p1_g0p1_lr5em6_e3|5e-6|3|MYTOFU_forget|trainer.method_args.beta1=5.0 trainer.method_args.beta2=1.0 trainer.method_args.alpha=0.1 trainer.method_args.gamma=0.1 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
)

get_runs() {
  if [ "${SWEEP_PROFILE}" = "smoke" ]; then
    printf '%s\n' "${SMOKE_RUNS[@]}"
  else
    printf '%s\n' "${EXT_SYNTH_RUNS[@]}"
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
      -J mytofu_synth_sum \
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
echo "MYTOFU synthesis extension tuning"
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

echo "===== MYTOFU SYNTHESIS EXTENSION RUN DONE ====="
