#!/bin/bash
#SBATCH -J mytofu_tune
#SBATCH -p compute
#SBATCH -N 1
#SBATCH --gres=gpu:a100-pcie-40gb:1
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

MODEL="${MODEL:-Llama-3.2-1B-Instruct}"
MODEL_PATH="${MODEL_PATH:-/home/zkzhang/unlearning/open-unlearning/saves/finetune/mytofu_Llama-3.2-1B-Instruct_full_e10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-saves/unlearn_tuned}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Use "0 1 2" for a robustness table after the first sweep identifies good
# regions. Keep the default at one seed so the initial sweep is affordable.
SEEDS="${SEEDS:-0}"

# Fields:
# method_label|trainer|experiment|tag|learning_rate|epochs|extra hydra args
#
# Tags are intentionally compact and filesystem-safe:
#   b=beta, a=alpha, g=gamma, sc=RMU steering_coeff, i=CEU ignored answer tokens.
# The set below is a small, defensible sweep around the current defaults:
# it varies the forget strength and retain regularization instead of searching
# every possible knob.
TUNED_RUNS=(
  "CEU|CEU|unlearn/mytofu/default.yaml|i1_lr5em6_e3|5e-6|3|trainer.method_args.ignore_first_n_answer_tokens=1"
  "CEU|CEU|unlearn/mytofu/default.yaml|i2_lr5em6_e3|5e-6|3|trainer.method_args.ignore_first_n_answer_tokens=2"
  "CEU|CEU|unlearn/mytofu/default.yaml|i1_lr1em5_e2|1e-5|2|trainer.method_args.ignore_first_n_answer_tokens=1"

  "NPO|NPO|unlearn/mytofu/default.yaml|b0p05_a1_g1_lr1em5_e3|1e-5|3|trainer.method_args.beta=0.05 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO|NPO|unlearn/mytofu/default.yaml|b0p1_a0p5_g1_lr1em5_e3|1e-5|3|trainer.method_args.beta=0.1 trainer.method_args.alpha=0.5 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "NPO_sago|NPO|unlearn/mytofu/default.yaml|b0p1_a1_g1_lr5em6_e3|5e-6|3|trainer.method_args.beta=0.1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"

  "RMU|RMU|unlearn/mytofu/default.yaml|sc1_a1_g1_lr5em6_e3|5e-6|3|trainer.method_args.steering_coeff=1 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "RMU|RMU|unlearn/mytofu/default.yaml|sc2_a1_g1_lr5em6_e3|5e-6|3|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0"
  "RMU|RMU|unlearn/mytofu/default.yaml|sc2_a2_g1_lr5em6_e3|5e-6|3|trainer.method_args.steering_coeff=2 trainer.method_args.alpha=2.0 trainer.method_args.gamma=1.0"

  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|none_a1_g1_lr5em6_e3|5e-6|3|trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"
  "GradDiff_sago|GradDiff|unlearn/mytofu/default.yaml|a1_g1_lr5em6_e3|5e-6|3|trainer.method_args.alpha=1.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "GradDiff|GradDiff|unlearn/mytofu/default.yaml|none_a2_g1_lr5em6_e3|5e-6|3|trainer.method_args.alpha=2.0 trainer.method_args.gamma=1.0 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=none"

  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|b2_g0p125_lr5em6_e3|5e-6|3|trainer.method_args.delta=0.0 trainer.method_args.beta=2.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=0.125 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SimNPO_sago|SimNPO|unlearn/mytofu/default.yaml|b4p5_g0p125_lr5em6_e3|5e-6|3|trainer.method_args.delta=0.0 trainer.method_args.beta=4.5 trainer.method_args.alpha=1.0 trainer.method_args.gamma=0.125 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=sago"
  "SimNPO_pcgrad|SimNPO|unlearn/mytofu/default.yaml|b2_g0p125_lr5em6_e3|5e-6|3|trainer.method_args.delta=0.0 trainer.method_args.beta=2.0 trainer.method_args.alpha=1.0 trainer.method_args.gamma=0.125 trainer.method_args.use_retain_loss=true trainer.method_args.gradient_synthesis=pcgrad"

  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|lr1em6_e1|1e-6|1|"
  "GradAscent|GradAscent|unlearn/mytofu/grad_ascent.yaml|lr2em6_e1|2e-6|1|"
)

echo "=================================================="
echo "HOSTNAME=$(hostname)"
echo "MODEL=${MODEL}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "SEEDS=${SEEDS}"
echo "RUN_FINAL_EVAL=${RUN_FINAL_EVAL}"
echo "SKIP_EXISTING=${SKIP_EXISTING}"
echo "=================================================="
nvidia-smi || true
mkdir -p logs "${OUTPUT_ROOT}"

for seed in ${SEEDS}; do
  for run_spec in "${TUNED_RUNS[@]}"; do
    IFS='|' read -r method_label trainer experiment tag learning_rate epochs extra_arg_string <<< "${run_spec}"

    full_tag="${tag}_s${seed}"
    task_name="mytofu_${MODEL}_${method_label}_${full_tag}_from_full_e10"
    run_dir="${OUTPUT_ROOT}/${task_name}"
    final_summary="${run_dir}/evals_final/MYTOFU_SUMMARY.json"

    echo "=================================================="
    echo "seed=${seed}"
    echo "method_label=${method_label}"
    echo "trainer=${trainer}"
    echo "experiment=${experiment}"
    echo "tag=${full_tag}"
    echo "learning_rate=${learning_rate}"
    echo "epochs=${epochs}"
    echo "run_dir=${run_dir}"
    echo "=================================================="

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${final_summary}" ]; then
      echo "[SKIP] Found final summary: ${final_summary}"
      continue
    fi

    extra_args=()
    if [ -n "${extra_arg_string}" ]; then
      read -r -a extra_args <<< "${extra_arg_string}"
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
      trainer.args.learning_rate=${learning_rate} \
      trainer.args.num_train_epochs=${epochs} \
      trainer.args.seed=${seed} \
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
  done
done

echo "===== MYTOFU TUNED SWEEP DONE ====="
echo "Collect with:"
echo "MYTOFU_RESULT_ROOT=${OUTPUT_ROOT} python sbatch/MYTOFU/unlearn/collect_mytofu_results.py"
