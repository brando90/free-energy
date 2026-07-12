#!/usr/bin/env bash
# TLDR: B2 — train the official EBT (xxs, paper recipe, from scratch) on VB-train; same formatting/tokenizer as B1.
# Usage: bash run_b2.sh [GPU_ID] [MAX_STEPS] [EBT_DIR] [SPLITS_DIR]
# Paper-recipe fallback (documented in results_b2.md): the EBT repo ships no pretrained
# LM checkpoints, so SFT-style training is not runnable as-is; we train from scratch
# with the canonical NLP recipe (job_scripts/nlp/pretrain/ebt_s1.sh) adapted to VB:
# Qwen tokenizer (CE comparability with B1), context 4096, small-data step budget.
set -euo pipefail
GPU_ID="${1:-1}"
MAX_STEPS="${2:-1400}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EBT_DIR="${3:-$SCRIPT_DIR/../../EBT}"   # default: EBT checkout at the workspace root (sibling of experiments/)
SPLITS_DIR="${4:-$SCRIPT_DIR/../08_vb_train_val_test/splits}"
EBT_COMMIT="19420cb"

if [ ! -d "$EBT_DIR" ]; then
  git clone https://github.com/alexiglad/EBT "$EBT_DIR"
fi
git -C "$EBT_DIR" checkout -q "$EBT_COMMIT"
python "$SCRIPT_DIR/ebt_integration/apply_ebt_integration.py" "$EBT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$EBT_DIR"
python train_model.py \
  --run_name "ebt-xxs-vb-steps${MAX_STEPS}" \
  --modality "NLP" \
  --model_name "ebt" \
  --model_size "xxs" \
  --dataset_name "veribench" \
  --dataset_dir "$(cd "$SPLITS_DIR" && pwd)" \
  --pretokenize_dataset \
  --tokenizer "Qwen/Qwen2.5-0.5B" \
  --context_length 2048 \
  --normalize_initial_condition \
  --ebt_type "time_embed" \
  --denoising_initial_condition "random_noise" \
  --mcmc_step_size_learnable \
  --mcmc_step_size 0.5 \
  --mcmc_step_size_lr_multiplier 1.5 \
  --mcmc_num_steps 2 \
  --gpus "-1" \
  --peak_learning_rate 0.0012 \
  --batch_size_per_device 1 \
  --accumulate_grad_batches 16 \
  --gradient_clip_val 1.0 \
  --weight_decay 0.01 \
  --min_lr_scale 10 \
  --max_steps "$MAX_STEPS" \
  --max_scheduling_steps "$MAX_STEPS" \
  --warm_up_steps 100 \
  --check_val_every_n_epoch 1 \
  --save_top_k_ckpts 1 \
  --float_precision "bf16-mixed" \
  --set_matmul_precision "high" \
  --num_workers 8 \
  --no_wandb
