#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

RULETAKER_MAX_ITEMS="${RULETAKER_MAX_ITEMS:-2048}"
MAX_STEPS="${MAX_STEPS:-50000}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-512}"
MODEL_SIZE="${MODEL_SIZE:-2xs}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-1}"
TRAIN_METRIC_LOG_BATCH_SIZE="${TRAIN_METRIC_LOG_BATCH_SIZE:-0}"
AGGREGATE_TRAIN_METRICS_OVER_ACCUMULATION="${AGGREGATE_TRAIN_METRICS_OVER_ACCUMULATION:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GPU_COUNT="${GPU_COUNT:-1}"
RUN_NAME="${RUN_NAME:-ruletaker_ebt_2xs_mcmc4_smoke}"
RULETAKER_TRAIN_AR_INTERVAL="${RULETAKER_TRAIN_AR_INTERVAL:-500}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-200}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-256}"
VALIDATE_BEFORE_TRAINING="${VALIDATE_BEFORE_TRAINING:-0}"

EXTRA_ARGS=()
if [[ "${AGGREGATE_TRAIN_METRICS_OVER_ACCUMULATION}" == "1" ]]; then
  EXTRA_ARGS+=(--aggregate_train_metrics_over_accumulation)
fi
if [[ "${VALIDATE_BEFORE_TRAINING}" == "1" ]]; then
  EXTRA_ARGS+=(--validate_before_training)
fi

"${PYTHON_BIN}" train_model.py \
  --run_name "${RUN_NAME}" \
  --modality NLP \
  --model_name ebt \
  --model_size "${MODEL_SIZE}" \
  --dataset_name ruletaker \
  --tokenizer EleutherAI/gpt-neox-20b \
  --context_length "${CONTEXT_LENGTH}" \
  --mcmc_num_steps 4 \
  --mcmc_step_size 0.75 \
  --denoising_initial_condition random_noise \
  --gaussian_random_noise_scaling 1.0 \
  --normalize_initial_condition \
  --vocab_to_embed_uses_prob_dist \
  --gpus "${GPU_COUNT}" \
  --distributed_strategy auto \
  --batch_size_per_device "${BATCH_SIZE}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES}" \
  --train_metric_log_batch_size "${TRAIN_METRIC_LOG_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --ruletaker_max_items "${RULETAKER_MAX_ITEMS}" \
  --ruletaker_train_ar_interval "${RULETAKER_TRAIN_AR_INTERVAL}" \
  --peak_learning_rate 0.0003 \
  --warm_up_steps 200 \
  --max_steps "${MAX_STEPS}" \
  --limit_val_batches "${LIMIT_VAL_BATCHES}" \
  --gradient_clip_val 1.0 \
  --checkpoint_monitor_string val/autoregressive_exact_accuracy \
  --checkpoint_monitor_mode max \
  --save_top_k_ckpts 3 \
  --wandb_project free-energy-ruletaker \
  --wandb_tags ruletaker ebt mcmc4 slash-metrics \
  --log_every_n_steps 1 \
  --val_check_interval "${VAL_CHECK_INTERVAL}" \
  "${EXTRA_ARGS[@]}"
