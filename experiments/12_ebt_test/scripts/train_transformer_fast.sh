#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Set WANDB_API_KEY before running this script."
  exit 1
fi

python generate_dataset.py --num-samples 10000
python train_transformer.py \
  wandb.enabled=true \
  wandb.mode=online \
  train.max_steps=2000 \
  loader.batch_size=128 \
  validation.every_steps=200 \
  train.log_every=20
