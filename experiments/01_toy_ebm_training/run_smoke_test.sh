#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  PYTHON_CMD=(arch -arm64 "${PYTHON_BIN}")
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi

"${PYTHON_CMD[@]}" experiments/01_toy_ebm_training/run_toy_ebm.py \
  --tag smoke \
  --output-dir experiments/01_toy_ebm_training/results/smoke \
  --models linear mlp \
  --seq-len 6 \
  --num-train-tasks 12 \
  --num-test-tasks 4 \
  --epochs 20 \
  --batch-size 4 \
  --pair-batch-size 512 \
  --hidden-dim 32 \
  --lr 0.01 \
  --device cpu \
  --require-improvement

