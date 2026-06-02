#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ "$(uname -s)" == "Darwin" && "${PYTHON_BIN}" == ".venv/bin/python" ]]; then
  PYTHON_CMD=(arch -arm64 "${PYTHON_BIN}")
else
  PYTHON_CMD=("${PYTHON_BIN}")
fi

"${PYTHON_CMD[@]}" experiments/05_energy_based_transformer_baseline/run_ebt_toy.py \
  --tag smoke \
  --output-dir experiments/05_energy_based_transformer_baseline/results/smoke_toy \
  --device cpu \
  --require-ebt-signal

VERIBENCH_ROOT="${VERIBENCH_ROOT:-$HOME/veribench}"
VERIBENCH_DATASET_ROOT="${VERIBENCH_DATASET_ROOT:-${VERIBENCH_ROOT}/veribench_dataset}"
SPLIT_DIR="experiments/02_ar_pros_cons/data/splits"
if [[ -d "${VERIBENCH_DATASET_ROOT}" ]]; then
  if [[ ! -f "${SPLIT_DIR}/train.jsonl" || ! -f "${SPLIT_DIR}/val.jsonl" || ! -f "${SPLIT_DIR}/test.jsonl" ]]; then
    (
      cd experiments/02_ar_pros_cons
      if [[ -x ../../.venv/bin/python ]]; then
        if [[ "$(uname -s)" == "Darwin" ]]; then
          SETUP_PYTHON_CMD=(arch -arm64 ../../.venv/bin/python)
        else
          SETUP_PYTHON_CMD=(../../.venv/bin/python)
        fi
      else
        SETUP_PYTHON_CMD=("${PYTHON_CMD[@]}")
      fi
      "${SETUP_PYTHON_CMD[@]}" -m data.setup \
        --veribench-root "${VERIBENCH_DATASET_ROOT}" \
        --include-generated-agents \
        --smoke
    )
  fi

  "${PYTHON_CMD[@]}" experiments/05_energy_based_transformer_baseline/run_veribench_ebt_ranking.py \
    --use-splits \
    --tag split_smoke \
    --output-dir experiments/05_energy_based_transformer_baseline/results/split_smoke \
    --split-dir "${SPLIT_DIR}" \
    --veribench-root "${VERIBENCH_DATASET_ROOT}" \
    --max-split-train-tasks 12 \
    --max-split-val-tasks 4 \
    --max-split-test-tasks 4 \
    --epochs 3 \
    --hidden-dim 24 \
    --num-layers 1 \
    --num-heads 4 \
    --max-length 512 \
    --batch-size 8 \
    --eval-batch-size 8 \
    --device cpu
else
  echo "[smoke] skipping VeriBench split reranker because VERIBENCH_DATASET_ROOT is missing: ${VERIBENCH_DATASET_ROOT}"
fi
