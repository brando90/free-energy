#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${HOME}/uv_envs/veribench"

source "${VENV}/bin/activate"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

exec python "${SCRIPT_DIR}/run_goedel8b_sglang_896.py" \
  --gpus "${CUDA_VISIBLE_DEVICES}" \
  --data-parallel-size 3 \
  --tensor-parallel-size 1 \
  "$@"
