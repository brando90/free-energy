#!/bin/bash
# E16 confirm driver: wait for a free GPU, run the confirm battery.
# Usage: run_e16_confirm.sh <model_key> '<conditions JSON>'
set -u
M="$1"
CONDS="$2"
E16=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e16_steering
LOG=$E16/logs
mkdir -p "$LOG"
PY=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/tooling/.venv-vllm/bin/python
export HF_HOME=/lfs/skampere2/0/eobbad/.cache/huggingface

echo "$(date -u +%FT%TZ) confirm: waiting for a free GPU" >> "$LOG/$M.confirm.log"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
  if [ -n "${FREE:-}" ]; then break; fi
  sleep 120
done
export CUDA_VISIBLE_DEVICES=$FREE
echo "$(date -u +%FT%TZ) confirm: claimed GPU $FREE" >> "$LOG/$M.confirm.log"
cd "$E16/code" || exit 1
"$PY" e16_run.py --model "$M" --stage confirm --conditions "$CONDS" >> "$LOG/$M.confirm.log" 2>&1 || { echo "$(date -u +%FT%TZ) CONFIRM_FAIL" >> "$LOG/$M.confirm.log"; exit 1; }
echo "$(date -u +%FT%TZ) CONFIRM_DONE" >> "$LOG/$M.confirm.log"
