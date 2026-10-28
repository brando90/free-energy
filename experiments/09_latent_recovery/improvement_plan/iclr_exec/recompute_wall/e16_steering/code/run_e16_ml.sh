#!/bin/bash
# E16 Amendment 2 driver: wait for GPU -> all-layer extraction + V3 probe ->
# multi-layer screen (V3 arm auto-gated by the probe result).
set -u
M="$1"
E16=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e16_steering
LOG=$E16/logs
mkdir -p "$LOG"
PY=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/tooling/.venv-vllm/bin/python
export HF_HOME=/lfs/skampere2/0/eobbad/.cache/huggingface

echo "$(date -u +%FT%TZ) ml: waiting for free GPU" >> "$LOG/$M.ml.log"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
  if [ -n "${FREE:-}" ]; then break; fi
  sleep 120
done
export CUDA_VISIBLE_DEVICES=$FREE
echo "$(date -u +%FT%TZ) ml: claimed GPU $FREE" >> "$LOG/$M.ml.log"
cd "$E16/code" || exit 1
echo "$(date -u +%FT%TZ) STAGE extract_ml (V3 probe)" >> "$LOG/$M.ml.log"
"$PY" e16_extract_ml.py --model "$M" >> "$LOG/$M.ml.log" 2>&1 || { echo "$(date -u +%FT%TZ) EXTRACT_ML_FAIL" >> "$LOG/$M.ml.log"; exit 1; }
echo "$(date -u +%FT%TZ) PROBE_DONE" >> "$LOG/$M.ml.log"
echo "$(date -u +%FT%TZ) STAGE screen_ml" >> "$LOG/$M.ml.log"
"$PY" e16_screen_ml.py --model "$M" >> "$LOG/$M.ml.log" 2>&1 || { echo "$(date -u +%FT%TZ) SCREEN_ML_FAIL" >> "$LOG/$M.ml.log"; exit 1; }
echo "$(date -u +%FT%TZ) SCREEN_ML_DONE" >> "$LOG/$M.ml.log"
