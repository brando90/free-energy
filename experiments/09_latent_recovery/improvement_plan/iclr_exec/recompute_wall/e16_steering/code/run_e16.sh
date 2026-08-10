#!/bin/bash
# E16 driver: wait for a free GPU, then extract -> audit -> screen.
# Usage: run_e16.sh <model_key>   (plain script, setsid-safe: no functions)
set -u
M="$1"
E16=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e16_steering
CODE=$E16/code
LOG=$E16/logs
mkdir -p "$LOG"
PY=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/tooling/.venv-vllm/bin/python
export HF_HOME=/lfs/skampere2/0/eobbad/.cache/huggingface

echo "$(date -u +%FT%TZ) waiting for a free GPU (<2000 MiB used)" >> "$LOG/$M.log"
while true; do
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1; exit}')
  if [ -n "${FREE:-}" ]; then break; fi
  sleep 120
done
export CUDA_VISIBLE_DEVICES=$FREE
echo "$(date -u +%FT%TZ) claimed GPU $FREE" >> "$LOG/$M.log"

cd "$CODE" || exit 1
echo "$(date -u +%FT%TZ) STAGE extract" >> "$LOG/$M.log"
"$PY" e16_extract.py --model "$M" >> "$LOG/$M.log" 2>&1 || { echo "$(date -u +%FT%TZ) EXTRACT_FAIL" >> "$LOG/$M.log"; exit 1; }
echo "$(date -u +%FT%TZ) STAGE audit" >> "$LOG/$M.log"
"$PY" e16_run.py --model "$M" --stage audit >> "$LOG/$M.log" 2>&1 || { echo "$(date -u +%FT%TZ) AUDIT_FAIL" >> "$LOG/$M.log"; exit 1; }
echo "$(date -u +%FT%TZ) STAGE screen" >> "$LOG/$M.log"
"$PY" e16_run.py --model "$M" --stage screen >> "$LOG/$M.log" 2>&1 || { echo "$(date -u +%FT%TZ) SCREEN_FAIL" >> "$LOG/$M.log"; exit 1; }
echo "$(date -u +%FT%TZ) SCREEN_DONE" >> "$LOG/$M.log"
