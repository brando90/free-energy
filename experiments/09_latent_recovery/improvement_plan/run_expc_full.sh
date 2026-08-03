#!/bin/bash
set -e
cd /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
PY=/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python
OUT=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/results/EXPC_POLARITY_CONTROL_FULL
export CUDA_VISIBLE_DEVICES=7
export HF_HOME=/lfs/skampere2/0/eobbad/.cache/huggingface
for stage in test prepare generate validate summarize manual-audit report; do
  echo "=== stage: $stage $(date -Is) ==="
  $PY src/expc_polarity_control.py $stage --out-dir $OUT --target 127 --device cuda:0
done
echo "=== ALL STAGES DONE $(date -Is) ==="
