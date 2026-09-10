#!/bin/bash
# run_model.sh <key> <dogold:yes|no>  -- gold(optional)+perturb+validate for one model key.
set -u
V=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/exph/.venv-exph/bin/python
D=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e10_topup_fable
source /lfs/skampere2/0/eobbad/.keys/latent_recovery_api.env
export PYTHONDONTWRITEBYTECODE=1
cd "$D"
KEY="$1"; DOGOLD="${2:-yes}"
LOG="$D/run.log"
echo "===== $KEY $(date -Is) dogold=$DOGOLD =====" >> "$LOG"
if [ "$DOGOLD" = yes ]; then "$V" e10topup.py gold --model "$KEY" >> "$LOG" 2>&1 || { echo "[$KEY] gold FAILED rc=$?" >> "$LOG"; exit 1; }; fi
"$V" e10topup.py perturb  --model "$KEY" >> "$LOG" 2>&1 || { echo "[$KEY] perturb FAILED rc=$?" >> "$LOG"; exit 1; }
"$V" e10topup.py validate --model "$KEY" >> "$LOG" 2>&1 || { echo "[$KEY] validate FAILED rc=$?" >> "$LOG"; exit 1; }
echo "===== $KEY DONE $(date -Is) =====" >> "$LOG"
