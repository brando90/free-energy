#!/bin/bash
set -u
V=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/exph/.venv-exph/bin/python
D=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e10_topup_fable
LOG="$D/run.log"; GATE=98.0
cd "$D"
echo "########## RUN_ALL start $(date -Is) ##########" >> "$LOG"
for spec in "claude-opus-4-8 yes" "claude-sonnet-5 yes" "claude-fable-5-off yes" "claude-fable-5-on no" "claude-sonnet-4-5 yes"; do
  set -- $spec; KEY="$1"; DG="$2"
  C=$("$V" cum.py 2>/dev/null)
  echo "[gate] before $KEY cumulative=\$$C (gate=$GATE) $(date -Is)" >> "$LOG"
  if awk "BEGIN{exit !($C >= $GATE)}"; then
    echo "[STOP] budget gate: cumulative \$$C >= \$$GATE; halting before $KEY" >> "$LOG"; break
  fi
  bash run_model.sh "$KEY" "$DG" || echo "[$KEY] run_model returned nonzero" >> "$LOG"
done
"$V" e10topup.py summarize >> "$LOG" 2>&1
echo "########## RUN_ALL complete $(date -Is) ##########" >> "$LOG"
