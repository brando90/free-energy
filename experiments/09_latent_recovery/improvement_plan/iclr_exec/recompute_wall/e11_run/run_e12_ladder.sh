#!/usr/bin/env bash
# E12 fixes-ladder haiku API driver: runs arms A0..A3 sequentially, then analysis.
# Cost-bounded per arm (--usd-cap) AND cumulatively (api_gen hard stop).
set -u
RW=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall
E11=$RW/e11_run
E12=$RW/e12_fixes
PY=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/exph/.venv-exph/bin/python
SHARED=$E12/programs.jsonl
LOG=$E12/ladder.log
mkdir -p $E12

# keys (never printed); the python refuses to run without ANTHROPIC_API_KEY
source /lfs/skampere2/0/eobbad/.keys/latent_recovery_api.env 2>/dev/null

echo "[ladder] start $(date -u +%FT%TZ)" | tee -a $LOG
for ARM in A0 A1 A2 A3; do
  echo "[ladder] === arm $ARM $(date -u +%FT%TZ) ===" | tee -a $LOG
  $PY $E11/e12_api.py --arm $ARM --out-base $E12 --programs $SHARED \
      --n-target 60 --R 3 --temp 0.7 --world-cap 110 --usd-cap 15 --seed 0 >>$LOG 2>&1
  rc=$?
  echo "[ladder] arm $ARM exit=$rc $(date -u +%FT%TZ)" | tee -a $LOG
  if [ $rc -ne 0 ]; then
    echo "[ladder] arm $ARM FAILED (rc=$rc); continuing to analysis with whatever completed" | tee -a $LOG
  fi
done

echo "[ladder] analysis $(date -u +%FT%TZ)" | tee -a $LOG
$PY $E11/e12_analyze.py --out-base $E12 >>$LOG 2>&1
echo "[ladder] done $(date -u +%FT%TZ)" | tee -a $LOG
touch $E12/.ladder_complete
