#!/usr/bin/env bash
# E13 API chain (launch agent, 2026-07-29): e13_api -> e13_inc -> e13_bridge.
# Bridge LAST + cell-cap 60 (spec allows 60-70): cumulative guard headroom to
# HARD_BUDGET_USD=150 measured at ~$26.4 pre-launch; projections ~$25.3.
# On BudgetExceeded the api_gen hard stop kills only the bridge tail.
set -u
RW=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall
PY=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/exph/.venv-exph/bin/python
E13=$RW/e13
LOG=$E13/logs/api_chain.log
STATUS=$E13/api_chain_status.json
export PYTHONDONTWRITEBYTECODE=1
# keys (never printed); each python hard-exits without the env keys
source /lfs/skampere2/0/eobbad/.keys/latent_recovery_api.env 2>/dev/null
cd "$E13"
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
setstat(){ printf '{"updated":"%s","stage":"%s","note":"%s","pid":%s}\n' "$(ts)" "$1" "$2" "$$" > "$STATUS"; }

setstat api_haiku start
echo "[chain] e13_api start $(ts)" >> "$LOG"
$PY e13_api.py --out-dir results/E13_haiku --n-target 60 --R 3 --temp 0.7 \
    --usd-cap 12 --seed 0 >> "$LOG" 2>&1
rc1=$?
echo "[chain] e13_api exit=$rc1 $(ts)" >> "$LOG"

setstat inc_A4 "api_rc=$rc1"
echo "[chain] e13_inc start $(ts)" >> "$LOG"
$PY e13_inc.py --out-base results/E13_inc --programs $RW/e12_fixes/programs.jsonl \
    --n-target 60 --world-cap 110 --R 3 --temp 0.7 --usd-cap 5 --seed 0 >> "$LOG" 2>&1
rc2=$?
echo "[chain] e13_inc exit=$rc2 $(ts)" >> "$LOG"

setstat bridge "api_rc=$rc1 inc_rc=$rc2"
echo "[chain] e13_bridge start $(ts)" >> "$LOG"
$PY e13_bridge.py --out-dir results/E13_bridge --cell-cap 60 --usd-cap 16 >> "$LOG" 2>&1
rc3=$?
echo "[chain] e13_bridge exit=$rc3 $(ts)" >> "$LOG"
setstat done "api_rc=$rc1 inc_rc=$rc2 bridge_rc=$rc3"
echo "[chain] DONE api=$rc1 inc=$rc2 bridge=$rc3 $(ts)" >> "$LOG"
