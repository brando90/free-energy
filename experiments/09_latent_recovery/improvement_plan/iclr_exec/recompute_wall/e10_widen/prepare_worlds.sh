#!/bin/bash
# E10 WIDEN -- CPU-only world preparation (run BEFORE arming the GPU).
# Runs the locked EXPG `prepare` stage for every model out-dir. Worlds are
# model-independent + deterministic (fixed per-shape seed base + knobs), so all
# five programs.jsonl are identical substrates; each model injects its OWN gold
# rollouts later in `generate`. prepare also runs the fail-closed generation-time
# audits (MAJOR-5 / true-values-distinct / no-axis-c-delta / MOD-9/10) and records
# them in each run_metadata.json's generator_funnels.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/e10_config.sh"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RESULTS" "$E10DIR/logs"

for entry in "${E10_MODELS[@]}"; do
  IFS='|' read -r slug repo rev <<< "$entry"
  OUT="$(e10_outdir "$slug")"
  mkdir -p "$OUT"
  log="$E10DIR/logs/${slug}.prepare.log"
  echo "=== [prepare $slug] $(date -Is) -> $OUT ===" | tee "$log"
  "$PY_MAIN" "$EXPG_DRIVER" prepare --out-dir "$OUT" \
    --model "$repo" --revision "$rev" \
    --cells "$CELLS" --cap "$CAP" --oversample "$OVERSAMPLE" --seed "$SEED" \
    ${1:+--overwrite} $CONFIRM_FLAGS >>"$log" 2>&1
  rc=$?
  echo "=== [prepare $slug] rc=$rc $(date -Is) ===" | tee -a "$log"
  [ "$rc" -ne 0 ] && { echo "PREPARE FAILED for $slug (rc=$rc); see $log"; exit "$rc"; }
done
echo "ALL WORLDS PREPARED"
