#!/usr/bin/env bash
# E11 open-model queue (skampere1). Sequential, one GPU (pin via CUDA_VISIBLE_DEVICES).
# Smallest/anchor first (qwen7b) then llama8b. Resume-safe per stage (raw_generations
# keyed by run_id). Rsyncs each model's results back to skampere2 (RESULTS RULE).
set -u
IP=/var/tmp/eobbad/recompute_wall/free-energy/experiments/09_latent_recovery/improvement_plan
PY=$IP/tooling/.venv-vllm/bin/python
RUN=$IP/iclr_exec/recompute_wall/e11_run
S2=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e11_run
SSH="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no"

# shellcheck disable=SC1090
source "$IP/tooling/env.sh"
export EXPG_LLM_MAX_MODEL_LEN=8192          # node-fit cap (FIX_NOTE precedent); prompts <<8192 -> byte-identical decoding
export TOKENIZERS_PARALLELISM=false
cd "$RUN" || exit 2

STATUS=$RUN/queue_status.json
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
setstat(){ printf '{"updated":"%s","current":"%s","note":"%s","gpu":"%s","pid":%s}\n' \
  "$(ts)" "$1" "$2" "${CUDA_VISIBLE_DEVICES:-unset}" "$$" > "$STATUS"; }

# RUN-PHASE solvability collapse (2-digit operands): measured qwen7b gold solve
# ~0.10 overall (k0 .155 -> k5 .01) vs EXPG pilot .68->.45. Heavy oversampling
# (spec S6.2) to power k0-k3 to n=150; k5 stays truncated (0.01 solve -> ~18
# eligible even at pool 1800) and is reported INVALID/truncated, not gap-filled.
MODELS=("Qwen/Qwen2.5-7B-Instruct:qwen7b:1800"
        "NousResearch/Meta-Llama-3.1-8B-Instruct:llama8b:1800")

setstat "START" "queue up"
for spec in "${MODELS[@]}"; do
  IFS=':' read -r MODEL TAG POOL <<< "$spec"
  OUT=$RUN/results/E11_$TAG
  LOG=$RUN/logs_$TAG
  mkdir -p "$OUT"

  setstat "$TAG:prepare" "pool=$POOL"
  $PY e11_run.py prepare  --out-dir "$OUT" --world-pool "$POOL" --n-target 150 --overwrite \
      > "$LOG.prepare.log" 2>&1

  setstat "$TAG:generate" "gold+cont R8+greedy"
  $PY e11_run.py generate --out-dir "$OUT" --model "$MODEL" --backend vllm \
      --n-target 150 --R 8 --temp 0.7 --batch 256 > "$LOG.generate.log" 2>&1
  grc=$?

  setstat "$TAG:validate" "rc_gen=$grc"
  $PY e11_run.py validate  --out-dir "$OUT" --R 8 > "$LOG.validate.log" 2>&1
  $PY e11_run.py summarize --out-dir "$OUT"        > "$LOG.summarize.log" 2>&1
  $PY e11_run.py report    --out-dir "$OUT"        > "$LOG.report.log" 2>&1

  setstat "$TAG:rsync" "rc_gen=$grc"
  $SSH -q true 2>/dev/null
  rsync -a -e "$SSH" "$OUT/" "skampere2:$S2/results/E11_$TAG/" >> "$LOG.rsync.log" 2>&1
  setstat "$TAG:done" "rc_gen=$grc"
done
setstat "ALL_DONE" "queue complete"
