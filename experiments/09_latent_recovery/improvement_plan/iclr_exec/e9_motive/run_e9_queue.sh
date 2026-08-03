#!/bin/bash
# E9 motive-experiment queued runner / waiter.
#
# Consumes NO GPU while waiting. Polls every 10 min for THREE gates, all of which
# must hold before any generation starts (registration sections 0.1, 6, 12):
#
#   GATE 1  E1 replication queue is TERMINAL
#           file: iclr_exec/e1_replication/queue_status.json, field .state in
#           {done, complete, completed, terminal, failed, aborted, error}.
#           FAIL-CLOSED: if the file is ABSENT or unparByteable, treat as NOT terminal
#           and keep waiting (the E1 dir does not yet exist as of 2026-07-24).
#   GATE 2  Some GPU is FREE (memory.used < E9_FREE_MB, default 1024 ~= idle/"0 MiB").
#   GATE 3  Human GO sentinel exists: <run-dir>/E9_HUMAN_GO  (a JSON file the human
#           writes ONLY after resolving G-D1..G-D4 in E9_REGISTRATION.md section 0.1).
#           Its fields drive the run: {"n_families":500,"final_n":300,
#           "gd2_in_family":true,"surprise_control":false,"verifications":["low","high"]}.
#           Missing fields fall back to the registered defaults below.
#
# THIS SCRIPT IS ARMED BUT HUMAN-GATED: without E9_HUMAN_GO it polls forever and
# never touches a GPU. Do NOT remove Gate 3 to "just run it" -- the two structural
# identification threats (surprise / compliance-interference) are unresolved until a
# human signs off (see REPORT.md).
#
# Usage (waiter, no GPU):   nohup bash run_e9_queue.sh >> waiter.out 2>&1 &
set -u

EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
OUT=$EXP/improvement_plan/iclr_exec/e9_motive
RUN_DIR=${E9_RUN_DIR:-$OUT/run_confirmatory}
E1_STATUS=$EXP/improvement_plan/iclr_exec/e1_replication/queue_status.json
GO=$RUN_DIR/E9_HUMAN_GO
STATUS=$RUN_DIR/queue_status.json
PY=$EXP/../../.venv/bin/python
VPY=$EXP/improvement_plan/tooling/.venv-vllm/bin/python
POLL=${E9_POLL_SEC:-600}
E9_FREE_MB=${E9_FREE_MB:-1024}
MODEL=Qwen/Qwen2.5-7B-Instruct
REV=a09a35458c702b33eeacc393d103063234e8bc28

mkdir -p "$RUN_DIR"
log() { echo "[$(date -Is)] $*" ; }
write_status() { echo "{\"state\":\"$1\",\"detail\":\"$2\",\"heartbeat\":\"$(date -Is)\",\"pid\":$$}" > "$STATUS" ; }

write_status "waiting" "startup"
log "E9 waiter armed (PID $$). run_dir=$RUN_DIR poll=${POLL}s"
log "gates: E1-terminal($E1_STATUS) + GPU-free(<${E9_FREE_MB}MB) + human-GO($GO)"

# ---------- gate checks --------------------------------------------------------
e1_terminal() {
  [ -f "$E1_STATUS" ] || return 1
  st=$("$PY" -c "import json;print(json.load(open('$E1_STATUS')).get('state','').lower())" 2>/dev/null) || return 1
  case "$st" in
    done|complete|completed|terminal|failed|aborted|error) return 0 ;;
    *) return 1 ;;
  esac
}

free_gpu() {   # echoes a free GPU index, or empty
  line=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null) || return 1
  echo "$line" | while IFS=, read -r idx used; do
    idx=$(echo "$idx" | tr -d ' '); used=$(echo "$used" | tr -d ' ')
    if [ -n "$used" ] && [ "$used" -lt "$E9_FREE_MB" ]; then echo "$idx"; break; fi
  done
}

# ---------- EngineCore cleanup trap (verbatim pattern from e7_judge/run_e7.sh) --
cleanup() {
  ec=$?
  log "cleanup: scanning descendant tree of PID $$ for orphaned EngineCore"
  frontier="$$"; all=""
  for _ in 1 2 3 4 5 6 7 8; do
    [ -z "$frontier" ] && break; next=""
    for p in $frontier; do next="$next $(pgrep -P "$p" -u "$(whoami)" 2>/dev/null)"; done
    next=$(echo "$next" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')
    [ -z "$(echo "$next" | tr -d ' ')" ] && break
    all="$all $next"; frontier="$next"
  done
  for p in $all; do
    cmd=$(ps -o args= -p "$p" 2>/dev/null || true)
    case "$cmd" in *EngineCore*) log "killing orphaned EngineCore pid=$p"; kill -9 "$p" 2>/dev/null ;; esac
  done
  exit $ec
}
trap cleanup EXIT INT TERM

# ---------- wait loop ----------------------------------------------------------
GPU=""
while : ; do
  g1=no; g2=no; g3=no
  e1_terminal && g1=yes
  GPU=$(free_gpu); [ -n "$GPU" ] && g2=yes
  [ -f "$GO" ] && g3=yes
  write_status "waiting" "E1=$g1 GPUfree=$g2(idx=${GPU:-none}) humanGO=$g3"
  log "gates: E1-terminal=$g1  GPU-free=$g2 (idx=${GPU:-none})  human-GO=$g3"
  if [ "$g1" = yes ] && [ "$g2" = yes ] && [ "$g3" = yes ]; then
    log "ALL GATES CLEAR -> launching E9 on GPU $GPU"; break
  fi
  sleep "$POLL"
done

# ---------- read human decisions ----------------------------------------------
NF=$("$PY" -c "import json;print(json.load(open('$GO')).get('n_families',500))" 2>/dev/null || echo 500)
GD2=$("$PY" -c "import json;print(str(json.load(open('$GO')).get('gd2_in_family',True)).lower())" 2>/dev/null || echo true)
SCTRL=$("$PY" -c "import json;print('--surprise-control' if json.load(open('$GO')).get('surprise_control',False) else '')" 2>/dev/null || echo "")
VERIFS=$("$PY" -c "import json;print(' '.join(json.load(open('$GO')).get('verifications',['low','high'])))" 2>/dev/null || echo "low high")
log "decisions: n_families=$NF gd2_in_family=$GD2 surprise_control='$SCTRL' verifications='$VERIFS'"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
source "$EXP/improvement_plan/tooling/env.sh" 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=$GPU   # env.sh may reset; re-pin

run() { log "RUN: $*"; "$@" || { write_status "failed" "step:$*"; log "STEP FAILED: $*"; exit 1; } ; }

write_status "running" "prepare"
run "$PY"  "$OUT/e9_run.py" prepare  --run-dir "$RUN_DIR" --seed 20260724 --n-families "$NF" $SCTRL
write_status "running" "gold"
run "$VPY" "$OUT/e9_run.py" gold     --run-dir "$RUN_DIR" --model "$MODEL" --model-revision "$REV"
write_status "running" "pair"
run "$PY"  "$OUT/e9_run.py" pair     --run-dir "$RUN_DIR"
write_status "running" "rollout"
run "$VPY" "$OUT/e9_run.py" rollout  --run-dir "$RUN_DIR" --R 8 --temperature 0.7 --verification $VERIFS
write_status "running" "score"
run "$PY"  "$OUT/e9_score.py"   --run-dir "$RUN_DIR"
write_status "running" "analyze"
GD2FLAG=""; [ "$GD2" = true ] && GD2FLAG="--gd2-in-family"
run "$PY"  "$OUT/e9_analyze.py" --run-dir "$RUN_DIR" $GD2FLAG

write_status "done" "complete"
log "E9 run COMPLETE. results: $RUN_DIR/e9_results.json"
