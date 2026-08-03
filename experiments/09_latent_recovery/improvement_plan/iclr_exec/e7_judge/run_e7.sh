#!/bin/bash
# E7 judge_v2 completion pass -- full launcher.
# Judges (a) EXPD_MATCHED_GRADIENT full grid (11,577 rows) and (b) EXPB's 118
# unparsed rows with the Qwen2.5-32B-Instruct judge (doubt_judge_v2.py),
# mirroring the exact invocation pattern of the prior judge_v2 runs
# (improvement_plan/tooling/judge_v2/run.log, results/EXPE_EVIDENCE_MOVER/
# full_run_judge.log).
#
# DOES NOT RUN TODAY: refuses to start unless a GPU shows <5GB used
# (nvidia-smi check below). GPUs are occupied by alexspan/sahasras as of
# 2026-07-22 -- do not touch their processes, do not force past this check.
#
# Usage (once a GPU frees):
#   bash run_e7.sh
# or pin a specific free GPU:
#   E7_GPU=3 bash run_e7.sh
set -e

EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
TOOLING=$EXP/improvement_plan/tooling
OUT=$EXP/improvement_plan/iclr_exec/e7_judge
PY=$TOOLING/.venv-vllm/bin/python
JOBS=$OUT/jobs_e7.json
LOG=$OUT/run_e7.log

MIN_FREE_GPU_CHECK_MB=5000   # refuse unless some GPU has <5GB used

# ---------------------------------------------------------------- GPU gate
gpu_line=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
if [ -z "$gpu_line" ]; then
  echo "REFUSING: nvidia-smi unavailable or returned nothing; cannot verify a GPU is free." >&2
  exit 1
fi

echo "nvidia-smi memory.used per GPU:"
echo "$gpu_line"

free_gpu=""
while IFS=, read -r idx used; do
  idx=$(echo "$idx" | tr -d ' ')
  used=$(echo "$used" | tr -d ' ')
  if [ "$used" -lt "$MIN_FREE_GPU_CHECK_MB" ]; then
    free_gpu=$idx
    break
  fi
done <<< "$gpu_line"

if [ -z "$free_gpu" ]; then
  echo "REFUSING: no GPU has <${MIN_FREE_GPU_CHECK_MB}MB used (all GPUs occupied -- lab members' jobs are running). Not launching vLLM." >&2
  exit 1
fi

GPU=${E7_GPU:-$free_gpu}
echo "Using CUDA_VISIBLE_DEVICES=$GPU (detected free: $free_gpu)"

# ------------------------------------------------------------ cleanup trap
# Known gotcha (JUDGE_V2_REPORT.md / STATUS.md): killed/interrupted vLLM
# runs leave orphaned EngineCore child processes holding GPU memory.
# EngineCore is a DIRECT CHILD of the vllm python process (confirmed on this
# node: ps shows EngineCore's ppid == the launching python's pid), which
# itself runs a level or two below this script's $$ (extra level if piped
# through `tee`) -- so a same-depth-only check misses it. Walk the FULL
# descendant tree of $$ (own user only) and kill any EngineCore found in it.
# This never touches other users' (alexspan/sahasras) EngineCore processes:
# their PPID chains do not trace back to this script's PID.
cleanup() {
  ec=$?
  echo "=== cleanup: scanning descendant tree of PID $$ for orphaned EngineCore ==="
  # BFS over children (this user's processes only)
  frontier="$$"
  all_descendants=""
  for _ in 1 2 3 4 5 6 7 8; do  # depth cap, generous for this pipeline depth
    [ -z "$frontier" ] && break
    next=""
    for p in $frontier; do
      kids=$(pgrep -P "$p" -u "$(whoami)" 2>/dev/null || true)
      next="$next $kids"
    done
    next=$(echo "$next" | tr ' ' '\n' | sed '/^$/d' | sort -u | tr '\n' ' ')
    [ -z "$(echo "$next" | tr -d ' ')" ] && break
    all_descendants="$all_descendants $next"
    frontier="$next"
  done
  killed_any=0
  for p in $all_descendants; do
    cmd=$(ps -o args= -p "$p" 2>/dev/null || true)
    case "$cmd" in
      *EngineCore*)
        echo "killing orphaned EngineCore pid=$p (descendant of script PID $$): $cmd"
        kill -9 "$p" 2>/dev/null || true
        killed_any=1
        ;;
    esac
  done
  [ "$killed_any" = 0 ] && echo "no EngineCore descendants of PID $$ found (clean)"
  exit $ec
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------- run
mkdir -p "$OUT/outputs"
source "$TOOLING/env.sh"
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

echo "=== E7 judge_v2 completion pass $(date -Is) ===" | tee "$LOG"
echo "jobs file: $JOBS" | tee -a "$LOG"
T0=$(date +%s)
cd "$TOOLING"
"$PY" doubt_judge_v2.py --jobs "$JOBS" 2>&1 | tee -a "$LOG"
T1=$(date +%s)
echo "=== E7 judge_v2 completion pass DONE $(date -Is), wall=${T1}-${T0}=$((T1-T0))s ===" | tee -a "$LOG"
