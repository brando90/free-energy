#!/usr/bin/env bash
# E13 open-model queue (launch agent, 2026-07-29): llama8b on GPU 3, qwen7b on
# GPU 6, in PARALLEL (both verified free pre-launch; GPUs 0,1,2,4,5 belong to
# another user -- never touch). Status file: e13/queue_status.json.
set -u
IP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan
RW=$IP/iclr_exec/recompute_wall
E13=$RW/e13
PY=$IP/tooling/.venv-vllm/bin/python
STATUS=$E13/queue_status.json
# pin BEFORE sourcing env.sh (its default would be GPU 0 = not ours)
export CUDA_VISIBLE_DEVICES=3
# shellcheck disable=SC1090
source "$IP/tooling/env.sh"
export EXPG_LLM_MAX_MODEL_LEN=8192
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
cd "$E13"
ts(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
stat_write(){ printf '{"updated":"%s","llama8b":{"gpu":3,"pid":%s,"state":"%s"},"qwen7b":{"gpu":6,"pid":%s,"state":"%s"},"parent_pid":%s}\n' \
  "$(ts)" "${LP:-0}" "$1" "${QP:-0}" "$2" "$$" > "$STATUS"; }

CUDA_VISIBLE_DEVICES=3 $PY e13_run.py --model llama8b --gpus 3 \
  --out-dir results/E13_open_llama8b --n-target 150 --R 8 --temp 0.7 --seed 0 \
  > logs/open_llama8b.log 2>&1 &
LP=$!
CUDA_VISIBLE_DEVICES=6 $PY e13_run.py --model qwen7b --gpus 6 \
  --out-dir results/E13_open_qwen7b --n-target 150 --R 8 --temp 0.7 --seed 0 \
  > logs/open_qwen7b.log 2>&1 &
QP=$!
stat_write running running
wait $LP; LRC=$?
stat_write "exit=$LRC" running
wait $QP; QRC=$?
stat_write "exit=$LRC" "exit=$QRC"
echo "[open_queue] DONE llama8b=$LRC qwen7b=$QRC $(ts)" >> logs/open_queue.log
