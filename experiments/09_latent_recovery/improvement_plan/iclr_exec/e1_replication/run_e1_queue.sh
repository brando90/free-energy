#!/bin/bash
# E1 replication — sequential queue driver (one pinned GPU, models run one at a time).
# Order: qwen1p5b (fastest, validates pipeline) -> llama8b -> olmo7b -> qwen32b (largest).
# Each model: prepare (CPU) -> generate pass1/2/3 (GPU) -> validate/summarize/report (CPU).
# queue_status.json is rewritten after EVERY stage transition (start + done, with rc).
#
# Launch (pin GPU to a FULLY FREE device -- 0 MiB used at launch):
#   CUDA_VISIBLE_DEVICES=<free_gpu> nohup bash run_e1_queue.sh > queue.out 2>&1 &
#
# The GPU id is taken from CUDA_VISIBLE_DEVICES in the environment; this script does
# NOT choose a GPU (the operator pins a verified-free one before launch).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/e1_config.sh"

export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
source "$EXP/improvement_plan/tooling/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=1   # re-assert after env.sh (which sets it to 0)
: "${CUDA_VISIBLE_DEVICES:?refusing to launch without a pinned CUDA_VISIBLE_DEVICES}"

STATUS="$RESULTS/queue_status.json"
LOGDIR="$E1DIR/logs"
mkdir -p "$RESULTS" "$LOGDIR"

QUEUE_ORDER="qwen1p5b,llama8b,olmo7b,qwen32b"

status() {  # model stage event rc
  "$PY_MAIN" "$E1DIR/status_writer.py" --file "$STATUS" \
    --model "$1" --stage "$2" --event "$3" ${4:+--rc "$4"} \
    ${INIT_QUEUE:+--queue "$QUEUE_ORDER"}
  INIT_QUEUE=""
}
INIT_QUEUE=1

run_stage() {  # model stage cmd...
  local model="$1" stage="$2"; shift 2
  local log="$LOGDIR/${model}.${stage}.log"
  status "$model" "$stage" start
  echo "=== [$model/$stage] $(date -Is) :: $* ===" | tee -a "$log"
  "$@" >>"$log" 2>&1
  local rc=$?
  status "$model" "$stage" done "$rc"
  echo "=== [$model/$stage] rc=$rc $(date -Is) ===" | tee -a "$log"
  return $rc
}

GEN() {  # out cells positions model revision
  "$PY_VLLM" "$EXPD/expd_matched_gradient.py" generate \
    --out-dir "$1" --backend vllm --world-config full \
    --model "$4" --model-revision "$5" \
    --cells "$2" --positions "$3" --cap "$CAP" \
    --batch-size "$BATCH" --max-new-tokens "$MAXNEW"
}

for entry in "${E1_MODELS[@]}"; do
  IFS='|' read -r slug repo rev seeds <<< "$entry"
  OUT="$RESULTS/$slug"
  mkdir -p "$OUT"
  echo "########## MODEL $slug ($repo @ $rev, seeds=$seeds) $(date -Is) ##########"

  run_stage "$slug" prepare \
    "$PY_MAIN" "$E1DIR/prepare_e1.py" --out "$OUT" --seeds "$seeds" \
    --model "$repo" --revision "$rev" --reg-commit "$E1_REG_COMMIT" \
    || { echo "[$slug] prepare failed; skipping model"; continue; }

  run_stage "$slug" generate_pass1 GEN "$OUT" "$PASS1_CELLS" "$PASS1_POS" "$repo" "$rev" \
    || { echo "[$slug] pass1 failed; skipping model"; continue; }
  run_stage "$slug" generate_pass2 GEN "$OUT" "$PASS2_CELLS" "$PASS2_POS" "$repo" "$rev" \
    || { echo "[$slug] pass2 failed; skipping model"; continue; }
  run_stage "$slug" generate_pass3 GEN "$OUT" "$PASS3_CELLS" "$PASS3_POS" "$repo" "$rev" \
    || { echo "[$slug] pass3 failed; skipping model"; continue; }

  run_stage "$slug" validate \
    "$PY_MAIN" "$EXPD/expd_matched_gradient.py" validate --out-dir "$OUT" \
    || { echo "[$slug] validate failed; skipping model"; continue; }
  run_stage "$slug" summarize \
    "$PY_MAIN" "$EXPD/expd_matched_gradient.py" summarize --out-dir "$OUT" \
    || { echo "[$slug] summarize failed; skipping model"; continue; }
  run_stage "$slug" report \
    "$PY_MAIN" "$EXPD/expd_matched_gradient.py" report --out-dir "$OUT" || true

  status "$slug" model_complete done 0
  echo "########## MODEL $slug DONE $(date -Is) ##########"
done

status ALL queue_complete done 0
echo "########## E1 QUEUE COMPLETE $(date -Is) ##########"
