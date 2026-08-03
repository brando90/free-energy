#!/bin/bash
# Stress-test runs (author-directed): QwQ-32B matched pair + OLMo-2 stage ladder.
# Mirrors e10_widen queue invocations exactly (same cells/caps/decoding/seed);
# NEW output tree stress_reasoning/; skips the judge stage (doubt_lex suffices
# for absorbed; disclosed in README). Usage: run_stress.sh {qwq|olmo}
set -u
EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
EXPG=$EXP/improvement_plan/expg
RW=$EXP/improvement_plan/iclr_exec/recompute_wall
OUTBASE=$RW/stress_reasoning
PY_VLLM=$EXP/improvement_plan/tooling/.venv-vllm/bin/python
PY_MAIN=/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python
DRIVER=$EXPG/expg_progtrace.py
WORLDS_SRC=$RW/e10_widen/results_skampere1/EXPG_PROGTRACE_qwen32b/programs.jsonl

CELLS="benign_paraphrase,true_interruption,adjacent_contradiction,opfree_kr1,opfree_kr8,onehop_kc1,deep_kc5"
FLAGS="--cells $CELLS --cap 150 --oversample 240 --seed 0"
GENFLAGS="--max-new-gold 512 --max-new-cont 384 --batch-size 256 --confirm-timestamped-plan-progtrace --ignore-gate-a"

source $EXP/improvement_plan/tooling/env.sh 2>/dev/null || true
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
# OLMo-2 max_position_embeddings=4096: only cap ctx for the 32B models
[ "${1:-}" = qwq ] && export EXPG_LLM_MAX_MODEL_LEN=8192 || unset EXPG_LLM_MAX_MODEL_LEN

run_model() {  # slug repo gpu
  local slug=$1 repo=$2 gpu=$3
  local out=$OUTBASE/EXPG_PROGTRACE_$slug
  local log=$OUTBASE/logs/$slug.log
  mkdir -p "$out" "$OUTBASE/logs"
  [ -s "$out/programs.jsonl" ] || cp "$WORLDS_SRC" "$out/programs.jsonl"
  echo "=== $slug generate start $(date -Is) gpu=$gpu ===" >> "$log"
  CUDA_VISIBLE_DEVICES=$gpu setsid $PY_VLLM $DRIVER generate --out-dir "$out" \
    --backend vllm --model "$repo" --revision main $FLAGS $GENFLAGS >> "$log" 2>&1
  local rc=$?
  echo "=== $slug generate rc=$rc $(date -Is) ===" >> "$log"
  [ $rc -ne 0 ] && return $rc
  $PY_MAIN $DRIVER validate  --out-dir "$out" >> "$log" 2>&1 || return $?
  $PY_MAIN $DRIVER summarize --out-dir "$out" >> "$log" 2>&1 || return $?
  echo "=== $slug COMPLETE $(date -Is) ===" >> "$log"
}

mkdir -p "$OUTBASE"
cat > "$OUTBASE/README_STRESS.md" <<'EOF'
Stress-test runs (2026-07-31, author-directed, results discussed in chat before paper use).
Protocol: identical to e10_widen queue (expg_progtrace.py, cells/cap/oversample/decoding/seed
byte-matched to e10_config.sh); worlds copied from EXPG_PROGTRACE_qwen32b (identical substrate,
fingerprint 64583365b3ad48dc46b168be) so pairs share exact programs. Judge stage skipped
(flag channel = frozen doubt_lex only, as in E11). EXPG_LLM_MAX_MODEL_LEN=8192.
- EXPG_PROGTRACE_qwq32b:    Qwen/QwQ-32B (reasoning-RL on the roster's Qwen2.5-32B base).
  Prefill continuation mode = forced-direct arm (thinking suppressed by prefill); a
  think-channel arm may follow. Matched-base pair: compare to roster EXPG_PROGTRACE_qwen32b.
- EXPG_PROGTRACE_olmo7b_sft / _dpo: allenai/OLMo-2-1124-7B-{SFT,DPO} — training-stage ladder;
  compare to roster EXPG_PROGTRACE_olmo7b (final Instruct). Base checkpoint pending (no chat
  template; needs raw-prompt adapter).
EOF

case "${1:?usage: run_stress.sh qwq|olmo}" in
  qwq)  run_model qwq32b     "Qwen/QwQ-32B" 0 ;;
  tulu) run_model tulu_sft   "allenai/Llama-3.1-Tulu-3-8B-SFT" ${2:-0} && \
        run_model tulu_dpo   "allenai/Llama-3.1-Tulu-3-8B-DPO" ${2:-0} && \
        run_model tulu_final "allenai/Llama-3.1-Tulu-3-8B"     ${2:-0} ;;
  olmo) run_model olmo7b_sft "allenai/OLMo-2-1124-7B-SFT" 6 && \
        run_model olmo7b_dpo "allenai/OLMo-2-1124-7B-DPO" 6 ;;
esac
