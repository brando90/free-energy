#!/bin/bash
# E1 smoke — ~10 problems end-to-end on Qwen2.5-1.5B (gold -> plant -> continue ->
# validate) on the pinned GPU. Uses the small PILOT world config (~200 golds) so the
# gold phase is fast, and caps perturbed worlds at 10/cell. Verifies the validator
# parses the 1.5B output format and that gold attrition is nonzero-but-viable.
#
#   CUDA_VISIBLE_DEVICES=<free_gpu> bash smoke_e1.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/e1_config.sh"
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
source "$EXP/improvement_plan/tooling/env.sh" 2>/dev/null || true
export HF_HUB_OFFLINE=1
: "${CUDA_VISIBLE_DEVICES:?pin a FREE GPU before smoke}"

SREPO="Qwen/Qwen2.5-1.5B-Instruct"
SREV="989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SOUT="$E1DIR/smoke/qwen1p5b_smoke"
SMOKE_CELLS="aff_false_attr_d1,cat_false_usable_d1,cat_false_inert_d1,benign_paraphrase"
rm -rf "$SOUT"; mkdir -p "$SOUT"

echo "=== smoke prepare (pilot, seed 0) ==="
"$PY_MAIN" "$E1DIR/prepare_e1.py" --out "$SOUT" --seeds 0 --config pilot \
  --model "$SREPO" --revision "$SREV" --reg-commit "$E1_REG_COMMIT" || exit 2

echo "=== smoke generate (cap 10, mid) ==="
"$PY_VLLM" "$EXPD/expd_matched_gradient.py" generate \
  --out-dir "$SOUT" --backend vllm --world-config full \
  --model "$SREPO" --model-revision "$SREV" \
  --cells "$SMOKE_CELLS" --positions mid --cap 10 \
  --batch-size 64 --max-new-tokens "$MAXNEW" || exit 3

echo "=== smoke validate + summarize ==="
"$PY_MAIN" "$EXPD/expd_matched_gradient.py" validate  --out-dir "$SOUT" || exit 4
"$PY_MAIN" "$EXPD/expd_matched_gradient.py" summarize --out-dir "$SOUT" || exit 5

echo "=== smoke checks ==="
"$PY_MAIN" "$E1DIR/check_smoke.py" --out "$SOUT"
