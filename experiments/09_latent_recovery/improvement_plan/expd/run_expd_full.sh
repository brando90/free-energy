#!/bin/bash
# EXPD full confirmatory run driver (PLAN2 v1.1 section C.5 core grid).
# Pass A: attribute quartet x d{0,1,2,3,5} x 3 positions + anchors x 3 positions.
# Pass B: cat_false {usable,inert} x d{1,3,inf} x 2 positions (early,mid).
# Then validate + summarize + runner report. Timings written to stage_timings.json.
set -e
EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
EXPD=$EXP/improvement_plan/expd
OUT=$EXP/results/EXPD_MATCHED_GRADIENT
PY=$EXP/improvement_plan/tooling/.venv-vllm/bin/python
MAINPY=/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python

source $EXP/improvement_plan/tooling/env.sh
export CUDA_VISIBLE_DEVICES=${EXPD_GPU:-6}
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

REV=a09a35458c702b33eeacc393d103063234e8bc28
EVID="--confirm-timestamped-plan2 --plan2-commit 545b35db420d1826b69ca9bdd6e47b5720bb54de --plan2-commit-timestamp 2026-07-02T14:05:29-07:00 --plan2-sha256 112da1a56a16745181e6ed7d2dac3b1e7d0dd3e7643e78e2bfdf16ca572e2b2e"

ATTR_CELLS="aff_false_attr_d0,aff_false_attr_d1,aff_false_attr_d2,aff_false_attr_d3,aff_false_attr_d5,neg_false_attr_d0,neg_false_attr_d1,neg_false_attr_d2,neg_false_attr_d3,neg_false_attr_d5,aff_true_attr_d0,aff_true_attr_d1,aff_true_attr_d2,aff_true_attr_d3,aff_true_attr_d5,neg_true_attr_d0,neg_true_attr_d1,neg_true_attr_d2,neg_true_attr_d3,neg_true_attr_d5,benign_paraphrase,true_interruption"
CAT_CELLS="cat_false_usable_d1,cat_false_usable_d3,cat_false_usable_dinf,cat_false_inert_d1,cat_false_inert_d3,cat_false_inert_dinf"

cd $EXPD
T0=$(date +%s)
echo "=== PASS A (attr quartet + anchors, early/mid/late) $(date -Is) ==="
$PY expd_matched_gradient.py generate --out-dir $OUT --world-config full $EVID \
  --model-revision $REV --cells "$ATTR_CELLS" --positions early,mid,late \
  --cap 150 --batch-size 256
T1=$(date +%s)
echo "=== PASS B (cat cells, early/mid) $(date -Is) ==="
$PY expd_matched_gradient.py generate --out-dir $OUT --world-config full $EVID \
  --model-revision $REV --cells "$CAT_CELLS" --positions early,mid \
  --cap 150 --batch-size 256
T2=$(date +%s)
echo "=== VALIDATE $(date -Is) ==="
$MAINPY expd_matched_gradient.py validate --out-dir $OUT
T3=$(date +%s)
echo "=== SUMMARIZE $(date -Is) ==="
$MAINPY expd_matched_gradient.py summarize --out-dir $OUT
T4=$(date +%s)
cat > $OUT/stage_timings.json <<EOF
{
  "gpu": "1x NVIDIA H200 (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)",
  "generate_passA_attr_anchors_seconds": $((T1-T0)),
  "generate_passB_cat_seconds": $((T2-T1)),
  "validate_seconds": $((T3-T2)),
  "summarize_seconds": $((T4-T3)),
  "gpu_stage_total_seconds": $((T2-T0)),
  "note": "prepare (world gen, CPU) ran separately; engine load included in each pass"
}
EOF
echo "=== RUNNER REPORT $(date -Is) ==="
$MAINPY expd_matched_gradient.py report --out-dir $OUT
echo "=== ALL RUNNER STAGES DONE $(date -Is) ==="
