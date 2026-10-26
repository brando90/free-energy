#!/usr/bin/env bash
# TLDR: Full B1 pipeline on one GPU — zeroshot ref, overfit sanity, LR sweep, 3-seed finals, aggregate to results_b1.md.
# Usage: bash run_b1.sh [GPU_ID]   (default GPU 0; uses the venv it's launched in)
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${1:-0}"
cd "$(dirname "$0")"

python train_b1.py --mode zeroshot
python train_b1.py --mode overfit --lr 5e-5

for lr in 1e-5 2e-5 5e-5; do
  python train_b1.py --mode train --lr "$lr" --seed 0
done

BEST_LR=$(python - <<'EOF'
import json, pathlib
runs = [json.loads(p.read_text()) for p in pathlib.Path("results").glob("b1_lr*_seed0.json")]
print(f"{min(runs, key=lambda r: r['val_ce'])['lr']:g}")
EOF
)
echo "[run_b1] best LR on val: $BEST_LR"

for s in 1 2; do
  python train_b1.py --mode train --lr "$BEST_LR" --seed "$s"
done

python train_b1.py --mode aggregate
