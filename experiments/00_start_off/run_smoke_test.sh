#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python experiments/00_start_off/pilot_ebm_ranking.py \
  --write-candidate-files \
  --require-gold-top
