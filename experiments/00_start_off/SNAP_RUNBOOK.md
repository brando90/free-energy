# SNAP Runbook

TL;DR: first run the local-style smoke test on SNAP to verify paths and candidate pools, then run the transformer energy starter on the same three VeriBench tasks.

## Assumed Layout

Expected repos:

```bash
~/free-energy
~/veribench
```

If VeriBench lives elsewhere, set:

```bash
export VERIBENCH_ROOT=/path/to/veribench
```

## Setup

From the SNAP login node:

```bash
cd ~/free-energy
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

If SNAP uses modules, load the local Python/CUDA modules first. The exact module names vary by cluster image, so check `module avail` if `python3.11` or CUDA is not visible.

## Smoke Test

Run this before requesting a GPU:

```bash
cd ~/free-energy
source .venv/bin/activate
export VERIBENCH_ROOT=${VERIBENCH_ROOT:-$HOME/veribench}

python experiments/00_start_off/pilot_ebm_ranking.py \
  --veribench-root "$VERIBENCH_ROOT" \
  --write-candidate-files \
  --require-gold-top
```

Expected artifacts:

```text
experiments/00_start_off/results/smoke_test_rankings.json
experiments/00_start_off/results/candidate_pools/
```

This is only a wiring test. The toy energy has a source prior and is not evidence that EBMs work.

## Transformer Energy Starter

Run on a GPU node or interactive GPU session:

```bash
cd ~/free-energy
source .venv/bin/activate
export VERIBENCH_ROOT=${VERIBENCH_ROOT:-$HOME/veribench}

python experiments/00_start_off/train_transformer_energy.py \
  --veribench-root "$VERIBENCH_ROOT" \
  --model-name microsoft/codebert-base \
  --epochs 5 \
  --batch-size 2 \
  --max-length 512 \
  --device cuda
```

Expected artifacts:

```text
experiments/00_start_off/results/transformer_energy/training_report.json
experiments/00_start_off/results/transformer_energy/energy_model_state.pt
experiments/00_start_off/results/transformer_energy/tokenizer/
```

## First Success Criterion

The first run is successful if:

- the smoke test passes;
- the transformer script trains without crashing;
- `training_report.json` reports the gold candidate rank for each of the three tasks;
- the final ranking is inspected manually, not just trusted because the pool is tiny.

## Next Scientific Step

After the three-example run works, remove the source-prior toy baseline from the story and scale to a real held-out VeriBench split:

- train/dev: small subset for energy fitting and hyperparameters;
- test: most VeriBench tasks, untouched until evaluation;
- negatives: generated agent attempts, corruptions, verifier-failing proofs, and later short-run MCMC/local-edit proposals.

Bottom TL;DR: do not start with full EBM likelihood training. Start with finite-pool candidate ranking, verify the pipeline, then add short-run MCMC only as a hard-negative generator.
