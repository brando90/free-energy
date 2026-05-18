# Agent Prompt: SNAP Three-Example EBM Pilot

TL;DR: Get the three-example VeriBench energy-ranking pilot running on SNAP. First prove the candidate-pool plumbing works. Then train the starter transformer scalar energy model and write a concise report with rankings and failure modes.

You are working in:

```text
~/free-energy
```

The related VeriBench repo should be at:

```text
~/veribench
```

If it is somewhere else, set `VERIBENCH_ROOT`.

## Goal

Run a minimal EBM-style candidate-ranking experiment for Lean/VeriBench artifacts. Do not attempt the full benchmark yet. The point is to get one reproducible end-to-end run working so we can later scale the split and add MCMC hard negatives.

## Files To Use

```text
experiments/00_start_off/veribench_three_example_manifest.json
experiments/00_start_off/pilot_ebm_ranking.py
experiments/00_start_off/train_transformer_energy.py
experiments/00_start_off/ebm_mcmc_veribench_notes.md
```

The source paper context is Song and Kingma, "How to Train Your Energy-Based Models", arXiv:2101.03288. The research question is whether biased/short-run negative sampling can still train a useful energy ranker for formal artifacts.

## Required Commands

Set up the repo:

```bash
cd ~/free-energy
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export VERIBENCH_ROOT=${VERIBENCH_ROOT:-$HOME/veribench}
```

Run the smoke test:

```bash
python experiments/00_start_off/pilot_ebm_ranking.py \
  --veribench-root "$VERIBENCH_ROOT" \
  --write-candidate-files \
  --require-gold-top
```

Run the transformer starter on a GPU:

```bash
python experiments/00_start_off/train_transformer_energy.py \
  --veribench-root "$VERIBENCH_ROOT" \
  --model-name microsoft/codebert-base \
  --epochs 5 \
  --batch-size 2 \
  --max-length 512 \
  --device cuda
```

If `microsoft/codebert-base` is unavailable or too slow, use a smaller public Hugging Face model for the first run and record the substitution explicitly.

## What To Check

1. The manifest resolves all three tasks:
   - `easy_set/1_MyAdd`
   - `easy_set/21_is_palindrome`
   - `cs_set/binary_search`

2. Candidate pools include:
   - the gold Lean file;
   - generated agent attempts from VeriBench;
   - simple corrupted negatives.

3. The transformer training report includes:
   - loss per epoch;
   - gold rank per task;
   - final mean reciprocal rank;
   - top candidate per task.

4. Inspect the output manually. Since there are only three tasks, do not overclaim from a pass.

## Deliverable

Write a short markdown report at:

```text
experiments/00_start_off/results/snap_three_example_report.md
```

Include:

- setup details: machine/GPU, model, command, date;
- smoke test outcome;
- transformer training outcome;
- table of final rankings by task;
- obvious failure modes;
- next concrete step for adding short-run MCMC/local-edit hard negatives.

## Constraints

- Do not scale to the full benchmark until the three-example run is clean.
- Do not silently change the task set.
- Do not report the toy smoke-test energy as a scientific result.
- Keep generated outputs under `experiments/00_start_off/results/`.

Bottom TL;DR: smoke test first, transformer scalar energy second, concise report third. The scientific claim is only whether this path is worth scaling, not whether EBMs solve VeriBench yet.
