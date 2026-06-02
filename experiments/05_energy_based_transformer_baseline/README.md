# `05_energy_based_transformer_baseline` -- EBT baseline for the free-energy project

This experiment asks whether the Energy-Based Transformer (EBT) idea from
Gladstone et al., "Energy-Based Transformers are Scalable Learners and Thinkers"
can be a real baseline for this repo's Lean/free-energy project.

Paper and code references:

- arXiv: <https://arxiv.org/abs/2507.02092>
- paper HTML mirror used for implementation notes: <https://ar5iv.labs.arxiv.org/html/2507.02092v1>
- official code: <https://github.com/alexiglad/EBT>

## What is implemented here

Two deliberately small rungs:

1. `run_ebt_toy.py`
   - A binary sequence toy task.
   - A feed-forward direct transformer baseline.
   - A tiny EBT that scores `(context, candidate distribution)` with a scalar
     energy and refines candidate logits by differentiating energy with respect
     to the candidate.
   - This exercises the key EBT feature: second-order training through energy
     descent.

2. `run_veribench_ebt_ranking.py`
   - A VeriBench/Lean candidate reranker.
   - Default smoke can still use the existing three-example pilot.
   - The real path is `--use-splits`, which trains on
     `experiments/02_ar_pros_cons/data/splits/train.jsonl` and reports validation
     and test metrics separately.
   - A char-level transformer scores `E(task, candidate Lean artifact)`.
   - This is a learned verifier / best-of-N selection baseline, not full Lean
     generation by gradient descent.

## Current read

This does **not** show that EBTs solve the project. It gives us a baseline and
a falsifiable next step.

Current smoke outcomes:

- Toy: EBT refinement gives a real signal on the small seq-len-4 task. In the
  checked run, test token accuracy improved from `0.656` at one refinement step
  to `0.846` at two refinement steps, with a positive energy drop.
- VeriBench split smoke: with a capped split run (`12` train tasks, `4` val
  tasks, `4` test tasks), test MRR improved from `0.237` to `0.279`, but gold
  was not strictly above all negatives. It especially confuses
  task-compatible-looking wrong gold/corrupted candidates. This is useful
  negative evidence: the cheap verifier baseline is not enough yet.

## Run locally

Use the repo environment:

```bash
cd ~/free-energy
uv run python --version
```

Toy smoke:

```bash
cd ~/free-energy
.venv/bin/python experiments/05_energy_based_transformer_baseline/run_ebt_toy.py \
  --tag smoke_default \
  --output-dir experiments/05_energy_based_transformer_baseline/results/smoke_toy_default \
  --device cpu \
  --require-ebt-signal
```

Generate VeriBench train/val/test split files if they are missing:

```bash
cd ~/free-energy/experiments/02_ar_pros_cons
../../.venv/bin/python -m data.setup \
  --veribench-root ~/veribench/veribench_dataset \
  --include-generated-agents \
  --smoke
```

VeriBench split smoke:

```bash
cd ~/free-energy
.venv/bin/python experiments/05_energy_based_transformer_baseline/run_veribench_ebt_ranking.py \
  --use-splits \
  --tag split_smoke \
  --output-dir experiments/05_energy_based_transformer_baseline/results/split_smoke \
  --split-dir experiments/02_ar_pros_cons/data/splits \
  --veribench-root ~/veribench/veribench_dataset \
  --max-split-train-tasks 12 \
  --max-split-val-tasks 4 \
  --max-split-test-tasks 4 \
  --epochs 3 \
  --hidden-dim 24 \
  --num-layers 1 \
  --num-heads 4 \
  --max-length 512 \
  --batch-size 8 \
  --eval-batch-size 8 \
  --device cpu
```

Or run both:

```bash
cd ~/free-energy
./experiments/05_energy_based_transformer_baseline/run_smoke_test.sh
```

## What would count as progress

- Toy: show monotonically improving accuracy as refinement steps increase on
  seq-len 8 or harder tasks, not just seq-len 4.
- VeriBench reranking: strict gold rank `1` with positive margins, where ties
  count against the model.
- VeriBench generation: replace candidate reranking with an actual discrete
  refinement/generation loop plus Lean verification. Until then, this is a
  verifier baseline, not a solver.
