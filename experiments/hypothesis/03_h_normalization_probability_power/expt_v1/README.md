# H03 Experiment v1 - Normalization Probability Power

This experiment tests whether probability normalization buys concrete
capabilities beyond unnormalized scores.

The setup is deliberately finite and dependency-free. We define candidate
scores, apply rank-preserving transformations, and compare downstream tasks
that either only need ordering or require calibrated probabilities.

## Run

From the repository root:

```bash
python3 experiments/hypothesis/03_h_normalization_probability_power/expt_v1/src/run_experiment.py
```

The script writes:

- `results/summary.json`
- `results/task_summary.csv`
- `results/capability_matrix.csv`
- `results/capability_matrix.md`
- `results/verdict.md`

## What This Tests

The experiment distinguishes four levels of information:

- rank-only ordering
- raw unnormalized scores
- score differences or ratios
- normalized probabilities

Rank-preserving transforms keep argmax and top-k decisions intact. They do not
preserve calibrated probabilities, entropy, sampling distributions,
latent-variable marginals, or expected-utility decisions.

## Status

Executed locally with standard-library Python. SNAP was not needed for this
finite-world proof-of-concept.
