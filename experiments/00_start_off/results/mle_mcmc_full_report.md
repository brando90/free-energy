# SNAP Full Finite-Support MLE vs MCMC Report

Date: 2026-05-19

Commit: `e2607fb158d133ed9481e3fa89ff806a7b1da477`

Host: `skampere1.stanford.edu`

Output directory on SNAP: `~/free-energy/experiments/00_start_off/results/mle_mcmc_full_e2607fb_20260519_072622`

## TLDR

This is the intended sanity check, not the earlier three-example ranking pilot.
It trains the same CodeBERT scalar energy model from the same initialization in
two ways over the same finite VeriBench candidate pools:

1. exact finite-support MLE;
2. persistent Metropolis-Hastings MCMC gradient estimation.

Both methods reached identical ranking accuracy on the finite candidate pools,
and their finite-support NLLs were close. MCMC was about 3x slower on this run.

## Command

```bash
cd ~/free-energy
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/00_start_off/run_mle_mcmc_experiment.py \
  --veribench-root ~/veribench \
  --output-dir experiments/00_start_off/results/mle_mcmc_full_e2607fb_20260519_072622 \
  --model-name microsoft/codebert-base \
  --subsets easy_set cs_set humaneval_set \
  --epochs 5 \
  --task-batch-size 4 \
  --max-length 256 \
  --mcmc-steps 25 \
  --device cuda
```

## Dataset

| Split | Tasks | Candidates | Subsets |
| --- | ---: | ---: | --- |
| all | 117 | 788 | easy_set=41, cs_set=20, humaneval_set=56 |
| train | 94 | 636 | humaneval_set=45, easy_set=32, cs_set=17 |
| test | 23 | 152 | humaneval_set=11, easy_set=9, cs_set=3 |

Each task pool contains the gold Lean file, up to three generated-agent Lean
files, and synthetic corruptions. This is a finite-support EBM experiment over
candidate pools, not an unconstrained model over all Lean programs.

## Results

| Method | Split | Top-1 | MRR | Mean finite-support NLL | Seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| exact finite-support MLE | train | 1.0000 | 1.0000 | 0.7365 | 58.94 |
| exact finite-support MLE | test | 1.0000 | 1.0000 | 0.7298 | 58.94 |
| persistent MH MCMC | train | 1.0000 | 1.0000 | 0.7391 | 174.03 |
| persistent MH MCMC | test | 1.0000 | 1.0000 | 0.7344 | 174.03 |

## Interpretation

The result matches the expected sanity-check behavior: when exact MLE and the
MCMC-estimated negative phase train on the same finite candidate support, the
final ranking metrics and finite-support NLL are close.

The wall-clock cost difference is also visible: the MCMC run took about 174s
for training versus about 59s for exact finite-support MLE, because each MCMC
batch runs 25 Metropolis-Hastings proposal steps before taking the gradient
step.

The next experiment should make the negative phase harder by increasing the
candidate support, using harder negatives, and/or moving from finite candidate
pools toward a generator/proposal distribution that better approximates the
actual model support.
