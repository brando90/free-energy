# SNAP EBM Equals MLE Sanity Check

Date: 2026-05-19

Commit run: `833c927d62fb3008cf60513c7ba770ac10f69a1c`

Host: `skampere1.stanford.edu`

Output directory on SNAP: `~/free-energy/experiments/00_start_off/results/ebm_mle_sanity_833c927_20260519_073752`

## Hypothesis From The Paper Notes

MCMC-based EBM training is usually expensive because the model expectation is
hard to sample exactly. The working hypothesis from the handwritten notes is:
for Lean/VeriBench candidate ranking, it may still be useful to "do it anyway"
with finite candidate pools, short-run MCMC, or otherwise biased negative
sampling, if the resulting gradient is close enough to MLE to learn a useful
verifier/proof ranker.

## Setting

For each VeriBench task `t`, the finite candidate pool `C_t` contains the gold
Lean file, up to three generated-agent files, and synthetic corruptions. The
conditional EBM is

```math
p_\theta(y \mid t)
  = \frac{\exp(-E_\theta(t, y))}
         {\sum_{z \in C_t} \exp(-E_\theta(t, z))}.
```

The exact finite-support MLE loss is

```math
E_\theta(t, y^+) + \log \sum_{y \in C_t} \exp(-E_\theta(t, y)).
```

The sanity check compares that exact MLE gradient to the EBM negative-phase
gradient

```math
\nabla_\theta E_\theta(t, y^+)
  - \mathbb{E}_{y \sim p_\theta(\cdot \mid t)}
    [\nabla_\theta E_\theta(t, y)].
```

## Dataset

| Split | Tasks | Candidates | Subsets |
| --- | ---: | ---: | --- |
| all | 117 | 788 | easy_set=41, cs_set=20, humaneval_set=56 |
| train / gradient check | 94 | 636 | humaneval_set=45, easy_set=32, cs_set=17 |
| test | 23 | 152 | humaneval_set=11, easy_set=9, cs_set=3 |

## Command

```bash
cd ~/free-energy
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/00_start_off/run_ebm_mle_sanity_check.py \
  --veribench-root ~/veribench \
  --output-dir experiments/00_start_off/results/ebm_mle_sanity_833c927_20260519_073752 \
  --model-name microsoft/codebert-base \
  --subsets easy_set cs_set humaneval_set \
  --task-batch-size 4 \
  --negative-batch-size 64 \
  --max-length 256 \
  --exact-sample-counts 1 4 16 64 256 \
  --mcmc-steps-list 0 1 5 25 100 \
  --mcmc-samples 128 \
  --device cuda
```

Runtime: `961.99s`.

## Results

The exact EBM negative phase matches the exact finite-support MLE gradient:

| Comparison | Cosine vs exact MLE | Relative L2 | Norm ratio |
| --- | ---: | ---: | ---: |
| exact EBM negative phase | 1.000000000 | 0.000000238 | 0.999999977 |

Exact model samples converge toward the exact MLE gradient as sample count
increases:

| Exact model samples per task | Cosine | Relative L2 | Norm ratio |
| ---: | ---: | ---: | ---: |
| 1 | 0.900582 | 0.443214 | 0.987105 |
| 4 | 0.996939 | 0.164227 | 1.141364 |
| 16 | 0.998912 | 0.047126 | 1.005663 |
| 64 | 0.999679 | 0.034996 | 1.023842 |
| 256 | 0.999953 | 0.012379 | 0.992229 |

Finite-support MH-MCMC samples are also close in this initial-model finite-pool
setting:

| MH steps | Samples per task | Cosine | Relative L2 | Norm ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 128 | 0.999491 | 0.032664 | 0.992539 |
| 1 | 128 | 0.999947 | 0.013226 | 0.991603 |
| 5 | 128 | 0.999784 | 0.022158 | 0.992038 |
| 25 | 128 | 0.999416 | 0.037551 | 0.983838 |
| 100 | 128 | 0.999856 | 0.020638 | 1.011622 |

## Interpretation

The identity check passes: in the finite candidate-pool setting, exact EBM
negative-phase training is the same gradient as exact finite-support MLE.

The sampled estimators behave as expected: more exact model samples reduce the
gradient error. MCMC is already close here because the model is at initialization
and the candidate pools are small, so even zero-step uniform samples are not far
from the initial model distribution. This does not prove short-run MCMC will
remain good after training or with larger/harder supports; it verifies the
sanity-check case and gives us the next stress test.

Next stress test: repeat the gradient comparison after several MLE/EBM training
epochs, then expand the candidate support with harder generated negatives.
