# Coding Agent Prompt - H03: Does Probability Normalization Buy Real Power?

**TLDR:** Build a finite-world experiment/proof suite deciding whether
normalizing energies into probabilities gives mathematical or computational
capabilities that unnormalized scores cannot reproduce.

## Context

- GitHub issue: https://github.com/brando90/free-energy/issues/53
- Hypothesis notes: `pre_prompt.md`
- Source image: `assets/source_photo.jpg`

## Compact Hypothesis

- **Goal:** Test whether probability normalization gives concrete mathematical
  or computational power that unnormalized scores cannot reproduce.
- **Confidence:** Answer unknown; roughly 80% confidence that this is an
  important question to settle.
- **Importance:** High, roughly 8/10.

## Your Task

Create `expt_v1/` under this folder and implement a reproducible finite-world
experiment. Do not only write an essay. The deliverable must include runnable
code, machine-readable results, a capability matrix, and a short verdict.

## Required Experiment

1. Build a finite candidate-world simulator:
   - candidates `y in {1, ..., K}`
   - energies or scores for each candidate
   - optional hidden variables `h` mapping to observed candidates
   - utility values for downstream decisions
2. Compare four information regimes:
   - rank-only oracle
   - raw unnormalized score/energy
   - score-difference or ratio-only oracle
   - normalized probability `p(y)`
3. Include at least these downstream tasks:
   - argmax and top-k ranking
   - calibrated thresholding or abstention
   - expected-utility decision making
   - entropy/uncertainty measurement
   - exact sampling from the model distribution
   - marginalization over hidden variables
   - Bayesian update or product-of-evidence composition
4. Apply monotone transforms to the same base scores:
   - affine scale/shift
   - temperature changes
   - nonlinear monotone transform
   - badly calibrated but rank-preserving transform
5. Report which tasks are invariant and which tasks break under these
   transforms.

## Metrics

- rank agreement and top-k agreement
- calibration error, Brier score, or log score where probabilities are needed
- total variation or KL divergence for sampling/marginal distributions
- decision regret for expected-utility tasks
- a binary capability matrix stating the minimal object needed by each task

## Implementation Notes

- Prefer standard-library Python so the experiment works even if local NumPy or
  PyTorch wheels are broken.
- If using NumPy/PyTorch locally, include a standard-library fallback or a clear
  failure note.
- Do not use raw LLM API calls or API keys.

## SNAP Cluster Escalation

SNAP should not be necessary for the first version. Use it only if you expand
the finite-world simulator into a large sweep that is too slow locally.

If using SNAP:

1. Use one GPU or one CPU job for the main run.
2. Record `hostname`, Python version, start/end time, and exact commands.
3. Save logs under `expt_v1/results/`.

## Deliverables

- `expt_v1/README.md` with setup, commands, and status.
- `expt_v1/src/` with the finite-world simulator and evaluation code.
- `expt_v1/results/` with JSON/CSV results and the capability matrix.
- `expt_v1/results/verdict.md` answering the hypothesis in one paragraph.

## Verdict Criteria

The hypothesis is strengthened if normalization is necessary for at least one
important workflow: calibrated abstention, expected utility, entropy, exact
sampling, marginalization, or Bayesian composition. It is weakened if the target
research workflow only uses ranking, argmax, top-k search, or score ratios that
remain valid without normalized probabilities.
