# Coding Agent Prompt - H01: Is One Scalar Energy Too Lossy?

**TLDR:** Build a controlled sequence experiment testing whether one scalar
energy is too compressed, and whether token/tag/decomposed energies improve
credit assignment, localization, or task performance. Start with synthetic data
so the ground-truth tags are known.

## Context

- GitHub issue: https://github.com/brando90/free-energy/issues/47
- Hypothesis notes: `pre_prompt.md`
- Source image: `assets/source_photo.jpg`

## Compact Hypothesis

- **Goal:** Test whether reducing a rich sequence or representation to one
  scalar real-valued energy is too lossy for EBMs, unless the energy is
  structured or decomposed.
- **Confidence:** ~50%.
- **Importance:** Medium, 5/10.

## Your Task

Create `expt_v1/` under this folder and implement a minimal but real experiment.
The experiment should decide whether scalar energy is the failure mode, not just
whether attribution plots look plausible.

## Required Experiment

1. Create a synthetic sequence dataset with known latent tags or concepts.
   Examples that are acceptable:
   - bracket/grammar validity with known invalid token positions
   - arithmetic traces with known step types
   - sequence classification where a small number of tagged spans determine the
     label
2. Train and compare at least three models:
   - scalar EBM: one scalar energy for the full sequence
   - decomposed EBM: token-additive or tag-additive energy contributions
   - autoregressive or classifier baseline with comparable capacity
3. Evaluate:
   - task accuracy or AUROC
   - calibration
   - localization F1 or rank correlation against the known responsible tags
   - intervention effect when tagged spans are removed, swapped, or corrupted
4. Include a seed sweep with at least 3 seeds unless the smoke run shows the
   setup is broken.

## SNAP Cluster Escalation

Prefer a local CPU/GPU implementation first. Escalate to SNAP only for a seed
sweep or larger model sweep.

If using SNAP:

1. Use one GPU unless a clear memory estimate justifies more.
2. Set `CUDA_VISIBLE_DEVICES=<chosen_id>`.
3. Save the exact launch command, environment, logs, and `nvidia-smi` output in
   `expt_v1/results/`.
4. Do not use raw LLM provider API calls.

## Deliverables

- `expt_v1/README.md` with the scientific question and exact run commands.
- `expt_v1/src/` with dataset generation, model definitions, training, and
  evaluation code.
- `expt_v1/results/` with JSON/CSV metrics and plots/tables.
- `expt_v1/results/verdict.md` explaining whether scalar energy is actually too
  lossy.

## Verdict Criteria

The hypothesis is strengthened if decomposed energies consistently improve
localization/intervention metrics or task performance while scalar energies
fail despite matched capacity. It is weakened if scalar energy matches the
decomposed model and attribution/localization claims are unstable across seeds.
