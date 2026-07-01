# Q02: Test dynamic weighted contrastive divergence for continuous EBMs

## Goal

Investigate whether a time-weighted mixture of short Langevin trajectory states can improve EBM training over CD-1 while preserving the correct interpretation of the partition function and negative phase.

## Why

The notes ask whether EBMs really need explicit partition-function computation, whether MCMC is unavoidable, and whether bad short-run samples can be used productively instead of discarded. This is a plausible scaling direction if many short chains can be run in parallel.

## Deliverables

- Theory and literature summary in `questions/02_ebm_dynamic_cd_langevin/research_brief.md`.
- Transcription in `questions/02_ebm_dynamic_cd_langevin/transcription.md`.
- Protocol in `questions/02_ebm_dynamic_cd_langevin/PROTOCOL.md`.
- Runnable prototype in `questions/02_ebm_dynamic_cd_langevin/expt_v1/src/dynamic_weighted_cd.py`.
- Result artifacts from a smoke run under `questions/02_ebm_dynamic_cd_langevin/expt_v1/results/`.

## Acceptance Criteria

- CD-1 and Dynamic Weighted CD train with identical model classes.
- The script writes plots and machine-readable metrics.
- The writeup explicitly corrects the "later samples as positives" sign issue.
- The result verdict states whether dynamic weighting helped on the toy task.
