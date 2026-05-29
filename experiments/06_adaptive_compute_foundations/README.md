# 06 - Adaptive Compute Foundations

**TLDR:** Foundations audit for the claim that EBMs can spend more inference-time compute on harder problems. The goal is to separate what is distinctive about energy-based iterative inference from what autoregressive chain-of-thought, thinking models, verifier search, and best-of-N already do.

## Core Question

Is it really true that EBMs have a special advantage because they can spend
more computation on more difficult problems?

The immediate objection is that modern autoregressive systems already vary
inference-time compute through chain-of-thought, hidden "thinking" tokens,
self-refinement, best-of-N sampling, tool calls, and verifier-guided search. If
those mechanisms already allocate more compute to harder instances, then the
EBM claim needs a sharper form than "variable compute exists."

## Why This Exists

The repo thesis already includes:

> Adaptive-compute inference (energy descent, iterative refinement) beats
> equivalent additional training compute on variable-difficulty tasks.

This experiment tests the foundations of that claim before building another
benchmark. The right question is not "can EBMs iterate?" but:

1. what kind of computation is being allocated;
2. what objective that computation optimizes;
3. whether autoregressive thinking/search is an equivalent mechanism;
4. what empirical signature would distinguish the two.

## Source Prompt

- Raw seed: [`pre_prompt.md`](pre_prompt.md)
- Plan: [`PLAN.md`](PLAN.md)
- Inspiration: [Energy-Based Transformers explained | How EBTs and EBMs work](https://www.youtube.com/watch?v=18Fn2m99X1k)

## Reuse From Existing Experiments

- `experiments/02_ar_pros_cons/PROBE_SPECS.md` already contains Probe 05:
  fixed compute per token and the role of scratchpad / CoT.
- `experiments/02_ar_pros_cons/integrated/README.md` already defines a
  compute toggle (`fixed` versus `scratchpad`) for integrated comparisons.
- `experiments/00_start_off/` already has finite-support EBM and MCMC-style
  inference code paths that can be repurposed as variable-step EBM baselines.
- `experiments/01_toy_ebm_training/` is the clean teaching baseline for exact
  scoring over a finite candidate set.

## Status

Scoping only. No experiment has been run yet.
