# 06 - Adaptive Compute Foundations

**TLDR:** Foundations audit for the claim that EBMs are "adaptive": they can spend more inference-time compute on harder problems. The goal is to separate what is distinctive about energy-based iterative inference from what autoregressive chain-of-thought, thinking models, verifier search, and best-of-N already do.

## Core Question

Are EBMs distinctively **adaptive**, meaning they can allocate more inference
compute to harder problems in a way that is not already captured by CoT,
thinking models, verifier search, best-of-N, or self-refinement?

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

## Key Questions

1. What does "adaptive" mean operationally: more steps, more candidates, more
   verifier calls, lower energy, or better success per unit compute?
2. Is adaptivity an EBM-specific property, or a general property of iterative
   inference/search systems?
3. Does EBM adaptivity improve a global compatibility objective more
   efficiently than compute-matched AR thinking/search?
4. When should we call the result "adaptive EBMs" versus "adaptive inference"
   more generally?

## Source Prompt

- Raw seed: [`pre_prompt.md`](pre_prompt.md)
- Plan: [`PLAN.md`](PLAN.md)
- Questions: [`QUESTIONS.md`](QUESTIONS.md)
- Term note: Letitia / AI coffee phrasing: EBMs are "adaptive"; this experiment
  tests whether that term names a distinctive mechanism or just variable
  inference-time compute.
- Inspiration: [Energy-Based Transformers explained | How EBTs and EBMs work](https://www.youtube.com/watch?v=18Fn2m99X1k)

## Reuse From Existing Experiments

- `experiments/00_ar_pros_cons/PROBE_SPECS.md` already contains Probe 05:
  fixed compute per token and the role of scratchpad / CoT.
- `experiments/00_ar_pros_cons/integrated/README.md` already defines a
  compute toggle (`fixed` versus `scratchpad`) for integrated comparisons.
- `experiments/00_start_off/` already has finite-support EBM and MCMC-style
  inference code paths that can be repurposed as variable-step EBM baselines.
- `experiments/01_toy_ebm_training/` is the clean teaching baseline for exact
  scoring over a finite candidate set.

## Status

Scoping only. No experiment has been run yet.
