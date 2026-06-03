# Pre-Prompt - Adaptive Compute Foundations

**TLDR:** Raw seed for a foundations experiment: test whether "EBMs can spend more compute on harder problems" is a real differentiator, given that CoT/thinking autoregressive models can also spend variable inference-time compute.

## Raw User Seed

> let's make a git issue on free energy about checking if it's really true that
> EBMs can spend more computation on more difficult problems or not... from a
> foundations perspective. eg can't COT or thinking COT models do this? why not
> use that? What the diff. This is worthy of an experiment folder, create and a
> pre_prompt.md this msg basically and a plan for this and push to main please.
> If the rest of the experiments are reusable to do this check fine let's do it.
> inspired from https://www.youtube.com/watch?v=18Fn2m99X1k

## Prompt To Future Agent

Do not answer with the slogan "EBMs do inference as optimization" unless you
make clear what that buys beyond CoT, hidden reasoning tokens, verifier-guided
retry/search, best-of-N sampling, and self-refinement.

The first task is conceptual hygiene:

1. Define "spend more computation on harder problems" in measurable terms.
2. List every mechanism that can already do this for autoregressive systems.
3. State the strongest version of the EBM claim that survives those baselines.
4. Design a controlled experiment where EBM-style energy descent and AR-style
   thinking/search are compute-matched, verifier-matched, and difficulty-binned.
5. Decide what outcome would make the EBM claim false, partially true, or
   genuinely distinctive.

Treat this as a foundations audit before implementing new code.
