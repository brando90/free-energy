# Questions - Adaptive EBMs

**TLDR:** "Adaptive" is the right term to test, but not yet a conclusion. This experiment asks whether EBMs are distinctively adaptive or whether they are one member of a broader class of adaptive inference/search systems.

## Core Question

Are EBMs **adaptive** in a way that matters scientifically, or do CoT/thinking
autoregressive models already provide the same kind of adaptive compute?

## Definitions To Nail Down

1. **Adaptive compute:** inference-time work increases with instance difficulty.
2. **Useful adaptivity:** the extra work improves hard-case success more than it
   wastes compute on easy cases.
3. **Distinctive EBM adaptivity:** the advantage comes from a global energy /
   compatibility objective and iterative improvement of candidates, not just
   from generic search, retries, or longer reasoning traces.

## Research Questions

1. Do EBMs automatically allocate more steps to harder examples, or do we have
   to design the stopping rule / sampler to make that happen?
2. Can AR systems with visible CoT, hidden thinking tokens, best-of-N,
   self-refinement, or verifier search match the same compute-difficulty curve?
3. At fixed average inference compute, do EBMs improve hard-bin success more
   than AR thinking/search baselines?
4. Is the difference the energy objective, the iterative inference procedure,
   the verifier, or just extra compute?
5. When should the paper say "adaptive EBMs" versus the more general "adaptive
   inference"?

## Falsification Hooks

- If AR+CoT/search matches EBM under compute and verifier matching, adaptivity
  is not an EBM-specific advantage.
- If EBMs help only with exact enumeration, the score may be good but inference
  is still the bottleneck.
- If all iterative methods help similarly, the result supports adaptive
  inference generally, not EBMs specifically.
