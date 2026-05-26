# Review paper plan

This plan is deliberately concrete. The review paper and the experiment suite
should evolve together, not as separate artifacts.

## Paper structure

The canonical paper source is `paper_latex/`. The target structure is:

1. **Introduction.**
   Explain that AR/LLM critiques bundle several different commitments. Separate
   factorization, softmax/local normalization, MLE, inference, trained behavior,
   and external data limits.
2. **What AR LLMs get right.**
   Exact chain-rule factorization, exact likelihood, dense token supervision,
   teacher forcing, streaming generation, scaling laws, and systems maturity.
3. **Catalog of objections.**
   Strength-rate each claim: theorem, robust empirical, weak empirical,
   conjectural, or external.
4. **What survives the audit.**
   Identify which claims actually justify alternative architectures.
5. **Experimental program.**
   Run the claim audit on toy controls, VeriBench/Lean, and MNIST first. Add a
   later vision dataset once the methodology is stable.
6. **Energy-based models.**
   Explain global energies, the partition function, training methods, inference,
   and the AR-local-EBM identity.
7. **JEPA and non-likelihood alternatives.**
   Treat JEPA as a representation-space energy program, not as just another
   generative model.
8. **Other alternatives.**
   Diffusion, masked/any-order models, SSMs/Mamba, flow matching, verifier +
   search around AR models.
9. **Synthesis.**
   The question is not "AR vs EBM" but "where should the hard part live?"

## Empirical program

The first empirical chapter is the AR/LLM pros/cons suite in
`experiments/02_ar_pros_cons/`.

### Track A: toy controls

Goal: show the assumptions and failure modes in settings where the ground truth
is known.

- LeCun `(1-e)^T` blind rollout control.
- Recoverable-error process.
- Verifier-resampling process.
- More VeriBench-like toy proposed by `@eobbad`.
- MNIST toy for sequence/order effects:
  - raster-order AR baseline;
  - any-order/masked reconstruction baseline;
  - EBM or energy classifier/ranker baseline;
  - measure whether left-to-right ordering hurts global constraint satisfaction
    after controlling for parameter count and compute.

### Track B: VeriBench/Lean

Goal: test whether the claims survive a real domain with a hard verifier.

- Split `~/veribench/veribench_dataset` into train/val/test at task level.
- Use Lean pass/fail as the trained-behavior metric wherever possible.
- Fit geometric `(1-e)^T`, constant-pass, and recoverable-Markov models to success
  versus length/depth.
- Measure proof/program length, tactic-depth proxy, compile status, retry count,
  and verifier-rejected intermediate candidates.
- Report whether the unrecoverability premise is empirically true.

### Track C: integrated ablations

Goal: determine which mechanisms matter once everything is trained together.

- Train small AR models on VeriBench-style Lean token data.
- Instrument hidden-state rank, attention entropy, bottleneck rank deficit,
  invalid-token mass, spectral-norm product, and pass@k.
- Run a factorial grid:
  - softmax attention vs sigmoid/linear attention;
  - MLE vs margin/energy objective;
  - blind AR vs verifier-guided retry/search;
  - optional any-order/masked objective.
- Estimate main effects and interactions with bootstrap CIs.

### Track D: vision first pass

Digits/MNIST is acceptable for now because the goal is methodology, not benchmark
leadership. The working scaffold lives in `experiments/04_vision_energy_comparison/`.

- Treat images as sequences under several orderings.
- Compare CNN, tiny ViT/patch transformer, AR next-pixel/token models,
  masked/iterative models, conventional EBMs, the novel EBM prototype, and a
  diffusion/denoising baseline.
- Track whether global digit validity or classifier agreement decays with
  sequence length/order.
- Later replace or supplement MNIST with a more realistic vision dataset.

## Blog series

The blog posts are paper-body drafts. Each post must use the website header rule:

```text
*Brando Miranda — Month YYYY · ~X min read*

**TL;DR.** Single paragraph.

---
```

Planned posts:

1. Testing LeCun's error-compounding argument.
2. The honest pros and cons of autoregressive language models.
3. The AR factorization is exact; the softmax/MLE/inference stack is not.
4. Why EBMs are attractive and why their partition function is hard.
5. JEPA, diffusion, SSMs, and verifier-guided AR as different relocations of the
   same hard part.
6. What the VeriBench experiments show.

## Review responsibilities

- `@eobbad`: toy examples, VeriBench-like control tasks, and interpretation of the
  LeCun `(1-e)^T` test.
- `@Srivatsava`: VeriBench split protocol, Lean verification metrics, and
  integration with the benchmark data.

If GitHub assignment fails because either account is still a pending collaborator,
tag both in the PR body and retry assignment after they accept the invitations.

## Done criteria for this experiment

- The paper has an explicit experimental-program section.
- The blog series has at least one review-paper-plan post.
- `paper_latex/main.tex` compiles.
- Every claim in the paper maps to at least one experiment, literature-only
  argument, or explicit "not tested yet" status.
- The first real VeriBench result is written in both `FINDINGS.md` and the paper.
