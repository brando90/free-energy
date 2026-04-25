# free-energy: Freedom with Enrergy from the Transformer

Post-softmax architectures via free-energy minimization, with Lean as the test bed.

## Motivation

The transformer's softmax — in attention, in the output head, and in the
implicit Z of MLE training — is a partition function: a global normalizer
that couples local computations and rules out flexible energy
parameterizations. The same exponentiation also drives well-known numerical
pathologies: overflow without log-sum-exp, attention sinks,
length-extrapolation breakdown, and the bookkeeping cost of streaming-stable
kernels (FlashAttention's running-max correction). Removing the exp removes
these too.

This repo investigates alternatives via the free-energy / variational / EBM
correspondence:

    F[Q] = E_Q[log P̃(X)] + H(Q)         (maximize over a tractable Q)

Inference becomes optimization, Z disappears or becomes a one-time fitting
cost, and the deployed forward pass is evaluation on a fitted Q̂. Lean is the test bed because formal proving has variable
per-instance difficulty (good for adaptive-compute architectures) and
verifiable correctness (clean training signal).

## Thesis

Four claims to support or falsify:

1. Softmax attention is a mean-field message-passing step. Going up the
   variational ladder (structured MF, Bethe, Kikuchi) captures pairwise
   structure attention can't, at competitive cost.
2. Energy-based heads (contrastive / score-matching) match or exceed
   softmax cross-entropy on structured-output tasks.
3. Adaptive-compute inference (energy descent, iterative refinement) beats
   equivalent additional training compute on variable-difficulty tasks.
4. Exp-free attention substrates (sigmoid, softplus, kernel, energy) match
   or beat softmax attention while removing the numerical-stability tax.

Each claim has a baseline and a falsification criterion in `MOTIVATION.md`.

## Status

Exploratory. Bets, not promises. The thesis is that the partition function
and softmax are worth questioning; the experiments will say whether any of
this cashes out.

## Against this thesis

The strongest counter-evidence is that transformers + scaling loss may
already be all we need: scaling laws (Kaplan, Chinchilla) keep paying out,
softmax attention has survived every proposed replacement at scale, and
"just add compute and tokens" has beaten architectural cleverness for a
decade. If that trend holds, the partition function isn't a bug to remove —
it's the substrate that scales. This repo is a bet that it isn't, but the
prior should be against us.

## Roadmap: four pillars

The thesis is too big to attack head-on, so the work is decomposed into
four pillars. Each has a concrete deliverable and lives in its own
subdirectory.

1. **Softmax graveyard** — `docs/softmax_graveyard.md`. Synthesis of
   why every proposed softmax-attention replacement has either failed or
   been surpassed by softmax at scale, with named failure modes per
   architecture. Frames what *not* to repeat.
2. **Signal propagation** — `experiments/signal_prop/`. Reproduce the
   two failure modes of attention at initialization: rank collapse
   (Dong et al. 2021, arXiv:2103.03404) and entropy collapse / phase
   diagram (arXiv:2505.24333). These are the diagnostic instruments any
   new substrate must pass.
3. **ML systems** — `kernels/flashattention_v1_triton/`. FlashAttention v1
   forward pass implemented from scratch in Triton. Internalizes what
   softmax is doing on the chip (online-softmax recurrence, tiling,
   memory hierarchy) so we know what we'd need to change in silicon for
   an exp-free substrate. See also issue: hardware co-design.
4. **Energy-based modeling** — `experiments/ebm_baseline/`. Toy JEPA with
   side-by-side contrastive (partition-function-explicit) and VICReg-style
   (non-contrastive) training, to feel where Z actually leaks back into
   the cost.

Side quest: `lean/RankCollapse/` — formalize Dong et al.'s rank-collapse
theorem in Lean 4. Unique target at the intersection of formal methods
and architecture analysis; nobody has done it.

## Hopfield reframe

Ramsauer et al. 2020 (*Hopfield Networks Is All You Need*, arXiv:2008.02217)
proved softmax attention **is** a continuous modern Hopfield network with

    E(ξ) = -lse(β · X ξ) + ½ ξᵀξ + const.

So this project is not "EBMs vs. transformers." It's: **a different
energy than Hopfield's, or a different way to fit / deploy the same
energy without paying for the partition function at every forward pass.**
Sharper framing, narrower target.

## Definition of done (per architectural proposal)

Any candidate substrate proposed in this repo must, before being
called a result:

1. State the energy function (or explain why it's exp-free / Z-free).
2. Pass the signal-propagation diagnostics from Pillar 2 (no rank
   collapse, no entropy collapse) at depth ≥ 12.
3. Run on a streaming-tile kernel analogue of FlashAttention (Pillar 3),
   so wall-clock isn't a free pass given to the baseline.
4. Hit a baseline on at least one of: Lean tactic prediction, image
   classification, language-model perplexity at matched compute.
5. **Failure-mode inheritance contract.** For each architecture in the
   softmax graveyard (Pillar 1 table), name whether this proposal
   inherits, partially inherits, or avoids that failure mode, and
   give an experiment that would detect it. Without this, the
   proposal is "another headstone."

## Lineage

- Koller & Friedman, *PGM*, Ch. 11 (inference as optimization).
- LeCun, *A Path Towards Autonomous Machine Intelligence* (2022), EBMs.
- Hinton's free-energy view of Boltzmann machines.
- Yedidia–Freeman–Weiss, Bethe / generalized belief propagation.
- Ramsauer et al., *Hopfield Networks Is All You Need*.
- Mean-field-type transformers (Tembine et al.) — adjacent but
  reparameterizes Z rather than removing it.
- Numerical-stability literature on softmax / FlashAttention / softplus
  attention (e.g. Gao et al. 2025); the practical case that exp is the
  wrong primitive.

Not Friston. "Free energy" here is Helmholtz / variational / ELBO.

## Non-goals

- Beating GPT-class models at general LM.
- A polished library.
- Renaming standard concepts.

## Appendix B: citation

If this repo is useful to you:

\`\`\`bibtex
@misc{miranda2026freeenergy,
  author       = {Miranda, Brando},
  title        = {free-energy: post-softmax architectures via free-energy minimization},
  year         = {2026},
  howpublished = {\url{https://github.com/<your-username>/free-energy}},
  note         = {Exploratory research repository}
}
\`\`\`

## Appendix A: names considered

For the record. Picked `free-energy` because it names the central object
(F[Q]) and carries the lineage (Helmholtz → Boltzmann machines → ELBO → EBM)
without describing the project by its opposition.

- `free-energy` ✓
- `partition-zero`, `no-Z`, `sans-Z` — too cute
- `argmin` — clean but underspecified
- `bethe`, `mean-field-up` — too narrow
- `descent`, `iterate`, `fixpoint` — generic
- `post-transformer`, `after-attention` — defines by opposition
- `freedom-from-transformer`, `free-energy-post-transformer` — pun / overspecified
- `qed-net`, `tactic-energy`, `lean-ebm` — locks in the test bed

## License

Apache-2.0. See `LICENSE`.
