# free-energy: Freedom with Energy from the Transformer

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

Each claim has a baseline and a falsification criterion in `MOTIVATION.md`
(forthcoming).

## Status

Exploratory. Bets, not promises. The thesis is that the partition function
and softmax are worth questioning; the experiments will say whether any of
this cashes out.

## Paper strategy

The project now has a three-paper publication stack under `paper_latex/`.
The point is to make the project publishable even if the riskiest architecture
does not work.

| Track | Role | Source |
|---|---|---|
| **1. Review paper** | "Guarantees" a publication by making the literature map, AR/LLM claim audit, and experiment protocol useful on their own. | `paper_latex/main.tex` |
| **2. Data-centric comparison paper** | "Guarantees" a publication by comparing AR/LLM, normal EBM, our novel EBM, and diffusion/iterative baselines under the same data, verifier, and compute protocol. | `paper_latex/papers/data_centric_architecture_comparison/` |
| **3. Novel EBM paper** | Upside paper if the novel EBM actually works. | `paper_latex/papers/novel_ebm/` |

Here "guarantees" means publication-floor strategy, not automatic acceptance:
the review and data-centric papers should still make defensible contributions
even if the novel model result is negative.

Paper PRs should assign `@brando90`, `@eobbad`, and `@Srivatsava`.
Elyas should be asked to pressure-test toy controls and EBM framing; Sri should
be asked to pressure-test VeriBench splits, Lean verifier metrics, and pass@k;
Brando owns the paper thesis and final strategy.

## Against this thesis

The strongest counter-evidence is that transformers + scaling loss may
already be all we need: scaling laws (Kaplan, Chinchilla) keep paying out,
softmax attention has survived every proposed replacement at scale, and
"just add compute and tokens" has beaten architectural cleverness for a
decade. If that trend holds, the partition function isn't a bug to remove —
it's the substrate that scales. This repo is a bet that it isn't, but the
prior should be against us.

## Hopfield reframe

Ramsauer et al. 2020 (*Hopfield Networks Is All You Need*, arXiv:2008.02217)
proved softmax attention **is** a continuous modern Hopfield network with

    E(ξ) = -lse(β · X ξ) + ½ ξᵀξ + const.

So this project is not "EBMs vs. transformers." It's: **a different
energy than Hopfield's, or a different way to fit / deploy the same
energy without paying for the partition function at every forward pass.**
See `docs/hopfield_equivalence.md` for what's literal vs metaphor.

## Current state

Literature-first phase, deliberately. No experiments are running yet.
The bar to clear before proposing a substrate is high (see *Definition
of Done* below); the homework is in progress.

Done:

- `docs/softmax_graveyard.md` — 15-row table of prior softmax-replacement
  attempts with named failure modes, plus the five properties of softmax
  nothing else replicates simultaneously.
- `docs/hopfield_equivalence.md` — careful statement of the Hopfield
  reframe and a seeded catalog of candidate energies.

In progress: hardware-co-design reading, the broader reading list,
expansion of the candidate-energy catalog, and `MOTIVATION.md`
(per-claim falsification criteria). See the meta tracker issue.

## Definition of done (per architectural proposal)

Any candidate substrate proposed in this repo must, before being
called a result:

1. State the energy function (or explain why it's exp-free / Z-free).
   See `docs/hopfield_equivalence.md` §4 for the candidate catalog.
2. Pass signal-propagation diagnostics at depth ≥ 12 (rank collapse
   per Dong et al. 2021, entropy collapse per arXiv:2505.24333).
3. Run on a streaming-tile kernel analogue of FlashAttention, so
   wall-clock isn't a free pass given to the baseline.
4. Hit a baseline on at least one of: Lean tactic prediction, image
   classification, language-model perplexity at matched compute.
5. **Failure-mode inheritance contract.** For each row in
   `docs/softmax_graveyard.md` Table A, name whether this proposal
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
