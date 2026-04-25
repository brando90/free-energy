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
cost, and the deployed forward pass is evaluation on a fitted Q̂. This is
the framing of Koller & Friedman Ch. 11 and the spine of LeCun's EBM
program. Lean is the test bed because formal proving has variable
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

## License

Apache-2.0. See `LICENSE`.
