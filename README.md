# free-energy: Freedom with Energy from the Transformer

**TLDR:** This repo investigates whether post-softmax, energy-based, and
adaptive-compute architectures can overcome limits of autoregressive systems,
using Lean/formal verification as the hard correctness test bed.

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

## AGI research strategy

Project lead context: Brando Miranda and Srivatsava (Sri) Daruru.

Core objective: define a high-impact independent research project that tests
whether autoregressive (AR) architectures are sufficient for AGI, where AGI is
understood economically as an AI system capable of performing any viable human
activity.

Source docs:

- [Strategy slide deck 1](https://docs.google.com/presentation/d/1kyQWQ74nqj7yPcn2LKFBjYti4Wl0h691vR8_AHbSDL8/edit?slide=id.g3f28065e0d7_0_116#slide=id.g3f28065e0d7_0_116)
- [Strategy slide deck 2](https://docs.google.com/presentation/d/1mjoYhbMNyGDN87YuUqGWZhS0q42eUhzDNTV-Zys-sOc/edit?slide=id.p1#slide=id.p1)
- [Working Google Doc](https://docs.google.com/document/d/14uMW6HMjVYEfFlgDlMRe8czxPbMHe5e035DKQk_I9Ts/edit?tab=t.0)

### Core premise

The working assumption is that scaling laws keep holding: vast compute,
expansive data, and large models remain prerequisites. Given that compute,
data, and large-model scale are available -- e.g. FineWebEdu-style data,
B200-class compute, and Marin-style infrastructure -- the remaining variable
that may determine the capability ceiling is architecture.

Current AR systems are limited by token-by-token generation and the need for
constant human oversight: agents can produce useful work, but a human still
has to verify, correct, and restart them. The key question is whether this
factorization limitation is an intrinsic mathematical barrier or a flaw that
scaling, tools, verifiers, and feedback can eventually overcome.

### Three project tracks

| Track | Hypothesis | Methodology and execution |
|---|---|---|
| **A. Scalable oversight** | AR models are sufficient for AGI, but the human verification bottleneck must be automated. | Develop certifying judges that autonomously verify outputs, such as code unit tests and mathematical theorem proofs. If solved broadly, scalable oversight becomes close to AGI-complete. |
| **B. Proving insufficiency** | AR models are not sufficient because token-by-token factorization has fundamental mathematical limits. | Provide empirical and theoretical evidence for those limits, using formal environments such as Lean 4 as a forcing function for contradiction-style or impossibility-style arguments. |
| **C. The EBM alternative** | AR models are not sufficient, so the architecture must change to bypass the sequential generation trap. | Replace AR generation with energy-based models that score candidate sequences holistically. Use Lean as the rigorous benchmark for comparing EBMs against AR systems. |

### Ownership and credibility

Sri should take primary ownership of one track rather than spread focus across
all three. Cross-collaboration is expected, but deep individual ownership is
the path to a credible contribution.

The broader goal is real-world technical impact that establishes credibility
for future investors, collaborators, and venture-building. First authorship is
valuable, but joint impact on a foundational paper can be equally useful if the
work becomes a durable signal of technical judgment.

### Immediate next steps

1. Literature review: Sri reviews the free-energy paper to build intuition for
   EBMs.
2. Issue triage: Sri reviews the relevant GitHub issues covering current
   technical hurdles.
3. Project selection: Sri selects one track based on either intrinsic interest
   or a practical view of which path reaches AGI fastest.
4. Reconvene by Monday, June 29, 2026, to finalize the project choice and map
   concrete engineering and research milestones.

### Theoretical anchors

- **AR bottleneck:** AR models generate one token at a time, conditioned only
  on the prefix. The concern is that this sequential factorization causes
  compounding errors on long-horizon exact tasks and forces a human verifier
  back into the loop.
- **EBM alternative:** EBMs assign energies to whole configurations rather
  than requiring normalized next-token probabilities at every step. This makes
  holistic scoring natural, while moving the hard part into inference,
  sampling, and normalization.
- **No Free Lunch and scaling:** No Free Lunch assumes a uniform distribution
  over all possible problems, which is not the real world. Massive compute and
  data can bake in useful priors over the task distribution humans actually
  care about, much as evolution did through search.
- **Benchmark saturation:** Public benchmarks may saturate before agents become
  robust at real work. The gap between benchmark scores and one-shot agentic
  reliability motivates harder formal test beds.
- **Lean as forcing function:** Lean 4 and formal verification provide a clean
  setting where long-horizon reasoning has an external correctness signal. If a
  scalable-oversight or EBM system works there, it becomes stronger evidence
  for broader economically useful AGI tasks.

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

## Blog and website workflow

This repo includes Brando's website as a git submodule at
`website/brandomiranda`, because some blog drafts are canonical experiment
artifacts in `free-energy` while others are canonical website-side drafts.

Use `BLOG_WORKFLOW.md` for the rule of thumb:

- experiment reports start in `experiments/<NN_name>/blog/` in this repo;
- website-native drafts live in `website/brandomiranda`;
- anything that should be visible on GitHub in the website must be committed as
  a real markdown file in the website repo, not only as a local symlink;
- after merging a website PR, update and merge the `website/brandomiranda`
  submodule pointer in `free-energy`.

Initialize the website submodule from a fresh clone with:

```bash
git submodule update --init --recursive website/brandomiranda
```

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
