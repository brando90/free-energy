# Free Energy paper drafts

**TLDR:** This folder holds the Free Energy paper portfolio. The root-level `main.tex` is the review paper. `papers/` holds the data-centric comparison paper and the conditional novel-EBM paper. The strategy is to have two publication-floor papers (review + data-centric comparison) plus one upside paper if the novel EBM works. **Status: DRAFT** — do not submit, do not cite externally yet.

## Three-paper strategy

| Track | Publication role | Source |
|---|---|---|
| **1. Review paper** | "Guarantees" a publication by making the literature map, claim audit, and experimental protocol useful on their own. | `main.tex` |
| **2. Data-centric comparison paper** | "Guarantees" a publication by comparing AR/LLM, normal EBM, and our novel EBM on the same toy, VeriBench, and MNIST-first protocol. | `papers/data_centric_architecture_comparison/main.tex` |
| **3. Novel EBM paper** | Conditional upside paper if the novel EBM actually works. | `papers/novel_ebm/main.tex` |

The quoted "guarantees" are a project-management target: the first two papers
should remain publishable even if the novel architecture result is negative.

Paper-track PRs should assign `@brando90`, `@eobbad`, and `@Srivatsava`.
Use the PR body to ask `@eobbad` for toy/EBM framing review and `@Srivatsava`
for VeriBench/Lean protocol review.

## Layout

```
paper_latex/
├── main.tex                       entry point; \input{}'s the rest
├── preamble.tex                   packages, macros, theorem env
├── 00_abstract.tex
├── 01_introduction.tex
├── 02_ar_advantages.tex           the honest pros
├── 03_critiques_catalog.tex       layer-tagged catalog of objections
├── 04_what_holds.tex               which objections survive the audit
├── 04a_experimental_program.tex    toy / VeriBench / MNIST empirical plan
├── 05_ebm_motivation.tex          EBMs + partition function
├── 06_ebm_training.tex            CD / NCE / score matching / Stein
├── 07_ebm_inference.tex           Langevin / variational / energy descent
├── 08_other_alternatives.tex      diffusion / JEPA / SSMs / AR + verifier
├── 09_conclusion.tex
├── 97_acknowledgments.tex
├── 98_appendix_open_questions.tex
├── 98_appendix_proofs.tex
├── refs.bib
├── Makefile
├── papers/
│   ├── README.md
│   ├── review_paper/README.md
│   ├── data_centric_architecture_comparison/
│   │   ├── main.tex
│   │   ├── Makefile
│   │   └── README.md
│   └── novel_ebm/
│       ├── main.tex
│       ├── Makefile
│       └── README.md
├── figures/                       (empty for now)
└── README.md
```

## Build

```bash
cd paper_latex
make             # latexmk (recommended; resolves bib automatically)
# or
make full        # explicit pdflatex / bibtex / pdflatex / pdflatex
```

`make watch` runs latexmk in continuous-preview mode.

`make clean` removes intermediates; `make distclean` also removes the PDF.

## Companion empirical work

This paper is the literature-grounded half of a two-part project. The probe-by-probe empirical companion lives at

```
../experiments/02_ar_pros_cons/
```

with claim definitions in `CLAIMS.md`, probe specs in `PROBE_SPECS.md`, and a living results log in `FINDINGS.md`. The review-paper coordination layer lives at

```
../experiments/03_review_paper/
```

with the Codex runbook, section map, blog drafts, and paper synchronization plan.
The paper references these folders by relative path throughout.

## Status

- Initial draft scaffolded 2026-05-25 (this branch: `paper_latex/ar_vs_ebm_initial_draft`).
- Bibliography includes best-effort bibtex entries; several are marked `% TODO(verify)` and should be checked against canonical sources before any external submission.
- Open questions live in `98_appendix_open_questions.tex` and are pre-registered as measurement targets for the companion suite.
- Data-centric and novel-EBM paper scaffolds were added as planning drafts in `papers/`.
