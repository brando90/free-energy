# Free Energy paper draft

**TLDR:** Literature-grounded audit of the standard objections to autoregressive language models, the partition-function obstacle of energy-based alternatives, and the broader space of architectures (diffusion, JEPA, SSMs, AR + verifier). Companion to the empirical suite in `../experiments/02_ar_pros_cons/` and the review-paper coordination layer in `../experiments/03_review_paper/`. **Status: DRAFT** — do not submit, do not cite externally yet.

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
