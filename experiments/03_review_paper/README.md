# `03_review_paper` -- review paper coordination layer

This experiment is the coordination layer for the Free Energy review paper:

```text
Free Energy: An Honest Audit of Autoregressive Language Models
and the Energy-Based Alternative
```

The actual LaTeX source lives at the repository-level `paper_latex/` directory so
there is only one canonical paper draft. This folder links the paper to the
experiment program, the blog-post series, and the Codex runbook for turning the
outline into results.

## Thesis

The review paper should not argue that autoregressive language models are
"doomed." It should make a sharper claim:

1. The chain-rule factorization is exact and is one reason LLMs scaled.
2. The real objections live at different layers: softmax/local normalization, MLE,
   inference, trained behavior, and external data limits.
3. Only some objections have theorem-level or robust empirical support.
4. The strongest alternatives, including EBMs and JEPA-style methods, should be
   evaluated by which AR/LLM cons they address and which costs they relocate.
5. The first empirical target is the LeCun `(1-e)^T` hypothesis, tested with toy
   controls, VeriBench/Lean, and a simple vision domain such as MNIST.

## What this folder owns

| File | Purpose |
|---|---|
| `PLAN.md` | Detailed work plan for the review paper and companion experiments. |
| `agent/CODEX_RUNBOOK.md` | Precise instructions for future Codex agents. |
| `paper/SECTION_MAP.md` | Mapping from paper sections to experiment/blog outputs. |
| `blog/2026-05-26-free-energy-review-paper-plan.md` | First review-paper blog draft; also usable as early paper prose. |
| `notes/OPEN_QUESTIONS.md` | Living list of unresolved literature/experiment questions. |

## Canonical sources

- Review paper LaTeX: `../../paper_latex/main.tex`
- AR/LLM pros/cons experiment: `../02_ar_pros_cons/`
- AR error-compounding blog source:
  `../02_ar_pros_cons/blog/2026-05-26-ar-error-compounding-real-or-fiction.md`
- Website submodule: `../../website/brandomiranda`

The blog drafts in this experiment should be written so that their body can be
lifted into the paper with minimal editing.

## Current status

- `paper_latex/` already contains a first full review-paper scaffold.
- `experiments/02_ar_pros_cons/` contains the first concrete empirical suite.
- This experiment adds the missing coordination layer and updates the paper to
  make the empirical program explicit: toy controls, VeriBench/Lean, MNIST first,
  later vision dataset to decide.

## Immediate next PR after this one

Implement the first paper-backed result:

```text
experiments/02_ar_pros_cons/probes/probe_06_error_compounding.py
```

against the real VeriBench split and write the result into:

```text
experiments/02_ar_pros_cons/FINDINGS.md
paper_latex/04a_experimental_program.tex
experiments/03_review_paper/blog/
```
