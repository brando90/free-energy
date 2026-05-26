# Paper portfolio

This folder tracks the three-paper publication strategy for the Free Energy
project. The existing root-level `paper_latex/main.tex` remains the canonical
review paper. The folders here add paper-specific plans and, where needed, new
standalone LaTeX scaffolds.

## Strategy

The project should be publishable even if the most ambitious model fails. The
three-paper stack is deliberately staged:

| Track | Publication role | Status | Canonical source |
|---|---|---|---|
| 1. Review paper | "Guarantees" a publication by making the literature map, claim audit, and experimental protocol useful on their own. | Active draft | `../main.tex` and `review_paper/README.md` |
| 2. Data-centric comparison paper | "Guarantees" a publication by comparing three architectures on the same data/protocol: AR/LLM baseline, normal EBM, and our novel EBM. | Scaffolded | `data_centric_architecture_comparison/main.tex` |
| 3. Novel EBM paper | Conditional upside paper if the novel EBM actually works. | Scaffolded | `novel_ebm/main.tex` |

The word "guarantees" is a strategy label, not a claim that acceptance is
automatic. It means the paper has a defensible contribution even if the novel
architecture result is negative.

## Assignment rule

For PRs touching these paper tracks, assign:

- `@brando90`
- `@eobbad`
- `@Srivatsava`

Use PR bodies to request specific review:

- `@eobbad`: toy controls, EBM framing, and whether the claimed AR failure/pro
  examples are fair.
- `@Srivatsava`: VeriBench data splits, Lean verifier metrics, pass@k protocol,
  and leakage checks.
- `@brando90`: paper thesis, publication strategy, and final prose.

## Promotion rule

Results move from experiments into papers only when they have:

1. a reproducible command;
2. a saved JSON/table;
3. a figure or table;
4. uncertainty or an explicit caveat;
5. a paragraph in the relevant `FINDINGS.md` or paper-track notes.
