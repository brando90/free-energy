# Paper 1: review paper

Canonical LaTeX source:

```text
paper_latex/main.tex
```

Working title:

```text
Free Energy: An Honest Audit of Autoregressive Language Models
and the Energy-Based Alternative
```

## Publication role

This is the "guaranteed" publication track: even if the novel architecture does
not work, the review paper should still be useful because it contributes:

1. a layer-separated audit of AR/LLM objections;
2. a clean account of what AR models get right;
3. an EBM/JEPA/diffusion/SSM taxonomy organized by which AR cons they address;
4. an explicit experimental protocol tying theory claims to toy, VeriBench, and
   MNIST-style tests.

## Inputs

- `../../main.tex` and section files.
- `../../../experiments/00_ar_pros_cons/`
- `../../../experiments/01_review_paper/`
- Blog drafts under `../../../experiments/01_review_paper/blog/`

## Next paper tasks

- Keep LeCun's `(1-e)^T` argument framed as a testable hypothesis, not a theorem.
- Add the first real VeriBench result once probe 06 is wired.
- Add the MNIST/order-effect smoke result once it exists.
- Keep Sutton's critique orthogonal unless the paper grows an appendix on agency,
  goals, and continual learning.
