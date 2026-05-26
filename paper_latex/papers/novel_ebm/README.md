# Paper 3: novel EBM paper

Working title:

```text
Free-Energy Inference for Verifier-Guided Structured Generation
```

## Publication role

This is the upside paper. It should be written if the novel EBM actually works:
better pass@k, better cost per verified solution, better scaling with difficulty,
or a clearly superior failure profile relative to AR and conventional EBM
baselines.

## Core claim

A novel energy/free-energy model can use verifier feedback and adaptive inference
more effectively than the standard AR/softmax/MLE stack on structured generation
tasks.

## Minimum evidence before this becomes a submission

- A precise energy/free-energy objective.
- A tractable training algorithm.
- An inference algorithm with a bounded budget.
- Toy controls showing the mechanism.
- VeriBench result against AR and normal EBM baselines.
- At least one ablation identifying which component matters.

## Build

```bash
cd paper_latex/papers/novel_ebm
make
```
