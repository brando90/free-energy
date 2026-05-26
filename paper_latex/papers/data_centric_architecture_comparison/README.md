# Paper 2: data-centric architecture comparison

Working title:

```text
A Data-Centric Comparison of Autoregressive, Energy-Based, and Novel Energy Models
for Verifier-Guided Generation
```

## Publication role

This is the second publication-floor paper. It should be publishable even if the
novel EBM is only competitive or partially successful, because the contribution is
the controlled dataset/protocol comparison across architectures.

## Core comparison

Compare three architecture families under the same data splits, metrics, and
compute budget:

1. **AR/LLM baseline.**
   Standard autoregressive transformer trained/evaluated on the same tasks.
2. **Normal EBM baseline.**
   A conventional energy model using known training/inference methods.
3. **Our novel EBM.**
   The proposed free-energy / post-softmax variant.

## Data

- Primary: VeriBench / Lean.
- Toy controls: from `experiments/02_ar_pros_cons/`.
- Vision first pass: MNIST.
- Later vision dataset: to decide once the methodology is stable.

## Success criteria

- A clean train/val/test split protocol.
- Same candidate pools and verification budget across methods where possible.
- Matched compute and parameter reporting.
- Pass@k, verifier survival, error-vs-length, and cost-per-verified-solution.
- Negative result still publishable if it explains which mechanism failed.

## Build

```bash
cd paper_latex/papers/data_centric_architecture_comparison
make
```
