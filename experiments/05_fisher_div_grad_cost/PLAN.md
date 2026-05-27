# Plan — Fisher Divergence Gradient Cost (Score Matching Profiling)

## Core Question

From the notes (see `TRANSCRIPTION.md`):

> **Is computing the gradient of the Fisher divergence (a.k.a. the explicit
> score-matching loss of Hyvärinen 2005) really as hard as the literature
> claims?**

The objective is:

```text
L_SM(θ) = E_{x ~ p_data}[ Σ_{i=1}^{D} ( (1/2) (∂ log p̂_θ / ∂ x_i)²
                                       +     (∂² log p̂_θ / ∂ x_i²) ) ]
        = E_{x ~ p_data}[ (1/2) ||∇_x log p̂_θ(x)||²  +  tr( ∇²_x log p̂_θ(x) ) ]
```

For an EBM `p̂_θ(x) = e^{−E_θ(x)} / Z_θ`, the partition function vanishes from
the gradient wrt x:

```text
∇_x log p̂_θ(x) = −∇_x E_θ(x)
```

so the SM loss reduces to a closed-form quantity in just the energy `E_θ`:

```text
L_SM(θ) = E_{x ~ p_data}[ (1/2) ||∇_x E_θ(x)||²  −  tr(∇²_x E_θ(x)) ]
                                                ↑
                                       (sign flips because we took
                                        2nd derivative of −E_θ)
```

Training requires `∇_θ L_SM(θ)` — i.e. *backprop through a quantity that
already contains second derivatives in x*. That's the textbook "hard part."

## Why Brando Is Skeptical

1. The 2nd derivative is over `x`, not `θ`. Input dimension `d` is usually
   orders of magnitude smaller than parameter dimension `|θ|` for any
   non-toy network.
2. In Brando's *beyond-scale-language-data-diversity* paper, the **trace of
   the Hessian wrt θ** was approximated cheaply via Hutchinson's estimator
   (elementwise square of stochastic gradients), and that scaled fine.
3. Modern autodiff (PyTorch `torch.func.{grad,vmap,hessian}`, JAX
   `jax.{grad, jvp, vjp, hessian}`) is far better at higher-order than the
   2005-era setup Hyvärinen complained about.

The whole point of this experiment is to **measure**, not reason about,
that cost.

## Methodology

### Models

A simple continuous EBM:

```python
E_θ(x) : R^d → R,  parameterized as MLP(d → h → h → 1)
```

We sweep:

- input dim `d ∈ {2, 8, 32, 128, 512, 2048}`
- hidden width `h ∈ {64, 256, 1024}`
- batch size `B ∈ {32, 128}`

### Losses to compare

For one forward+backward training step:

1. **Exact SM** — explicit Hyvärinen loss with exact `tr(∇²_x E_θ(x))`:
   - per-coordinate second derivative, computed via `torch.func.hessian` /
     `jax.hessian` then take diagonal, OR
   - per-coordinate `grad ∘ grad` (double backward).
2. **Hutchinson SM** — replace `tr(H_x)` with one (or `k`) samples of
   `vᵀ H_x v` with `v ~ Rademacher`. This costs **1 HVP**, not d HVPs.
3. **Denoising SM (DSM, baseline)** — Vincent (2011) trick:
   `E[ (1/2) || σ ∇_x log p̂_θ(x̃) + (x̃ − x)/σ ||² ]`. Only first derivatives.
4. **MLE-style baseline** — just `E_θ(x)` for comparison (no x-grad needed).

### Measurements

For each (loss × d × h × backend × device):

- wall clock per step (forward + ∇_θ L), averaged with warmup
- peak memory (`torch.mps.driver_allocated_memory` / `psutil` rss / jax
  `live_arrays()`)
- ratio vs DSM baseline (the "how much extra work?" answer)
- ratio Exact/Hutchinson

### Backends

- PyTorch on `mps` (Apple Silicon) and `cpu`
- JAX on `cpu` (and `metal` if available; otherwise note skipped)

If `cuda` were available we'd add it, but local laptop is M-series.

## Success Criteria

This experiment is **descriptive**, not pass/fail. We accept it as done when:

- both backends produce numerically consistent SM loss values for the same
  `(θ, x)` (cross-validation),
- we have a wall-clock + memory table for every (loss × d × backend), and
- a written-up README + blog summarising whether SM is "really" expensive at
  the scales we tested.

A *secondary* success: if Hutchinson is within ~2× of DSM at d = 2048, the
notes' skepticism is empirically vindicated.

## Tie-in to existing experiments

- `experiments/01_toy_ebm_training/` trains a **discrete** EBM (binary
  sequences). For Fisher divergence / SM we need **continuous** densities,
  so we either:
  - relax the toy via Gaussian-perturbed one-hot embeddings, or
  - just use a synthetic continuous target (e.g. mixture of Gaussians) for
    the timing study and bring SM back to the toy in a follow-up.
- "Lean EBM" — not yet implemented. The notes flag it as the second target
  for SM training once timing looks viable.

## Out of scope (this experiment)

- training convergence of SM vs MLE (separate experiment)
- correctness proof of Hyvärinen's identity (in the transcription)
- comparison to MCMC / contrastive divergence (existing `00_start_off`)
- GPT-5 / Codex asymptotic analysis (delegated)
