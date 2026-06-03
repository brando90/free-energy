# Transcribed Notes — Fisher-divergence Gradient Cost

Transcription of 6 handwritten / annotated pages on score matching, Fisher divergence,
and the empirical-experiment plan that motivates this experiment folder.

Image files live in `experiments/02_fisher_div_grad_cost/assets/`:
- `note_01_derive_fisher_divergence.jpg`
- `note_02_score_matching_motivation.jpg`
- `note_03_min_DF_grad_theta.jpg`
- `note_04_hyvarinen_trace_hessian_and_experiment_plan.jpg`
- `note_05_hyvarinen_paper_annotations.jpg`
- `note_06_song_paper_annotations.jpg`

Math is rendered with LaTeX-style inline notation. Some handwriting was ambiguous;
guesses are marked `?` and `[…]` indicates illegible text.

---

## Note 01 — Why Fisher divergence equals matching log-densities (asset: `note_01_derive_fisher_divergence.jpg`)

Claim: if for all `x ∈ 𝒳`, `∇_x f(x) = ∇_x g(x)`, then `f(x) = g(x) + c` for some
constant `c`. Apply this to `f := log p_θ` and `g := log p̂_q` (or `log p*`).

Suppose `f(x) = g(x) + c`. If both `exp(f)` and `exp(g)` are densities then both
integrate to 1:

```
∫_𝒳 exp(f(x)) dx = ∫_𝒳 exp(g(x) + c) dx = exp(c) · ∫_𝒳 exp(g(x)) dx = exp(c) · 1.
```

Since `∫ exp(f) = 1` we get `exp(c) = 1 ⇒ c = 0`, hence `f(x) = g(x)` and so
`p_θ(x) = q̂(x)` for all `x ∈ 𝒳`. ✅

So matching the gradients ∇log p of the model and the data automatically matches
the normalized densities — no partition-function evaluation required. This is the
whole motivation for **score matching** (Hyvärinen, 2005): match
`∇_x log p_θ(x)` to `∇_x log p_data(x)` rather than matching the densities
themselves.

Note that we want grad **wrt the input x** (called the **score**), not wrt
parameters θ. The data-score `∇_x log p_data` is *different* from
`∇_θ log p_data` (the Fisher-information style object).

We formalize the score-matching objective as the **Fisher divergence**:

```
D_F(p* ‖ p̂_θ) := E_{x ~ p*} [ (1/2) ‖ ∇_x log p*(x) − ∇_x log p̂_θ(x) ‖² ].
```

We drive this to zero.

---

## Note 02 — Score matching motivation (asset: `note_02_score_matching_motivation.jpg`)

Question: why does the score `∇_x log p_θ(x)` even exist / is it useful? It is
the derivative wrt something that *matters*, namely the input x (the data /
sample). MLE and a few related estimators have failed for unnormalized models
because `log Z_θ` requires integrating over 𝒳, but **score** sidesteps that.

MLE has the issue: `(d/dθ) log ∫ exp(−E_θ(x)) dx = −∇_θ log Z_θ`, which equals
`E_{p_θ}[ −∇_θ E_θ(x) ]`. This requires sampling from the model and an estimate
of the partition function.

Instead use the observation: if for all `x ∈ 𝒳`, `∇_x f(x) = ∇_x g(x)` then the
two functions differ by a constant `f(x) = g(x) + c`. So if we match
`∇_x log p_θ = ∇_x log p*` we get `log p_θ(x) = log p*(x) + c`.

For an EBM: `p_θ(x) = exp(−E_θ(x)) / Z_θ`, so

```
log p_θ(x) = −E_θ(x) − log Z_θ,
∇_x log p_θ(x) = −∇_x E_θ(x),
```

which does **not** involve `Z_θ` at all. Score matching becomes:

minimize `D_F(p* ‖ p_θ) = E_{p*}[ (1/2) ‖ ∇_x log p*(x) + ∇_x E_θ(x) ‖² ]`
over θ.

This is a more tractable functional form than MLE for unnormalized EBMs.

---

## Note 03 — Solving `min_θ D_F(p* ‖ p̂_θ)` and the algorithm (asset: `note_03_min_DF_grad_theta.jpg`)

Given `p*`, find θ such that `q̂_θ = arg min_q D_F(p* ‖ q̂)`:

```
q̂ = arg min_q D_F(p* ‖ q̂) = arg min_q E_{p*} [ (1/2) ‖ ∇_x log p*(x) − ∇_x log q̂(x) ‖² ].
```

When is `D_F(p* ‖ q) = 0`? Iff `E_{p*}[(1/2) ‖ ∇_x log p*(x) − ∇_x log q̂(x)‖²] = 0`,
which (for `p* > 0` on 𝒳) requires
`‖ ∇_x log p*(x) − ∇_x log q̂(x) ‖² = 0` ∀ x ∈ 𝒳, i.e. matching scores
everywhere. **Cool — so let's do SGD / GD on it.**

Plan: find params θ for q̂ such that the scores match,
`∇_x log p* ≈ ∇_x log q̂_θ`. With learning rate η:

```
θ^{(t)} := θ^{(t−1)} − η ∇_θ D_F(p* ‖ q̂_{θ^{(t−1)}})
        = θ^{(t−1)} − η ∇_θ E_{x ~ p*} [ (1/2) ‖ ∇_x log p*(x) − ∇_x log q̂_θ(x) ‖² ].
```

For an EBM `q̂_θ = p_θ(x) = exp(−E_θ(x))/Z_θ`, we have
`∇_x log q̂_θ(x) = −∇_x E_θ(x)`, so the score is computable without `Z_θ`
(✅ the whole point).

But `∇_x log p*(x)` is also unknown in general — the data score. Hyvärinen's
trick (next note) turns the objective into one that does not require it, at the
cost of needing **second** derivatives wrt x.

---

## Note 04 — Hyvärinen's trace-of-Hessian form, and the experiment plan (asset: `note_04_hyvarinen_trace_hessian_and_experiment_plan.jpg`)

Continuing from above. With the gradient still depending on `∇_x log p*(x)`,
Hyvärinen (2005) shows that under mild boundary conditions integration by parts
gives a form that *doesn't* depend on the data-score at all:

```
θ^{(t)} := θ^{(t−1)} − η ∇_θ E_{x ~ p*}
            [ Σ_{d=1..D} ( (∂E_θ(x)/∂x_d)² / 2  +  ∂²E_θ(x)/∂x_d² ) ].
```

i.e. the loss is now

```
L_SM(θ) := E_{x ~ p*}[ (1/2) ‖ ∇_x E_θ(x) ‖²  +  tr( ∇_x² E_θ(x) ) ]    (★)
```

and the gradient `∇_θ L_SM` is what we need.

This still depends on `∇_x = [∂/∂x_1, …, ∂/∂x_D]`, i.e. second derivatives wrt
the input.

### Experimental question

Q: But, at this point, is it really that bad / is it truly hard to compute?

The conjecture in the score-matching literature is "yes, computing the Hessian
trace `tr(∇_x² E_θ(x))` is `O(D)` extra autodiff calls / `O(D²)` for the full
Hessian, so it doesn't scale to high D" — this is exactly the motivation for
sliced score matching, denoising score matching, etc. **I'm skeptical.** ∇_x is
much cheaper than ∇_θ in practice (input dim ≪ param dim for LLMs), so let's
actually measure.

Plan:

1. **Run it & profile it** — see how long it actually takes.
2. **Mathematical analysis** of `O(·)` etc. — GPT-5 / Codex will do the
   asymptotic analysis; here we focus on empirical numbers, but having both is
   what we need.
3. **(modeling, to help me think)** if (1) is good, let's do score matching on
   a) a toy EBM example (we already have one — `experiments/01_toy_ebm_training`),
   b) the lean EBM that we've been developing.

Save:
- raw images as assets in the new experiment folder,
- transcription of the notes,
- profiling code in PyTorch and JAX,
- results (csv / md / plots).

---

## Note 05 — Hyvärinen paper annotations (asset: `note_05_hyvarinen_paper_annotations.jpg`)

This is a printed page from Hyvärinen (2005, "Estimation of Non-Normalized
Statistical Models by Score Matching") with handwritten margin notes. The key
highlighted points:

- **Score matching (SM)**: `ψ(x; θ) = ∇_x log p̂(x; θ)` with the normalization
  constant dropped. The objective is
  `D_F(p_data ‖ p_θ) = (1/2) E_{p_data}[ ‖ψ(x; θ) − ∇_x log p_data(x)‖² ]`.
- The first derivative of `log p̂(x; θ)` captures the data distribution exactly
  ("if `log p̂(x; θ)` has the same first derivatives as `log p_data(x)`, they
  differ by a constant — which equals zero by the normalization requirement,
  giving `p̂(x; θ) = p_data(x)` exactly").
- It is then typically tractable to simulate the discrepancy between
  derivatives of `log p_data` and `log p̂`, since the discrepancy can be
  reformulated as integrals of partials of `log p̂(x; θ)` only (Hyvärinen 2005).
- The result: `D_F = E_{p_data}[ Σ_d (1/2)(∂ψ_d/∂x_d) + (1/2)ψ_d² ]` —
  Theorem 1 of the paper.
- This requires computing the diagonal of the Hessian of `log p̂(x; θ)` (or
  equivalently `E_θ(x)`) wrt x.
- An important downside (from the paper / margin note): in general,
  computation of full second derivatives is quadratic in the dimensionality of
  x. Although SM only requires the *trace* of the Hessian, this often requires
  expensive backward passes and prevents SM from being applied to very-high-D
  data (e.g. images). **My margin note:** "Q: really? in practice ∇_x is
  cheap" — **this is exactly the conjecture we are testing.**
- Bottom-right: SM assumes a parameter family with positive density on a
  connected support; it can be generalized to discrete-data settings (Hyvärinen
  2007). Question: how does generalization-error theory carry over?

---

## Note 06 — Song et al. / energy-based-model paper annotations (asset: `note_06_song_paper_annotations.jpg`)

This is a printed first page of "How to Train Your Energy-Based Models" (Song &
Kingma, 2021, arXiv 2101.03288) with my margin notes. Key bits:

- Probabilistic models with a tractable likelihood are a double-edged sword:
  enforcing tractable likelihood often comes at modeling cost (autoregressive,
  flows, etc.). **EBMs do not place a restriction on the tractability of the
  normalizing constants**, so they are more flexible.
- A tractable likelihood is related to "low-dim manifold" assumptions etc.;
  EBMs avoid these.
- These assumptions are not always natural to data. EBMs are *the most flexible*
  in functional form, instead of specifying a normalized probability function,
  EBMs specify the unnormalized negative log-density (the *energy*), allowing
  the model to learn the "quantity of fit" without `Z_θ`.
- My margin questions:
  - "Curious: forward-forward vs. EBM? Hopf-Rosenblatt RNN-like-EBM hybrid? RL
    capable nets?"
  - "From Hopfield → fast theorem of MTML being mentioned … RBM is not energy-based or something … this looks like recent Geom?"
  - "I don't understand the conjecture — why do GP / Bowl-fits be better for
    EBM, could it be better than say AR after the Lp norm? AR is infinite-depth
    so latents can have a perfect *zero* fit. EBM is finite-depth but if model
    is misspecified, EBM has no easy way to recover."
  - "Curious: how does generalization theory look for EBMs?"
  - "Q: Z₀ Partition needs sampling from `∇_θ log p_data(x)` is — `−E_θ(x)`
    Conjecture/intuition: it would have been called `Z(θ)`, Score, no `Z_θ`."
  - "Q: Curious why `p_θ(x) p_θ(x)dx = ∇_x [E_θ(x)] · p_θ(x)dx` is called
    'score' is the score of `log p_θ(x)` (since `−E_θ(x)` is the unnormalized
    log-density)?"
  - "Q: really understand covariance / 2nd-deriv ∂²/∂x²..."

The last question is the explicit motivation for *this* experiment folder:
**measure how expensive the second-derivative-wrt-x really is**.

---

## Experiment plan (concretely, what we'll actually code & profile)

Setup: small MLP energy `E_θ: R^D → R` (parameter count Pθ). Sample
`x ~ N(0, I_D)`. Compute the **two pieces** of the SM gradient and time them:

For batch `x ∈ R^{B × D}`:

1. Score term `(1/2) ‖∇_x E_θ(x)‖²` — one backward pass per sample wrt x.
2. Hessian-trace term `tr(∇_x² E_θ(x)) = Σ_d ∂²E_θ/∂x_d²` — `D` backward passes
   over the score (naive), or `1` stochastic estimate via Hutchinson:
   `E_v[ vᵀ ∇_x² E_θ(x) v ]` with `v ~ N(0, I_D)` or Rademacher.

Estimators benchmarked:
- (E1) **Exact SM** — explicit Hessian-trace via PyTorch
  `torch.autograd.functional.hessian` *diagonal* (or `jacrev(grad)` in JAX).
- (E2) **Hutchinson SM** — single-sample trace estimator
  `vᵀ (∇_x score) v` via Hessian–vector product (HVP).
- (E3) **Sliced SM** — projection onto random direction `v`, then `∂(vᵀ score)/∂(vᵀ x)`.
- (E4) **Denoising SM** — perturb `x → x + σε`, regress
  `s_θ(x̃) ≈ −ε/σ`. No second derivatives.
- (E5) **Baseline** — MLE-gradient `∇_θ E_θ(x)` only (no Z) — pure first-order
  reference cost.

Sweep `D ∈ {2, 8, 64, 512, 2048}`; batch `B = 64`; report mean step-time and
peak memory across 20 warmed steps.

Backends:
- **PyTorch** (`src/profile_sm_pytorch.py`)
- **JAX** (`src/profile_sm_jax.py`)

Apply to:
- toy EBM from `experiments/01_toy_ebm_training/`
- ("lean ebm" placeholder — see TODO in `src/`).

Output: `results/profile.csv` + `results/profile.md` summary.
