# Transcription — Fisher Divergence / Score Matching Notes

Source images (in logical order):

- `assets/note_01_derive_fisher_divergence.jpg`
- `assets/note_02_score_matching_motivation.jpg`
- `assets/note_03_min_DF_grad_theta.jpg`
- `assets/note_04_hyvarinen_trace_hessian_and_experiment_plan.jpg`
- `assets/note_05_hyvarinen_paper_annotations.jpg`
- `assets/note_06_song_paper_annotations.jpg`

Best-effort transcription. Unclear handwriting is marked `[unclear]`. Symbols
expanded for searchability: ∇_x, ∇_θ, log, ∫, E_p[·], tr(·).

---

## Note 01 — From Fisher equality of gradients to a Fisher-divergence-style objective

Top derivation: if for all `x ∈ X`, `∇_x f(x) = ∇_x g(x)`, then by integration

```text
f(x) = g(x) + c       (*)
```

(Equivalently: `f = log p_θ`, `g = log p_θ̂`.)

Now check the normalization. If `p, q` are both densities on `X` with
`∫_X exp(f(x)) dx = 1` and `∫_X exp(g(x)) dx = 1`, then using (*):

```text
∫ exp(f(x)) dx = ∫ exp(g(x) + c) dx
                = exp(c) · ∫ exp(g(x)) dx
                = exp(c) · 1.
```

So `exp(c) = 1` ⇒ `c = 0`.

Therefore `∫ exp(f(x)) dx = ∫ exp(g(x)) dx = 1` and via (*), `f(x) = g(x)`,
i.e. **`p(x) = q(x)`** pointwise.

Conclusion (boxed):
> Matching the **scores** `∇_x log p` matches the pdfs, **without** evaluating
> the normalization constant. So let's try to match the gradients `∇_x` of the
> log-densities (instead of the densities themselves). [Q: explain why.]

> "Conf: are we trying to learn/match `p(x) = q(x)` for all `x ∈ X`?"

So we optimize **the Fisher divergence**:

```text
D_F(p* || p̂) = E_{p*}[ (1/2) || ∇_x log p*(x) - ∇_x log p̂(x) ||² ]
```

→ "drive this to zero."

---

## Note 02 — Why match scores (not densities); the partition function disappears

Heading: **Score matching**.

Margin annotation (green): "the derivative wrt something `p̂_θ` really matters
[unclear] — `p̂_θ`'s effect from this MLE + LOG[?] [unclear] is hard."

Body:
- Why does this exist? Is it equivalent to MLE? Score matching: let's try
  anything else — and it happens that `∇_x [log p]` (= score) does NOT contain
  the partition function. So maybe that works.
- Q: why is `∇_x log p` called the **score**?

[Margin Q: "Are we trying to learn it differently than this objective? Look at
[unclear]."]

**MLE recap.** It's hard due to `Z_θ`:
- `(d/dθ) ∫ log p_θ` = `∇_θ E_{p_θ}[· - ∇_θ E_θ(x)]` requires sampling from
  the model.

**Instead use Lemma: if for all `x ∈ X`, `f(x) = g(x) + c`, then the two
functions differ only by a constant if and only if `∇ f = ∇ g`.** So setting
`log p̂_θ(x) = g(x) + c` and taking ∇_x makes the `+c` drop out:

```text
log p_θ(x) = g(x) + c
log p_θ(x) = -E_θ(x) + C            [for an EBM]
∇_x log p_θ = -∇_x E_θ(x)           (no Z_θ!)
```

Why: pdfs in EBMs:

```text
p_θ(x) = e^{-E_θ(x)} / Z_θ.
log p_θ(x) = log(e^{-E_θ(x)} / Z_θ) = -E_θ(x) - log Z_θ
                                      \_____  C  _____/
```

Check normalization: `∫ exp(log p_θ(x)) dx = ∫ p_θ(x) dx = 1`. ✓

---

## Note 03 — Minimizing the Fisher divergence; derive the ∇_θ update

Setup: given `p*`, find `θ̂` such that:

```text
θ̂ = argmin_θ D_F(p* || p̂_θ)
  = argmin_θ E_{p*}[ (1/2) || ∇_x log p*(x) - ∇_x log p̂_θ(x) ||² ]
```

When is this 0?

```text
D_F(p*||q) = 0  ⇔  E_{p*}[ (1/2) || ∇_x log p*(x) - ∇_x log q(x) ||² ] = 0.
              p* > 0
              Σ_{x ∈ X} p*(x) · (1/2) || ∇_x log p*(x) - ∇_x log q(x) ||² = 0.
```

Because each term is ≥ 0 and `||·||² ≥ 0`, the only way for the sum to be zero is:

```text
||∇_x log p*(x) - ∇_x log q(x)||² = 0    ∀ x ∈ X.
```

Cool — so let's do SGD / FD / GA on this.

(Boxed) goal: find params `θ` for `q` such that
`scores match (∇_x log p̂_θ ≈ ∇_x log p*)`, i.e. minimize over `θ`.

Gradient-descent update:

```text
θ^{t+1} := θ^t − η ∇_θ D_F(p* || p̂_{θ^{(t)}})
        := θ^t − η ∇_θ E_{x ~ p*}[ (1/2) || ∇_x log p*(x) - ∇_x log p̂_θ(x) ||² ]
```

Cool!

```text
∇_x log p̂_θ(x) = −∇_x E_θ(x) − ∇_x log Z_θ   (the Z log term has no x-dep!)
                = −∇_x E_θ(x).
```

(Margin: "computable via integration", "wait — `∇_x` then bar wrt `x`?" —
the `∇_x log Z_θ = 0` because `Z_θ` does not depend on `x`.)

So the update reduces to:

```text
θ^{(t+1)} := θ^{(t)} − η ∇_θ E_{x ~ p*}[ (1/2) || ∇_x log p*(x) − ∇_x E_θ(x) ||² ]
                                            [boxed: depends on ∇_x log p*]
```

---

## Note 04 — Expanding the score-matching loss + Hyvärinen's identity + experiment plan

Top: expanding the squared norm and using Hyvärinen's integration-by-parts
trick to eliminate the unknown data score `∇_x log p*(x)`.

```text
θ^{(t+1)} := θ^{(t)} − η ∇_θ E_{x~p*}[ (1/2) || ∇_x log p*(x) − ∇_x log p̂_θ(x) ||² ]

θ^{(t+1)} := θ^{(t)} − η ∇_θ E_{x~p*}[ (1/2) || ∇_x log p*(x) ||²
                                       + (1/2) || ∇_x log p̂_θ(x) ||²    ← data-free
                                       − ⟨∇_x log p*(x), ∇_x log p̂_θ(x)⟩ ]
                                                                          ↑
                                                                  "Q: hmm? it's
                                                                   not data-free."
```

The first term doesn't depend on θ so it drops out (constant in θ). The
problematic cross term involves the unknown data score `∇_x log p*(x)`. But
Hyvärinen's trick rewrites it via integration by parts using only the model:

```text
−E_{x~p*}[ ⟨∇_x log p*(x), ∇_x log p̂_θ(x)⟩ ]
   = E_{x~p*}[ tr(∇²_x log p̂_θ(x)) ]      (under mild boundary conditions)
```

→ "Some condition by Hyvärinen states the [unclear] is computable just from
the model `q` and under derivatives of it [unclear] so no foul (oh)."

So the update becomes:

```text
θ^{(t+1)} := θ^{(t)} − η ∇_θ E_{x~p*}[ Σ_{d=1}^{D} ( (∂ E_θ(x) / ∂ x_d)²
                                                    + ∂² E_θ(x) / ∂ x_d² ) ]
                                                                  └────┬────┘
                                                                still depends on
                                                                ∇_x = [ ∂/∂x_d ]
```

(Margin Q: "is it the [unclear] data we [unclear]?")

> Q: but enough trace — is it truly the best? Is it really hard to compute?

**Plan:**

1. Let's try it in PyTorch.
2. ... and Jax. I'm skeptical, let's just try.
   - (1) Run it & "profile" how long it actually takes.
   - (2) Do mathematical analysis of GPT-5 / Codex / Pro to see how long it
     takes... asymptotically. But empirical is what's true.
   - (3) Maybe to help us think.
3. If (1) is good, let's do score matching.
   - (a) On a toy ebm example we've developed → quad.
   - (b) The lean ebm we've been developing.

---

## Note 05 — Annotations on Hyvärinen (2005) "Estimation of Non-Normalized Statistical Models by Score Matching"

Highlighted phrases (printed text underlined / boxed in pen):

- **Score Matching (SM):** "ψ(x; θ) = ∇_x log p_θ(x)"
  (Marginalia: "Let me try working through derivations.")

- Definition of the SM objective (Hyvärinen, Eq. 2):
  ```text
  D_F(p_x(·) || p_{x;θ}(·)) = E_{p_x}[ (1/2) || ψ(x; θ) − ψ_x(x) ||² ]
  ```
  with `ψ_x(x) = ∇_x log p_x(x)` (the data score).

- Hyvärinen's identity (Eq. 4 in the paper):
  ```text
  D_F(p_x || p_{x;θ}) = E_{p_x}[ Σ_i ( ∂_i ψ_i(x; θ) + (1/2) ψ_i(x; θ)² ) ]
                       + const(θ)
  ```
  with two non-trivial derivatives in `x`: a **first derivative** (`ψ²`) and a
  **second derivative** (`∂_i ψ_i = ∂²/∂x_i² log p_θ`).

  (Marginalia: "Why does this require [unclear]? Honestly, do we need to
  compute the full second derivative? Is it the diagonal of the Hessian? Yes,
  it is — diagonal!")

- Score matching assumes some regularity conditions on the model density
  `p_θ`, with positive density at every data point.

  (Marginalia in red: "Really?? Hutchinson? Doable!" referring to using a
  Hutchinson-style stochastic estimator for the trace of the Hessian, which
  is the standard trick to avoid computing all D diagonal entries.)

- "An important drawback of the objective above is that, in general,
  computation of both second derivatives is expensive in high dimensions"
  (Marginalia: "So is it 'doable'? Let's go check this on the toy & lean
  ebms.").

- "Although SM only requires first-order gradients and not second
  derivatives, [unclear] makes scoring difficult with [unclear] dimensions."

- "For this reason, the implicit SM formulation of Eq. 4 has only been
  applied to relatively simple energy functions where computation of the
  second derivatives is tractable."

- "Score Matching assumes a continuous data distribution with positive
  density over the support, in case data is just real-valued."
  (Marginalia: "Note: that's why DSM denoising score matching, then jam
  noise that smears [unclear] for continuous? Q: but what about
  [unclear]? Note also `f` here is `R^N → R`.")

---

## Note 06 — Annotations on Song & Kingma (2021) "How to Train Your Energy-Based Models"

Highlights and margin notes on the title and abstract:

- Title boxed: **"How to Train Your Energy-Based Models"** — Yang Song,
  Diederik P. Kingma. Stanford / Google.

  Margin: "Q: contrastive forward-forward vs EBM, Hopf, Boltzmann hyperloop
  Mod[unclear] DL / capsule nets?"
  Margin: "from Hinton, but this 2 hr lecture is leading exemplary &
  [unclear]."

- Abstract phrases highlighted: "Energy-Based Models (EBMs); flexible
  parameterization; tractable likelihood; unlike most other probabilistic
  models, EBMs do not place a restriction on the tractability of the
  normalization constants, thus are more flexible to parametrize and can
  model a more expressive family of probability distributions. However, the
  [unclear] requires of EBMs makes training notoriously difficult."

- Margin: "I don't understand the conjecture; why do GPT think this EBM could
  be better/has fee/Y[unclear] lecture re: ∞ infinite [unclear]?"
  "looking at part of #4 / Adv. Cas: too where I don't have a perfect script?"

- Section 1 (Introduction) highlights:
  "Probabilistic models with a tractable likelihood are a double-edged sword.
  Tractable likelihood allows for straightforward [unclear] of the model
  parameters in [unclear] the log likelihood of the data… Through similar
  tractable models such as autoregressive [unclear] or normalizing flow…
  generative models the data is modeled as a transformed latent variable
  with a tractable likelihood. … Synthesis of pseudo-data from the model can
  be done with a specified, tractable procedure."

- "These assumptions are not always natural in [unclear]"
  (Margin: "block fact for models 1 is barely any model")

  "Energy-based models (EBMs) are much less restrictive in functional form:
  instead of specifying a normalized probability, they only specify the
  unnormalized negative log probability, called the **energy function**,
  E(x): R^D → R, [unclear] that the marginal density of the variables of
  interest is so derived as:
  p_θ(x) = e^{−E_θ(x)} / Z_θ
  where the normalizing constant Z_θ, also known as the **partition
  function**, is..."

- Bottom-page margin Q&A:
  "Q: are pdf scores matching here? Are ∫ p_θ(x)dx = 1, ∫ E_θ(x)dx, etc.
  required as a 'quantity of theirs'? Is the [unclear] of [unclear] called
  the score?"
  "Q: is `p_θ(x) = ∇_θ E_θ(x)` called 'score'?"
  "Q: Curious: how does generalization theory look for EBMs?"

---

## Cross-cutting open questions in the notes

1. **Is the trace of the Hessian really hard?** Notes repeatedly question
   whether `Σ_i ∂²/∂x_i² log p̂_θ(x)` is intractable. Hutchinson estimator is
   flagged in red as the obvious mitigation.
2. **Diagonal vs full Hessian.** Confirmed: only the diagonal sum (trace) is
   needed — but exact computation costs D forward/backward passes through
   `∇_x E_θ`.
3. **Gradient wrt x vs wrt θ.** All second derivatives in the SM loss are
   in `x`, not in θ. The final SGD step takes a `∇_θ` of this scalar — so
   asymptotic cost question is: how much more expensive is the SM loss
   compared to a standard log-likelihood / DSM training step?
4. **Targets to profile.**
   - PyTorch and JAX
   - Toy EBM in `experiments/01_toy_ebm_training/` (continuous variant)
   - Lean EBM (to be developed)
5. **Connection to prior work.** Brando's *beyond-scale-language-data-diversity*
   paper computed the trace of the parameter-space Hessian via the elementwise
   square of the parameter gradient (a Hutchinson-style estimator with
   variance reduction). The intuition: `∇_x` (input dim) is typically much
   smaller than `∇_θ` (network params), so the bound here is *more* favorable.
