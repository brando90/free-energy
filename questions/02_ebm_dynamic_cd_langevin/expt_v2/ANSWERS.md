# Answers to the 9 questions

Each answer gives a direct verdict, the evidence from our experiments (numbers
live in `results/RESULTS.md`), and the connection to the literature. The
experiments are tiny and tractable *on purpose*: where possible the partition
function and the exact maximum-likelihood gradient are computed by brute force,
so claims about bias/variance/`Z` are measured against ground truth, not
asserted. The two most error-prone results (E2/E4) were independently
re-implemented by Codex from a spec; the two implementations agree to ~2–3
significant figures (`results/crosscheck_codex/`).

---

## Core theory — EBMs & the partition function

### RQ1 — If the manifold hypothesis holds, why deal with `Z(θ)` at all? Does it only *seem* intractable because we never *see* `Z`?

**`Z` is genuinely, provably intractable — but it is irrelevant to everything
the manifold intuition actually cares about.** Both halves matter.

- *It is really intractable* (E5): computing `log Z` exactly by enumeration
  costs `2^{nv}` — the measured time doubles with every added bit (fit slope
  ≈ ×2/bit), so it is sub-millisecond at `nv=14` and extrapolates to ~`10^8 s`
  (years) at `nv=50` and longer than the age of the universe at `nv=100`. This
  is not an artifact of "not looking at it."
- *But you never need it to traverse the world* (E1–E4): the manifold — the
  energy landscape, its minima, the ranking of states, the gradient `∇_x E`
  used to sample — is **invariant to `Z`**, because `log p(x) = −E(x) − log Z`
  and `log Z` is an additive constant in `x`. E1 trains and samples a continuous
  EBM without ever calling `Z`; E2/E4 compute gradients whose `Z`-dependence is
  the *negative phase*, estimated by sampling, not by evaluating `Z`.

So the sharpened answer to the note: `Z` *seems* unnecessary because it is
invisible to sampling and optimization — it cancels. It stops being optional the
moment you ask for a **normalized probability** of a specific `x` (model
comparison, calibrated likelihood, density evaluation). Then it is exponentially
hard, and AIS (E5) is the standard "messy bypass" that estimates `log Z` to
~0.15% here. Manifold-for-use: yes, forget `Z`. Probability-for-evaluation: you
pay for `Z`. (Refs: Neal 2001 AIS; Salakhutdinov & Murray 2008 RBM AIS.)

### RQ2 — Is MCMC a *fundamental* requirement for EBMs, or just the default? Can we bypass it (SAGE/GPTs-style)?

**Not fundamental. It is the default for *unnormalized* EBMs, and it is
fully avoidable for training.**

- *MCMC-free training exists and works* (E1): denoising **score matching**
  trains a valid continuous EBM with **zero** model sampling in the loss and no
  `Z` — the objective only needs the data and the model's `∇_x E`. The note's
  "do it messily / SAGE / GPTs" instinct is exactly right: **autoregressive**
  models factor `p(x)=∏_i p(x_i|x_{<i})` so each factor is *locally* normalized
  by a small softmax — there is no global `Z` and no MCMC at all. Other
  MCMC-free trainers: sliced score matching (Song et al. 2019), noise-
  contrastive estimation (Gutmann & Hyvärinen 2010).
- *Sampling need not be MCMC either*: a learned score defines a deterministic
  **probability-flow ODE** sampler (Song et al. 2021); flows/autoregressive
  models sample by direct ancestral sampling.
- *Where MCMC does show up, its weakness is mixing* (E3): a single chain on a
  multimodal target is sticky (τ≈100, visited 1 of 3 modes). That is the real
  reason to want to bypass it.

Verdict: MCMC (Langevin/Gibbs) is a convenient default tied to the *unnormalized*
energy parameterization, not a law. Score matching bypasses it for training;
ODE/ancestral methods bypass it for sampling.

### Q3 — Is the PMF/PDF restriction an unnecessary limitation for pure EBMs?

**Two of its three assumptions are droppable; one is not.**
- *Discreteness/countability* — drop it. E1 is a pure EBM on continuous `R²`;
  going continuous is in fact the key enabler (Q5).
- *Explicit normalization for every task* — drop it for ranking, retrieval,
  argmax/MAP, and optimization, which use only **relative** energies (the
  free-energy / EBM correspondence this repo is built on).
- *Normalizability `∫ e^{−E} < ∞`* — **keep it.** The instant you want an actual
  probability (a calibrated likelihood, a sampling distribution, an expectation),
  the energy must be normalizable and you are back to `Z`. You can hide `Z`, you
  cannot abolish it while still claiming "this is a probability." (This is the
  same boundary E5 draws and the companion experiment 03 draws for normalization
  in general.)

---

## Algorithms — MCMC & Langevin

### Q4 — Parallel / semi-sequential chains so mixing improves *and* cost drops?

**Yes, and it is the right lever — with one honest caveat.** (E3)
- *Cost drops*: a **batched** MCMC step is nearly free on vectorized hardware —
  measured Langevin throughput rose ~10,000× from `M=1` to `M=16384` chains with
  almost flat ms/step until the device saturated. Parallel chains are essentially
  a free axis.
- *Mixing/variance improves*: one long chain on a 3-mode mixture was stuck
  (τ≈103, **0.97%** effective samples, recovered weights `[0,1,0]` instead of
  `[⅓,⅓,⅓]`). `M` parallel chains cut the mode-weight error ~4× at fixed total
  samples.
- *Semi-sequential = persistence*: **PCD** carries chain state across SGD steps
  so burn-in is paid once. In E2, PCD matched CD-20's bias (`0.004`) at **one**
  Gibbs step per iteration — the amortization the note is reaching for.
- *Caveat*: parallel *short* chains do not cross energy barriers either; they
  inherit the initialization's basin coverage (E3 error floor). Genuine
  between-mode mixing still needs tempering/annealing or score-based sampling.

This is precisely how modern EBMs scale: persistent replay buffer + many
parallel Langevin chains (Tieleman 2008; Du & Mordatch 2019).

### Q5 — Go strictly continuous to dissolve the undirected chicken-and-egg?

**Yes — this is the single most useful move, and E1 demonstrates the whole
"pure EBM" recipe working.** In continuous space the energy is differentiable,
so:
- you sample by Langevin/SGLD using `∇_x E(x)` directly — no `Z`, and no need
  for the directed `p(x|h)`/`p(h|x)` conditional structure whose absence creates
  the undirected chicken-and-egg; you just descend one scalar energy;
- you can train with **score matching** (E1), which removes model-sampling from
  the training loop entirely, dissolving the *training-time* chicken-and-egg
  (you no longer need samples-to-get-the-gradient-to-get-samples).

E1's `E_θ:R²→R` trains via both PCD-Langevin and MCMC-free DSM and samples
cleanly on a connected target. **Caveat surfaced by E1:** well-separated modes
(the eight-Gaussians case) remain hard to *sample/score-match* cleanly with a
single energy — single-scale score matching blurs isolated modes into a ridge;
the standard fix is multi-scale/annealed noise (Song & Ermon 2019), still
continuous and still MCMC-free for training.

### Q6 — In the log-likelihood gradient (Eq. 14.62), are we sampling the *model*? Is `p0` constant?

**Yes to both, and here is exactly what each piece is.** The MLE gradient is

```
∇_θ log p_θ(x) = −∇_θ E_θ(x)  −  ∇_θ log Z_θ ,
        with     ∇_θ log Z_θ =  E_{x'~p_θ}[ −∇_θ E_θ(x') ]   (the negative phase)
```

so the gradient is **positive phase (data)** − **negative phase (model)**, and
the negative phase is an expectation under the model `p_θ` — you must **sample
the very model you are training** (the training-time chicken-and-egg). The
"`p0` is constant" remark: in CD the chain is initialized at `p0 = p_data`; within
one gradient step the positive/data term is a fixed reference, and what moves is
the negative-phase sample drifting from `p0` toward `p_θ` as the chain runs. E2
makes this literal — we compute the negative phase exactly by enumerating the
model, and CD approximates it with a short chain from `p0`.

---

## Contrastive Divergence & scaling

### Q7 — Why is short-run CD "bad" — not enough data-manifold coverage, or too noisy (1 step)? Bias or variance?

**Dominantly BIAS** (E2, cross-checked by Codex). At `K=1` the CD gradient has
**47% relative bias**, sits **23° off** the true gradient (cosine 0.92), and that
bias is **96% of its MSE**. As `k` grows the bias falls monotonically
(`0.47 → 0.062 → 0.003` for `k=1,5,50`) while variance stays roughly flat, so
beyond `k≈10` CD becomes variance-limited. Mechanism: a `k`-step chain has not
reached `p_θ`, so the negative samples come from a distribution *between* data
and model — the negative phase is **systematically wrong**, not merely noisy.
This is Carreira-Perpiñán & Hinton (2005)'s CD-bias result, measured directly.
Two footnotes the note anticipates: (i) both "insufficient coverage" and "too
noisy" are partially true, but bias dominates at `k=1`; (ii) CD bias *vanishes at
the optimum* (a chain started at data is already at equilibrium once the model
matches the data) — which is why CD still trains usable models despite the bias.

### Q8 — Dynamic weighting `α^(T)` over the chain (early=negative, late=positive) + Zipf alignment — can "bad" samples become useful negatives?

**Yes, and it beats vanilla CD by ~2× on gradient MSE — with a correction to the
note's direction.** (E4, cross-checked by Codex.) We weighted the whole
`K=20`-step trajectory's negative phase, `Σ_t w_t·stats(v^{(t)})`, at equal
compute:

| schedule | rel. bias | variance | gradient MSE |
| --- | ---: | ---: | ---: |
| `zipf_late` `w_t∝1/(K−t+1)` | 0.019 | 0.059 | **0.062** |
| `geom_late` `w_t∝ρ^{K−t}` | 0.004 | 0.067 | **0.065** |
| `uniform` | 0.056 | 0.051 | 0.093 |
| `last` (vanilla CD-K) | 0.004 | 0.123 | 0.123 |
| `early` | 0.23 | 0.063 | 0.79 |

The mechanism: vanilla CD-K (`last`) has very low bias but the **highest
variance** because it discards `K−1` samples. **Late-weighted averaging keeps the
low bias (late samples ≈ the model = good negatives) while halving the variance**
→ ~2× lower MSE. `early` is the worst (early samples ≈ data, so they make biased
negatives). End-to-end weighted-CD training evaluated by exact NLL is essentially
tied among the uniform and late-weighted schedules, with last-only slightly
worse; it does **not** cleanly preserve the gradient-MSE ranking. The robust,
cross-validated effect is the gradient-MSE reduction.

**Correction to the note's wording.** With **data-initialized** chains the roles
are the *reverse* of "early=negative, late=positive": early samples are near the
**data** (poor negatives), late samples are near the **model** (good negatives).
The note's "further `T` → higher weight" instinct is right *for bias*; combining
several late samples is what buys the variance win. The "Zipf alignment" reads
naturally as a heavy-tailed weight over steps-from-the-end — that is exactly
`zipf_late`, the MSE-best schedule. (If chains were **noise**-initialized, late =
converged = the model's best draws, a different regime where "late=positive" is
closer to right.) So the proposal is validated, with the schedule that works
being *late/Zipf-weighted*, not early-weighted.

### Q9 — Combine the weighting with parallelism so Langevin/"MD" EBM training scales?

**Yes — they are complementary and compose into the practical scaling recipe.**
(E3 + E4.) Once CD's bias is controlled (`k≳10`, E2), the negative-phase
**variance** is what limits the gradient, and there are two cheap, orthogonal
ways to cut it:
- **across chains** — many parallel/persistent chains (E3), nearly free on
  vectorized hardware (10,000× throughput at flat ms/step) and giving ~`1/√M`
  variance plus better coverage;
- **along each chain** — late/Zipf trajectory weighting (E4), ~2× variance
  reduction for free because those intermediate samples were already computed.

Stacking them is the modern deep-EBM recipe — persistent replay buffer + many
parallel Langevin chains (Du & Mordatch 2019) — with the late/Zipf weight as a
zero-cost add-on. The one thing neither fixes is between-mode mixing (E3 floor);
for that you escalate to tempering/annealing or score-based sampling (E1).

*Scope note:* the two variance-reduction effects are each measured directly (E3
across-chain, E4 along-chain), but their **composition** in one combined training
run is argued, not measured here — a single script that stacks parallel/persistent
chains *and* late/Zipf weighting is the natural follow-up.

---

## One-paragraph synthesis

`Z` is exponentially hard but only matters for normalized likelihood; the energy
landscape, sampling, and training never need it (E5, E1). MCMC is the default for
unnormalized EBMs, not a necessity: score matching trains them MCMC-free, and
going continuous + Langevin is the clean "pure EBM" recipe (E1, E2-Q6). Short-run
CD is bad because it is **biased**, not just noisy (E2), and the bias vanishes at
the optimum. The note's two scaling ideas both check out: trajectory weighting
with a **late/Zipf** schedule beats vanilla CD-K by ~2× gradient MSE (E4), and it
composes with **parallel/persistent** chains, which are nearly free and the
actual way EBMs scale (E3) — the residual hard problem being multimodal mixing,
which is exactly where score-based and tempered methods take over.

## Selected references

- Hinton 2002 — Contrastive Divergence. Carreira-Perpiñán & Hinton 2005 — CD bias.
- Tieleman 2008 — Persistent CD. Du & Mordatch 2019 — IGEBM (persistent + parallel Langevin).
- Hyvärinen 2005 — score matching; Vincent 2011 — denoising SM; Song et al. 2019 — sliced SM / NCSN; Song et al. 2021 — score SDE / probability-flow ODE.
- Welling & Teh 2011 — SGLD. Neal 2001 — AIS; Salakhutdinov & Murray 2008 — RBM AIS.
- Gutmann & Hyvärinen 2010 — noise-contrastive estimation. Nijkamp et al. 2019/2020 — short-run / non-convergent MCMC EBMs.
