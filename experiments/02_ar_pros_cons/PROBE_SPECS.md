# Probe specifications

Each probe states a **prediction**, a **null hypothesis**, the **measurement**, a
**positive control**, and the **real (VeriBench) test**. A probe is only trusted
once its control reproduces the predicted effect on known ground truth.

Conventions: `d` = hidden dim, `V` = vocab, `N` = sequence length, `L` = depth,
`f_θ(x)` = model logit/score, `e` = per-step error rate.

---

## Probe 01 — Softmax bottleneck
**Claim 1 · Layer: architecture + data · No training.**

- **Prediction:** a single softmax head over a `d`-dim state cannot fit a target
  log-prob matrix of rank `r > d+1`; fit error is lower-bounded by the truncated-SVD
  tail (energy beyond the top `d+1` singular values). A mixture-of-softmaxes closes
  the gap.
- **Null:** single softmax fits the rank-`r` target to within tolerance for `d < r`.
- **Measurement (toy / control):** construct a target log-prob matrix `T` with
  *known* rank `r`. Fit (i) single softmax with hidden dim `d ∈ {r/2, r, 2r}`,
  (ii) mixture-of-`K`-softmaxes. Plot achieved KL vs `d`; overlay the SVD-tail lower
  bound. The single-softmax curve must hug the bound for `d < r`.
- **Real test:** assemble VeriBench next-token log-prob matrices (contexts × `V`)
  from a small open model; estimate effective rank (SVD, 99% energy). Plot data rank
  vs the head's `d`. **Bites iff data rank > d.**
- **Verdict logic:** CONFIRMED if control hugs the bound *and* real data rank > d;
  PARTIAL if data rank ≈ d; NOT-SUPPORTED if data rank ≪ d (the bottleneck is slack).

---

## Probe 02 — Mode-covering (forward-KL is zero-avoiding)
**Claim 2 · Layer: objective · Small controlled training run.**

- **Prediction:** minimizing forward KL spreads probability mass into low-density
  regions between modes (covers), where reverse KL concentrates on a mode (seeks).
- **Null:** forward and reverse KL learn indistinguishable densities.
- **Measurement (toy / control):** explicit bimodal 1-D target. Train one model with
  forward KL (MLE) and one with reverse KL, identical capacity. Plot both learned
  densities over the target; measure mass placed in the inter-mode valley. Forward
  KL must place visibly more.
- **Real test (proxy):** on VeriBench, measure probability mass the trained model
  assigns to **verifier-rejected** next tokens (Lean tokens that fail to compile)
  vs accepted ones, as a function of training. Rising mass-on-wrong is the realized
  signature of mode-covering.
- **Verdict logic:** CONFIRMED if control shows valley-filling *and* real model puts
  non-trivial mass on rejected tokens; PARTIAL if only the toy shows it.

---

## Probe 03 — Rank collapse with depth
**Claim 3 · Layer: architecture · No training (random init).**

- **Prediction:** pure stacked self-attention drives the token representation matrix
  toward rank 1 doubly-exponentially in depth `L`. Residual + MLP slow it.
- **Null:** rank is stable across depth in all configs.
- **Measurement (control):** random-init stacked attention, three configs —
  (a) pure attention, (b) + residual, (c) + residual + MLP. Feed random token
  embeddings; measure effective rank of the output matrix vs `L`. Config (a) must
  collapse fast (the positive control).
- **Real test:** measure effective rank of a **trained** model's hidden states
  across its layers. **The integrated question:** does collapse survive training and
  residuals, or is it masked?
- **Verdict logic:** CONFIRMED (pure) always expected; the reportable result is
  whether (c) and the trained model retain meaningful rank — likely **MASKED** in
  practice, which the bridge analysis then quantifies.

---

## Probe 04 — Partition function is a removable per-step tax
**Claim 4 · Layer: architecture (ablation) · Small training run.**

- **Prediction:** replacing softmax attention with sigmoid or linear/kernel
  attention matches task loss within CI — softmax is not necessary.
- **Null:** softmax strictly dominates; non-softmax variants incur higher loss
  beyond CI.
- **Measurement:** fix a small Lean-token task. Train three otherwise-identical
  models: `attn ∈ {softmax, sigmoid, linear}`. Report task loss + throughput,
  **compute-matched and param-matched** (see METHODOLOGY §5).
- **Real test:** retrain the VeriBench head/body without softmax; compare pass rate.
- **Verdict logic:** "removable" CONFIRMED iff non-softmax matches within CI on both
  matchings. If softmax wins only compute-matched but not param-matched (or vice
  versa), report the asymmetry rather than a verdict.

---

## Probe 05 — Fixed compute per token is a representational ceiling
**Claim 5 · Layer: architecture + complexity · No training for the toy.**

- **Prediction:** a fixed-depth transformer's single-pass accuracy collapses past a
  problem-size threshold on tasks requiring serial depth (parity,
  graph-connectivity); a scratchpad/CoT that externalizes serial steps recovers it.
- **Null:** single-pass accuracy is flat in problem size; CoT gives no lift.
- **Measurement (toy / control):** parity and graph-connectivity at increasing input
  length `n`. Plot single-pass accuracy vs `n`, with and without scratchpad. The
  single-pass curve must fall off a cliff; scratchpad must lift it.
- **Real test:** VeriBench proof success vs **proof depth** (number of tactic
  steps), single-pass vs iterative/multi-step. Does deep-proof success require the
  serial token budget?
- **Verdict logic:** CONFIRMED if both toy and VeriBench show a depth threshold that
  CoT/iteration moves outward.

---

## Probe 06 — Error compounding, the (1−e)ⁿ test  ⚑ most likely to falsify
**Claim 6 · Layer: trained behavior + verifier · The key probe.**

- **Prediction (LeCun):** `P(proof compiles)` decays geometrically in proof length,
  `(1−e)ⁿ`, because errors are unrecoverable.
- **Alternative (ours):** with a hard verifier, errors are *recoverable* (backtrack
  / resample), so a **recoverable-Markov** model — a 2-state chain with a nonzero
  recovery probability per step — fits better than geometric, and a constant model
  may even suffice.
- **Measurement:** on VeriBench, generate proofs and record **per-step validity**
  using the Lean verifier. Fit `P(compiles)` vs length to three models:
  1. geometric `(1−e)ⁿ`
  2. constant `p`
  3. recoverable-Markov (states {on-manifold, off-manifold}, transition probs fit)
  Compare by held-out log-likelihood / AIC, with bootstrap CIs.
- **Verdict logic:** if recoverable-Markov wins, the `(1−e)ⁿ` claim is
  **NOT-SUPPORTED in the verifier setting** — a genuine, reportable negative result.
  If geometric wins, LeCun's argument survives here and we say so.
- **PS / open question (BM, 2026-05-27):** the `(1−e)ⁿ → 0` story is true for *any*
  approximate generative model, not just autoregressive ones — so it isn't clear how
  EBMs (or any non-AR sampler) would escape it on their own. The whole bite of the
  argument should come from the **error model** (independent + unrecoverable),
  not from the AR factorization. If that's right, the right contrast is
  AR-without-verifier vs. AR-with-verifier (recovery changes the exponent), not
  AR vs. EBM. Worth designing the recoverable-Markov fit so it makes that contrast
  explicit, and worth checking whether an EBM-style sampler with no verifier shows
  the same `(1−e)ⁿ` decay on the toy.

---

## Probe 07 — Reversal curse
**Claim 7 · Layer: trained behavior.**

- **Prediction:** a model trained on "A is B" does not reliably produce "B is A".
- **Null:** forward and reverse query accuracy are equal.
- **Measurement (control):** synthesize a set of "A is B" facts, train, test "B is
  A" held out. Forward accuracy high, reverse near chance is the predicted
  asymmetry.
- **Real test:** scan VeriBench-FTP for lemmas usable in both directions; measure
  asymmetric success rate.
- **Verdict logic:** CONFIRMED if reverse ≪ forward on both toy and real.

---

## Probe 08 — Brittleness / Lipschitz–margin
**Claim 8 · Layer: trained behavior.**

- **Prediction:** high output confidence does not imply a large input margin; the
  true margin obeys `‖δ‖ ≥ f_θ(x) / L_global`, `L_global ≤ ∏_i ‖W_i‖₂`. Minimal
  token perturbations flip outputs when `∏‖W_i‖₂` is large.
- **Null:** flip rate is independent of `∏‖W_i‖₂`; output margin tracks input margin.
- **Measurement:** measure output margin and `∏_i ‖W_i‖₂`; apply minimal-perturbation
  token edits and record flip rate. Plot empirical input margin vs the bound
  `f_θ(x)/L_global`; the bound must lie below measured margins (it's a lower bound).
- **Real test:** adversarial token edits on VeriBench sequences; correlate flip rate
  with the spectral-norm product across checkpoints.
- **Verdict logic:** CONFIRMED if flip rate rises with `∏‖W_i‖₂` and the bound holds.

---

## Probe — Data wall (context)
**Layer: external · No model claim.**

- **Prediction:** loss follows a power law in data; high-quality data is finite, so
  projected gains flatten as the supply is consumed.
- **Measurement:** fit a scaling curve (loss vs tokens) on VeriBench subsets of
  increasing size; extrapolate to the available-data ceiling. Report the
  exponential-data-for-linear-loss tradeoff explicitly.
- **Verdict logic:** descriptive — report the fitted exponent and the extrapolated
  flattening point; no CONFIRMED/NOT verdict, it's a resource curve.

---

## Summary of expected verdicts (pre-registered)

| Probe | Pre-registered expectation |
|---|---|
| 01 bottleneck | CONFIRMED on toy; real depends on measured data rank vs d |
| 02 mode-covering | CONFIRMED on toy; PARTIAL/realized on VeriBench |
| 03 rank collapse | CONFIRMED pure; **MASKED** in trained model |
| 04 partition removable | likely CONFIRMED (loss matches within CI) |
| 05 fixed compute | CONFIRMED (depth threshold, CoT lifts) |
| 06 error compounding | **NOT-SUPPORTED in verifier setting** (recoverable-Markov wins) |
| 07 reversal curse | CONFIRMED |
| 08 Lipschitz–margin | CONFIRMED (flip rate ∝ spectral-norm product) |
| data wall | descriptive |

Surprises against this table are the most valuable outputs — flag them in
`FINDINGS.md`.
