# Verdict — one line per question

Measured on tractable testbeds (exact `Z` / exact MLE gradient by enumeration);
E2/E4 independently cross-checked by Codex (`crosscheck_codex/`). Numbers in
`RESULTS.md`, reasoning in `../ANSWERS.md`.

| # | question | verdict |
| --- | --- | --- |
| RQ1 | why deal with `Z` at all? | `Z` is provably exponential (E5, ×2/bit) yet needed **only** for normalized likelihood; the energy landscape, sampling, and training are `Z`-invariant. "We don't see `Z`" = it cancels for *use*, not for *evaluation* (AIS bridges that). |
| RQ2 | is MCMC fundamental? | **No.** Score matching trains a valid EBM with **no MCMC and no `Z`** (E1); autoregressive (GPT-style) and probability-flow-ODE samplers bypass it as well. MCMC is the default for *unnormalized* energies, not a law. |
| Q3 | discard the PMF/PDF restriction? | Partly: discard discreteness (E1 is continuous) and explicit normalization (ranking/optimization use relative energy); **keep** normalizability `∫e^{−E}<∞` — required the moment you want a probability. |
| Q4 | parallel / semi-sequential chains? | **Yes**, the right lever: one chain is stuck (E3: τ≈103, 0.97% effective, 1/3 modes); parallel cuts error ~4× and is ~free on MPS (10⁴× throughput, flat ms/step); persistence (PCD) amortizes burn-in. Caveat: short parallel chains still don't cross barriers. |
| Q5 | go continuous to dissolve chicken-and-egg? | **Yes — the key move.** Differentiable energy ⇒ Langevin via `∇_x E` (no `Z`, no `p(x|h)`/`p(h|x)`); score matching removes model-sampling from training (E1). Caveat: well-separated modes still need multi-scale/annealed methods. |
| Q6 | sampling the model? is `p0` constant? | **Yes** and **yes**: `∇_θ log Z = E_{p_θ}[−∇_θE]` (negative phase) is an expectation under the *model*; the data/positive phase is the fixed reference, and CD inits the chain at `p0=p_data` (E2). |
| Q7 | why is short-run CD bad — bias or noise? | **Bias.** CD-1: 47% relative bias, 23° off, **96% of MSE is bias**; falls monotonically with `k` to a variance floor (E2). Both note-intuitions partly true; bias dominates at `k=1`. CD bias vanishes at the optimum. |
| Q8 | dynamic CD weighting + Zipf? | **Yes — and it works:** late/Zipf-weighted trajectory averaging keeps CD-K's low bias but halves variance ⇒ **~2× lower gradient MSE** than vanilla last-only CD (E4). Direction fix: with data-init chains, *late*=good negatives, *early*=poor (reverse of the note's wording). |
| Q9 | weighting + parallel ⇒ scaling? | **Yes, complementary.** Along-chain (weighting, E4) × across-chain (parallel/persistent, E3) variance reduction; the modern persistent-replay + parallel-Langevin recipe, with late/Zipf weighting as a near-free add-on. Each effect is measured separately; their joint composition is argued, not tested in one combined script. Residual hard part: multimodal mixing. |
