# Codex cross-check spec — independent re-implementation of E2/E4

You (Codex) are the **independent second implementation** for a numerical claim,
mirroring the two-implementation cross-check culture of
`experiments/02_fisher_div_grad_cost`. Implement this **from scratch in your own
style** — do NOT read or import the sibling `e2_cd_bias.py` / `e4_cd_weighting.py`
(they are the implementation you are cross-checking). Use only `numpy` (no torch
needed). Write outputs to the scratch dir given in the dispatch prompt.

## Model (Restricted Boltzmann Machine, all variables binary {0,1})

- visible `v ∈ {0,1}^{nv}`, hidden `h ∈ {0,1}^{nh}`, params `W (nv×nh)`, `b (nv)`, `c (nh)`.
- Use the **score = negative energy** convention:
  `score(v,h) = vᵀ W h + bᵀ v + cᵀ h`, `p(v,h) ∝ exp(score)`.
- Free energy `F(v) = −log Σ_h exp(score(v,h)) = −bᵀv − Σ_j softplus(c_j + (vᵀW)_j)`.
- `p(v) ∝ exp(−F(v))`. With `nv ≤ 16` you can enumerate all `2^{nv}` visible
  states to get exact `p(v)` and exact model expectations.

## IMPORTANT: evaluate at a model that DIFFERS from the data distribution

CD bias **vanishes at the data-generating params** (the chain starts at data and
data is already the stationary distribution, so any `k` is unbiased). To see the
bias→0-as-`k`-grows trend you must measure the gradient of an `eval` RBM that is
**different** from the `data` RBM. So: one RBM generates the data; a second,
independent RBM is the point at which we measure the gradient and run the Gibbs
chains (this mimics being mid-training, model ≠ data).

## Exact MLE gradient (ground truth)

For log-likelihood of the data under the **eval** model, the gradient w.r.t.
each parameter is `E_data[∂(−F)/∂θ] − E_{p_eval(v)}[∂(−F)/∂θ]` (positive −
negative phase). All `F`, stats, and Gibbs steps below use the **eval** params. For an RBM
the per-sample sufficient statistics use `p(h_j=1|v) = sigmoid(c_j + (vᵀW)_j)`:

- `∂(−F)/∂W_ij = v_i · sigmoid(c_j + (vᵀW)_j)`
- `∂(−F)/∂b_i  = v_i`
- `∂(−F)/∂c_j  = sigmoid(c_j + (vᵀW)_j)`

`g_exact` uses the **exact** negative phase `E_{p_θ(v)}[·]` computed by
enumerating all `2^{nv}` states weighted by `p(v)`.

## CD-k and PCD estimators

- Block Gibbs step: `h ~ Bernoulli(sigmoid(c + vᵀW))`, then
  `v ~ Bernoulli(sigmoid(b + W h))`.
- **CD-k:** start the chain at a data minibatch, run `k` Gibbs steps, use the
  final `v` for the negative phase. **PCD:** keep a persistent set of chains
  across minibatches, 1 Gibbs step each call.
- Estimate `E[ĝ]` by averaging the estimator over many minibatch + RNG seeds.

## E2 metrics (sweep k ∈ {1,2,5,10,20,50}, plus PCD)

For each estimator report, on the flattened concatenation of (W,b,c):
- `bias_rel = ‖E[ĝ] − g_exact‖ / ‖g_exact‖`
- `cosine   = cos(E[ĝ], g_exact)`
- `var      = tr Cov(ĝ)` (sum of per-component variances across seeds)
- `mse      = ‖E[ĝ] − g_exact‖² + var`

## E4 weighted-trajectory negative phase (equal compute K)

Record the chain states `v^(1..K)` and form the negative phase as
`Σ_{t=1}^{K} w_t · stats(v^(t))`, `Σ_t w_t = 1`. Schedules (K=20):
- `last`  : `w = [0,…,0,1]` (vanilla CD-K)
- `uniform`: `w_t = 1/K`
- `geom_late`: `w_t ∝ ρ^{K−t}` with `ρ=0.7` (emphasize late)
- `zipf_late`: `w_t ∝ 1/(K−t+1)`
- `early` : `w_t ∝ ρ^{t}` (emphasize early; expected worst)
Report bias_rel / var / mse for each schedule vs `g_exact`.

## Config (use exactly so the two runs are comparable)

`nv=14, nh=6, n_data=4000, n_seeds=800, batch=128`. Build TWO RBMs with weight
scale **`w_scale=1.5`** (sharper weights => slower Gibbs mixing => short-run CD
is meaningfully biased, so the trend is visible above the MC noise floor):

- `data_rbm` from `numpy.random.default_rng(0)`: `W=1.5*randn(nv,nh)`,
  `b=0.2*randn(nv)`, `c=0.2*randn(nh)`. Draw the dataset by sampling its exact
  `p(v)` (enumerate all `2^nv`).
- `eval_rbm` from `numpy.random.default_rng(1)` with the SAME scales
  (independent draw; `W=1.5*randn`, biases `0.2*randn`). **All gradients, the exact negative phase, the chains,
  and `g_exact` use `eval_rbm`.** Chains are still initialized from data
  minibatches. No training — this is a fixed-point bias measurement at a model
  that differs from the data (so `g_exact` is solidly nonzero).

## Output (write both, into the in-repo crosscheck dir given in the dispatch)

1. `codex_e2e4.json` — `{"e2": {k or "pcd": {bias_rel,cosine,var,mse}}, "e4": {schedule: {bias_rel,var,mse}}, "meta": {...}}`
2. `codex_verdict.md` — 8–15 lines: do you confirm (a) bias_rel decreases
   monotonically in `k` and cosine→1; (b) PCD bias < CD-1 bias; (c) among E4
   schedules, `last`/`geom_late`/`zipf_late` beat `uniform`/`early` on MSE, and
   which single schedule minimizes MSE. State agreement or disagreement plainly.
