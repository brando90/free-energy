# Codex QA Report

Overall verdict: PASS, with prose corrections applied; no experiment-code math bug found.

Common RBM: PASS. `free_energy = -(b·v) - sum_j softplus(c_j + (vW)_j)` matches `p(v,h)∝exp(vWh+bv+ch)`. `grad_stats` are `d(-F)/dtheta`; `exact_model_stats`, `log_Z`, and Gibbs conditionals are correct. Finite-difference check of exact mean log-likelihood gradient: max abs diff `1.1e-9`.

E1 `e1_langevin_vs_sm.py`: PASS. Langevin/SGLD sign is correct; DSM uses `score=-grad E` and target `(x-xtilde)/sigma^2`. MMD/precision-recall are reasonable sample metrics. Claim supported as “DSM is similar/within 2x of PCD and uses no MCMC in training,” not as DSM strictly matching or beating PCD.

E2 `e2_cd_bias.py`: PASS. Exact-gradient-vs-CD-k bias/variance/MSE decomposition is sound; `eval_rbm` seed 1 on data from `data_rbm` seed 0 correctly avoids the optimum where CD bias vanishes. Data support bias-dominated CD-1 and monotone bias decrease: `0.4679 -> 0.0030`.

E3 `e3_parallel_chains.py`: PASS. GMM score, Langevin sign, IAT, mode-weight TV, and MPS synchronization are correct for the stated experiment. The writeup is honest about the short-chain initialization-bias floor: TV improves to `M=256` then plateaus/slightly worsens at `M=1024`.

E4 `e4_cd_weighting.py`: PASS. Trajectory-weighted negative phase is correctly `sum_t w_t stats(v_t)` after each Gibbs step; all schedules are normalized and match their descriptions. Data support late/Zipf gradient MSE improvement: `zipf_late=0.0640` vs `last=0.1233` (~1.9x lower). Part B is weighted-CD training with exact-NLL evaluation, not exact-gradient/NLL training.

E4 prose fixes applied: `ANSWERS.md` no longer says last-only has the lowest bias or that exact-NLL training preserves the gradient-MSE ranking. README wording changed from “exact-NLL training” to “weighted-CD training ... exact-NLL evaluation.”

E5 `e5_partition_ais.py`: PASS. AIS uses the visible tempered marginal `f_beta(v)=exp(-beta F(v))`, `logZ0=nv*log 2`, and symmetric bit-flip MH transitions invariant to `exp(-beta F)`. Estimates track exact `log_Z=30.531`; mean error is not strictly monotone at 1000 levels (`0.0455` vs `0.0251` at 200), so “converges” should be read statistically, not row-by-row monotone.

Cross-cutting: no sign errors, off-by-one Gibbs-step issue, or biased exact enumerators found. Q9 composition of weighting + parallelism is plausible but not directly tested in one combined script.

Verification: `arch -arm64 ../../../.venv/bin/python src/verify_claims.py` exits 0 (path updated after the experiment was folded into `questions/02/expt_v2/`). Direct venv invocation from this shell is x86_64 and cannot import the arm64 NumPy wheel.
