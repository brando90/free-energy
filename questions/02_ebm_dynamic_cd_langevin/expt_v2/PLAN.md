# Plan — answering the 9 questions with runnable experiments

The 9 questions are partly conceptual, but every one of them reduces to a claim
we can *measure* on a small model. The strategy: use **tractable testbeds**
(where the partition function `Z` and the exact MLE gradient can be computed by
brute force) so we can quote ground-truth error, not vibes.

Five experiments, each emitting JSON + a plot into `results/`. A final
`make_report.py` rolls them into `results/RESULTS.md`, and `ANSWERS.md`
addresses each numbered question in prose, grounded in these numbers + the
literature.

| Exp | File | Testbed | Answers |
| --- | --- | --- | --- |
| E1 | `e1_langevin_vs_sm.py` | 2-D toy density (8-Gaussians / moons / pinwheel), MLP-EBM | RQ1, RQ2, Q3, Q5 |
| E2 | `e2_cd_bias.py` | tiny RBM, `Z` + exact grad by enumeration | Q6, Q7 |
| E3 | `e3_parallel_chains.py` | same RBM + 2-D EBM | Q4, Q9 |
| E4 | `e4_cd_weighting.py` | tiny RBM (exact-grad ground truth) | Q8, Q9 |
| E5 | `e5_partition_ais.py` | tiny RBM, enumerable `Z`; AIS estimator | RQ1, Q6 |

## E1 — Continuous EBM training never touches Z; MCMC is replaceable

Train an MLP energy `E_θ : R² → R` on a 2-D target with two recipes:

- **PCD-Langevin** (the modern EBM recipe, Du & Mordatch 2019): persistent
  SGLD negative chains, gradient `∇_θ(E_θ(x⁺) − E_θ(x⁻))`. Never computes `Z`.
- **Denoising / sliced score matching** (Hyvärinen 2005; Vincent 2011;
  Song et al. 2019): an **MCMC-free** objective; never samples the model and
  never computes `Z`.

Metrics: model-energy heatmap vs data, sample scatter, and a quantitative
fit score (kernel MMD between model samples and data; SM loss). **Claim under
test (RQ2/Q5):** both train a usable continuous EBM with `Z` untouched, and
score matching does it with *no MCMC at all*.

## E2 — Why short-run CD is "bad": measure the bias directly (Q6, Q7)

Tiny RBM (`n_v ≤ 16`, `n_h` small) so the model expectation — and hence the
**exact** maximum-likelihood gradient — is computable by enumerating `2^{n_v}`
states. Compare the exact gradient against CD-`k` (`k ∈ {1,2,5,10,20,50}`) and
PCD, each averaged over many minibatch/chain seeds, decomposing the error into:

- **bias** `‖E[ĝ] − g_exact‖`, and `cos(E[ĝ], g_exact)`;
- **variance** `tr Cov(ĝ)`;
- **MSE** `= bias² + variance`.

Predicted: bias falls monotonically with `k`, cosine → 1; CD-1 is *biased*, not
merely noisy. This isolates Q7's "is it bias or variance?" with real numbers.
Q6 is settled in `ANSWERS.md`: the negative phase samples the **model**; the
data term is the constant-`Z`-free "positive phase" (the `p0`-constant remark is
about the score-function identity `∇_θ log Z = E_{p_θ}[∇_θ(−E)]`).

## E3 — Parallel / persistent chains: better mixing per wall-clock (Q4, Q9)

Negative-phase expectation estimated with (a) one long sequential chain vs
(b) `M` parallel chains of length `L` with `M·L` fixed. Measure effective
sample size (ESS via integrated autocorrelation), gradient-estimate variance,
and wall-clock. Vectorized parallel chains are near-free on GPU/MPS, so we
expect far lower variance per second. Also test **persistent** chains (carry
state across SGD steps) = amortized burn-in = the "semi-sequential" idea.

## E4 — The dynamic CD weighting scheme (Q8, Q9)

The user's idea: weight chain samples `X^(t)` along the trajectory with
`α^(t)` instead of using only the last one. On the E2 RBM (exact-grad ground
truth), form a **trajectory-weighted negative phase**
`Σ_t w_t ⟨∇E(x^(t))⟩`, `Σ_t w_t = 1`, and sweep schedules at **equal compute**
(same `K` Langevin/Gibbs steps):

- `last` (vanilla CD-`K`), `uniform`, `geometric-late`, `zipf-late`
  (`w_t ∝ 1/(K−t+1)`), `early`.

Report bias / variance / MSE of each schedule's gradient vs `g_exact`, plus
end-task NLL of a model trained with each. Hypothesis (sharpened from the note):
*later* samples are the better negatives (closer to the model), so late-weighted
averaging trades a little bias for a real **variance** drop → lower gradient
**MSE** than last-only at equal compute; uniform/early *raise* bias. We also
clarify the note's "early=negative / late=positive" wording against the two
chain-initialization regimes (data-init vs noise-init).

## E5 — How intractable is Z, really? Enumeration blow-up + AIS (RQ1, Q6)

On the RBM: (a) time exact `log Z` by enumeration as `n_v` grows → the `2^{n_v}`
wall hits fast (concrete "intractable"); (b) estimate `log Z` with **Annealed
Importance Sampling** (Neal 2001) and show it tracks the exact value where
enumeration is still feasible, then scales past it. Payoff for RQ1: `Z` is
genuinely (provably) exponential to compute, but it is **only needed to report
normalized likelihood** — never for the energy landscape, for sampling, or for
score-matching training. So "we don't *see* Z" is right for *use*, wrong for
*evaluation*; AIS is the "messy bypass" for evaluation.

## Cross-validation & QA

- **Codex** independently re-implements the E2/E4 bias measurement from a
  spec (`src/codex_crosscheck_spec.md`) in a scratch dir; we compare the
  scientific conclusions (monotone bias↓ in `k`; ordering of weight schedules).
  This mirrors the two-implementation cross-check culture of exp 02.
- A final Codex QA pass reviews the whole folder (correctness + structure).

## Environment

Repo `.venv` is arm64 with `torch 2.12` + MPS. All scripts run CPU/MPS in a few
minutes total. Run from the experiment dir:

```bash
cd questions/02_ebm_dynamic_cd_langevin/expt_v2
../../../.venv/bin/python src/e1_langevin_vs_sm.py
../../../.venv/bin/python src/e2_cd_bias.py
../../../.venv/bin/python src/e3_parallel_chains.py
../../../.venv/bin/python src/e4_cd_weighting.py
../../../.venv/bin/python src/e5_partition_ais.py
../../../.venv/bin/python src/make_report.py
```
