# expt_v2 — MCMC, Contrastive Divergence & the Partition Function (full 9-question suite)

Nine questions, transcribed from three handwritten textbook pages on Monte Carlo
methods / Markov chains / Langevin sampling / Contrastive Divergence, turned into
five runnable experiments and answered against ground truth.

> **Context.** This is `expt_v2` of the `02_ebm_dynamic_cd_langevin` packet — the
> rigorous, full-suite companion to [`expt_v1`](../expt_v1/) (the continuous
> 8-Gaussians prototype, where dynamic-weighted CD improved mean nearest-mode
> distance `0.90 → 0.78` at higher per-step cost). The two agree and complement:
> **expt_v1 shows *that*** dynamic weighting helps end-task sample quality on a
> continuous task; **expt_v2 measures *why*** — against an exact-`Z` / exact-MLE-
> gradient RBM testbed (E4 here: late/Zipf weighting → ~2× lower gradient MSE at
> *equal* compute) — and answers all 9 of the packet's questions (E1–E5). See the
> [packet README](../README.md) and [`research_brief.md`](../research_brief.md).

The through-line of the notes: *is MCMC (and the partition function `Z`) actually
fundamental for training EBMs, or can we go continuous / messy / parallel and
still win?* See `TRANSCRIPTION.md` for the notes, `PLAN.md` for the mapping,
**`ANSWERS.md` for the prose answers to all 9 questions**, and
`results/RESULTS.md` for the auto-generated number tables.

## Strategy

Use **tractable testbeds** so claims are measured, not asserted:
- a tiny **RBM** (`nv=14`) whose partition function and **exact** maximum-
  likelihood gradient are computable by enumerating `2^{nv}` states — ground
  truth for CD bias (E2), trajectory weighting (E4), and `Z` cost (E5);
- a continuous **MLP energy** on a 2-D target for the "pure EBM" recipe (E1) and
  parallel-chain mixing (E3).

## Files

`src/`:
- `common.py` — shared: 2-D toy datasets, `MLPEnergy` + SGLD/Langevin sampler,
  the numpy `RBM` (exact `log Z`, exact gradient, Gibbs), MMD, autocorrelation.
- `e1_langevin_vs_sm.py` — continuous EBM trained two ways: PCD-Langevin and
  **MCMC-free** denoising score matching; `Z` never computed. → RQ1, RQ2, Q3, Q5
- `e2_cd_bias.py` — exact MLE gradient vs CD-`k` / PCD: bias / variance / MSE,
  measured against enumeration. → Q6, Q7
- `e3_parallel_chains.py` — one long Langevin chain (stuck) vs many parallel /
  persistent chains; ESS, mode-weight error, batched throughput. → Q4, Q9
- `e4_cd_weighting.py` — the note's dynamic trajectory-weighting idea (`last`,
  `uniform`, `geom_late`, `zipf_late`, `early`) on the exact-grad RBM, plus an
  end-to-end weighted-CD training comparison with exact-NLL evaluation. → Q8, Q9
- `e5_partition_ais.py` — exact `log Z` enumeration cost (exponential) + AIS
  estimator that tracks it. → RQ1, Q6
- `make_report.py` — rolls `results/*.json` into `results/RESULTS.md`.
- `codex_crosscheck_spec.md` — spec for the independent second implementation.

`results/`:
- `e{1..5}_*.json` + `e{1..5}_*.png` — per-experiment numbers and plots.
- `RESULTS.md` — merged tables. `verdict.md` — one-line answer per question.
- `crosscheck_codex/` — Codex's independent from-scratch E2/E4 implementation
  (`codex_e2e4.py/json`, `codex_verdict.md`); agrees with `src/` to ~2–3 s.f.

## Quick start (repo `.venv`, arm64 + MPS)

```bash
cd questions/02_ebm_dynamic_cd_langevin/expt_v2
P=../../../.venv/bin/python
$P src/e1_langevin_vs_sm.py   # ~2 min on MPS
$P src/e2_cd_bias.py          # ~4 s
$P src/e3_parallel_chains.py  # ~15 s
$P src/e4_cd_weighting.py     # ~14 s
$P src/e5_partition_ais.py    # ~1 s
$P src/make_report.py
```

## Headline findings

| # | question | answer (measured) |
| --- | --- | --- |
| RQ1 | why fight `Z`? | `Z` is exp-hard (E5: ×2/bit, nv=50 ≈ 10⁸ s) but needed **only** for normalized likelihood — sampling/training never touch it. |
| RQ2 | is MCMC fundamental? | No. Score matching trains a valid EBM **MCMC-free** (E1); autoregressive/ODE samplers bypass it too. |
| Q3 | drop PMF/PDF? | Drop discreteness + explicit normalization; keep normalizability. |
| Q4 | parallel chains help? | Yes: one chain is stuck (E3: τ≈100, 1% effective); parallel cuts error ~4× and is ~free on MPS (10⁴× throughput, flat ms/step). |
| Q5 | go continuous? | Yes — the key move; `∇_x E` + Langevin needs no `Z` and dissolves the chicken-and-egg (E1). |
| Q6 | sampling the model? `p0` const? | Yes; negative phase = `E_{p_θ}[−∇_θE]`; data/positive term is the constant reference (E2). |
| Q7 | why short-run CD bad? | **Bias**, not noise: CD-1 has 47% bias = 96% of MSE; falls monotonically with `k` (E2). |
| Q8 | dynamic CD weighting? | Yes: **late/Zipf** trajectory weighting beats vanilla CD-K by **~2× gradient MSE** (E4). |
| Q9 | weighting + parallel scaling? | Complementary variance reduction (along-chain × across-chain); the modern persistent+parallel recipe (E3+E4). |

## Status

| step | done |
| --- | --- |
| Notes transcribed (3 pages) | ✅ `TRANSCRIPTION.md` |
| 9 questions → 5 experiments | ✅ `PLAN.md` |
| E1 continuous EBM (PCD-Langevin + MCMC-free DSM) | ✅ |
| E2 CD bias vs exact gradient | ✅ bias-dominated, monotone in `k` |
| E3 parallel/persistent chains + throughput | ✅ |
| E4 trajectory weighting (+ exact-NLL evaluation) | ✅ late/Zipf ≈ 2× MSE win |
| E5 partition-function cost + AIS | ✅ |
| Independent cross-check (Codex, from scratch) | ✅ agree to ~2–3 s.f. |
| Prose answers to all 9 questions | ✅ `ANSWERS.md` |
