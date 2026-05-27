# 05 — Fisher-Divergence Gradient Cost (Score-Matching Profiling)

Is computing `∇_θ` of the Hyvärinen score-matching loss really expensive?

The literature (Hyvärinen 2005; Song & Kingma 2021) repeatedly says "yes, the
trace of the Hessian wrt `x` is the bottleneck." Brando's intuition is that
`∇_x` is over **input dim** (small) rather than parameter dim (huge), and
modern autodiff handles HVPs cheaply — so the cost should be modest, and
Hutchinson should nearly close the gap with denoising score matching.

This experiment measures that, in PyTorch and JAX, side by side.

## Files

This experiment was run twice in parallel by two Claude instances (a deliberate
cross-validation requested by Brando). Both sets of files are kept and the
cross-check confirms they agree to numerical precision.

- `PLAN.md` — full experiment plan.
- `TRANSCRIPTION.md` — transcription of the 6 handwritten note pages.
- `notes/transcribed_notes.md` — parallel transcription.
- `assets/` — raw note photos (`note_01_…` through `note_06_…`).

`src/`:
- `ebm_models.py` — `TinyEBM` (shared by `profile_sm_*.py` + `train_sm_toy.py`).
- `profile_pytorch.py` / `profile_jax.py` — PyTorch + JAX sweeps over
  (d × hidden × batch × loss), with `sm_hutch1` / `sm_hutch4` / `sm_exact`.
- `profile_sm_pytorch.py` / `profile_sm_jax.py` — parallel sweeps with
  `exact_sm` / `hutch_sm` / `sliced_sm` / `dsm` / `mle_like`.
- `cross_check_backends.py` — numerical agreement check across backends
  (✅ exact_sm, dsm, mle_like, sm_hutch1 all match to ≤ 1e-6).
- `make_summary.py` — generate `results/RESULTS.md` + `results/wallclock_vs_dim.png`.
- `train_sm_toy.py` — end-to-end SM training on a 2-mixture Gaussian target
  to validate that the gradient actually trains an EBM.

`results/`:
- `profile_pytorch.csv` / `profile_jax.csv` / `*.log` — raw timings from `profile_sm_*.py`.
- `pt_full.json` / `pt_full.log` / `jax_full.json` / `jax_full.log` — full
  sweep over (D × hidden × batch) from `profile_pytorch.py` / `profile_jax.py`.
- `pt_smoke.json` / `jax_smoke.json` — smaller MPS smoke runs.
- `SUMMARY.md` — merged 422-row summary across both backends, written by
  `summarize_results.py`.
- `RESULTS.md` + `wallclock_vs_dim.png` — compact summary table + plot from
  `make_summary.py`.
- `train_sm_toy*.json` — sanity-check training curves.

## Losses compared

For an EBM `p_θ(x) = e^{-E_θ(x)} / Z_θ` with `E_θ` an MLP `R^d → R`:

| name | formula | x-derivative order |
| --- | --- | --- |
| `mle_like`  | `E[E_θ(x)]`                                                        | none |
| `dsm`       | Vincent denoising SM (Gaussian noise)                              | 1st |
| `sm_hutch1` | exact `½‖∇_x E‖²` + Hutchinson `tr(∇²_x E)` with 1 Rademacher probe | 2nd (1 HVP) |
| `sm_hutch4` | same, 4 probes (variance reduction)                                | 2nd (4 HVPs) |
| `sm_exact`  | exact `½‖∇_x E‖²` + `tr(∇²_x E)` via `d` backward passes           | 2nd (d HVPs / dense H) |

All five reduce to the same quantity in expectation for `dsm`/`sm_*`, modulo
constants and noise variance — but their compute footprints differ
dramatically. See `TRANSCRIPTION.md` for the underlying math.

## Quick start (Apple Silicon)

```bash
# arm64 python with torch+jax — the repo .venv is x86_64, so we use uv
uv venv --python 3.13 /tmp/fisher_bench_env/.venv-fisher
source /tmp/fisher_bench_env/.venv-fisher/bin/activate
uv pip install torch jax numpy

cd experiments/05_fisher_div_grad_cost

# 1. sanity check: PyTorch and JAX agree on all four losses
python src/cross_check_backends.py

# 2. quick smoke sweeps
python src/profile_pytorch.py --quick --tag pt_smoke
python src/profile_jax.py     --quick --tag jax_smoke

# 3. full sweeps
python src/profile_pytorch.py --tag pt_full --n-warmup 5 --n-runs 10
python src/profile_jax.py     --tag jax_full --n-warmup 5 --n-runs 10

# 4. roll up
python src/summarize_results.py
```

`SUMMARY.md` will appear at `results/SUMMARY.md`.

## Sweep grid

- `in_dim d ∈ {2, 8, 32, 128, 512, 2048}`
- `hidden h ∈ {64, 256, 1024}`
- `batch B ∈ {32, 128}`
- `sm_exact` is capped at `d ≤ 512` to keep total runtime bounded
  (cost scales linearly with `d`).

## Cross-backend agreement

`cross_check_backends.py` builds one MLP-EBM from a numpy RNG, copies the
exact same parameters into PyTorch and JAX, and confirms `mle_like`, `dsm`,
`sm_hutch1`, and `sm_exact` all agree to `≤ rtol·max(|a|,|b|) + atol`
(defaults: `rtol=1e-3, atol=1e-4`). This guards against subtle issues like
the trace-sign flip from `log p = -E - log Z` (the `tr(H)` term enters with
a minus sign because we are taking second derivatives of `−E`).

## Status

| step | done |
| --- | --- |
| Notes transcribed (2 independent passes) | ✅ |
| PyTorch sweep (2 independent impls) | ✅ |
| JAX sweep (2 independent impls) | ✅ |
| Cross-backend numerical check (PT ↔ JAX) | ✅ ≤ 1e-6 abs diff on all 4 losses |
| Wall-clock summary table + plot | ✅ (`results/RESULTS.md`, `wallclock_vs_dim.png`) |
| End-to-end SM training sanity check (GMM target) | ✅ score-err 2.7 → 0.5 |
| Apply to discrete-toy EBM (exp 01) | N/A — exp 01 is discrete; SM needs continuous data |
| Apply to lean EBM | follow-up — lean EBM still under development |

## Headline numbers

`exact_sm / hutch_sm` slowdown vs input dim D, batch=64, CPU:

| backend | D=2 | D=8 | D=64 | D=512 | D=2048 |
| --- | --- | --- | --- | --- | --- |
| pytorch | 1.4× | 4.3× | 31.2× | 188× | **729×** |
| jax | 1.2× | 1.2× | 6.1× | 37× | **419×** |

The naive "trace of Hessian via D backward passes" is exactly as bad as the
literature warns — and **Hutchinson eliminates that scaling completely**.

## Connection to prior work

In Brando's *beyond-scale-language-data-diversity* paper, the trace of the
Hessian wrt **θ** was approximated using the elementwise square of stochastic
parameter gradients (a Hutchinson-style estimator with variance reduction);
that worked at LLM scale. Here we are looking at `tr(∇²_x ·)`, which lives in
input dim — even more favorable since `d ≪ |θ|`. The numbers in
`results/SUMMARY.md` make this concrete.
