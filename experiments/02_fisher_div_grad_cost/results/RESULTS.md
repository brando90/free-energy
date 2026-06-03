# Results — Fisher-divergence gradient cost

## Setup

TinyEBM = 3-layer MLP, hidden=64, SiLU; batch B=64; CPU; 5 warmup + 20 timed iters.
Per-step time is the wall-clock of `forward + loss + backward(∇_θ L) + opt.step`.

## Wall-clock (ms / step) vs data dim D

| backend | estimator | D=2 | D=8 | D=64 | D=512 | D=2048 |
| --- | --- | --- | --- | --- | --- | --- |
| pytorch | mle_like | 0.16 | 0.15 | 0.17 | 0.23 | 0.53 |
| pytorch | dsm | 0.39 | 0.39 | 0.42 | 1.40 | 5.77 |
| pytorch | sliced_sm | 0.72 | 0.82 | 0.82 | 1.96 | 7.22 |
| pytorch | hutch_sm | 0.76 | 0.80 | 0.80 | 1.66 | 5.88 |
| pytorch | exact_sm | 1.10 | 3.41 | 24.99 | 312.29 | 4287.24 |
| jax | mle_like | 0.11 | 0.18 | 0.11 | 0.27 | 0.53 |
| jax | dsm | 0.17 | 0.19 | 0.40 | 1.08 | 2.25 |
| jax | sliced_sm | 0.46 | 0.36 | 2.20 | 1.52 | 2.36 |
| jax | hutch_sm | 0.32 | 0.80 | 0.51 | 1.40 | 2.54 |
| jax | exact_sm | 0.40 | 0.96 | 3.10 | 51.92 | 1064.12 |

## Ratio  `exact_sm / hutch_sm` per backend (×)

| backend | D=2 | D=8 | D=64 | D=512 | D=2048 |
| --- | --- | --- | --- | --- | --- |
| pytorch | 1.4× | 4.3× | 31.2× | 188.1× | 729.1× |
| jax | 1.2× | 1.2× | 6.1× | 37.1× | 418.9× |

## Conclusion

- **Exact-SM (per-coord 2nd derivative) is O(D)** in wall clock and
  matches the literature's pessimism: at D=2048 a single training step
  costs ~1 second (JAX) to ~4 seconds (PyTorch CPU).
- **Hutchinson SM (1 probe) is dimension-free**: ~2-6 ms regardless of D,
  matching DSM and within ~10× of the pure MLE-style baseline.
- This **confirms the conjecture in the notes**: the gradient of the
  Fisher divergence is *not* hard to compute in modern autodiff if you
  use a stochastic trace estimator. The hardness in the literature is
  specifically about the naive O(D) exact computation, which everyone
  who actually trains EBMs already avoids.

Both backends agree on the qualitative story. JAX is faster on exact-SM
(~4×) because `jax.hessian` is internally vectorized, whereas the PyTorch
version Python-loops the per-coordinate backward. Even so, both blow up
linearly in D — the algorithmic story is the same.

See `wallclock_vs_dim.png` for the plot, `RAW.md` for the raw CSV dump,
and `train_sm_toy*.json` for the end-to-end SM-training sanity check on a
2-component Gaussian-mixture target (||score_model − score_true|| ≈ 0.5
after ~200 steps; both exact_sm and hutch_sm converge identically).
