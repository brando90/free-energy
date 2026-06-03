"""JAX twin of profile_sm_pytorch.py.

Same five estimators of d/dtheta of the Fisher / score-matching loss, but
written in JAX so we can compare wall-clock + memory.

Model is a small MLP energy E_theta : R^D -> R, parameters held in a
plain pytree (no Flax/Haiku/Equinox dependency).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


# -----------------------------------------------------------------------------
# Model: a pure-functional MLP energy. params is a list of (W, b) tuples.
# -----------------------------------------------------------------------------


def init_params(key, dim: int, hidden: int = 64, layers: int = 3):
    keys = jax.random.split(key, layers)
    params = []
    last = dim
    for i, k in enumerate(keys):
        out = 1 if i == layers - 1 else hidden
        w_key, b_key = jax.random.split(k)
        W = jax.random.normal(w_key, (last, out)) * jnp.sqrt(1.0 / last)
        b = jnp.zeros((out,))
        params.append((W, b))
        last = out
    return params


def energy(params, x):
    """Scalar energy per sample. x: (D,) -> R."""
    h = x
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h.squeeze(-1)


def energy_batched(params, x):
    """x: (B, D) -> (B,)."""
    return jax.vmap(energy, in_axes=(None, 0))(params, x)


def num_params(params) -> int:
    return int(sum(np.prod(W.shape) + np.prod(b.shape) for W, b in params))


# -----------------------------------------------------------------------------
# Per-sample loss functions. Each takes (params, x_i) where x_i is (D,) and
# returns a scalar; the batched loss = vmap mean.
# -----------------------------------------------------------------------------


def _score(params, x):
    """nabla_x log p_theta(x) = - nabla_x E_theta(x); shape (D,)."""
    return -jax.grad(energy, argnums=1)(params, x)


def loss_exact_sm_sample(params, x):
    """Exact Hyvärinen SM: 0.5 ||score||^2 + tr( d score / d x )."""
    s = _score(params, x)
    # Hessian of E wrt x via jax.hessian — D x D matrix
    H = jax.hessian(energy, argnums=1)(params, x)
    # tr( d score / d x ) = tr( - H )  = - tr(H)
    return 0.5 * jnp.sum(s * s) + (-jnp.trace(H))


def loss_hutch_sm_sample(params, x, key):
    """Hutchinson SM via HVP of E (note: HVP of score = -HVP of E)."""
    s = _score(params, x)
    v = jax.random.rademacher(key, x.shape).astype(x.dtype)
    # JvP of grad(E)_x wrt x in direction v == Hessian @ v
    _, Hv = jax.jvp(lambda y: jax.grad(energy, argnums=1)(params, y), (x,), (v,))
    # tr(d score/dx) = -tr(H) ; v^T (-H) v
    return 0.5 * jnp.sum(s * s) + (-jnp.sum(v * Hv))


def loss_sliced_sm_sample(params, x, key):
    """Sliced SM with Gaussian projection."""
    s = _score(params, x)
    v = jax.random.normal(key, x.shape)
    _, Hv = jax.jvp(lambda y: jax.grad(energy, argnums=1)(params, y), (x,), (v,))
    return 0.5 * jnp.sum(s * s) + (-jnp.sum(v * Hv))


def loss_dsm_sample(params, x, key, sigma: float = 0.1):
    """Denoising score matching."""
    eps = jax.random.normal(key, x.shape)
    x_tilde = x + sigma * eps
    s = _score(params, x_tilde)
    target = -eps / sigma
    return 0.5 * jnp.sum((s - target) ** 2)


def loss_mle_like_sample(params, x):
    return energy(params, x)


# Batched losses (mean over batch).


def make_loss(name: str):
    if name == "exact_sm":
        def loss(params, x, key):
            del key
            return jnp.mean(jax.vmap(lambda xi: loss_exact_sm_sample(params, xi))(x))
        return loss
    if name == "hutch_sm":
        def loss(params, x, key):
            keys = jax.random.split(key, x.shape[0])
            return jnp.mean(jax.vmap(lambda xi, k: loss_hutch_sm_sample(params, xi, k))(x, keys))
        return loss
    if name == "sliced_sm":
        def loss(params, x, key):
            keys = jax.random.split(key, x.shape[0])
            return jnp.mean(jax.vmap(lambda xi, k: loss_sliced_sm_sample(params, xi, k))(x, keys))
        return loss
    if name == "dsm":
        def loss(params, x, key):
            keys = jax.random.split(key, x.shape[0])
            return jnp.mean(jax.vmap(lambda xi, k: loss_dsm_sample(params, xi, k))(x, keys))
        return loss
    if name == "mle_like":
        def loss(params, x, key):
            del key
            return jnp.mean(jax.vmap(lambda xi: loss_mle_like_sample(params, xi))(x))
        return loss
    raise ValueError(name)


LOSS_NAMES = ["exact_sm", "hutch_sm", "sliced_sm", "dsm", "mle_like"]


# -----------------------------------------------------------------------------
# Benchmark harness.
# -----------------------------------------------------------------------------


def make_step(loss_fn, lr: float = 1e-3):
    """Return a jit'd one-step SGD update returning new params and loss."""
    grad_fn = jax.value_and_grad(loss_fn, argnums=0)

    @jax.jit
    def step(params, x, key):
        l, grads = grad_fn(params, x, key)
        new_params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return new_params, l

    return step


def benchmark(name: str, loss_fn, params, x, key, warmup: int, iters: int) -> dict:
    step = make_step(loss_fn)
    # warmup (also triggers jit compile + caches)
    for _ in range(warmup):
        params, l = step(params, x, key)
        _ = jax.block_until_ready(l)
    # timed
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        params, l = step(params, x, key)
        _ = jax.block_until_ready(l)
        times.append(time.perf_counter() - t0)
    return {
        "estimator": name,
        "iters": iters,
        "mean_ms": 1e3 * statistics.mean(times),
        "p50_ms": 1e3 * statistics.median(times),
        "p90_ms": 1e3 * (sorted(times)[int(0.9 * (iters - 1))]),
        "min_ms": 1e3 * min(times),
        "max_ms": 1e3 * max(times),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dims", type=int, nargs="+", default=[2, 8, 64, 512, 2048])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "profile_jax.csv",
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "profile_jax.json",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"[jax] devices={jax.devices()}", flush=True)
    rng = jax.random.PRNGKey(args.seed)
    rows: list[dict] = []
    for D in args.dims:
        rng, init_key, data_key, step_key = jax.random.split(rng, 4)
        params = init_params(init_key, D, args.hidden, args.layers)
        x = jax.random.normal(data_key, (args.batch, D))
        n_p = num_params(params)
        for name in LOSS_NAMES:
            try:
                r = benchmark(
                    name,
                    make_loss(name),
                    params,
                    x,
                    step_key,
                    args.warmup,
                    args.iters,
                )
            except Exception as e:  # pylint: disable=broad-except
                r = {
                    "estimator": name,
                    "iters": 0,
                    "mean_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p90_ms": float("nan"),
                    "min_ms": float("nan"),
                    "max_ms": float("nan"),
                    "error": str(e)[:200],
                }
            r.update(
                {
                    "backend": "jax",
                    "device": str(jax.devices()[0]),
                    "D": D,
                    "B": args.batch,
                    "n_params": n_p,
                    "hidden": args.hidden,
                    "layers": args.layers,
                }
            )
            rows.append(r)
            print(
                f"[jax] D={D:<5} {name:<10} "
                f"mean={r['mean_ms']:.3f} ms  p50={r['p50_ms']:.3f} ms",
                flush=True,
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as fh:
        cols = [
            "backend",
            "device",
            "D",
            "B",
            "n_params",
            "hidden",
            "layers",
            "estimator",
            "iters",
            "mean_ms",
            "p50_ms",
            "p90_ms",
            "min_ms",
            "max_ms",
            "error",
        ]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with args.out_json.open("w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"[jax] wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
