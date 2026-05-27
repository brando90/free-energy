#!/usr/bin/env python3
"""JAX mirror of profile_pytorch.py.

Same EBM, same losses, same sweep — to cross-check the PyTorch results and
see whether JAX's functional autodiff (jacfwd / hessian / jvp / vjp) makes
the second-derivative term meaningfully cheaper.

Usage
-----
    python profile_jax.py [--tag TAG] [--out-dir DIR] [--quick]
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import random


# ---------------------------------------------------------------------------
# Model — flax-free Stax-style for minimal deps
# ---------------------------------------------------------------------------


def init_mlp(key, in_dim: int, hidden: int, depth: int = 2) -> list[dict]:
    keys = random.split(key, depth + 1)
    sizes = [in_dim] + [hidden] * depth + [1]
    params = []
    for k, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:]):
        w = random.normal(k, (fan_in, fan_out)) * (1.0 / np.sqrt(fan_in))
        b = jnp.zeros((fan_out,))
        params.append({"W": w, "b": b})
    return params


def apply_mlp(params, x):
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["W"] + layer["b"]
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h.squeeze(-1)


def energy_single(params, x_single):
    """E_θ(x) for a single x of shape (d,)."""
    return apply_mlp(params, x_single[None, :])[0]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def sm_exact_loss(params, x_batch):
    """Explicit SM with exact tr(H_x) via jax.hessian."""
    def per_example(x):
        grad_x = jax.grad(energy_single, argnums=1)(params, x)        # (d,)
        H = jax.hessian(energy_single, argnums=1)(params, x)           # (d, d)
        return 0.5 * jnp.dot(grad_x, grad_x) - jnp.trace(H)
    return jnp.mean(jax.vmap(per_example)(x_batch))


def sm_hutchinson_loss(params, x_batch, vs):
    """SM with Hutchinson trace estimator.

    vs : (n_probes, B, d) array of ±1.
    """
    def per_example(x, v_stack):
        grad_x_fn = lambda y: jax.grad(energy_single, argnums=1)(params, y)
        grad_x = grad_x_fn(x)
        sq = 0.5 * jnp.dot(grad_x, grad_x)

        def one_probe(v):
            _, hvp = jax.jvp(grad_x_fn, (x,), (v,))
            return jnp.dot(hvp, v)

        tr_est = jnp.mean(jax.vmap(one_probe)(v_stack))
        return sq - tr_est

    return jnp.mean(jax.vmap(per_example)(x_batch, vs.swapaxes(0, 1)))


def dsm_loss(params, x_batch, noise, sigma: float):
    def per_example(x, n):
        xt = x + n
        grad_xt = jax.grad(energy_single, argnums=1)(params, xt)
        target = -n / (sigma ** 2)
        diff = (-grad_xt) - target
        return 0.5 * jnp.dot(diff, diff) * (sigma ** 2)
    return jnp.mean(jax.vmap(per_example)(x_batch, noise))


def mle_like_loss(params, x_batch):
    return jnp.mean(jax.vmap(lambda x: energy_single(params, x))(x_batch))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    loss_name: str
    in_dim: int
    hidden: int
    batch: int
    backend: str
    device: str
    n_params: int
    fwd_bwd_ms: float
    peak_mem_mb: float
    loss_value: float
    n_runs: int
    n_warmup: int


def _params_count(params) -> int:
    return int(sum(np.prod(np.array(p.shape)) for layer in params for p in layer.values()))


def _peak_mem_mb() -> float:
    # JAX-CPU: report process RSS as a stand-in.
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return float(ru / (1024 ** 2))
        return float(ru / 1024)
    except Exception:
        return float("nan")


def time_step(
    loss_name: str,
    in_dim: int,
    hidden: int,
    batch: int,
    n_warmup: int = 3,
    n_runs: int = 5,
    n_hutch_probes: int = 1,
    sigma: float = 0.1,
) -> StepResult:
    key = random.PRNGKey(0)
    k_init, k_data, k_probe, k_noise = random.split(key, 4)
    params = init_mlp(k_init, in_dim, hidden)
    x = random.normal(k_data, (batch, in_dim))

    if loss_name == "sm_exact":
        loss_fn = jax.jit(sm_exact_loss)
        loss_grad = jax.jit(jax.grad(sm_exact_loss))
        call = lambda: (loss_fn(params, x), loss_grad(params, x))
    elif loss_name == "sm_hutch1":
        vs_fixed = (random.bernoulli(k_probe, 0.5, (1, batch, in_dim)).astype(jnp.float32) * 2 - 1)

        def loss_fn(p, xx, v): return sm_hutchinson_loss(p, xx, v)
        loss_fn_j = jax.jit(loss_fn)
        loss_grad_j = jax.jit(jax.grad(loss_fn))
        call = lambda: (loss_fn_j(params, x, vs_fixed), loss_grad_j(params, x, vs_fixed))
    elif loss_name == "sm_hutch4":
        vs_fixed = (random.bernoulli(k_probe, 0.5, (4, batch, in_dim)).astype(jnp.float32) * 2 - 1)

        def loss_fn(p, xx, v): return sm_hutchinson_loss(p, xx, v)
        loss_fn_j = jax.jit(loss_fn)
        loss_grad_j = jax.jit(jax.grad(loss_fn))
        call = lambda: (loss_fn_j(params, x, vs_fixed), loss_grad_j(params, x, vs_fixed))
    elif loss_name == "dsm":
        noise = random.normal(k_noise, (batch, in_dim)) * sigma

        def loss_fn(p, xx, n): return dsm_loss(p, xx, n, sigma)
        loss_fn_j = jax.jit(loss_fn)
        loss_grad_j = jax.jit(jax.grad(loss_fn))
        call = lambda: (loss_fn_j(params, x, noise), loss_grad_j(params, x, noise))
    elif loss_name == "mle_like":
        loss_fn_j = jax.jit(mle_like_loss)
        loss_grad_j = jax.jit(jax.grad(mle_like_loss))
        call = lambda: (loss_fn_j(params, x), loss_grad_j(params, x))
    else:
        raise ValueError(loss_name)

    last_loss = float("nan")
    for _ in range(n_warmup):
        loss, grads = call()
        loss.block_until_ready()
        # block on grads pytree
        jax.tree.map(lambda g: g.block_until_ready(), grads)
        last_loss = float(loss)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        loss, grads = call()
        loss.block_until_ready()
        jax.tree.map(lambda g: g.block_until_ready(), grads)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

    peak = _peak_mem_mb()
    n_params = _params_count(params)
    devices = jax.devices()

    del params, x
    gc.collect()

    return StepResult(
        loss_name=loss_name,
        in_dim=in_dim,
        hidden=hidden,
        batch=batch,
        backend="jax",
        device=str(devices[0]),
        n_params=n_params,
        fwd_bwd_ms=elapsed_ms,
        peak_mem_mb=peak,
        loss_value=last_loss,
        n_runs=n_runs,
        n_warmup=n_warmup,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


SM_EXACT_MAX_D = 512  # exact jax.hessian is O(d²) elements; cap to keep sweep sane


def sweep(
    dims: list[int], hiddens: list[int], batches: list[int],
    losses: list[str], n_warmup: int, n_runs: int,
) -> list[StepResult]:
    out: list[StepResult] = []
    for d in dims:
        for h in hiddens:
            for b in batches:
                for ln in losses:
                    if ln == "sm_exact" and d > SM_EXACT_MAX_D:
                        print(f"  - skip sm_exact d={d} (> {SM_EXACT_MAX_D})")
                        continue
                    try:
                        r = time_step(ln, d, h, b, n_warmup, n_runs)
                    except Exception as exc:
                        r = StepResult(
                            loss_name=ln, in_dim=d, hidden=h, batch=b,
                            backend="jax", device=str(jax.devices()[0]),
                            n_params=-1, fwd_bwd_ms=float("nan"),
                            peak_mem_mb=float("nan"), loss_value=float("nan"),
                            n_runs=0, n_warmup=n_warmup,
                        )
                        print(f"  ! {ln} d={d} h={h} b={b}: {type(exc).__name__}: {exc}")
                    print(
                        f"  {ln:>9s} d={d:5d} h={h:5d} b={b:4d}  "
                        f"{r.fwd_bwd_ms:8.2f} ms  {r.peak_mem_mb:7.1f} MB"
                    )
                    out.append(r)
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="jax_sweep")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=5)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        dims = [2, 8, 32]
        hiddens = [64]
        batches = [32]
        losses = ["mle_like", "dsm", "sm_hutch1", "sm_exact"]
    else:
        dims = [2, 8, 32, 128, 512, 2048]
        hiddens = [64, 256, 1024]
        batches = [32, 128]
        losses = ["mle_like", "dsm", "sm_hutch1", "sm_hutch4", "sm_exact"]

    print(f"backend=jax devices={jax.devices()} jax={jax.__version__}")
    print(f"dims={dims} hiddens={hiddens} batches={batches} losses={losses}")
    results = sweep(dims, hiddens, batches, losses, args.n_warmup, args.n_runs)

    payload = {
        "backend": "jax",
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "platform": platform.platform(),
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "results": [asdict(r) for r in results],
    }
    out_json = args.out_dir / f"{args.tag}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
