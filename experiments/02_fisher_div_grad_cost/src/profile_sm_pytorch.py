"""Profile cost of d/dtheta of the Fisher / score-matching loss in PyTorch.

We benchmark five training-step estimators for an EBM E_theta : R^D -> R:

  (E1) Exact SM  — explicit tr( d^2 E_theta / d x^2 ) via per-coordinate
                   second derivatives (D backward passes per sample).
  (E2) Hutch SM  — Hutchinson trace estimator of the Hessian, single probe
                   v ~ Rademacher, computed by an HVP on score(x).
  (E3) Sliced SM — single-direction surrogate from Song et al. 2019,
                   uses one (vector,Jacobian-vector) product per sample.
  (E4) DSM       — denoising score matching, no 2nd-order autodiff.
  (E5) MLE-like  — pure first-order gradient of E_theta(x) wrt theta (no Z),
                   the cheapest possible reference baseline.

For each estimator we measure:
  - wall-clock time of one (forward + loss + backward + step) iteration,
  - peak resident memory used by the autograd graph.

The point is to see whether Hyvärinen's "trace-of-Hessian" form (E1) is in
fact prohibitively expensive vs. the cheaper alternatives — i.e. whether
the dim D matters in practice, because we only differentiate wrt x
(small), not theta (large).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Make sibling module 'ebm_models.py' importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch import Tensor  # noqa: E402

from ebm_models import TinyEBM, num_params  # noqa: E402


# -----------------------------------------------------------------------------
# Loss implementations.
# Each function returns a SCALAR loss whose .backward() yields d L / d theta.
# Inputs:
#   model : TinyEBM
#   x     : (B, D) tensor of data samples (already on the right device)
# -----------------------------------------------------------------------------


def _score(model: torch.nn.Module, x: Tensor) -> Tensor:
    """nabla_x log p_theta(x) = - nabla_x E_theta(x); shape (B, D)."""
    x = x.detach().requires_grad_(True)
    e = model(x).sum()
    score = -torch.autograd.grad(e, x, create_graph=True)[0]
    return score, x


def loss_exact_sm(model: torch.nn.Module, x: Tensor) -> Tensor:
    """Exact Hyvärinen SM loss: 0.5 ||score||^2 + tr( d score / d x )."""
    score, x = _score(model, x)
    B, D = x.shape
    sq = 0.5 * (score**2).sum(dim=-1).mean()
    trace = 0.0
    # Diagonal of Jacobian-of-score (= -Hessian of E) — D backward passes.
    for d in range(D):
        grad_d = torch.autograd.grad(
            score[:, d].sum(), x, create_graph=True, retain_graph=True
        )[0][:, d]
        trace = trace + grad_d
    trace_mean = trace.mean() if isinstance(trace, Tensor) else torch.tensor(0.0)
    return sq + trace_mean


def loss_hutch_sm(model: torch.nn.Module, x: Tensor) -> Tensor:
    """Hutchinson SM: tr(J) ≈ v^T J v for Rademacher v, ONE HVP per sample."""
    score, x = _score(model, x)
    sq = 0.5 * (score**2).sum(dim=-1).mean()
    v = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2 - 1
    vTscore = (v * score).sum()
    # nabla_x (v^T score) ; then dot with v -> v^T J v.
    jv = torch.autograd.grad(vTscore, x, create_graph=True)[0]
    hutch = (v * jv).sum(dim=-1).mean()
    return sq + hutch


def loss_sliced_sm(model: torch.nn.Module, x: Tensor) -> Tensor:
    """Sliced score matching (Song et al. 2019): single random direction."""
    score, x = _score(model, x)
    v = torch.randn_like(x)
    vTs = (v * score).sum()
    vTjv = (v * torch.autograd.grad(vTs, x, create_graph=True)[0]).sum(dim=-1)
    sq = 0.5 * (score**2).sum(dim=-1).mean()
    return sq + vTjv.mean()


def loss_dsm(model: torch.nn.Module, x: Tensor, sigma: float = 0.1) -> Tensor:
    """Denoising SM: no second derivatives. score should match -eps/sigma."""
    eps = torch.randn_like(x)
    x_tilde = x + sigma * eps
    score, _ = _score(model, x_tilde)
    target = -eps / sigma
    return 0.5 * ((score - target) ** 2).sum(dim=-1).mean()


def loss_mle_like(model: torch.nn.Module, x: Tensor) -> Tensor:
    """Pure first-order baseline: E_theta(x).mean()."""
    return model(x).mean()


LOSSES = {
    "exact_sm": loss_exact_sm,
    "hutch_sm": loss_hutch_sm,
    "sliced_sm": loss_sliced_sm,
    "dsm": loss_dsm,
    "mle_like": loss_mle_like,
}


# -----------------------------------------------------------------------------
# Benchmark harness.
# -----------------------------------------------------------------------------


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_mem(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    if device.type == "mps":
        try:
            return torch.mps.driver_allocated_memory() / 1e6
        except Exception:
            return None
    return None


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def benchmark(
    name: str,
    loss_fn,
    model: torch.nn.Module,
    x: Tensor,
    device: torch.device,
    n_warmup: int = 5,
    n_iters: int = 20,
) -> dict:
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    # warmup
    for _ in range(n_warmup):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model, x)
        loss.backward()
        opt.step()
        _sync(device)
    _reset_peak(device)
    # timed
    times = []
    for _ in range(n_iters):
        opt.zero_grad(set_to_none=True)
        _sync(device)
        t0 = time.perf_counter()
        loss = loss_fn(model, x)
        loss.backward()
        opt.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return {
        "estimator": name,
        "iters": n_iters,
        "mean_ms": 1e3 * statistics.mean(times),
        "p50_ms": 1e3 * statistics.median(times),
        "p90_ms": 1e3 * (sorted(times)[int(0.9 * (n_iters - 1))]),
        "min_ms": 1e3 * min(times),
        "max_ms": 1e3 * max(times),
        "peak_mem_mb": _peak_mem(device),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dims", type=int, nargs="+", default=[2, 8, 64, 512, 2048]
    )
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--out_csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "profile_pytorch.csv",
    )
    p.add_argument(
        "--out_json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "results"
        / "profile_pytorch.json",
    )
    return p.parse_args(argv)


def _resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = _resolve_device(args.device)
    print(f"[pytorch] device={device}", flush=True)
    rows: list[dict] = []
    for D in args.dims:
        model = TinyEBM(D, args.hidden, args.layers).to(device)
        x = torch.randn(args.batch, D, device=device)
        n_p = num_params(model)
        for name, fn in LOSSES.items():
            try:
                r = benchmark(
                    name, fn, model, x, device, args.warmup, args.iters
                )
            except RuntimeError as e:
                # MPS fallback / OOM
                r = {
                    "estimator": name,
                    "iters": 0,
                    "mean_ms": float("nan"),
                    "p50_ms": float("nan"),
                    "p90_ms": float("nan"),
                    "min_ms": float("nan"),
                    "max_ms": float("nan"),
                    "peak_mem_mb": None,
                    "error": str(e)[:200],
                }
            r.update(
                {
                    "backend": "pytorch",
                    "device": str(device),
                    "D": D,
                    "B": args.batch,
                    "n_params": n_p,
                    "hidden": args.hidden,
                    "layers": args.layers,
                }
            )
            rows.append(r)
            print(
                f"[pytorch] D={D:<5} {name:<10} "
                f"mean={r['mean_ms']:.3f} ms  "
                f"p50={r['p50_ms']:.3f} ms  "
                f"peak_mem={r['peak_mem_mb']}",
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
            "peak_mem_mb",
            "error",
        ]
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with args.out_json.open("w") as fh:
        json.dump(rows, fh, indent=2, default=str)
    print(f"[pytorch] wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
