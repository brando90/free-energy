#!/usr/bin/env python3
"""Profile cost of the explicit score-matching gradient ∇_θ L_SM in PyTorch.

For an EBM p_θ(x) = exp(-E_θ(x)) / Z_θ, Hyvärinen's score-matching loss is:

    L_SM(θ) = E_x[ (1/2) ||∇_x E_θ(x)||² − tr(∇²_x E_θ(x)) ]    (1)

(sign of the trace because ∇_x log p̂ = −∇_x E.)

We measure wall clock + peak memory for one training step of:

    SM-Exact     — exact tr(∇²_x E_θ(x)) via d backward-of-backward calls
    SM-Hutch(k)  — stochastic estimator using k Hessian-vector products
                   with Rademacher probes
    DSM          — denoising score matching: only first derivatives in x
    MLE-like     — plain mean energy (no x-derivatives at all); a floor

Backends: cpu and mps.

Usage
-----
    python profile_pytorch.py [--tag TAG] [--device {cpu,mps,cuda,auto}]
                              [--out-dir DIR] [--quick]
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

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLPEnergy(nn.Module):
    """E_θ : R^d → R, scalar energy via a small MLP."""

    def __init__(self, in_dim: int, hidden: int, depth: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Score-matching losses
# ---------------------------------------------------------------------------


def sm_exact_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Explicit Hyvärinen score matching with exact trace(Hessian_x)."""
    x = x.detach().requires_grad_(True)
    e = model(x).sum()
    grad_x = torch.autograd.grad(e, x, create_graph=True)[0]      # (B, d)
    sq = 0.5 * grad_x.pow(2).sum(dim=-1)                          # (B,)
    d = x.shape[-1]
    tr_terms = []
    for i in range(d):
        gi = grad_x[:, i].sum()
        d2 = torch.autograd.grad(gi, x, create_graph=True)[0][:, i]
        tr_terms.append(d2)
    tr_hess = torch.stack(tr_terms, dim=-1).sum(dim=-1)           # (B,)
    return (sq - tr_hess).mean()


def sm_hutchinson_loss(model: nn.Module, x: torch.Tensor, n_probes: int = 1) -> torch.Tensor:
    """Score matching with Hutchinson trace estimator: tr(H) ≈ E[ vᵀ H v ]."""
    x = x.detach().requires_grad_(True)
    e = model(x).sum()
    grad_x = torch.autograd.grad(e, x, create_graph=True)[0]
    sq = 0.5 * grad_x.pow(2).sum(dim=-1)
    tr_est = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for _ in range(n_probes):
        v = torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype).mul_(2).sub_(1)
        gv = (grad_x * v).sum()
        hvp = torch.autograd.grad(gv, x, create_graph=True)[0]
        tr_est = tr_est + (hvp * v).sum(dim=-1)
    tr_est = tr_est / n_probes
    return (sq - tr_est).mean()


def dsm_loss(model: nn.Module, x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    """Denoising SM (Vincent 2011) — only first-order in x."""
    noise = torch.randn_like(x) * sigma
    xt = (x + noise).detach().requires_grad_(True)
    e = model(xt).sum()
    grad_x = torch.autograd.grad(e, xt, create_graph=True)[0]
    # ∇_x log p = -∇_x E, target = -noise / σ²
    target = -noise / (sigma ** 2)
    return 0.5 * ((-grad_x) - target).pow(2).sum(dim=-1).mean() * (sigma ** 2)


def mle_like_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Floor for timing: just the mean energy, no x-derivatives."""
    return model(x).mean()


LOSSES: dict[str, Callable[..., torch.Tensor]] = {
    "mle_like": lambda m, x: mle_like_loss(m, x),
    "dsm": lambda m, x: dsm_loss(m, x),
    "sm_hutch1": lambda m, x: sm_hutchinson_loss(m, x, 1),
    "sm_hutch4": lambda m, x: sm_hutchinson_loss(m, x, 4),
    "sm_exact": lambda m, x: sm_exact_loss(m, x),
}


# ---------------------------------------------------------------------------
# Timing harness
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


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _peak_mem_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    if device.type == "mps":
        try:
            return float(torch.mps.driver_allocated_memory() / (1024 ** 2))
        except AttributeError:
            return float("nan")
    # cpu — fall back to process RSS
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, linux reports KB
        if platform.system() == "Darwin":
            return float(ru / (1024 ** 2))
        return float(ru / 1024)
    except Exception:
        return float("nan")


def _reset_peak_mem(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def time_step(
    loss_name: str,
    in_dim: int,
    hidden: int,
    batch: int,
    device: torch.device,
    n_warmup: int = 3,
    n_runs: int = 5,
) -> StepResult:
    torch.manual_seed(0)
    model = MLPEnergy(in_dim, hidden).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss_fn = LOSSES[loss_name]
    x_base = torch.randn(batch, in_dim, device=device)

    last_loss = float("nan")
    for _ in range(n_warmup):
        optim.zero_grad(set_to_none=True)
        x = x_base.clone()
        loss = loss_fn(model, x)
        loss.backward()
        optim.step()
        last_loss = float(loss.detach().cpu().item())

    _device_sync(device)
    _reset_peak_mem(device)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        optim.zero_grad(set_to_none=True)
        x = x_base.clone()
        loss = loss_fn(model, x)
        loss.backward()
        optim.step()
    _device_sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

    peak = _peak_mem_mb(device)
    n_params = sum(p.numel() for p in model.parameters())

    del model, optim, x_base
    gc.collect()
    if device.type == "mps":
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

    return StepResult(
        loss_name=loss_name,
        in_dim=in_dim,
        hidden=hidden,
        batch=batch,
        backend="pytorch",
        device=str(device),
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


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


SM_EXACT_MAX_D = 512  # exact SM is O(d) backward passes; cap to keep sweep tractable


def sweep(
    device: torch.device,
    dims: list[int],
    hiddens: list[int],
    batches: list[int],
    losses: list[str],
    n_warmup: int,
    n_runs: int,
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
                        r = time_step(ln, d, h, b, device, n_warmup, n_runs)
                    except Exception as exc:  # OOM, autodiff failure, etc.
                        r = StepResult(
                            loss_name=ln, in_dim=d, hidden=h, batch=b,
                            backend="pytorch", device=str(device), n_params=-1,
                            fwd_bwd_ms=float("nan"), peak_mem_mb=float("nan"),
                            loss_value=float("nan"),
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
    p.add_argument("--tag", default="pt_sweep")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    p.add_argument("--quick", action="store_true",
                   help="Tiny sweep for smoke test (cuts dims/batches).")
    p.add_argument("--n-warmup", type=int, default=3)
    p.add_argument("--n-runs", type=int, default=5)
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    device = resolve_device(args.device)
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

    print(f"backend=pytorch device={device} torch={torch.__version__}")
    print(f"dims={dims} hiddens={hiddens} batches={batches} losses={losses}")

    results = sweep(device, dims, hiddens, batches, losses, args.n_warmup, args.n_runs)

    payload = {
        "backend": "pytorch",
        "torch_version": torch.__version__,
        "device": str(device),
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
