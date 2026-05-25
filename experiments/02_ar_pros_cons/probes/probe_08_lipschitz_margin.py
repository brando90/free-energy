"""Probe 08 — Brittleness / Lipschitz-margin.

The Lipschitz lower bound on the input perturbation needed to flip a prediction is
    ||delta|| >= f(x) / L_global,   L_global <= prod_i ||W_i||_2.
Two predictions:
    (a) the bound holds (it's a *lower* bound on the smallest flipping perturbation);
    (b) when we train models that have larger L_global at the same data, the
        empirically smallest flipping perturbation gets smaller -- output margin
        alone is not protective.

Method: train K MLP classifiers on the same synthetic binary task with varying
weight decay (and matched epochs). Low/zero decay gives large L_global (and
typically a sharper decision boundary); large decay gives small L_global.
Across the K models, plot empirical median-min-flip-norm vs L_global; expect
a downward trend.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from probes._common import (
    ProbeResult,
    add_common_args,
    default_output_dir,
    gpu_name,
    resolve_device,
    seed_everything,
    write_result,
)


def make_synthetic(n: int, d: int, device: torch.device, sep: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.randint(0, 2, (n,), device=device)
    mu = torch.zeros(2, d, device=device)
    mu[1, 0] = sep
    x = mu[y] + 0.5 * torch.randn(n, d, device=device)
    return x, y


class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = d_in
        for _ in range(depth):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.GELU())
            prev = hidden
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def linear_weights(self) -> List[torch.Tensor]:
        return [m.weight for m in self.net if isinstance(m, nn.Linear)]


def spectral_norm_product(model: MLP) -> float:
    prod = 1.0
    for w in model.linear_weights():
        with torch.no_grad():
            s = torch.linalg.svdvals(w).max().item()
        prod *= s
    return prod


def output_margin(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    correct = logits.gather(1, y.unsqueeze(1)).squeeze(1)
    masked = logits.scatter(1, y.unsqueeze(1), float("-inf"))
    other = masked.max(dim=1).values
    return correct - other


def find_min_flip_norm(
    model: MLP, x: torch.Tensor, y: torch.Tensor, max_radii: List[float], steps_per_radius: int
) -> torch.Tensor:
    model.eval()
    N = x.size(0)
    flipped_at = torch.full((N,), float(max_radii[-1]) + 1.0, device=x.device)
    for radius in max_radii:
        delta = torch.zeros_like(x).normal_(std=min(radius / 10.0, 0.01)).detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=max(radius / 20.0, 1e-3))
        for _ in range(steps_per_radius):
            opt.zero_grad()
            logits = model(x + delta)
            loss = -F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            opt.step()
            with torch.no_grad():
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                factor = (radius / norms).clamp_max(1.0)
                delta.mul_(factor)
        with torch.no_grad():
            pred = model(x + delta).argmax(dim=-1)
            newly_flipped = (pred != y) & (flipped_at > radius)
            flipped_at[newly_flipped] = radius
    return flipped_at


def train_with_wd(d: int, hidden: int, depth: int, x: torch.Tensor, y: torch.Tensor, steps: int, lr: float, wd: float, device: torch.device) -> MLP:
    model = MLP(d_in=d, hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    model.train()
    for _ in range(steps):
        idx = torch.randperm(x.size(0), device=device)[:256]
        logits = model(x[idx])
        loss = F.cross_entropy(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 08: Lipschitz margin")
    add_common_args(parser)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--d", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[0.0, 1e-4, 1e-3, 1e-2, 1e-1])
    parser.add_argument("--eval-n", type=int, default=256)
    parser.add_argument("--attack-radii", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])
    parser.add_argument("--attack-steps", type=int, default=80)
    args = parser.parse_args()

    if args.smoke:
        args.n = 2048
        args.steps = 600
        args.weight_decays = [0.0, 1e-3, 1e-1]
        args.eval_n = 128
        args.attack_radii = [0.05, 0.2, 0.8, 3.2]
        args.attack_steps = 40

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_08", args.tag)

    result = ProbeResult(
        probe="probe_08_lipschitz_margin",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    x_train, y_train = make_synthetic(args.n, args.d, device)
    x_eval, y_eval = make_synthetic(args.eval_n, args.d, device)

    runs: List[Dict] = []
    for wd in args.weight_decays:
        seed_everything(args.seed)  # same init across decays
        model = train_with_wd(args.d, args.hidden, args.depth, x_train, y_train, args.steps, args.lr, wd, device)
        model.eval()
        with torch.no_grad():
            logits = model(x_eval)
            acc = (logits.argmax(-1) == y_eval).float().mean().item()
        L = spectral_norm_product(model)
        correct = logits.argmax(-1) == y_eval
        if correct.sum().item() < 4:
            runs.append({"weight_decay": wd, "spectral_norm_product": L, "eval_acc": acc, "skipped": True})
            continue
        x_c = x_eval[correct]
        y_c = y_eval[correct]
        margin = output_margin(model(x_c), y_c)
        bound = (margin / max(L, 1e-9)).clamp_min(1e-9)
        flip_norms = find_min_flip_norm(model, x_c.clone(), y_c, sorted(args.attack_radii), args.attack_steps)
        ever_flipped = flip_norms <= args.attack_radii[-1]
        median_flip = float(flip_norms[ever_flipped].median().item()) if ever_flipped.any() else float("inf")
        min_flip = float(flip_norms[ever_flipped].min().item()) if ever_flipped.any() else float("inf")
        min_bound = float(bound.min().item())
        bound_holds = bool(min_flip + 1e-3 >= min_bound) if ever_flipped.any() else True
        runs.append(
            {
                "weight_decay": wd,
                "spectral_norm_product": L,
                "eval_acc": acc,
                "mean_output_margin": float(margin.mean().item()),
                "median_lipschitz_bound": float(bound.median().item()),
                "min_lipschitz_bound": min_bound,
                "median_min_flip_norm": median_flip,
                "min_flip_norm": min_flip,
                "flip_rate_at_max_radius": float(ever_flipped.float().mean().item()),
                "bound_holds_at_min": bound_holds,
                "skipped": False,
            }
        )

    valid_runs = [r for r in runs if not r.get("skipped")]
    # sort by spectral product ascending
    valid_runs_by_L = sorted(valid_runs, key=lambda r: r["spectral_norm_product"])
    # We expect median_min_flip_norm to DECREASE as L grows.
    # Replace inf with a large sentinel (twice max attack radius) for comparison.
    sentinel = 2.0 * max(args.attack_radii)
    flips = [r["median_min_flip_norm"] if r["median_min_flip_norm"] != float("inf") else sentinel for r in valid_runs_by_L]
    Ls = [r["spectral_norm_product"] for r in valid_runs_by_L]
    # The key prediction: the run with the LARGEST L_global has the SMALLEST flip norm.
    # Spearman rank corr is a softer secondary check (negative trend).
    largest_L_is_most_brittle = flips[-1] <= min(flips)
    spearman_negative = _spearman(np.asarray(Ls), np.asarray(flips)) < -0.2
    bound_holds_all = all(r["bound_holds_at_min"] for r in valid_runs)

    control_passed = bool(largest_L_is_most_brittle and spearman_negative and bound_holds_all)

    result.control_passed = control_passed
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "n_train": args.n,
        "weight_decays": args.weight_decays,
        "runs": runs,
        "ranked_L_global": Ls,
        "ranked_median_min_flip_norm": flips,
        "largest_L_is_most_brittle": largest_L_is_most_brittle,
        "spearman_L_vs_flip_norm": float(_spearman(np.asarray(Ls), np.asarray(flips))),
        "bound_holds_across_all_runs": bound_holds_all,
    }
    result.notes = {
        "interpretation": (
            "Across models trained with the same architecture but varying weight decay, larger spectral"
            " norm product (low / zero decay) should yield smaller empirical min flip norms. Output"
            " margin alone is not protective; the relevant ratio is f(x)/L_global."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_08] wrote {path}  control_passed={control_passed}")
    return 0 if control_passed else 1


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2:
        return 0.0
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    ar = ar - ar.mean()
    br = br - br.mean()
    denom = float(np.sqrt((ar ** 2).sum() * (br ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((ar * br).sum() / denom)


if __name__ == "__main__":
    raise SystemExit(main())
