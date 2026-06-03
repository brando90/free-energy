"""Probe — Data wall.

Descriptive scaling-curve probe. Fits L(N) = a * N^{-alpha} + b on a clean
linear-regression task where we know the asymptotic loss is the noise floor.

The point of this probe in the suite is a methodology check on the scaling fit
itself: when the underlying generator is power-law-on-N (closed-form holds for
ridge regression on Gaussian-noise targets), we should recover alpha > 0 and
a smooth, near-monotone curve. Real scaling curves on language modelling will
plug into the same fit pipeline.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
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


def sample_w_true(d: int, device: torch.device, seed: int) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(d, generator=g, device=device) / math.sqrt(d)


def make_regression_data(n: int, d: int, sigma: float, device: torch.device, seed: int, w_true: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n, d, generator=g, device=device)
    noise = sigma * torch.randn(n, generator=g, device=device)
    y = x @ w_true + noise
    return x, y


def fit_eval_ridge(x_train: torch.Tensor, y_train: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, ridge: float) -> float:
    d = x_train.size(1)
    A = x_train.T @ x_train + ridge * torch.eye(d, device=x_train.device)
    b = x_train.T @ y_train
    w = torch.linalg.solve(A, b)
    pred = x_test @ w
    return float(F.mse_loss(pred, y_test).item())


def fit_power_law(ns: np.ndarray, losses: np.ndarray, noise_floor: float) -> Dict:
    """Fit L = a * N^{-alpha} + b where b is the known noise floor."""
    excess = np.clip(losses - noise_floor, 1e-6, None)
    log_n = np.log(ns)
    log_l = np.log(excess)
    slope, intercept = np.polyfit(log_n, log_l, 1)
    alpha = float(-slope)
    a = float(np.exp(intercept))
    pred = a * ns ** (-alpha) + noise_floor
    sse = float(np.sum((losses - pred) ** 2))
    return {"alpha": alpha, "a": a, "b": noise_floor, "sse": sse, "pred": pred.tolist()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe data wall: scaling curve")
    add_common_args(parser)
    parser.add_argument("--ns", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--d", type=int, default=16)
    parser.add_argument("--noise-sigma", type=float, default=0.3)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--eval-n", type=int, default=8192)
    args = parser.parse_args()

    if args.smoke:
        args.ns = [64, 128, 256, 512, 1024, 2048]
        args.repeats = 3
        args.eval_n = 4096

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_data_wall", args.tag)

    result = ProbeResult(
        probe="probe_data_wall",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    noise_floor = float(args.noise_sigma ** 2)  # MSE asymptote for unbiased linear regression
    per_n: List[Dict] = []
    for n in args.ns:
        losses = []
        for r in range(args.repeats):
            # One shared w_true per (seed, repeat), used for both train and test of THIS run.
            w_true = sample_w_true(args.d, device, seed=1000 * args.seed + 17 * r)
            x_tr, y_tr = make_regression_data(n, args.d, args.noise_sigma, device, seed=2000 + 31 * r, w_true=w_true)
            x_te, y_te = make_regression_data(args.eval_n, args.d, args.noise_sigma, device, seed=3000 + 41 * r, w_true=w_true)
            losses.append(fit_eval_ridge(x_tr, y_tr, x_te, y_te, args.ridge))
        per_n.append({"n": n, "losses": losses, "mean": float(np.mean(losses)), "std": float(np.std(losses))})

    ns = np.asarray([p["n"] for p in per_n], dtype=np.float64)
    means = np.asarray([p["mean"] for p in per_n], dtype=np.float64)
    fit = fit_power_law(ns, means, noise_floor)

    # Control: loss must trend down toward the noise floor as N grows, with positive alpha.
    # Use relative drop because we may already be close to the floor at smoke sizes.
    drop_pct = float((means[0] - means[-1]) / max(abs(means[0]), 1e-9))
    overall_decrease = bool(drop_pct > 0.1)
    total_drop = max(means[0] - means[-1], 1e-6)
    no_huge_jump_up = bool(np.all(np.diff(means) <= 0.3 * total_drop + 1e-3))
    above_floor = bool(means[-1] >= noise_floor - 0.05)
    alpha_positive = bool(fit["alpha"] > 0.05)
    control_passed = bool(overall_decrease and no_huge_jump_up and alpha_positive and above_floor)

    result.control_passed = control_passed
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "ns": args.ns,
        "repeats": args.repeats,
        "d": args.d,
        "noise_sigma": args.noise_sigma,
        "noise_floor": noise_floor,
        "per_n": per_n,
        "fit": {k: v for k, v in fit.items() if k != "pred"},
        "fit_pred_at_ns": fit["pred"],
        "loss_overall_decreases": overall_decrease,
        "no_huge_jump_up": no_huge_jump_up,
        "above_noise_floor": above_floor,
        "alpha_positive": alpha_positive,
    }
    result.notes = {
        "interpretation": (
            "Descriptive: report fitted alpha and the known noise floor b. The smooth, monotone scaling"
            " curve validates the fit pipeline used later on real (LM) loss curves."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_data_wall] wrote {path}  alpha={fit['alpha']:.3f}  control_passed={control_passed}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
