"""Probe 02 — Mode covering (forward-KL is zero-avoiding).

Forward KL (MLE) is zero-avoiding: it heavily penalises log q(x) -> -infty where
p(x) > 0, so an under-parameterised model is forced to cover both modes of a
bimodal target, putting non-trivial mass in the inter-mode valley.

Reverse KL is mode-seeking: it collapses on whichever mode the optimiser finds
first because penalty regions are weighted by q(x), not p(x).

Demonstrating this requires a model that *cannot* fit both modes perfectly --
otherwise both KLs converge to the target. We use a single Gaussian q(x) =
N(mu, sigma^2) (2 parameters) against a bimodal target. Then:
  - forward KL drives mu toward the average of the two modes, sigma large -> valley mass high
  - reverse KL drives (mu, sigma) toward one mode -> valley mass low

This is the textbook demonstration; with enough capacity both losses match the
target and you cannot distinguish them.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict

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


def make_bimodal_target(bins: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    left = torch.exp(-0.5 * ((bins + mu) / sigma) ** 2)
    right = torch.exp(-0.5 * ((bins - mu) / sigma) ** 2)
    p = 0.5 * left + 0.5 * right
    p = p / p.sum()
    return p


def gaussian_logp_on_bins(bin_centers: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    sigma = log_sigma.exp()
    logits = -0.5 * ((bin_centers - mu) / sigma) ** 2
    return F.log_softmax(logits, dim=-1)


def train_forward_kl(target_p: torch.Tensor, bin_centers: torch.Tensor, steps: int, lr: float, device: torch.device) -> Dict:
    target_p = target_p.to(device)
    bin_centers = bin_centers.to(device)
    mu = torch.nn.Parameter(torch.tensor(-0.3, device=device))  # init off-centre
    log_sigma = torch.nn.Parameter(torch.tensor(math.log(0.1), device=device))
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logq = gaussian_logp_on_bins(bin_centers, mu, log_sigma)
        # KL(p||q) = sum p (log p - log q); minimise w.r.t. q -> MLE
        kl = (target_p * (target_p.clamp_min(1e-12).log() - logq)).sum()
        kl.backward()
        opt.step()
    with torch.no_grad():
        logq = gaussian_logp_on_bins(bin_centers, mu, log_sigma)
    return {
        "log_q": logq.detach().cpu().numpy().tolist(),
        "mu": float(mu.item()),
        "sigma": float(log_sigma.exp().item()),
        "final_forward_kl": float(kl.item()),
    }


def train_reverse_kl(target_p: torch.Tensor, bin_centers: torch.Tensor, steps: int, lr: float, device: torch.device) -> Dict:
    target_p = target_p.to(device)
    bin_centers = bin_centers.to(device)
    target_logp = target_p.clamp_min(1e-12).log()
    mu = torch.nn.Parameter(torch.tensor(-0.3, device=device))
    log_sigma = torch.nn.Parameter(torch.tensor(math.log(0.1), device=device))
    opt = torch.optim.Adam([mu, log_sigma], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        logq = gaussian_logp_on_bins(bin_centers, mu, log_sigma)
        q = logq.exp()
        # KL(q||p) = sum q (log q - log p)
        kl = (q * (logq - target_logp)).sum()
        kl.backward()
        opt.step()
    with torch.no_grad():
        logq = gaussian_logp_on_bins(bin_centers, mu, log_sigma)
    return {
        "log_q": logq.detach().cpu().numpy().tolist(),
        "mu": float(mu.item()),
        "sigma": float(log_sigma.exp().item()),
        "final_reverse_kl": float(kl.item()),
    }


def valley_mass(log_q: list, bin_edges: np.ndarray, mu: float, sigma: float) -> float:
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    q = np.exp(np.asarray(log_q))
    q = q / q.sum()
    valley = (centers > -mu + 1.5 * sigma) & (centers < mu - 1.5 * sigma)
    return float(q[valley].sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 02: mode covering")
    add_common_args(parser)
    parser.add_argument("--n-bins", type=int, default=512)
    parser.add_argument("--mu", type=float, default=0.6)
    parser.add_argument("--sigma", type=float, default=0.06)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()

    if args.smoke:
        args.n_bins = 256
        args.steps = 2000

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_02", args.tag)

    result = ProbeResult(
        probe="probe_02_mode_covering",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    bin_edges = np.linspace(-1.0, 1.0, args.n_bins + 1)
    centers = torch.from_numpy(0.5 * (bin_edges[:-1] + bin_edges[1:])).float()
    target_p = make_bimodal_target(centers, mu=args.mu, sigma=args.sigma)

    forward = train_forward_kl(target_p, centers, args.steps, args.lr, device)
    reverse = train_reverse_kl(target_p, centers, args.steps, args.lr, device)

    target_valley = valley_mass(target_p.log().tolist(), bin_edges, args.mu, args.sigma)
    fwd_valley = valley_mass(forward["log_q"], bin_edges, args.mu, args.sigma)
    rev_valley = valley_mass(reverse["log_q"], bin_edges, args.mu, args.sigma)

    # Predictions for the under-parameterised Gaussian:
    # forward KL  -> mu_q ~ 0, sigma_q wide, lots of valley mass
    # reverse KL  -> mu_q ~ +- args.mu, sigma_q narrow, ~0 valley mass
    # Tail leakage from a narrow Gaussian on one mode contributes a small but non-zero valley mass;
    # what matters is that forward covers the valley *much* more than reverse.
    forward_covers = fwd_valley > 0.1
    forward_much_more_than_reverse = fwd_valley > 3.0 * max(rev_valley, 1e-6)
    forward_close_to_centre = abs(forward["mu"]) < 0.2
    reverse_near_a_mode = abs(abs(reverse["mu"]) - args.mu) < args.sigma * 4

    control_passed = bool(
        forward_covers and forward_much_more_than_reverse and forward_close_to_centre and reverse_near_a_mode
    )

    result.control_passed = control_passed
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "n_bins": args.n_bins,
        "mu_target": args.mu,
        "sigma_target": args.sigma,
        "target_valley_mass": target_valley,
        "forward_kl_valley_mass": fwd_valley,
        "reverse_kl_valley_mass": rev_valley,
        "forward_fit_mu": forward["mu"],
        "forward_fit_sigma": forward["sigma"],
        "reverse_fit_mu": reverse["mu"],
        "reverse_fit_sigma": reverse["sigma"],
        "forward_covers_valley": forward_covers,
        "forward_much_more_valley_than_reverse": forward_much_more_than_reverse,
        "forward_close_to_centre": forward_close_to_centre,
        "reverse_near_a_mode": reverse_near_a_mode,
        "forward_final_kl": forward["final_forward_kl"],
        "reverse_final_kl": reverse["final_reverse_kl"],
    }
    result.notes = {
        "interpretation": (
            "Single-Gaussian q is forced to compromise. Forward KL (mode-covering) pulls mu toward the"
            " mid-point and widens sigma. Reverse KL (mode-seeking) collapses onto one of the two modes."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_02] wrote {path}  control_passed={control_passed}  fwd_mu={forward['mu']:.3f}  rev_mu={reverse['mu']:.3f}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
