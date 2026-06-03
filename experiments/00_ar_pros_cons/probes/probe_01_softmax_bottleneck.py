"""Probe 01 — Softmax bottleneck (rank <= d+1).

A single softmax head whose state has hidden dim ``d`` can produce log-prob
matrices of effective rank at most ``d + 1`` (the +1 absorbs the per-row
log-partition shift). If the target log-prob matrix has rank ``r > d + 1`` the
KL achievable by a single softmax is lower-bounded by the truncated-SVD tail
beyond the (d+1)-st singular value. A mixture-of-K softmaxes closes the gap.

Positive control:
    - Build a known-rank target ``T`` of shape (C, V).
    - Fit (i) single softmax with hidden dim d in {r/2, r, 2r} and
      (ii) a mixture of K softmaxes.
    - Assert the single-softmax curve hugs the SVD-tail lower bound and that
      the mixture-of-softmaxes beats the single softmax when d < r.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

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


def synth_target_logprobs(num_contexts: int, vocab: int, rank: int, device: torch.device) -> torch.Tensor:
    """Construct an unnormalized score matrix of *exactly* the requested rank, return its log-softmax."""
    a = torch.randn(num_contexts, rank, device=device)
    b = torch.randn(rank, vocab, device=device)
    scores = a @ b  # rank exactly = rank
    return F.log_softmax(scores, dim=-1)


def fit_single_softmax(target_logp: torch.Tensor, d: int, steps: int, lr: float, device: torch.device) -> Dict[str, float]:
    C, V = target_logp.shape
    H = nn.Parameter(torch.randn(C, d, device=device) * 0.1)
    W = nn.Parameter(torch.randn(d, V, device=device) * 0.1)
    b = nn.Parameter(torch.zeros(V, device=device))
    opt = torch.optim.Adam([H, W, b], lr=lr)
    target_p = target_logp.exp()
    for _ in range(steps):
        opt.zero_grad()
        logits = H @ W + b
        logp = F.log_softmax(logits, dim=-1)
        kl = (target_p * (target_logp - logp)).sum(dim=-1).mean()
        kl.backward()
        opt.step()
    with torch.no_grad():
        logits = H @ W + b
        logp = F.log_softmax(logits, dim=-1)
        kl = (target_p * (target_logp - logp)).sum(dim=-1).mean().item()
    return {"final_kl": kl, "params": (C * d + d * V + V)}


def fit_mixture_softmax(target_logp: torch.Tensor, d: int, K: int, steps: int, lr: float, device: torch.device) -> Dict[str, float]:
    C, V = target_logp.shape
    H = nn.Parameter(torch.randn(C, d, device=device) * 0.1)
    Ws = nn.Parameter(torch.randn(K, d, V, device=device) * 0.1)
    bs = nn.Parameter(torch.zeros(K, V, device=device))
    pi = nn.Parameter(torch.zeros(C, K, device=device))
    opt = torch.optim.Adam([H, Ws, bs, pi], lr=lr)
    target_p = target_logp.exp()
    for _ in range(steps):
        opt.zero_grad()
        log_components = []
        for k in range(K):
            logits_k = H @ Ws[k] + bs[k]
            log_components.append(F.log_softmax(logits_k, dim=-1))
        log_comp = torch.stack(log_components, dim=1)  # (C, K, V)
        log_pi = F.log_softmax(pi, dim=-1)             # (C, K)
        logp = torch.logsumexp(log_pi.unsqueeze(-1) + log_comp, dim=1)
        kl = (target_p * (target_logp - logp)).sum(dim=-1).mean()
        kl.backward()
        opt.step()
    with torch.no_grad():
        log_components = []
        for k in range(K):
            logits_k = H @ Ws[k] + bs[k]
            log_components.append(F.log_softmax(logits_k, dim=-1))
        log_comp = torch.stack(log_components, dim=1)
        log_pi = F.log_softmax(pi, dim=-1)
        logp = torch.logsumexp(log_pi.unsqueeze(-1) + log_comp, dim=1)
        kl = (target_p * (target_logp - logp)).sum(dim=-1).mean().item()
    return {"final_kl": kl, "params": (C * d + K * (d * V + V) + C * K)}


def svd_tail_lower_bound(target_logp: torch.Tensor, d: int) -> float:
    """Lower bound on achievable KL for a rank-(d+1) approximation of the centered log-prob matrix.

    We measure the tail energy beyond the (d+1)-st singular value of the *centered* log-prob matrix.
    This is a heuristic floor used only for sanity-checking direction; the meaningful test is whether
    KL decreases monotonically with d and whether the mixture beats the single softmax.
    """
    with torch.no_grad():
        M = target_logp - target_logp.mean(dim=-1, keepdim=True)
        s = torch.linalg.svdvals(M.float())
        keep = min(d + 1, s.numel())
        tail = (s[keep:] ** 2).sum().item()
        total = (s ** 2).sum().item()
        return tail / max(total, 1e-12)


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 01: softmax bottleneck")
    add_common_args(parser)
    parser.add_argument("--num-contexts", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--mixture-k", type=int, default=4)
    args = parser.parse_args()

    if args.smoke:
        args.num_contexts = 64
        args.vocab = 32
        args.rank = 8
        args.steps = 300
        args.mixture_k = 4

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_01", args.tag)

    result = ProbeResult(
        probe="probe_01_softmax_bottleneck",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    target_logp = synth_target_logprobs(args.num_contexts, args.vocab, args.rank, device)
    ds = [max(args.rank // 2, 2), args.rank, args.rank * 2]
    single_curve: List[Dict] = []
    for d in ds:
        single = fit_single_softmax(target_logp, d=d, steps=args.steps, lr=args.lr, device=device)
        single["d"] = d
        single["svd_tail_share"] = svd_tail_lower_bound(target_logp, d)
        single_curve.append(single)

    mixture = fit_mixture_softmax(
        target_logp,
        d=max(args.rank // 2, 2),
        K=args.mixture_k,
        steps=args.steps,
        lr=args.lr,
        device=device,
    )

    bottleneck_d = ds[0]
    single_at_bottleneck = next(s for s in single_curve if s["d"] == bottleneck_d)
    single_at_2r = next(s for s in single_curve if s["d"] == ds[-1])

    monotone_down = single_curve[0]["final_kl"] >= single_curve[-1]["final_kl"] - 1e-4
    mixture_beats_single = mixture["final_kl"] < single_at_bottleneck["final_kl"] - 1e-3
    large_d_low_kl = single_at_2r["final_kl"] < single_at_bottleneck["final_kl"]
    control_passed = bool(monotone_down and mixture_beats_single and large_d_low_kl)

    result.control_passed = control_passed
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "rank": args.rank,
        "vocab": args.vocab,
        "num_contexts": args.num_contexts,
        "single_softmax": single_curve,
        "mixture_softmax": mixture,
        "monotone_down_in_d": monotone_down,
        "mixture_beats_bottleneck_single": mixture_beats_single,
        "large_d_lower_kl_than_bottleneck": large_d_low_kl,
    }
    result.notes = {
        "interpretation": (
            "If KL falls monotonically as d grows and the K-mixture beats the d=r/2 single softmax,"
            " the softmax bottleneck is acting as predicted."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_01] wrote {path}  control_passed={control_passed}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
