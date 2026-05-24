"""Probe 03 — Rank collapse with depth (pure attention vs residual vs residual+MLP).

Pure stacked self-attention drives the token representation matrix toward rank 1
doubly-exponentially in depth. Residual + MLP slow this. We measure the effective
rank of the output token matrix as a function of layer depth L for three random-init
configs:
    (a) pure attention
    (b) attention + residual
    (c) attention + residual + MLP

Positive control: (a) collapses fast; (b) and (c) retain meaningfully higher rank.
"""
from __future__ import annotations

import argparse
import math
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
    effective_rank,
    gpu_name,
    resolve_device,
    seed_everything,
    write_result,
)


class AttnBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, residual: bool, mlp: bool, mlp_mult: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, bias=False)
        self.residual = residual
        self.mlp = (
            nn.Sequential(nn.Linear(d_model, mlp_mult * d_model), nn.GELU(), nn.Linear(mlp_mult * d_model, d_model))
            if mlp
            else None
        )
        self.norm1 = nn.LayerNorm(d_model) if residual else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if mlp else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = a if not self.residual else self.norm1(x + a)
        if self.mlp is not None:
            x = self.norm2(x + self.mlp(x))
        return x


def measure_rank_curve(
    config: str,
    depth: int,
    d_model: int,
    n_heads: int,
    seq_len: int,
    device: torch.device,
) -> List[Dict]:
    residual = config in {"residual", "residual_mlp"}
    mlp = config == "residual_mlp"
    blocks = nn.ModuleList(
        [AttnBlock(d_model=d_model, n_heads=n_heads, residual=residual, mlp=mlp).to(device) for _ in range(depth)]
    )
    for p in blocks.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, seq_len, d_model, device=device)
    curve = []
    with torch.no_grad():
        h = x
        for layer_idx, block in enumerate(blocks):
            h = block(h)
            rank = effective_rank(h[0])
            curve.append({"layer": layer_idx + 1, "effective_rank": rank})
    return curve


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 03: rank collapse with depth")
    add_common_args(parser)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=64)
    args = parser.parse_args()

    if args.smoke:
        args.depth = 12
        args.d_model = 64
        args.n_heads = 4
        args.seq_len = 32

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_03", args.tag)

    result = ProbeResult(
        probe="probe_03_rank_collapse",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    configs = ["pure", "residual", "residual_mlp"]
    curves: Dict[str, List[Dict]] = {}
    for cfg in configs:
        curves[cfg] = measure_rank_curve(cfg, args.depth, args.d_model, args.n_heads, args.seq_len, device)

    pure_final = curves["pure"][-1]["effective_rank"]
    residual_final = curves["residual"][-1]["effective_rank"]
    mlp_final = curves["residual_mlp"][-1]["effective_rank"]
    upper_bound = min(args.seq_len, args.d_model)
    pure_collapsed = pure_final <= max(2, int(0.25 * upper_bound))
    residual_keeps_rank = residual_final >= max(pure_final + 1, int(0.5 * upper_bound))
    mlp_keeps_rank = mlp_final >= max(pure_final + 1, int(0.5 * upper_bound))
    control_passed = bool(pure_collapsed and residual_keeps_rank and mlp_keeps_rank)

    result.control_passed = control_passed
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "depth": args.depth,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "seq_len": args.seq_len,
        "upper_bound": upper_bound,
        "curves": curves,
        "final": {"pure": pure_final, "residual": residual_final, "residual_mlp": mlp_final},
        "pure_collapsed": pure_collapsed,
        "residual_keeps_rank": residual_keeps_rank,
        "residual_mlp_keeps_rank": mlp_keeps_rank,
    }
    result.notes = {
        "interpretation": (
            "Pure attention's final-layer effective rank should drop near 1; configs with residuals (and MLP)"
            " should retain a meaningful fraction of the upper bound."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_03] wrote {path}  control_passed={control_passed}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
