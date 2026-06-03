"""Probe 04 — Partition function is a removable per-step tax.

Replace softmax attention with sigmoid (bounded but unnormalised) or linear /
kernel attention. If the partition function is a removable per-step tax, the
non-softmax variants should match task loss within CI on a small copy / shift
task, both compute- and param-matched.

Task: synthetic next-token sequence on a small vocab where the model has to
shift / copy tokens at fixed offsets. Tiny, so we can fit each variant in
under a minute on a single GPU.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

VOCAB = 16


def make_copy_batch(batch: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Source tokens at positions [0, n), target at position t is source[t-1] (a right-shift copy task)."""
    n = seq_len // 2
    src = torch.randint(1, VOCAB, (batch, n), device=device)
    sep = torch.zeros(batch, 1, device=device, dtype=torch.long)
    target = torch.cat([sep, src[:, :-1]], dim=1)
    inputs = torch.cat([src, sep, target], dim=1)
    labels = torch.cat([torch.full((batch, n + 1), -100, device=device, dtype=torch.long), src], dim=1)
    return inputs, labels


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, attn_kind: str):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.h = n_heads
        self.dh = d // n_heads
        self.attn_kind = attn_kind
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        if self.attn_kind == "softmax":
            attn = F.softmax(scores, dim=-1)
        elif self.attn_kind == "sigmoid":
            attn = torch.sigmoid(scores)
            attn = attn.masked_fill(mask, 0.0)
        elif self.attn_kind == "linear":
            phi_q = F.elu(q) + 1
            phi_k = F.elu(k) + 1
            # causal linear attention via cumulative sums
            kv = torch.einsum("bhtd,bhte->bhtde", phi_k, v).cumsum(dim=2)
            denom = phi_k.cumsum(dim=2)
            num = torch.einsum("bhtd,bhtde->bhte", phi_q, kv)
            den = torch.einsum("bhtd,bhtd->bht", phi_q, denom).clamp_min(1e-6).unsqueeze(-1)
            out = num / den
            out = out.transpose(1, 2).reshape(B, T, self.d)
            return self.out(out)
        else:
            raise ValueError(self.attn_kind)
        out = (attn @ v).transpose(1, 2).reshape(B, T, self.d)
        return self.out(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int, n_heads: int, depth: int, max_len: int, attn_kind: str):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, attn_kind) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(pos)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


def train_and_eval(
    attn_kind: str,
    d: int,
    n_heads: int,
    depth: int,
    seq_len: int,
    steps: int,
    batch: int,
    lr: float,
    device: torch.device,
) -> Dict:
    model = TinyLM(VOCAB, d=d, n_heads=n_heads, depth=depth, max_len=seq_len, attn_kind=attn_kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: List[float] = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        x, y = make_copy_batch(batch, seq_len, device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_wall = time.time() - t0
    model.eval()
    with torch.no_grad():
        x, y = make_copy_batch(2048, seq_len, device)
        logits = model(x)
        eval_loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), ignore_index=-100).item()
        valid = (y != -100)
        acc = (logits.argmax(-1)[valid] == y[valid]).float().mean().item()
    params = sum(p.numel() for p in model.parameters())
    return {"eval_loss": eval_loss, "eval_acc": acc, "params": params, "train_wall_s": train_wall, "loss_curve": losses}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 04: partition removable")
    add_common_args(parser)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=33)  # 2*16+1
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    if args.smoke:
        args.d = 32
        args.steps = 500
        args.batch = 64

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_04", args.tag)

    result = ProbeResult(
        probe="probe_04_partition_removable",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    runs: Dict[str, Dict] = {}
    for kind in ("softmax", "sigmoid", "linear"):
        r = train_and_eval(
            attn_kind=kind,
            d=args.d,
            n_heads=args.n_heads,
            depth=args.depth,
            seq_len=args.seq_len,
            steps=args.steps,
            batch=args.batch,
            lr=args.lr,
            device=device,
        )
        r["attn_kind"] = kind
        # keep loss curve compact: keep first/last + 1-in-10
        curve = r.pop("loss_curve")
        r["loss_curve_decimated"] = curve[::max(1, len(curve) // 50)]
        runs[kind] = r

    soft = runs["softmax"]["eval_loss"]
    sig = runs["sigmoid"]["eval_loss"]
    lin = runs["linear"]["eval_loss"]
    # "within CI": within 15% of softmax loss is our smoke-grade tolerance
    tol = 0.15 * abs(soft) + 1e-3
    sig_close = abs(sig - soft) <= tol
    lin_close = abs(lin - soft) <= tol
    same_params = abs(runs["softmax"]["params"] - runs["sigmoid"]["params"]) < 10 and abs(
        runs["softmax"]["params"] - runs["linear"]["params"]
    ) < 10
    control_passed = bool(same_params)  # control is param-matching; verdict is the loss comparison

    result.control_passed = control_passed
    if control_passed and sig_close and lin_close:
        result.verdict = "CONFIRMED_REMOVABLE"
    elif control_passed:
        result.verdict = "PARTIAL_REMOVABLE"
    else:
        result.verdict = "CONTROL_FAIL"
    result.metrics = {
        "depth": args.depth,
        "d": args.d,
        "n_heads": args.n_heads,
        "seq_len": args.seq_len,
        "runs": runs,
        "tolerance": tol,
        "sigmoid_within_tolerance": sig_close,
        "linear_within_tolerance": lin_close,
        "param_matched": same_params,
    }
    result.notes = {
        "interpretation": (
            "If sigmoid and linear attention match softmax eval loss within tolerance under the same"
            " param count, softmax's partition function is a removable per-step tax for this task."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_04] wrote {path}  verdict={result.verdict}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
