"""Probe 05 — Fixed compute per token is a representational ceiling.

A fixed-depth transformer's single-pass accuracy on parity collapses as input
length grows past a threshold determined by depth; a scratchpad / chain-of-thought
that exposes intermediate XOR results gives the model serial budget and restores
near-perfect accuracy.

Positive control:
    - Train a small transformer to predict parity from a binary string of length n.
    - Plot single-pass accuracy as n grows; expect a cliff.
    - Train a second transformer on the *scratchpad* variant where intermediate
      cumulative XOR values are part of the target sequence; expect the cliff to
      move outward.
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

# Vocab layout
# 0 = pad, 1 = '0', 2 = '1', 3 = '|', 4 = '=' (end-of-prompt for direct mode)
PAD, ZERO, ONE, BAR, EQ = 0, 1, 2, 3, 4
VOCAB = 5


def make_direct_batch(batch_size: int, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    bits = torch.randint(0, 2, (batch_size, n), device=device)
    parity = bits.sum(dim=-1) % 2  # (B,)
    prompt = bits + 1  # map 0,1 -> 1,2
    eq = torch.full((batch_size, 1), EQ, device=device, dtype=torch.long)
    inputs = torch.cat([prompt, eq], dim=1)  # (B, n+1)
    target_token = (parity + 1).unsqueeze(-1)  # (B, 1)
    return inputs, target_token  # predict next-token at the EQ position


def make_scratchpad_batch(batch_size: int, n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sequence: b1 | c1 b2 | c2 ... bn | cn = parity
    where c_i is cumulative XOR up to position i.
    """
    bits = torch.randint(0, 2, (batch_size, n), device=device)
    cum = bits.cumsum(dim=-1) % 2
    parts: List[torch.Tensor] = []
    for i in range(n):
        parts.append(bits[:, i : i + 1] + 1)  # b_i
        parts.append(torch.full((batch_size, 1), BAR, device=device, dtype=torch.long))
        parts.append(cum[:, i : i + 1] + 1)   # c_i
    eq = torch.full((batch_size, 1), EQ, device=device, dtype=torch.long)
    parts.append(eq)
    seq = torch.cat(parts, dim=1)
    target_token = (cum[:, -1] + 1).unsqueeze(-1)
    return seq, target_token


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, d_model: int, n_heads: int, depth: int, max_len: int):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos(positions)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)


def train_direct(
    n: int,
    depth: int,
    d_model: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_len_pad: int = 8,
) -> Tuple[float, int]:
    max_len = n + max_len_pad
    model = TinyTransformer(VOCAB, d_model=d_model, n_heads=n_heads, depth=depth, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        inputs, target = make_direct_batch(batch_size, n, device)
        logits = model(inputs)
        pred = logits[:, -1, :]
        loss = F.cross_entropy(pred, target.squeeze(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        eval_b = 2048
        inputs, target = make_direct_batch(eval_b, n, device)
        logits = model(inputs)
        pred = logits[:, -1, :].argmax(dim=-1)
        acc = (pred == target.squeeze(-1)).float().mean().item()
    params = sum(p.numel() for p in model.parameters())
    return acc, params


def train_scratchpad(
    n: int,
    depth: int,
    d_model: int,
    n_heads: int,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_len_pad: int = 8,
) -> Tuple[float, int]:
    max_len = 3 * n + max_len_pad
    model = TinyTransformer(VOCAB, d_model=d_model, n_heads=n_heads, depth=depth, max_len=max_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        seq, target = make_scratchpad_batch(batch_size, n, device)
        # Teacher-forced next-token prediction on the whole sequence; loss on the final-token prediction is enough,
        # but we also include the per-step scratchpad supervision since it gives the model the serial signal.
        logits = model(seq)
        pred_final = logits[:, -1, :]
        loss = F.cross_entropy(pred_final, target.squeeze(-1))
        # add scratchpad CE: at each '|' position we want the next token (cumulative XOR) to be ZERO/ONE.
        bar_mask = seq == BAR
        if bar_mask.any():
            B, T, V = logits.shape
            shift_logits = logits[:, :-1, :]
            shift_targets = seq[:, 1:]
            shift_mask = bar_mask[:, :-1]
            scratch_loss = F.cross_entropy(
                shift_logits[shift_mask],
                shift_targets[shift_mask],
            )
            loss = loss + scratch_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        eval_b = 2048
        seq, target = make_scratchpad_batch(eval_b, n, device)
        logits = model(seq)
        pred = logits[:, -1, :].argmax(dim=-1)
        acc = (pred == target.squeeze(-1)).float().mean().item()
    params = sum(p.numel() for p in model.parameters())
    return acc, params


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 05: fixed-compute parity")
    add_common_args(parser)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4, 8, 16, 32])
    args = parser.parse_args()

    if args.smoke:
        args.depth = 2
        args.d_model = 32
        args.n_heads = 4
        args.steps = 400
        args.batch_size = 128
        args.lengths = [4, 8, 16]

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_05", args.tag)

    result = ProbeResult(
        probe="probe_05_fixed_compute",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    direct: List[Dict] = []
    scratch: List[Dict] = []
    for n in args.lengths:
        acc_d, p_d = train_direct(
            n=n,
            depth=args.depth,
            d_model=args.d_model,
            n_heads=args.n_heads,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        acc_s, p_s = train_scratchpad(
            n=n,
            depth=args.depth,
            d_model=args.d_model,
            n_heads=args.n_heads,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        direct.append({"n": n, "accuracy": acc_d, "params": p_d})
        scratch.append({"n": n, "accuracy": acc_s, "params": p_s})

    longest = max(args.lengths)
    direct_at_longest = next(d for d in direct if d["n"] == longest)["accuracy"]
    scratch_at_longest = next(d for d in scratch if d["n"] == longest)["accuracy"]
    direct_drops = direct[0]["accuracy"] - direct_at_longest > 0.15
    scratch_lifts = scratch_at_longest - direct_at_longest > 0.10
    control_passed = bool(direct_drops and scratch_lifts)

    result.control_passed = control_passed
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "lengths": args.lengths,
        "direct": direct,
        "scratchpad": scratch,
        "direct_drops_with_n": direct_drops,
        "scratchpad_lifts_at_longest": scratch_lifts,
    }
    result.notes = {
        "interpretation": (
            "Direct parity accuracy should drop as n grows past a depth-dependent threshold,"
            " and the scratchpad variant should restore accuracy."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_05] wrote {path}  control_passed={control_passed}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
