"""Probe 07 — Reversal curse.

A tiny LM trained on facts of the form "A is B" should fail to reliably
produce "B is A" when queried in the reverse direction. We synthesise unique
token pairs (A_i, B_i), train forward-only with cross-entropy, then evaluate
forward and reverse accuracy on a held-out reverse-only query set.

Forward acc >> reverse acc is the predicted asymmetry.
"""
from __future__ import annotations

import argparse
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

# Vocab layout:
# 0 = pad, 1 = IS, 2 = QUERY (== EQ), 3 = REV_QUERY
# 4..4+N-1 = A tokens, 4+N..4+2N-1 = B tokens
PAD, IS, QF, QR = 0, 1, 2, 3
SPECIALS = 4


def build_facts(n_facts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    a_ids = SPECIALS + torch.arange(n_facts)
    b_ids = SPECIALS + n_facts + torch.arange(n_facts)
    return a_ids, b_ids


def make_forward_train_sequences(a_ids: torch.Tensor, b_ids: torch.Tensor) -> torch.Tensor:
    # sequence: [A_i, IS, B_i, QF, A_i, B_i] with loss only on the final B_i token.
    n = a_ids.size(0)
    is_tok = torch.full((n,), IS)
    qf = torch.full((n,), QF)
    seqs = torch.stack([a_ids, is_tok, b_ids, qf, a_ids, b_ids], dim=1)
    return seqs  # (n, 6)


def make_reverse_eval_sequences(a_ids: torch.Tensor, b_ids: torch.Tensor) -> torch.Tensor:
    # sequence: [B_i, IS, A_i, QR, B_i] with model asked to produce A_i as next token.
    n = a_ids.size(0)
    is_tok = torch.full((n,), IS)
    qr = torch.full((n,), QR)
    seqs = torch.stack([b_ids, is_tok, a_ids, qr, b_ids], dim=1)
    return seqs  # (n, 5)


def make_forward_eval_sequences(a_ids: torch.Tensor, b_ids: torch.Tensor) -> torch.Tensor:
    n = a_ids.size(0)
    is_tok = torch.full((n,), IS)
    qf = torch.full((n,), QF)
    seqs = torch.stack([a_ids, is_tok, b_ids, qf, a_ids], dim=1)
    return seqs


class TinyLM(nn.Module):
    def __init__(self, vocab: int, d: int, n_heads: int, depth: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=n_heads,
            dim_feedforward=4 * d,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.tok(x) + self.pos(positions)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(h, mask=mask)
        return self.head(h)


def train_forward_only(model: nn.Module, train_seqs: torch.Tensor, steps: int, lr: float, batch: int, device: torch.device) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_seqs = train_seqs.to(device)
    n = train_seqs.size(0)
    losses: List[float] = []
    model.train()
    for step in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        seq = train_seqs[idx]
        logits = model(seq)
        # next-token CE on the WHOLE sequence is fine: the QF position is the relevant supervised target,
        # but training on the full sequence makes the model learn the A IS B pattern.
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def eval_completion(model: nn.Module, seqs: torch.Tensor, target_ids: torch.Tensor, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        seqs = seqs.to(device)
        target_ids = target_ids.to(device)
        logits = model(seqs)
        # predict the token AT the last position via next-token logits at the second-to-last position
        pred = logits[:, -1, :].argmax(dim=-1)
    return (pred == target_ids).float().mean().item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe 07: reversal curse")
    add_common_args(parser)
    parser.add_argument("--n-facts", type=int, default=128)
    parser.add_argument("--d", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    args = parser.parse_args()

    if args.smoke:
        args.n_facts = 48
        args.steps = 1500
        args.d = 32

    device = resolve_device(args.device)
    seed_everything(args.seed)
    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir("probe_07", args.tag)

    result = ProbeResult(
        probe="probe_07_reversal_curse",
        tag=args.tag,
        seed=args.seed,
        device=str(device),
        started_at=time.time(),
        gpu_name=gpu_name(),
    )

    vocab = SPECIALS + 2 * args.n_facts
    a_ids, b_ids = build_facts(args.n_facts)

    train_seqs = make_forward_train_sequences(a_ids, b_ids)
    fwd_eval_seqs = make_forward_eval_sequences(a_ids, b_ids)  # last position = A_i, predict B_i
    # but we want prediction of B_i: model's next-token prediction at position 3 (QF) should be A_i,
    # at position 4 (A_i) should be B_i. We use the existing forward 6-token train form for eval too:
    fwd_full = make_forward_train_sequences(a_ids, b_ids)  # last token B_i, predict B_i from prefix
    rev_eval_seqs = make_reverse_eval_sequences(a_ids, b_ids)  # last position = B_i, predict A_i

    model = TinyLM(vocab=vocab, d=args.d, n_heads=args.n_heads, depth=args.depth, max_len=train_seqs.size(1)).to(device)
    loss_curve = train_forward_only(model, train_seqs, args.steps, args.lr, args.batch, device)

    # Forward eval: feed [A,IS,B,QF,A] and ask for B_i
    fwd_eval_in = fwd_full[:, :-1]  # (n,5) ending in A_i
    fwd_eval_target = b_ids
    fwd_acc = eval_completion(model, fwd_eval_in, fwd_eval_target, device)

    # Reverse eval: feed [B,IS,A,QR,B] and ask for A_i
    rev_eval_target = a_ids
    rev_acc = eval_completion(model, rev_eval_seqs, rev_eval_target, device)

    chance = 1.0 / vocab
    control_passed = (fwd_acc > 0.8) and (rev_acc < max(fwd_acc - 0.4, 5 * chance))

    result.control_passed = bool(control_passed)
    result.verdict = "CONTROL_PASS" if control_passed else "CONTROL_FAIL"
    result.metrics = {
        "n_facts": args.n_facts,
        "vocab": vocab,
        "chance": chance,
        "forward_accuracy": fwd_acc,
        "reverse_accuracy": rev_acc,
        "final_train_loss": loss_curve[-1] if loss_curve else None,
        "loss_curve_decimated": loss_curve[:: max(1, len(loss_curve) // 50)],
    }
    result.notes = {
        "interpretation": (
            "Forward eval should be near 1.0 (model has memorised A->B). Reverse eval should be near"
            " chance: the model never sees B preceding A in training."
        )
    }
    path = write_result(result, out_dir)
    print(f"[probe_07] wrote {path}  fwd_acc={fwd_acc:.3f}  rev_acc={rev_acc:.3f}  control_passed={control_passed}")
    return 0 if control_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
