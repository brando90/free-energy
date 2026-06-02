#!/usr/bin/env python3
"""Toy Energy-Based Transformer baseline.

This is a deliberately small implementation of the core EBT loop from
arXiv:2507.02092:

1. embed a context sequence and a candidate prediction distribution;
2. score the pair with a transformer scalar energy;
3. update the candidate by gradient descent on that energy;
4. train through the unrolled refinement with a reconstruction loss.

The toy task is binary sequence transduction. It is cheap enough to run on CPU
while still exercising the second-order training path that makes EBTs different
from ordinary feed-forward transformers.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "toy"


@dataclass(frozen=True)
class ToyMetrics:
    loss: float
    token_accuracy: float
    sequence_accuracy: float
    energy_start: float | None = None
    energy_end: float | None = None
    energy_drop: float | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_targets(context: torch.Tensor) -> torch.Tensor:
    """A small structured target with local and global dependencies."""

    left = torch.roll(context, shifts=1, dims=1)
    right = torch.roll(context, shifts=-1, dims=1)
    parity = (context[:, ::2].sum(dim=1, keepdim=True) % 2).long()
    alternating_mask = (torch.arange(context.shape[1], device=context.device) % 2).view(1, -1)
    local_rule = context ^ left ^ right
    return local_rule ^ (parity & alternating_mask)


def make_dataset(num_examples: int, seq_len: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    context = torch.randint(0, 2, (num_examples, seq_len), generator=generator, dtype=torch.long)
    target = make_targets(context)
    return context, target


class ManualSelfAttention(nn.Module):
    """Small attention module with second-order gradients on CPU.

    PyTorch's fused scaled-dot-product attention is fast, but the CPU flash path
    currently does not expose the double backward EBT training needs. This
    unfused version is tiny and slow, but it keeps the smoke experiment honest.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        hidden = torch.matmul(attn, v).transpose(1, 2).reshape(batch, seq_len, dim)
        return self.out(hidden)


class ManualTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ManualSelfAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyTransformerBackbone(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [ManualTransformerBlock(dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class DirectTransformerBaseline(nn.Module):
    """Feed-forward transformer that predicts all target bits in one pass."""

    def __init__(self, seq_len: int, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(2, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        self.backbone = TinyTransformerBackbone(hidden_dim, num_layers, num_heads)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(context) + self.position[:, : context.shape[1]]
        hidden = self.backbone(hidden)
        return self.head(hidden)


class EnergyBasedTransformer(nn.Module):
    """Tiny autoregressive-style EBT over a binary candidate distribution."""

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        max_steps: int,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.max_steps = max_steps
        self.context_embedding = nn.Embedding(2, hidden_dim)
        self.candidate_to_embedding = nn.Linear(2, hidden_dim, bias=False)
        self.position = nn.Parameter(torch.zeros(1, 2 * seq_len, hidden_dim))
        self.segment = nn.Parameter(torch.zeros(2, hidden_dim))
        self.step_embedding = nn.Embedding(max_steps + 2, hidden_dim)
        self.backbone = TinyTransformerBackbone(hidden_dim, num_layers, num_heads)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)
        nn.init.normal_(self.segment, mean=0.0, std=0.02)

    def energy(self, context: torch.Tensor, candidate_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
        probs = F.softmax(candidate_logits, dim=-1)
        context_hidden = self.context_embedding(context)
        candidate_hidden = self.candidate_to_embedding(probs)

        hidden = torch.cat([context_hidden, candidate_hidden], dim=1)
        hidden = hidden + self.position[:, : hidden.shape[1]]
        hidden = hidden.clone()
        hidden[:, : self.seq_len] = hidden[:, : self.seq_len] + self.segment[0]
        hidden[:, self.seq_len :] = hidden[:, self.seq_len :] + self.segment[1]
        step = min(max(step_idx, 0), self.max_steps + 1)
        hidden = hidden + self.step_embedding.weight[step].view(1, 1, -1)

        encoded = self.backbone(hidden)
        candidate_encoded = encoded[:, self.seq_len :]
        per_position_energy = self.energy_head(candidate_encoded).squeeze(-1)
        return per_position_energy.mean(dim=1)

    def initial_logits(
        self,
        batch_size: int,
        device: torch.device,
        init_scale: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.seq_len,
            2,
            device=device,
            generator=generator,
        ) * init_scale

    def refine(
        self,
        context: torch.Tensor,
        num_steps: int,
        alpha: float,
        init_scale: float,
        create_graph: bool,
        detach_between_steps: bool,
        grad_clip: float,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        logits = self.initial_logits(context.shape[0], context.device, init_scale, generator)
        energies: list[torch.Tensor] = []

        for step in range(num_steps):
            if detach_between_steps:
                logits = logits.detach()
            logits = logits.requires_grad_(True)
            energy = self.energy(context, logits, step)
            energies.append(energy)
            grad = torch.autograd.grad(energy.sum(), logits, create_graph=create_graph)[0]
            if grad_clip > 0:
                grad = grad.clamp(min=-grad_clip, max=grad_clip)
            logits = logits - alpha * grad

        logits = logits.requires_grad_(True)
        energies.append(self.energy(context, logits, num_steps))
        return logits, energies


def batch_indices(num_examples: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    order = torch.randperm(num_examples, generator=generator)
    return [order[start : start + batch_size] for start in range(0, num_examples, batch_size)]


def classification_metrics(logits: torch.Tensor, target: torch.Tensor) -> tuple[float, float, float]:
    loss = F.cross_entropy(logits.reshape(-1, 2), target.reshape(-1)).detach()
    pred = logits.argmax(dim=-1)
    token_acc = (pred == target).float().mean()
    seq_acc = (pred == target).all(dim=1).float().mean()
    return float(loss.cpu()), float(token_acc.cpu()), float(seq_acc.cpu())


@torch.no_grad()
def evaluate_direct(
    model: DirectTransformerBaseline,
    context: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
) -> ToyMetrics:
    model.eval()
    losses = []
    token_accs = []
    seq_accs = []
    for start in range(0, context.shape[0], batch_size):
        logits = model(context[start : start + batch_size])
        loss, token_acc, seq_acc = classification_metrics(logits, target[start : start + batch_size])
        losses.append(loss)
        token_accs.append(token_acc)
        seq_accs.append(seq_acc)
    return ToyMetrics(
        loss=sum(losses) / len(losses),
        token_accuracy=sum(token_accs) / len(token_accs),
        sequence_accuracy=sum(seq_accs) / len(seq_accs),
    )


def evaluate_ebt(
    model: EnergyBasedTransformer,
    context: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    num_steps: int,
    alpha: float,
    init_scale: float,
    detach_between_steps: bool,
    grad_clip: float,
    samples: int,
    seed: int,
) -> ToyMetrics:
    model.eval()
    losses = []
    token_accs = []
    seq_accs = []
    energy_starts = []
    energy_ends = []

    with torch.enable_grad():
        for start in range(0, context.shape[0], batch_size):
            context_batch = context[start : start + batch_size]
            target_batch = target[start : start + batch_size]
            best_logits = None
            best_energy = None
            best_energies: list[torch.Tensor] | None = None
            for sample_idx in range(samples):
                generator = torch.Generator(device=context.device).manual_seed(seed + 10_000 * sample_idx + start)
                logits, energies = model.refine(
                    context_batch,
                    num_steps=num_steps,
                    alpha=alpha,
                    init_scale=init_scale,
                    create_graph=False,
                    detach_between_steps=detach_between_steps,
                    grad_clip=grad_clip,
                    generator=generator,
                )
                final_energy = energies[-1].detach()
                if best_energy is None:
                    best_energy = final_energy
                    best_logits = logits.detach()
                    best_energies = [energy.detach() for energy in energies]
                    continue
                take = final_energy < best_energy
                best_energy = torch.where(take, final_energy, best_energy)
                best_logits = torch.where(take.view(-1, 1, 1), logits.detach(), best_logits)
                best_energies = [
                    torch.where(take, new.detach(), old)
                    for old, new in zip(best_energies or [], energies)
                ]

            assert best_logits is not None
            assert best_energies is not None
            loss, token_acc, seq_acc = classification_metrics(best_logits, target_batch)
            losses.append(loss)
            token_accs.append(token_acc)
            seq_accs.append(seq_acc)
            energy_starts.append(float(best_energies[0].mean().cpu()))
            energy_ends.append(float(best_energies[-1].mean().cpu()))

    energy_start = sum(energy_starts) / len(energy_starts)
    energy_end = sum(energy_ends) / len(energy_ends)
    return ToyMetrics(
        loss=sum(losses) / len(losses),
        token_accuracy=sum(token_accs) / len(token_accs),
        sequence_accuracy=sum(seq_accs) / len(seq_accs),
        energy_start=energy_start,
        energy_end=energy_end,
        energy_drop=energy_start - energy_end,
    )


def train_direct(
    args: argparse.Namespace,
    train_context: torch.Tensor,
    train_target: torch.Tensor,
    test_context: torch.Tensor,
    test_target: torch.Tensor,
) -> dict[str, Any]:
    model = DirectTransformerBaseline(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(train_context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.direct_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 101)
    history = []
    started = time.time()

    for epoch in range(1, args.direct_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for idx in batch_indices(train_context.shape[0], args.batch_size, generator):
            idx = idx.to(train_context.device)
            logits = model(train_context[idx])
            loss = F.cross_entropy(logits.reshape(-1, 2), train_target[idx].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * idx.numel()
            total_count += idx.numel()
        if epoch == 1 or epoch == args.direct_epochs or epoch % args.log_every == 0:
            history.append({"epoch": epoch, "train_loss": total_loss / max(1, total_count)})

    train_metrics = evaluate_direct(model, train_context, train_target, args.eval_batch_size)
    test_metrics = evaluate_direct(model, test_context, test_target, args.eval_batch_size)
    return {
        "model": "direct_transformer",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.time() - started,
        "history": history,
        "train": asdict(train_metrics),
        "test": asdict(test_metrics),
    }


def train_ebt(
    args: argparse.Namespace,
    train_context: torch.Tensor,
    train_target: torch.Tensor,
    test_context: torch.Tensor,
    test_target: torch.Tensor,
) -> dict[str, Any]:
    model = EnergyBasedTransformer(
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_steps=max(args.ebt_steps, max(args.eval_steps)),
    ).to(train_context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.ebt_lr, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 202)
    history = []
    started = time.time()

    for epoch in range(1, args.ebt_epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for idx in batch_indices(train_context.shape[0], args.batch_size, generator):
            idx = idx.to(train_context.device)
            logits, energies = model.refine(
                train_context[idx],
                num_steps=args.ebt_steps,
                alpha=args.alpha,
                init_scale=args.init_scale,
                create_graph=True,
                detach_between_steps=args.detach_between_steps,
                grad_clip=args.prediction_grad_clip,
            )
            loss = F.cross_entropy(logits.reshape(-1, 2), train_target[idx].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * idx.numel()
            total_count += idx.numel()
        if epoch == 1 or epoch == args.ebt_epochs or epoch % args.log_every == 0:
            energy_start = float(energies[0].detach().mean().cpu())
            energy_end = float(energies[-1].detach().mean().cpu())
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, total_count),
                    "last_batch_energy_drop": energy_start - energy_end,
                }
            )

    eval_by_steps = {}
    for steps in args.eval_steps:
        eval_by_steps[str(steps)] = {
            "train": asdict(
                evaluate_ebt(
                    model,
                    train_context,
                    train_target,
                    args.eval_batch_size,
                    steps,
                    args.alpha,
                    args.init_scale,
                    args.detach_between_steps,
                    args.prediction_grad_clip,
                    args.eval_samples,
                    args.seed + 303,
                )
            ),
            "test": asdict(
                evaluate_ebt(
                    model,
                    test_context,
                    test_target,
                    args.eval_batch_size,
                    steps,
                    args.alpha,
                    args.init_scale,
                    args.detach_between_steps,
                    args.prediction_grad_clip,
                    args.eval_samples,
                    args.seed + 404,
                )
            ),
        }

    selected = eval_by_steps[str(args.ebt_steps)]
    return {
        "model": "energy_based_transformer",
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "train_seconds": time.time() - started,
        "alpha": args.alpha,
        "train_steps": args.ebt_steps,
        "eval_samples": args.eval_samples,
        "history": history,
        "train": selected["train"],
        "test": selected["test"],
        "eval_by_steps": eval_by_steps,
    }


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    direct = report["results"]["direct_transformer"]["test"]
    ebt = report["results"]["energy_based_transformer"]["test"]
    step_rows = []
    for steps, metrics_by_split in report["results"]["energy_based_transformer"]["eval_by_steps"].items():
        metrics = metrics_by_split["test"]
        step_rows.append(
            "| {steps} | {loss:.4f} | {tok:.3f} | {seq:.3f} | {drop:.4f} |".format(
                steps=steps,
                loss=metrics["loss"],
                tok=metrics["token_accuracy"],
                seq=metrics["sequence_accuracy"],
                drop=metrics["energy_drop"],
            )
        )

    lines = [
        "# Toy EBT Baseline Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(report["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Held-Out Test Metrics",
        "",
        "| model | loss | token acc | sequence acc | energy drop |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| direct_transformer | {loss:.4f} | {tok:.3f} | {seq:.3f} | n/a |".format(
            loss=direct["loss"],
            tok=direct["token_accuracy"],
            seq=direct["sequence_accuracy"],
        ),
        "| energy_based_transformer | {loss:.4f} | {tok:.3f} | {seq:.3f} | {drop:.4f} |".format(
            loss=ebt["loss"],
            tok=ebt["token_accuracy"],
            seq=ebt["sequence_accuracy"],
            drop=ebt["energy_drop"],
        ),
        "",
        "## EBT Test-Time Compute Sweep",
        "",
        "| refinement steps | loss | token acc | sequence acc | energy drop |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *step_rows,
        "",
        "The direct transformer is the feed-forward baseline. The EBT result uses the same",
        "task but predicts by descending the learned energy landscape over candidate logits.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.ebt_steps not in args.eval_steps:
        args.eval_steps = sorted({*args.eval_steps, args.ebt_steps})
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_context, train_target = make_dataset(args.num_train, args.seq_len, args.seed)
    test_context, test_target = make_dataset(args.num_test, args.seq_len, args.seed + 1)
    train_context = train_context.to(device)
    train_target = train_target.to(device)
    test_context = test_context.to(device)
    test_target = test_target.to(device)

    direct = train_direct(args, train_context, train_target, test_context, test_target)
    ebt = train_ebt(args, train_context, train_target, test_context, test_target)

    ebt_test = ebt["test"]
    status = "pass" if math.isfinite(ebt_test["loss"]) and ebt_test["token_accuracy"] > 0.55 else "warn"
    report = {
        "status": status,
        "config": jsonable_args(args) | {"device_resolved": str(device)},
        "dataset": {
            "target_rule": "local xor neighbors with an even-position global parity bit",
            "num_train": args.num_train,
            "num_test": args.num_test,
        },
        "results": {
            "direct_transformer": direct,
            "energy_based_transformer": ebt,
        },
    }

    json_path = args.output_dir / f"{args.tag}_toy_report.json"
    md_path = args.output_dir / f"{args.tag}_toy_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report, md_path)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)

    if args.require_ebt_signal and status != "pass":
        raise RuntimeError("EBT toy run did not clear the smoke threshold")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="local")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seq-len", type=int, default=4)
    parser.add_argument("--num-train", type=int, default=256)
    parser.add_argument("--num-test", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--direct-epochs", type=int, default=20)
    parser.add_argument("--ebt-epochs", type=int, default=60)
    parser.add_argument("--direct-lr", type=float, default=0.003)
    parser.add_argument("--ebt-lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--ebt-steps", type=int, default=2)
    parser.add_argument("--eval-steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--eval-samples", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--init-scale", type=float, default=0.0)
    parser.add_argument("--prediction-grad-clip", type=float, default=1.0)
    parser.add_argument("--detach-between-steps", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--require-ebt-signal", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
