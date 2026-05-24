#!/usr/bin/env python3
"""Finite-support toy EBM training experiment.

The experiment trains conditional energy models

    p_theta(x | c) = exp(-E_theta(c, x)) / Z_theta(c)

on a synthetic distribution where the candidate space is small enough to
enumerate exactly. This makes the EBM positive/negative phase update directly
checkable before adding approximate samplers.
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
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results"
MODEL_NAMES = ("linear", "mlp", "cnn", "resnet", "transformer")


@dataclass(frozen=True)
class Metrics:
    kl_pstar_model: float
    tv_distance: float
    nll_pstar: float
    target_mode_rank_mean: float
    target_mode_rank_median: float
    mode_match_rate: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def enumerate_binary_support(seq_len: int, device: torch.device | str = "cpu") -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if seq_len > 20:
        raise ValueError("exact enumeration is too large for seq_len > 20")
    device = torch.device(device)
    values = torch.arange(2**seq_len, device=device, dtype=torch.long)
    offsets = torch.arange(seq_len, device=device, dtype=torch.long)
    return ((values[:, None] >> offsets[None, :]) & 1).long()


def make_task_bank(num_tasks: int, seq_len: int, seed: int) -> torch.Tensor:
    support = enumerate_binary_support(seq_len, "cpu")
    if num_tasks > support.shape[0]:
        raise ValueError(f"num_tasks={num_tasks} exceeds support size={support.shape[0]}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(support.shape[0], generator=generator)[:num_tasks]
    return support[indices].clone()


def target_energy_table(tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Hidden teacher energy E_star(c, x), lower is more data-like."""

    c = tasks[:, None, :].long()
    x = candidates[None, :, :].long()
    seq_len = tasks.shape[-1]

    copy_score = (x == c).float().mean(dim=-1)
    endpoint_score = ((x[:, :, 0] == c[:, :, 0]) & (x[:, :, -1] == c[:, :, -1])).float()
    parity_score = ((x.sum(dim=-1) % 2) == (c.sum(dim=-1) % 2)).float()

    if seq_len > 1:
        shift_score = (x[:, :, 1:] == c[:, :, :-1]).float().mean(dim=-1)
        smooth_score = (x[:, :, 1:] == x[:, :, :-1]).float().mean(dim=-1)
    else:
        shift_score = torch.zeros_like(copy_score)
        smooth_score = torch.zeros_like(copy_score)

    return -(
        2.0 * copy_score
        + 1.0 * shift_score
        + 0.7 * smooth_score
        + 0.8 * parity_score
        + 0.5 * endpoint_score
    )


def target_log_probs(
    tasks: torch.Tensor,
    support: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    target_energy = target_energy_table(tasks, support) / temperature
    return -target_energy - torch.logsumexp(-target_energy, dim=-1, keepdim=True)


def pair_token_ids(tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return tasks.long() * 2 + candidates.long()


class LinearEnergy(nn.Module):
    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.position_token_energy = nn.Parameter(torch.zeros(seq_len, 4))
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.position_token_energy, mean=0.0, std=0.02)

    def forward(self, tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        tokens = pair_token_ids(tasks, candidates)
        positions = torch.arange(self.seq_len, device=tokens.device)
        return self.position_token_energy[positions[None, :], tokens].sum(dim=-1) + self.bias


class MLPEnergy(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(seq_len * 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        tokens = pair_token_ids(tasks, candidates)
        features = F.one_hot(tokens, num_classes=4).float().reshape(tokens.shape[0], -1)
        return self.net(features).squeeze(-1)


class CNNEnergy(nn.Module):
    def __init__(self, _seq_len: int, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, hidden_dim)
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        tokens = pair_token_ids(tasks, candidates)
        hidden = self.embedding(tokens).transpose(1, 2)
        hidden = self.net(hidden).mean(dim=-1)
        return self.head(hidden).squeeze(-1)


class ResidualConvBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(1, width)
        self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.gelu(x + residual)


class ResNetEnergy(nn.Module):
    def __init__(self, _seq_len: int, hidden_dim: int, num_blocks: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(4, hidden_dim)
        self.stem = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(ResidualConvBlock(hidden_dim) for _ in range(num_blocks)))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        tokens = pair_token_ids(tasks, candidates)
        hidden = self.embedding(tokens).transpose(1, 2)
        hidden = F.gelu(self.stem(hidden))
        hidden = self.blocks(hidden).mean(dim=-1)
        return self.head(hidden).squeeze(-1)


class TransformerEnergy(nn.Module):
    def __init__(self, seq_len: int, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.embedding = nn.Embedding(4, hidden_dim)
        self.position = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, tasks: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        tokens = pair_token_ids(tasks, candidates)
        hidden = self.embedding(tokens) + self.position[:, : tokens.shape[1], :]
        hidden = self.encoder(hidden).mean(dim=1)
        return self.head(hidden).squeeze(-1)


def build_model(name: str, seq_len: int, hidden_dim: int, num_layers: int, num_heads: int) -> nn.Module:
    if name == "linear":
        return LinearEnergy(seq_len)
    if name == "mlp":
        return MLPEnergy(seq_len, hidden_dim)
    if name == "cnn":
        return CNNEnergy(seq_len, hidden_dim)
    if name == "resnet":
        return ResNetEnergy(seq_len, hidden_dim, num_layers)
    if name == "transformer":
        return TransformerEnergy(seq_len, hidden_dim, num_layers, num_heads)
    raise ValueError(f"unknown model: {name}")


def energy_table_for_tasks(
    model: nn.Module,
    tasks: torch.Tensor,
    support: torch.Tensor,
    pair_batch_size: int,
) -> torch.Tensor:
    num_tasks, seq_len = tasks.shape
    support_size = support.shape[0]
    task_flat = tasks[:, None, :].expand(num_tasks, support_size, seq_len).reshape(-1, seq_len)
    cand_flat = support[None, :, :].expand(num_tasks, support_size, seq_len).reshape(-1, seq_len)

    chunks = []
    for start in range(0, task_flat.shape[0], pair_batch_size):
        end = min(start + pair_batch_size, task_flat.shape[0])
        chunks.append(model(task_flat[start:end], cand_flat[start:end]))
    return torch.cat(chunks, dim=0).reshape(num_tasks, support_size)


def exact_ebm_loss(
    model: nn.Module,
    tasks: torch.Tensor,
    support: torch.Tensor,
    log_probs_star: torch.Tensor,
    pair_batch_size: int,
) -> torch.Tensor:
    energies = energy_table_for_tasks(model, tasks, support, pair_batch_size)
    probs_star = log_probs_star.exp()
    data_energy = (probs_star * energies).sum(dim=-1)
    log_z = torch.logsumexp(-energies, dim=-1)
    return (data_energy + log_z).mean()


def metrics_from_log_probs(log_probs_star: torch.Tensor, log_probs_model: torch.Tensor) -> Metrics:
    probs_star = log_probs_star.exp()
    probs_model = log_probs_model.exp()

    kl = (probs_star * (log_probs_star - log_probs_model)).sum(dim=-1)
    tv = 0.5 * (probs_star - probs_model).abs().sum(dim=-1)
    nll = -(probs_star * log_probs_model).sum(dim=-1)

    target_modes = log_probs_star.argmax(dim=-1)
    model_modes = log_probs_model.argmax(dim=-1)
    order = torch.argsort(log_probs_model, dim=-1, descending=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(1, order.shape[1] + 1, device=order.device)[None, :]
    ranks.scatter_(1, order, rank_values.expand_as(order))
    target_mode_ranks = ranks.gather(1, target_modes[:, None]).squeeze(1).float()

    return Metrics(
        kl_pstar_model=float(kl.mean().detach().cpu().item()),
        tv_distance=float(tv.mean().detach().cpu().item()),
        nll_pstar=float(nll.mean().detach().cpu().item()),
        target_mode_rank_mean=float(target_mode_ranks.mean().detach().cpu().item()),
        target_mode_rank_median=float(target_mode_ranks.median().detach().cpu().item()),
        mode_match_rate=float((model_modes == target_modes).float().mean().detach().cpu().item()),
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    tasks: torch.Tensor,
    support: torch.Tensor,
    log_probs_star: torch.Tensor,
    pair_batch_size: int,
) -> Metrics:
    model.eval()
    energies = energy_table_for_tasks(model, tasks, support, pair_batch_size)
    log_probs_model = -energies - torch.logsumexp(-energies, dim=-1, keepdim=True)
    return metrics_from_log_probs(log_probs_star, log_probs_model)


def uniform_metrics(log_probs_star: torch.Tensor) -> Metrics:
    support_size = log_probs_star.shape[1]
    log_uniform = torch.full_like(log_probs_star, -math.log(support_size))
    return metrics_from_log_probs(log_probs_star, log_uniform)


def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    train_tasks: torch.Tensor,
    train_log_probs: torch.Tensor,
    test_tasks: torch.Tensor,
    test_log_probs: torch.Tensor,
    support: torch.Tensor,
    device: torch.device,
    model_seed: int,
) -> dict[str, Any]:
    set_seed(model_seed)
    model = build_model(
        model_name,
        seq_len=args.seq_len,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    indices = torch.arange(train_tasks.shape[0], device=device)
    generator = torch.Generator(device="cpu").manual_seed(model_seed + 10_000)
    history: list[dict[str, float]] = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(train_tasks.shape[0], generator=generator, device="cpu").to(device)
        total_loss = 0.0
        total_count = 0
        for start in range(0, train_tasks.shape[0], args.batch_size):
            batch_idx = indices[order[start : start + args.batch_size]]
            loss = exact_ebm_loss(
                model,
                train_tasks[batch_idx],
                support,
                train_log_probs[batch_idx],
                args.pair_batch_size,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * batch_idx.numel()
            total_count += batch_idx.numel()

        mean_loss = total_loss / max(1, total_count)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            history.append({"epoch": float(epoch), "train_loss": mean_loss})

    train_metrics = evaluate_model(model, train_tasks, support, train_log_probs, args.pair_batch_size)
    test_metrics = evaluate_model(model, test_tasks, support, test_log_probs, args.pair_batch_size)

    return {
        "model": model_name,
        "model_seed": model_seed,
        "num_parameters": sum(param.numel() for param in model.parameters()),
        "train_seconds": time.time() - started,
        "history": history,
        "train": asdict(train_metrics),
        "test": asdict(test_metrics),
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
    baseline = report["baseline"]["test"]
    rows = []
    for result in report["results"]:
        test = result["test"]
        rows.append(
            "| {model} | {params} | {kl:.4f} | {tv:.4f} | {nll:.4f} | {rank:.2f} | {mode:.2f} | {sec:.1f} |".format(
                model=result["model"],
                params=result["num_parameters"],
                kl=test["kl_pstar_model"],
                tv=test["tv_distance"],
                nll=test["nll_pstar"],
                rank=test["target_mode_rank_mean"],
                mode=test["mode_match_rate"],
                sec=result["train_seconds"],
            )
        )

    lines = [
        "# Toy EBM Training Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(report["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Uniform Baseline",
        "",
        f"- test KL(p_star || uniform): `{baseline['kl_pstar_model']:.4f}`",
        f"- test TV distance: `{baseline['tv_distance']:.4f}`",
        f"- test NLL: `{baseline['nll_pstar']:.4f}`",
        "",
        "## Held-Out Results",
        "",
        "| model | params | test KL | test TV | test NLL | target mode rank | mode match | seconds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "Lower KL/TV/NLL and lower target-mode rank are better. Higher mode match is better.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    support = enumerate_binary_support(args.seq_len, device=device)
    all_tasks = make_task_bank(args.num_train_tasks + args.num_test_tasks, args.seq_len, args.seed)
    train_tasks = all_tasks[: args.num_train_tasks].to(device)
    test_tasks = all_tasks[args.num_train_tasks :].to(device)

    train_log_probs = target_log_probs(train_tasks, support, args.target_temperature)
    test_log_probs = target_log_probs(test_tasks, support, args.target_temperature)

    baseline = {
        "train": asdict(uniform_metrics(train_log_probs)),
        "test": asdict(uniform_metrics(test_log_probs)),
    }

    results = []
    for offset, model_name in enumerate(args.models):
        result = train_one_model(
            model_name=model_name,
            args=args,
            train_tasks=train_tasks,
            train_log_probs=train_log_probs,
            test_tasks=test_tasks,
            test_log_probs=test_log_probs,
            support=support,
            device=device,
            model_seed=args.seed + 100 * (offset + 1),
        )
        results.append(result)
        print(
            "{model}: test_kl={kl:.4f} tv={tv:.4f} mode_rank={rank:.2f} seconds={sec:.1f}".format(
                model=model_name,
                kl=result["test"]["kl_pstar_model"],
                tv=result["test"]["tv_distance"],
                rank=result["test"]["target_mode_rank_mean"],
                sec=result["train_seconds"],
            ),
            flush=True,
        )

    best_test_kl = min(result["test"]["kl_pstar_model"] for result in results)
    baseline_test_kl = baseline["test"]["kl_pstar_model"]
    status = "pass" if best_test_kl < baseline_test_kl else "fail"

    report = {
        "status": status,
        "config": jsonable_args(args) | {
            "device_resolved": str(device),
            "support_size": support.shape[0],
        },
        "baseline": baseline,
        "results": results,
    }

    json_path = args.output_dir / f"{args.tag}_report.json"
    md_path = args.output_dir / f"{args.tag}_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report, md_path)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)

    if args.require_improvement and status != "pass":
        raise RuntimeError(
            f"best test KL {best_test_kl:.4f} did not improve over uniform {baseline_test_kl:.4f}"
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="real_exact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    parser.add_argument("--seq-len", type=int, default=9)
    parser.add_argument("--num-train-tasks", type=int, default=48)
    parser.add_argument("--num-test-tasks", type=int, default=16)
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pair-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--require-improvement", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

