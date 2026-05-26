#!/usr/bin/env python3
"""Small vision benchmark for AR/EBM/free-energy derisking.

This is intentionally modest. The default `digits` dataset is an 8x8 MNIST-like
proxy from sklearn so smoke and SNAP runs do not depend on external downloads.
Use `--dataset mnist` once torchvision data is available.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class RunConfig:
    dataset: str = "digits"
    tag: str = "smoke"
    output_dir: str = "experiments/04_vision_energy_comparison/results"
    seed: int = 0
    device: str = "auto"
    batch_size: int = 128
    epochs: int = 2
    diffusion_epochs: int = 2
    lr: float = 1e-3
    max_train: int | None = 512
    max_test: int | None = 256
    models: tuple[str, ...] = ("cnn", "tiny_vit", "ebm", "novel_ebm", "diffusion")
    smoke: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda requested but unavailable; falling back to cpu")
        return torch.device("cpu")
    return torch.device(name)


def _limit(x: torch.Tensor, y: torch.Tensor, max_n: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    if max_n is None or max_n >= len(x):
        return x, y
    return x[:max_n], y[:max_n]


def load_digits_like(dataset: str, max_train: int | None, max_test: int | None, seed: int):
    dataset = dataset.lower()
    if dataset == "digits":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        d = load_digits()
        x = torch.tensor(d.images[:, None, :, :] / 16.0, dtype=torch.float32)
        y = torch.tensor(d.target, dtype=torch.long)
        idx = np.arange(len(y))
        train_idx, test_idx = train_test_split(
            idx, test_size=0.25, random_state=seed, stratify=y.numpy()
        )
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]
    elif dataset in {"mnist", "fashionmnist"}:
        try:
            from torchvision import datasets, transforms
        except Exception as exc:  # pragma: no cover - env-dependent
            raise RuntimeError("MNIST/FashionMNIST requires torchvision") from exc

        root = Path.home() / ".cache" / "free_energy_vision_data"
        klass = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        transform = transforms.ToTensor()
        train = klass(root=str(root), train=True, download=True, transform=transform)
        test = klass(root=str(root), train=False, download=True, transform=transform)
        x_train = torch.stack([train[i][0] for i in range(len(train))])
        y_train = torch.tensor([train[i][1] for i in range(len(train))], dtype=torch.long)
        x_test = torch.stack([test[i][0] for i in range(len(test))])
        y_test = torch.tensor([test[i][1] for i in range(len(test))], dtype=torch.long)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    x_train, y_train = _limit(x_train, y_train, max_train)
    x_test, y_test = _limit(x_test, y_test, max_test)
    return x_train, y_train, x_test, y_test


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyViT(nn.Module):
    def __init__(self, image_size: int, patch: int, num_classes: int = 10, dim: int = 64):
        super().__init__()
        assert image_size % patch == 0
        self.patch = patch
        self.num_patches = (image_size // patch) ** 2
        self.proj = nn.Linear(patch * patch, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, self.num_patches + 1, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        p = self.patch
        patches = x.unfold(2, p, p).unfold(3, p, p).contiguous()
        patches = patches.view(b, 1, -1, p, p).squeeze(1).flatten(-2)
        tokens = self.proj(patches)
        cls = self.cls.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos[:, : tokens.shape[1] + 1]
        return self.head(self.encoder(tokens)[:, 0])


class EnergyMLP(nn.Module):
    """Conditional energy E(x, y); logits are -E(x, y)."""

    def __init__(self, image_size: int, num_classes: int = 10, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size * image_size, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def energies(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


class DiffusionDenoiser(nn.Module):
    def __init__(self, image_size: int, time_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.image_size = image_size
        d = image_size * image_size
        self.time = nn.Embedding(100, time_dim)
        self.net = nn.Sequential(
            nn.Linear(d + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        flat = x_t.flatten(1)
        return self.net(torch.cat([flat, self.time(t)], dim=1)).view_as(x_t)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == y).float().mean().item()


@torch.no_grad()
def eval_classifier(model: nn.Module, loader: DataLoader, device: torch.device, energy: bool = False):
    model.eval()
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    margins: list[float] = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        logits = -out if energy else out
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_acc += accuracy(logits, y) * len(y)
        total_n += len(y)
        if energy:
            true_e = out.gather(1, y[:, None]).squeeze(1)
            masked = out.clone()
            masked.scatter_(1, y[:, None], float("inf"))
            margins.extend((masked.min(dim=1).values - true_e).detach().cpu().tolist())
    out = {
        "loss": total_loss / max(total_n, 1),
        "accuracy": total_acc / max(total_n, 1),
    }
    if margins:
        out["energy_margin_mean"] = float(np.mean(margins))
    return out


def train_classifier(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    energy: bool = False,
    contrastive_weight: float = 0.0,
):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = -out if energy else out
            loss = F.cross_entropy(logits, y)
            if energy and contrastive_weight > 0:
                x_bad = (x + 0.35 * torch.randn_like(x)).clamp(0, 1)
                e_clean = out.gather(1, y[:, None]).squeeze(1)
                e_bad = model(x_bad).gather(1, y[:, None]).squeeze(1)
                loss = loss + contrastive_weight * F.relu(1.0 + e_clean - e_bad).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        metrics = eval_classifier(model, test_loader, device, energy=energy)
        metrics["epoch"] = epoch + 1
        history.append(metrics)
    final = dict(history[-1])
    final["model"] = name
    final["objective"] = "energy_ce" if energy else "cross_entropy"
    if contrastive_weight:
        final["contrastive_weight"] = contrastive_weight
    final["history"] = history
    return final


def make_beta_schedule(steps: int = 100):
    beta = torch.linspace(1e-4, 0.02, steps)
    alpha = 1.0 - beta
    return torch.cumprod(alpha, dim=0)


@torch.no_grad()
def eval_diffusion(model: DiffusionDenoiser, loader: DataLoader, device: torch.device, steps: int = 100):
    model.eval()
    alpha_bar = make_beta_schedule(steps).to(device)
    total, count = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        t = torch.randint(0, steps, (x.shape[0],), device=device)
        eps = torch.randn_like(x)
        ab = alpha_bar[t].view(-1, 1, 1, 1)
        x_t = ab.sqrt() * x + (1 - ab).sqrt() * eps
        pred = model(x_t, t)
        total += F.mse_loss(pred, eps, reduction="sum").item()
        count += eps.numel()
    return total / max(count, 1)


def train_diffusion(train_loader: DataLoader, test_loader: DataLoader, device: torch.device, image_size: int, epochs: int, lr: float):
    model = DiffusionDenoiser(image_size=image_size).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    alpha_bar = make_beta_schedule(100).to(device)
    history = []
    for epoch in range(epochs):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            t = torch.randint(0, 100, (x.shape[0],), device=device)
            eps = torch.randn_like(x)
            ab = alpha_bar[t].view(-1, 1, 1, 1)
            x_t = ab.sqrt() * x + (1 - ab).sqrt() * eps
            loss = F.mse_loss(model(x_t, t), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        history.append({"epoch": epoch + 1, "test_noise_mse": eval_diffusion(model, test_loader, device)})
    return {
        "model": "diffusion",
        "objective": "ddpm_noise_prediction",
        "test_noise_mse": history[-1]["test_noise_mse"],
        "history": history,
    }


def save_plot(results: list[dict], out_dir: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None
    cls = [r for r in results if "accuracy" in r]
    if not cls:
        return None
    names = [r["model"] for r in cls]
    accs = [r["accuracy"] for r in cls]
    plt.figure(figsize=(7, 4))
    plt.bar(names, accs)
    plt.ylabel("test accuracy")
    plt.ylim(0, 1)
    plt.title("Vision derisking baseline accuracy")
    plt.tight_layout()
    path = out_dir / "accuracy.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def write_report(out_dir: Path, cfg: RunConfig, results: list[dict], plot_path: str | None) -> None:
    metrics = {
        "config": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = [
        f"# Vision Energy Comparison Run: `{cfg.tag}`",
        "",
        f"Dataset: `{cfg.dataset}`",
        f"Seed: `{cfg.seed}`",
        f"Device: `{cfg.device}`",
        "",
        "## Results",
        "",
        "| model | objective | accuracy | loss / mse | energy margin |",
        "|---|---|---:|---:|---:|",
    ]
    for r in results:
        acc = f"{r.get('accuracy', float('nan')):.4f}" if "accuracy" in r else ""
        loss_key = "loss" if "loss" in r else "test_noise_mse"
        loss = f"{r.get(loss_key, float('nan')):.4f}"
        margin = f"{r.get('energy_margin_mean', float('nan')):.4f}" if "energy_margin_mean" in r else ""
        lines.append(f"| {r['model']} | {r['objective']} | {acc} | {loss} | {margin} |")
    if plot_path:
        lines.extend(["", f"Plot: `{plot_path}`"])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a derisking scaffold. Digits/MNIST results are not a publishable vision claim by themselves.",
        "They are a check that CNN, tiny ViT, EBM-style, novel-EBM-style, and diffusion objectives all run under one protocol.",
    ])
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="digits", choices=["digits", "mnist", "fashionmnist"])
    p.add_argument("--tag", default="smoke")
    p.add_argument("--output-dir", default="experiments/04_vision_energy_comparison/results")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--diffusion-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-train", type=int, default=512)
    p.add_argument("--max-test", type=int, default=256)
    p.add_argument("--models", nargs="+", default=["cnn", "tiny_vit", "ebm", "novel_ebm", "diffusion"])
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.tag = args.tag if args.tag != "smoke" else "smoke"
        args.epochs = min(args.epochs, 1)
        args.diffusion_epochs = min(args.diffusion_epochs, 1)
        args.max_train = min(args.max_train, 256)
        args.max_test = min(args.max_test, 128)
    models = tuple(args.models)
    if models == ("all",):
        models = ("cnn", "tiny_vit", "ebm", "novel_ebm", "diffusion")
    return RunConfig(
        dataset=args.dataset,
        tag=args.tag,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        diffusion_epochs=args.diffusion_epochs,
        lr=args.lr,
        max_train=args.max_train,
        max_test=args.max_test,
        models=models,
        smoke=args.smoke,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)

    x_train, y_train, x_test, y_test = load_digits_like(cfg.dataset, cfg.max_train, cfg.max_test, cfg.seed)
    image_size = int(x_train.shape[-1])
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=cfg.batch_size)

    results: list[dict] = []
    for name in cfg.models:
        if name == "cnn":
            results.append(train_classifier(name, SmallCNN(), train_loader, test_loader, device, cfg.epochs, cfg.lr))
        elif name == "tiny_vit":
            patch = 2 if image_size <= 8 else 4
            results.append(train_classifier(name, TinyViT(image_size, patch), train_loader, test_loader, device, cfg.epochs, cfg.lr))
        elif name == "ebm":
            results.append(train_classifier(name, EnergyMLP(image_size), train_loader, test_loader, device, cfg.epochs, cfg.lr, energy=True))
        elif name == "novel_ebm":
            results.append(train_classifier(name, EnergyMLP(image_size), train_loader, test_loader, device, cfg.epochs, cfg.lr, energy=True, contrastive_weight=0.2))
        elif name == "diffusion":
            results.append(train_diffusion(train_loader, test_loader, device, image_size, cfg.diffusion_epochs, cfg.lr))
        else:
            raise ValueError(f"unknown model: {name}")

    out_dir = Path(cfg.output_dir) / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_plot(results, out_dir)
    write_report(out_dir, cfg, results, plot_path)
    print(f"[done] wrote {out_dir / 'metrics.json'}")
    print(f"[done] wrote {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
