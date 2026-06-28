#!/usr/bin/env python3
"""Toy EBM comparison: CD-1 vs dynamic weighted contrastive divergence.

The script trains two continuous EBMs on an eight-Gaussians target and writes
plots plus JSON metrics. Dynamic weighted CD uses all intermediate Langevin
states as a weighted negative phase. Later states get larger weights because
they are closer to the current model chain distribution, but they are still
negative-phase samples.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import trange


@dataclass
class Config:
    steps: int = 800
    batch_size: int = 256
    lr: float = 1e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    depth: int = 3
    data_std: float = 0.08
    data_radius: float = 2.0
    langevin_step_size: float = 0.01
    langevin_noise_scale: float = 1.0
    langevin_steps: int = 8
    dynamic_ramp: float = 2.0
    init_noise: float = 0.03
    energy_l2: float = 1e-2
    grad_clip: float = 10.0
    clamp: float = 4.0
    eval_samples: int = 2048
    eval_langevin_steps: int = 300
    grid_size: int = 180
    seed: int = 7
    device: str = "auto"
    out_dir: str = "questions/02_ebm_dynamic_cd_langevin/expt_v1/results"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    for field_name, field_value in asdict(Config()).items():
        arg_name = "--" + field_name.replace("_", "-")
        if isinstance(field_value, bool):
            parser.add_argument(arg_name, action="store_true")
        else:
            parser.add_argument(arg_name, type=type(field_value), default=field_value)
    return Config(**vars(parser.parse_args()))


def resolve_device(config: Config) -> torch.device:
    if config.device != "auto":
        return torch.device(config.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def true_centers(radius: float, device: torch.device) -> torch.Tensor:
    angles = torch.arange(8, device=device, dtype=torch.float32) * (2.0 * math.pi / 8.0)
    return radius * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


def sample_eight_gaussians(
    n: int, radius: float, std: float, device: torch.device
) -> torch.Tensor:
    centers = true_centers(radius, device)
    idx = torch.randint(0, centers.shape[0], (n,), device=device)
    return centers[idx] + std * torch.randn(n, 2, device=device)


class EnergyNet(nn.Module):
    def __init__(self, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = 2
        for _ in range(depth):
            linear = nn.Linear(dim, hidden_dim)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers += [linear, nn.SiLU()]
            dim = hidden_dim
        out = nn.Linear(dim, 1)
        nn.init.normal_(out.weight, mean=0.0, std=0.02)
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def langevin_dynamics(
    model: nn.Module,
    x0: torch.Tensor,
    steps: int,
    step_size: float,
    noise_scale: float,
    clamp: float,
    collect: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    x = x0.detach()
    trajectory: list[torch.Tensor] = []
    noise_std = math.sqrt(2.0 * step_size) * noise_scale
    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        energy = model(x).sum()
        grad = torch.autograd.grad(energy, x, create_graph=False)[0]
        x = x - step_size * grad + noise_std * torch.randn_like(x)
        if clamp > 0:
            x = x.clamp(-clamp, clamp)
        x = x.detach()
        if collect:
            trajectory.append(x)
    return x, trajectory


def dynamic_weights(length: int, ramp: float, device: torch.device) -> torch.Tensor:
    if length <= 0:
        raise ValueError("length must be positive")
    if length == 1:
        return torch.ones(1, device=device)
    t = torch.linspace(0.0, 1.0, length, device=device)
    weights = torch.exp(ramp * t)
    return weights / weights.sum()


def train_one(method: str, config: Config, device: torch.device) -> tuple[nn.Module, dict]:
    model = EnergyNet(config.hidden_dim, config.depth).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    history: list[dict] = []
    start = time.time()

    iterator = trange(
        config.steps,
        desc=method,
        leave=False,
        dynamic_ncols=True,
    )
    for step in iterator:
        data = sample_eight_gaussians(
            config.batch_size, config.data_radius, config.data_std, device
        )
        x0 = data + config.init_noise * torch.randn_like(data)

        if method == "cd1":
            neg, _ = langevin_dynamics(
                model,
                x0,
                steps=1,
                step_size=config.langevin_step_size,
                noise_scale=config.langevin_noise_scale,
                clamp=config.clamp,
                collect=False,
            )
            neg_energy = model(neg).mean()
        elif method == "dynamic_weighted_cd":
            _, trajectory = langevin_dynamics(
                model,
                x0,
                steps=config.langevin_steps,
                step_size=config.langevin_step_size,
                noise_scale=config.langevin_noise_scale,
                clamp=config.clamp,
                collect=True,
            )
            weights = dynamic_weights(
                len(trajectory), config.dynamic_ramp, device=device
            )
            energies = torch.stack([model(x_t).mean() for x_t in trajectory])
            neg_energy = torch.sum(weights * energies)
        else:
            raise ValueError(f"unknown method: {method}")

        data_energy = model(data).mean()
        energy_l2 = config.energy_l2 * (data_energy.square() + neg_energy.square())
        loss = data_energy - neg_energy + energy_l2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()

        if step % max(1, config.steps // 20) == 0 or step == config.steps - 1:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "data_energy": float(data_energy.detach().cpu()),
                "negative_energy": float(neg_energy.detach().cpu()),
            }
            history.append(row)
            iterator.set_postfix(loss=f"{row['loss']:.3f}")

    train_seconds = time.time() - start
    return model, {"history": history, "train_seconds": train_seconds}


@torch.no_grad()
def nearest_mode_metrics(
    samples: torch.Tensor, centers: torch.Tensor, radius: float
) -> dict:
    dists = torch.cdist(samples, centers)
    nearest_dist, nearest_idx = dists.min(dim=1)
    counts = torch.bincount(nearest_idx, minlength=centers.shape[0]).float()
    probs = counts / counts.sum().clamp_min(1.0)
    uniform = torch.full_like(probs, 1.0 / centers.shape[0])
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum()
    entropy = entropy / math.log(centers.shape[0])
    tv_to_uniform = 0.5 * torch.abs(probs - uniform).sum()
    coverage = (counts > 0.01 * samples.shape[0]).float().sum()
    radial_error = (samples.norm(dim=1) - radius).abs().mean()
    return {
        "mode_counts": [int(x) for x in counts.cpu().tolist()],
        "mode_probs": [float(x) for x in probs.cpu().tolist()],
        "coverage_1pct": float(coverage.cpu()),
        "normalized_mode_entropy": float(entropy.cpu()),
        "tv_to_uniform_modes": float(tv_to_uniform.cpu()),
        "mean_nearest_mode_distance": float(nearest_dist.mean().cpu()),
        "mean_radial_error": float(radial_error.cpu()),
    }


def evaluate_model(
    model: nn.Module, config: Config, device: torch.device
) -> tuple[torch.Tensor, dict]:
    model.eval()
    start = torch.randn(config.eval_samples, 2, device=device)
    samples, _ = langevin_dynamics(
        model,
        start,
        steps=config.eval_langevin_steps,
        step_size=config.langevin_step_size,
        noise_scale=config.langevin_noise_scale,
        clamp=config.clamp,
        collect=False,
    )
    centers = true_centers(config.data_radius, device)
    data = sample_eight_gaussians(
        config.eval_samples, config.data_radius, config.data_std, device
    )
    noise = torch.empty_like(data).uniform_(-config.clamp, config.clamp)
    with torch.no_grad():
        metrics = nearest_mode_metrics(samples, centers, config.data_radius)
        metrics.update(
            {
                "eval_data_energy": float(model(data).mean().cpu()),
                "eval_generated_energy": float(model(samples).mean().cpu()),
                "eval_uniform_noise_energy": float(model(noise).mean().cpu()),
            }
        )
    return samples.detach().cpu(), metrics


def energy_grid(
    model: nn.Module, config: Config, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = torch.linspace(-config.clamp, config.clamp, config.grid_size, device=device)
    ys = torch.linspace(-config.clamp, config.clamp, config.grid_size, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    chunks = []
    with torch.no_grad():
        for chunk in points.split(8192):
            chunks.append(model(chunk).detach().cpu())
    zz = torch.cat(chunks).reshape(config.grid_size, config.grid_size).numpy()
    return xx.cpu().numpy(), yy.cpu().numpy(), zz


def save_energy_plot(
    models: dict[str, nn.Module],
    config: Config,
    device: torch.device,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), dpi=140)
    if len(models) == 1:
        axes = [axes]
    data = sample_eight_gaussians(600, config.data_radius, config.data_std, device)
    data_np = data.cpu().numpy()
    for ax, (name, model) in zip(axes, models.items()):
        xx, yy, zz = energy_grid(model, config, device)
        lo, hi = np.percentile(zz, [5, 95])
        zz_display = np.clip(zz, lo, hi)
        levels = np.linspace(lo, hi, 32)
        contour = ax.contourf(xx, yy, zz_display, levels=levels, cmap="viridis")
        ax.scatter(data_np[:, 0], data_np[:, 1], s=4, c="white", alpha=0.35)
        ax.set_title(name)
        ax.set_xlim(-config.clamp, config.clamp)
        ax.set_ylim(-config.clamp, config.clamp)
        ax.set_aspect("equal")
        fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_surfaces.png")
    plt.close(fig)


def save_sample_plot(
    generated: dict[str, torch.Tensor],
    config: Config,
    device: torch.device,
    out_dir: Path,
) -> None:
    cols = 1 + len(generated)
    fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 5), dpi=140)
    target = sample_eight_gaussians(
        config.eval_samples, config.data_radius, config.data_std, device
    ).cpu()

    panels = {"target": target, **generated}
    for ax, (name, samples) in zip(axes, panels.items()):
        arr = samples.numpy()
        ax.scatter(arr[:, 0], arr[:, 1], s=5, alpha=0.55)
        ax.set_title(name)
        ax.set_xlim(-config.clamp, config.clamp)
        ax.set_ylim(-config.clamp, config.clamp)
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "samples.png")
    plt.close(fig)


def write_verdict(report: dict, out_dir: Path) -> None:
    cd = report["methods"]["cd1"]["metrics"]
    dw = report["methods"]["dynamic_weighted_cd"]["metrics"]
    better_coverage = dw["coverage_1pct"] - cd["coverage_1pct"]
    better_distance = cd["mean_nearest_mode_distance"] - dw["mean_nearest_mode_distance"]
    better_entropy = dw["normalized_mode_entropy"] - cd["normalized_mode_entropy"]

    if better_coverage > 0 or (better_distance > 0 and better_entropy >= -0.05):
        headline = "Dynamic weighted CD helped on this run."
    elif better_coverage == 0 and abs(better_distance) < 0.03:
        headline = "Dynamic weighted CD was roughly tied on this run."
    else:
        headline = "Dynamic weighted CD did not clearly help on this run."

    text = f"""# Verdict

{headline} CD-1 covered {cd['coverage_1pct']:.1f}/8 modes at the 1% threshold with mean nearest-mode distance {cd['mean_nearest_mode_distance']:.3f}; dynamic weighted CD covered {dw['coverage_1pct']:.1f}/8 modes with mean nearest-mode distance {dw['mean_nearest_mode_distance']:.3f}. The dynamic run used a weighted negative phase over {report['config']['langevin_steps']} Langevin states, so it costs more per optimizer step; interpret this as a scaling heuristic, not an unbiased likelihood-gradient estimator.
"""
    (out_dir / "verdict.md").write_text(text)


def main() -> None:
    config = parse_args()
    set_seed(config.seed)
    device = resolve_device(config)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = ["cd1", "dynamic_weighted_cd"]
    trained: dict[str, nn.Module] = {}
    generated: dict[str, torch.Tensor] = {}
    report = {
        "config": asdict(config),
        "device": str(device),
        "torch_version": torch.__version__,
        "methods": {},
    }

    for method in methods:
        model, train_info = train_one(method, config, device)
        samples, metrics = evaluate_model(model, config, device)
        trained[method] = model
        generated[method] = samples
        report["methods"][method] = {**train_info, "metrics": metrics}

    save_energy_plot(trained, config, device, out_dir)
    save_sample_plot(generated, config, device, out_dir)

    with (out_dir / "report.json").open("w") as f:
        json.dump(report, f, indent=2)
    write_verdict(report, out_dir)

    print(json.dumps(report["methods"], indent=2))
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
