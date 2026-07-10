#!/usr/bin/env python3
"""Mini 35D Gaussian-energy/Langevin sanity check.

This isolates the proposed objective from the full EBT stack. It trains a small
energy model E(y) using only perturbed target samples:

    loss = exp(-||eps||^2 / attenuation_dim) * E(target + alpha * eps)

Then it tests whether fixed-step Langevin dynamics converges from random starts
to each trained target point.

Example:
    uv run test_gaussian_langevin_35d.py --num-targets 20 --train-steps 5000
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F


@dataclass
class Metrics:
    mean_initial_dist: float
    mean_final_dist: float
    mean_nearest_final_dist: float
    max_final_dist: float
    mean_initial_energy: float
    mean_final_energy: float
    mean_grad_alignment: float
    grad_agreement_rate: float
    assigned_success_rate: float
    nearest_success_rate: float


def gradient_alignment(
    model: EnergyMLP,
    targets: torch.Tensor,
    *,
    gaussian_alpha: float,
    samples_per_target: int,
) -> tuple[float, float]:
    target_batch = targets.repeat_interleave(samples_per_target, dim=0)
    eps = torch.randn_like(target_batch)
    y = (target_batch + gaussian_alpha * eps).detach().requires_grad_(True)
    energy = model(y).sum()
    grad = torch.autograd.grad(energy, y)[0]
    descent_direction = -grad
    target_direction = target_batch - y.detach()
    cosine = F.cosine_similarity(descent_direction, target_direction, dim=-1)
    return float(cosine.mean().item()), float((cosine > 0).float().mean().item())


class EnergyMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y).squeeze(-1)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_targets(num_targets: int, dim: int, radius: float, device: torch.device) -> torch.Tensor:
    targets = torch.randn(num_targets, dim, device=device)
    targets = F.normalize(targets, dim=-1) * radius
    return targets


def gaussian_energy_loss(
    model: EnergyMLP,
    targets: torch.Tensor,
    *,
    gaussian_alpha: float,
    samples_per_target: int,
    attenuation_dim: float,
    beta: float,
) -> torch.Tensor:
    num_targets, dim = targets.shape
    target_batch = targets.repeat_interleave(samples_per_target, dim=0)
    eps = torch.randn_like(target_batch)
    y = target_batch + gaussian_alpha * eps

    # Set --attenuation-dim 1 for the literal exp(-||eps||^2). The default 35
    # keeps gradients numerically visible in 35D.
    target_energy = (-beta * torch.exp(-eps.square().sum(dim=-1) / attenuation_dim)).detach()
    energy = model(y)
    return F.mse_loss(energy, target_energy)


def langevin_refine(
    model: EnergyMLP,
    init: torch.Tensor,
    *,
    steps: int,
    step_size: float,
    noise: float,
    annealed: bool,
    final_step_size: float,
    final_noise: float,
    return_trajectory: bool = False,
) -> torch.Tensor:
    y = init.detach().clone()
    trajectory = [y.detach().clone()]
    for step in range(steps):
        if annealed and steps > 1:
            frac = step / (steps - 1)
            current_step_size = step_size * ((final_step_size / step_size) ** frac)
            current_noise = noise * ((final_noise / noise) ** frac) if noise > 0.0 and final_noise > 0.0 else 0.0
        else:
            current_step_size = step_size
            current_noise = noise
        y = y.detach().requires_grad_(True)
        energy = model(y).sum()
        grad = torch.autograd.grad(energy, y)[0]
        y = y - current_step_size * grad
        if current_noise != 0.0:
            y = y + torch.randn_like(y) * current_noise
        if return_trajectory:
            trajectory.append(y.detach().clone())
    if return_trajectory:
        return torch.stack(trajectory, dim=0)
    return y.detach()


def save_2d_mp4(
    model: EnergyMLP,
    targets: torch.Tensor,
    path: str,
    *,
    starts_per_target: int,
    init_scale: float,
    langevin_steps: int,
    langevin_step_size: float,
    langevin_noise: float,
    annealed_langevin: bool,
    langevin_final_step_size: float,
    langevin_final_noise: float,
    grid_lim: float,
    grid_size: int,
    fps: int,
) -> None:
    if targets.shape[1] != 2:
        raise ValueError("--save-mp4 requires dim == 2")

    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_batch = targets.repeat_interleave(starts_per_target, dim=0)
    init = torch.randn_like(target_batch) * init_scale
    trajectory = langevin_refine(
        model,
        init,
        steps=langevin_steps,
        step_size=langevin_step_size,
        noise=langev:w
        in_noise,
        annealed=annealed_langevin,
        final_step_size=langevin_final_step_size,
        final_noise=langevin_final_noise,
        return_trajectory=True,
    ).detach()

    xs = torch.linspace(-grid_lim, grid_lim, grid_size, device=targets.device)
    ys = torch.linspace(-grid_lim, grid_lim, grid_size, device=targets.device)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    with torch.no_grad():
        energy = model(grid).reshape(grid_size, grid_size).detach().cpu().numpy()

    trajectory_cpu = trajectory.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(path_obj, fps=fps, codec="libx264", quality=8) as writer:
        for frame_idx in range(trajectory_cpu.shape[0]):
            fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
            ax.imshow(
                energy,
                origin="lower",
                extent=(-grid_lim, grid_lim, -grid_lim, grid_lim),
                cmap="viridis",
                aspect="equal",
            )
            upto = trajectory_cpu[: frame_idx + 1]
            for particle_idx in range(trajectory_cpu.shape[1]):
                ax.plot(upto[:, particle_idx, 0], upto[:, particle_idx, 1], color="white", alpha=0.35, linewidth=0.8)
            current = trajectory_cpu[frame_idx]
            ax.scatter(current[:, 0], current[:, 1], s=10, c="white", edgecolors="black", linewidths=0.2)
            ax.scatter(targets_cpu[:, 0], targets_cpu[:, 1], s=80, c="red", marker="*", edgecolors="black")
            ax.set_xlim(-grid_lim, grid_lim)
            ax.set_ylim(-grid_lim, grid_lim)
            ax.set_title(f"Annealed Langevin step {frame_idx}/{trajectory_cpu.shape[0] - 1}")
            ax.set_xlabel("y[0]")
            ax.set_ylabel("y[1]")
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(frame)
            plt.close(fig)


def evaluate(
    model: EnergyMLP,
    targets: torch.Tensor,
    *,
    starts_per_target: int,
    init_scale: float,
    langevin_steps: int,
    langevin_step_size: float,
    langevin_noise: float,
    annealed_langevin: bool,
    langevin_final_step_size: float,
    langevin_final_noise: float,
    gaussian_alpha: float,
    success_threshold: float,
) -> Metrics:
    target_batch = targets.repeat_interleave(starts_per_target, dim=0)
    init = torch.randn_like(target_batch) * init_scale
    final = langevin_refine(
        model,
        init,
        steps=langevin_steps,
        step_size=langevin_step_size,
        noise=langevin_noise,
        annealed=annealed_langevin,
        final_step_size=langevin_final_step_size,
        final_noise=langevin_final_noise,
    )
    initial_dists = torch.linalg.vector_norm(init - target_batch, dim=-1)
    dists = torch.linalg.vector_norm(final - target_batch, dim=-1)
    nearest_dists = torch.cdist(final, targets).min(dim=1).values
    with torch.no_grad():
        initial_energy = model(init)
        final_energy = model(final)
    mean_grad_alignment, grad_agreement_rate = gradient_alignment(
        model,
        targets,
        gaussian_alpha=gaussian_alpha,
        samples_per_target=starts_per_target,
    )
    return Metrics(
        mean_initial_dist=float(initial_dists.mean().item()),
        mean_final_dist=float(dists.mean().item()),
        mean_nearest_final_dist=float(nearest_dists.mean().item()),
        max_final_dist=float(dists.max().item()),
        mean_initial_energy=float(initial_energy.mean().item()),
        mean_final_energy=float(final_energy.mean().item()),
        mean_grad_alignment=mean_grad_alignment,
        grad_agreement_rate=grad_agreement_rate,
        assigned_success_rate=float((dists < success_threshold).float().mean().item()),
        nearest_success_rate=float((nearest_dists < success_threshold).float().mean().item()),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gaussian_langevin_35d.yaml")
    parser.add_argument("--dim", type=int, default=35)
    parser.add_argument("--num-targets", type=int, default=20)
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gaussian-alpha", type=float, default=1.0)
    parser.add_argument("--samples-per-target", type=int, default=16)
    parser.add_argument("--attenuation-dim", type=float, default=35.0)
    parser.add_argument("--beta", type=float, default=50.0)
    parser.add_argument("--langevin-steps", type=int, default=100)
    parser.add_argument("--langevin-step-size", type=float, default=0.1)
    parser.add_argument("--langevin-noise", type=float, default=0.0)
    parser.add_argument("--annealed-langevin", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--langevin-final-step-size", type=float, default=0.01)
    parser.add_argument("--langevin-final-noise", type=float, default=0.001)
    parser.add_argument("--starts-per-target", type=int, default=8)
    parser.add_argument("--init-scale", type=float, default=4.0)
    parser.add_argument("--success-threshold", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-mp4", type=str, default="")
    parser.add_argument("--viz-grid-lim", type=float, default=6.0)
    parser.add_argument("--viz-grid-size", type=int, default=160)
    parser.add_argument("--viz-fps", type=int, default=20)
    config_args, _ = parser.parse_known_args()
    cfg = OmegaConf.load(config_args.config)
    parser.set_defaults(**OmegaConf.to_container(cfg, resolve=True))
    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device)
    targets = make_targets(args.num_targets, args.dim, args.target_radius, device)
    model = EnergyMLP(args.dim, args.hidden_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(
        f"dim={args.dim} targets={args.num_targets} gaussian_alpha={args.gaussian_alpha} "
        f"langevin_steps={args.langevin_steps} step_size={args.langevin_step_size} noise={args.langevin_noise} "
        f"annealed={args.annealed_langevin}"
    )

    for step in range(1, args.train_steps + 1):
        loss = gaussian_energy_loss(
            model,
            targets,
            gaussian_alpha=args.gaussian_alpha,
            samples_per_target=args.samples_per_target,
            attenuation_dim=args.attenuation_dim,
            beta=args.beta,
        )
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step == 1 or step % args.log_every == 0 or step == args.train_steps:
            metrics = evaluate(
                model,
                targets,
                starts_per_target=args.starts_per_target,
                init_scale=args.init_scale,
                langevin_steps=args.langevin_steps,
                langevin_step_size=args.langevin_step_size,
                langevin_noise=args.langevin_noise,
                annealed_langevin=args.annealed_langevin,
                langevin_final_step_size=args.langevin_final_step_size,
                langevin_final_noise=args.langevin_final_noise,
                gaussian_alpha=args.gaussian_alpha,
                success_threshold=args.success_threshold,
            )
            print(
                f"step={step} loss={loss.item():.6f} "
                f"dist={metrics.mean_initial_dist:.4f}->{metrics.mean_final_dist:.4f} "
                f"nearest_dist={metrics.mean_nearest_final_dist:.4f} "
                f"energy={metrics.mean_initial_energy:.4f}->{metrics.mean_final_energy:.4f} "
                f"grad_align={metrics.mean_grad_alignment:.4f} grad_agree={metrics.grad_agreement_rate:.3f} "
                f"max_dist={metrics.max_final_dist:.4f} assigned_success={metrics.assigned_success_rate:.3f} "
                f"nearest_success={metrics.nearest_success_rate:.3f}"
            )

    with torch.no_grad():
        target_energy = model(targets).mean().item()
        random_energy = model(torch.randn_like(targets) * args.init_scale).mean().item()
    print(f"target_energy={target_energy:.6f} random_energy={random_energy:.6f}")

    if args.save_mp4:
        save_2d_mp4(
            model,
            targets,
            args.save_mp4,
            starts_per_target=args.starts_per_target,
            init_scale=args.init_scale,
            langevin_steps=args.langevin_steps,
            langevin_step_size=args.langevin_step_size,
            langevin_noise=args.langevin_noise,
            annealed_langevin=args.annealed_langevin,
            langevin_final_step_size=args.langevin_final_step_size,
            langevin_final_noise=args.langevin_final_noise,
            grid_lim=args.viz_grid_lim,
            grid_size=args.viz_grid_size,
            fps=args.viz_fps,
        )
        print(f"saved_mp4={args.save_mp4}")


if __name__ == "__main__":
    main()
