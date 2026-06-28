"""E1 — A continuous EBM trains without ever touching Z; MCMC is replaceable.

Answers RQ1 (why deal with Z at all), RQ2 (is MCMC fundamental), Q3 (PMF/PDF
restriction), Q5 (go continuous to dissolve the undirected chicken-and-egg).

We fit the SAME 2-D target two ways with an MLP energy E_theta : R^2 -> R:

  (1) PCD-Langevin  — the modern deep-EBM recipe (Du & Mordatch 2019):
      negative samples come from a persistent SGLD replay buffer; the loss is
      E_theta(x+) - E_theta(x-) (+ a small energy regularizer). Z never appears.
  (2) Denoising Score Matching (Vincent 2011) — an MCMC-FREE training objective:
      match -grad_x E_theta to the score of the noised data. The training loop
      never samples the model and never touches Z.

Both yield a usable continuous energy landscape; we draw samples from each with
the SAME low-noise Langevin sampler and score them against held-out data with
kernel MMD and a precision/recall metric. NB: score matching makes *training*
MCMC-free; drawing samples afterward still uses Langevin (or a deterministic
probability-flow ODE) — that is using the learned score, not the chicken-and-egg
training chain the note worries about. (Single-scale DSM blurs *well-separated*
modes — e.g. the eight-Gaussians case — into a ridge; the standard remedy is
multi-scale/annealed noise (Song & Ermon 2019), still MCMC-free for training.
We use a connected target here so the comparison is fair to both methods.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (DATASETS, MLPEnergy, get_device, mmd2_rbf, set_seed,
                    sgld_sample)

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DATASET = "two_moons"   # connected support: a fair head-to-head for both methods
N_TRAIN = 6000
STEPS = 3000
BATCH = 256
DEVICE = get_device()


# --------------------------------------------------------------------------- #
# (1) PCD-Langevin training
# --------------------------------------------------------------------------- #
def train_pcd_langevin(data: torch.Tensor, steps: int = STEPS):
    energy = MLPEnergy(2, hidden=128, n_layers=4).to(DEVICE)
    opt = torch.optim.Adam(energy.parameters(), lr=1e-3, betas=(0.9, 0.999))
    buf = (2.0 * torch.rand(1000, 2, device=DEVICE) - 1.0) * 2.0  # replay buffer
    alpha = 0.1  # energy L2 regularizer (IGEBM)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(data), (BATCH,), device=DEVICE)
        x_pos = data[idx]
        bidx = torch.randint(0, len(buf), (BATCH,), device=DEVICE)
        x0 = buf[bidx].clone()
        reinit = torch.rand(BATCH, device=DEVICE) < 0.05
        x0[reinit] = (2.0 * torch.rand(int(reinit.sum()), 2, device=DEVICE) - 1) * 2.0
        x_neg = sgld_sample(energy, x0, n_steps=40, step_size=0.02,
                            noise_scale=0.01, clamp_grad=1.0, clamp_data=4.0)
        buf[bidx] = x_neg.detach()
        e_pos, e_neg = energy(x_pos), energy(x_neg)
        loss = (e_pos.mean() - e_neg.mean()) + alpha * (e_pos**2 + e_neg**2).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            losses.append((step, float((e_pos.mean() - e_neg.mean()).detach())))
    return energy, losses


# --------------------------------------------------------------------------- #
# (2) Denoising Score Matching (MCMC-free training)
# --------------------------------------------------------------------------- #
def train_dsm(data: torch.Tensor, steps: int = STEPS, sigma: float = 0.08):
    energy = MLPEnergy(2, hidden=128, n_layers=4).to(DEVICE)
    opt = torch.optim.Adam(energy.parameters(), lr=1e-3)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, len(data), (BATCH,), device=DEVICE)
        x = data[idx]
        noise = torch.randn_like(x)
        x_tilde = (x + sigma * noise).requires_grad_(True)
        e = energy(x_tilde).sum()
        (grad,) = torch.autograd.grad(e, x_tilde, create_graph=True)
        score = -grad                                    # score = -grad_x E
        target = (x - x_tilde) / (sigma ** 2)            # grad log q_sigma(x~|x)
        loss = 0.5 * ((score - target) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 10.0)
        opt.step()
        if step % 50 == 0:
            losses.append((step, float(loss.detach())))
    return energy, losses


# --------------------------------------------------------------------------- #
# evaluation (shared sampler for both energies)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def energy_grid(energy, lim=2.5, n=200):
    xs = torch.linspace(-lim, lim, n)
    gx, gy = torch.meshgrid(xs, xs, indexing="xy")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1).to(DEVICE)
    e = energy(pts).reshape(n, n).cpu().numpy()
    return gx.numpy(), gy.numpy(), e


def sample_model(energy, n=2000, steps=500):
    x0 = (2.0 * torch.rand(n, 2, device=DEVICE) - 1) * 2.0
    return sgld_sample(energy, x0, n_steps=steps, step_size=0.02,
                       noise_scale=0.01, clamp_grad=1.0, clamp_data=4.0).cpu().numpy()


def precision_recall(samples: np.ndarray, data: np.ndarray, r: float = 0.2):
    """Generic sample-quality: precision = fraction of model samples within r of
    some real point (are samples on the manifold?); recall = fraction of real
    points within r of some model sample (is the manifold covered?)."""
    d = np.sqrt(((samples[:, None, :] - data[None, :, :]) ** 2).sum(-1))
    precision = float((d.min(1) < r).mean())
    recall = float((d.min(0) < r).mean())
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "radius": r}


def main() -> None:
    t0 = time.time()
    set_seed(0)
    rng = np.random.default_rng(0)
    data_np = DATASETS[DATASET](N_TRAIN, rng)
    held = DATASETS[DATASET](2000, np.random.default_rng(99))
    data = torch.tensor(data_np, device=DEVICE)

    print(f"device={DEVICE}  dataset={DATASET}")
    print("training PCD-Langevin EBM ...")
    e_pcd, l_pcd = train_pcd_langevin(data)
    print("training Denoising Score Matching EBM (MCMC-free training) ...")
    e_dsm, l_dsm = train_dsm(data)

    s_pcd = sample_model(e_pcd)
    s_dsm = sample_model(e_dsm)
    mmd_pcd = mmd2_rbf(s_pcd, held)
    mmd_dsm = mmd2_rbf(s_dsm, held)
    pr_pcd = precision_recall(s_pcd, held)
    pr_dsm = precision_recall(s_dsm, held)
    mmd_floor = mmd2_rbf(DATASETS[DATASET](2000, np.random.default_rng(7)), held)
    noise = (2.0 * np.random.default_rng(8).random((2000, 2)) - 1) * 2.5
    mmd_noise = mmd2_rbf(noise.astype(np.float32), held)

    print(f"\nMMD^2 to held-out data (lower=better):")
    print(f"  data vs data (floor)  : {mmd_floor:.4f}")
    print(f"  PCD-Langevin samples  : {mmd_pcd:.4f}")
    print(f"  DSM (MCMC-free train) : {mmd_dsm:.4f}")
    print(f"  uniform noise (ceil)  : {mmd_noise:.4f}")
    print(f"\nprecision / recall @ r={pr_pcd['radius']} (higher=better):")
    print(f"  PCD-Langevin : P={pr_pcd['precision']:.2f} R={pr_pcd['recall']:.2f}"
          f" F1={pr_pcd['f1']:.2f}")
    print(f"  DSM          : P={pr_dsm['precision']:.2f} R={pr_dsm['recall']:.2f}"
          f" F1={pr_dsm['f1']:.2f}")

    out = {"meta": {"dataset": DATASET, "device": str(DEVICE), "steps": STEPS,
                    "n_train": N_TRAIN, "runtime_s": round(time.time() - t0, 2)},
           "mmd2": {"floor_data_vs_data": mmd_floor, "pcd_langevin": mmd_pcd,
                    "dsm_mcmc_free": mmd_dsm, "ceiling_noise": mmd_noise},
           "precision_recall": {"pcd_langevin": pr_pcd, "dsm_mcmc_free": pr_dsm},
           "final_contrastive_gap_pcd": l_pcd[-1][1],
           "final_dsm_loss": l_dsm[-1][1],
           "z_was_computed": False,
           "mcmc_used_in_training": {"pcd_langevin": True, "dsm": False}}
    (RESULTS / "e1_langevin_vs_sm.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'e1_langevin_vs_sm.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    ax[0].scatter(held[:, 0], held[:, 1], s=4, alpha=.4, c="k")
    ax[0].set_title(f"target data ({DATASET})")
    panels = [(e_pcd, s_pcd, "PCD-Langevin (MCMC)", mmd_pcd, pr_pcd),
              (e_dsm, s_dsm, "Score Matching (MCMC-free train)", mmd_dsm, pr_dsm)]
    for a, (en, sm, name, mmd, pr) in zip(ax[1:], panels):
        gx, gy, e = energy_grid(en)
        a.contourf(gx, gy, np.exp(-(e - e.min())), levels=30, cmap="viridis")
        a.scatter(sm[:, 0], sm[:, 1], s=3, alpha=.25, c="white")
        a.set_title(f"{name}\nexp(-E)+samples  MMD²={mmd:.3f}  "
                    f"P/R={pr['precision']:.2f}/{pr['recall']:.2f}")
    for a in ax:
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5); a.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(RESULTS / "e1_langevin_vs_sm.png", dpi=130)
    print(f"wrote {RESULTS / 'e1_langevin_vs_sm.png'}  ({out['meta']['runtime_s']}s)")


if __name__ == "__main__":
    main()
