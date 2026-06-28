"""E3 — Parallel / persistent chains: better mixing per wall-clock (Q4, Q9).

The note asks: "Can we do sampling in parallel / semi-sequential s.t. mixing
improves & cost goes down?" Three measurements:

  (A) ESS — on a multimodal target, one long Langevin chain is sticky: it
      lingers in a mode and crosses barriers rarely, so its integrated
      autocorrelation time tau >> 1 and ESS = N/tau << N.
  (B) Coverage vs #chains at FIXED total samples — many parallel chains start
      in different basins and recover the true mode weights, while one long
      chain (and a few long chains) gives badly biased weights because it is
      stuck. Ground truth = the analytic 3-Gaussian mixture weights.
  (C) Throughput — a *batched* MCMC step costs almost the same wall-clock for
      M=1 or M=thousands of chains on vectorized hardware (MPS), so the mixing
      win in (B) is nearly free until the device saturates.

(D) is conceptual and handled by E2's PCD result: *persistent* chains carry
state across SGD steps, amortizing burn-in — the "semi-sequential" idea.
Together these say: parallelism is the right lever for Q4/Q9; the sequential
bottleneck is slow barrier-crossing / per-chain burn-in, which parallel and
persistent chains sidestep.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import MLPEnergy, get_device, integrated_autocorr_time, sgld_sample

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

DEVICE = get_device()

# 1-D 3-mode Gaussian mixture: well-separated => slow Langevin mode-hopping.
MUS = np.array([-4.0, 0.0, 4.0])
SIG = 0.5
WEIGHTS = np.array([1 / 3, 1 / 3, 1 / 3])
STEP = 0.01           # Langevin step (small => slow barrier crossing)


def gmm_score(x: np.ndarray) -> np.ndarray:
    """d/dx log p(x) for the 1-D GMM (vectorized over chains)."""
    comp = WEIGHTS[None, :] * np.exp(-0.5 * ((x[:, None] - MUS[None, :]) / SIG) ** 2)
    comp /= SIG * np.sqrt(2 * np.pi)
    r = comp / comp.sum(1, keepdims=True)              # responsibilities
    return (r * (MUS[None, :] - x[:, None]) / SIG ** 2).sum(1)


def langevin_step(x: np.ndarray, rng) -> np.ndarray:
    return x + 0.5 * STEP * gmm_score(x) + np.sqrt(STEP) * rng.standard_normal(x.shape)


def mode_weights(x: np.ndarray) -> np.ndarray:
    nearest = np.abs(x[:, None] - MUS[None, :]).argmin(1)
    return np.array([(nearest == k).mean() for k in range(len(MUS))])


def part_a_ess(rng) -> dict:
    n, burn = 40000, 2000
    x = rng.normal(0, 3, size=1)
    traj = np.empty(n)
    for t in range(n):
        x = langevin_step(x, rng)
        traj[t] = x[0]
    tau = integrated_autocorr_time(traj[burn:])
    return {"chain_len": n, "burn": burn, "tau": tau,
            "ess": (n - burn) / tau, "ess_frac": 1.0 / tau,
            "single_chain_weights": mode_weights(traj[burn:]).round(3).tolist(),
            "true_weights": WEIGHTS.tolist()}


def part_b_coverage(rng) -> dict:
    """Estimate the 3 mode weights with few-long vs many-short chains at FIXED
    total post-burn-in samples. Error = total-variation to the true weights."""
    N_total, burn, repeats = 12000, 500, 30
    Ms = [1, 4, 16, 64, 256, 1024]
    res = {}
    for M in Ms:
        L = N_total // M
        tv = []
        for _ in range(repeats):
            x = rng.normal(0, 3, size=M)               # broad init covers basins
            for _ in range(burn):
                x = langevin_step(x, rng)
            acc = np.zeros(len(MUS))
            for _ in range(L):
                x = langevin_step(x, rng)
                acc += mode_weights(x)
            w = acc / L
            tv.append(0.5 * np.abs(w - WEIGHTS).sum())
        res[str(M)] = {"L": L, "tv_mean": float(np.mean(tv)),
                       "tv_std": float(np.std(tv))}
        print(f"  M={M:<5d} L={L:<5d} TV(weights)={np.mean(tv):.4f} "
              f"+/- {np.std(tv):.4f}")
    return res


def part_c_throughput() -> dict:
    """Wall-clock per batched Langevin step vs #chains on vectorized hardware
    (MPS/torch), using an MLP energy. Throughput = chains*steps / second."""
    out = {"device": str(DEVICE)}
    energy = MLPEnergy(2, hidden=128, n_layers=4).to(DEVICE)
    ebm_tp = {}
    for M in [1, 64, 1024, 16384, 131072]:
        x0 = torch.randn(M, 2, device=DEVICE)
        _ = sgld_sample(energy, x0, n_steps=3, step_size=0.02, noise_scale=0.01)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        t0 = time.time(); steps = 20
        _ = sgld_sample(energy, x0, n_steps=steps, step_size=0.02, noise_scale=0.01)
        if DEVICE.type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        ebm_tp[str(M)] = {"sec_per_step": dt / steps,
                          "samples_per_sec": M * steps / dt}
    out["ebm_langevin_torch"] = ebm_tp
    return out


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(7)

    print("=== A: ESS of one long Langevin chain on a 3-mode mixture ===")
    a = part_a_ess(rng)
    print(f"  tau={a['tau']:.1f}  ESS={a['ess']:.0f} / {a['chain_len']-a['burn']}"
          f"  ({100*a['ess_frac']:.2f}% effective)")
    print(f"  single-chain mode weights={a['single_chain_weights']} "
          f"vs true {a['true_weights']}")

    print("=== B: mode-weight error, few-long vs many-short (N=12000 fixed) ===")
    b = part_b_coverage(rng)

    print("=== C: batched-step throughput (parallel ~ free until saturation) ===")
    c = part_c_throughput()
    for M, d in c["ebm_langevin_torch"].items():
        print(f"  EBM/MPS M={M:<7s} {d['sec_per_step']*1e3:7.2f} ms/step  "
              f"{d['samples_per_sec']:.2e} samples/s")

    out = {"meta": {"target": "1D-3-gaussian-mixture", "mus": MUS.tolist(),
                    "sigma": SIG, "step": STEP,
                    "runtime_s": round(time.time() - t0, 2)},
           "part_a_ess": a, "part_b_coverage": b, "part_c_throughput": c}
    (RESULTS / "e3_parallel_chains.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'e3_parallel_chains.json'}")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    Ms = [int(m) for m in b]
    errs = [b[str(m)]["tv_mean"] for m in Ms]
    stds = [b[str(m)]["tv_std"] for m in Ms]
    ax[0].errorbar(Ms, errs, yerr=stds, marker="o", capsize=3, c="crimson")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("# parallel chains M  (total samples fixed = 12000)")
    ax[0].set_ylabel("TV error of estimated mode weights")
    ax[0].set_title(f"B — one long chain is stuck (τ={a['tau']:.0f}, "
                    f"{a['ess_frac']*100:.1f}% effective);\nmany parallel chains "
                    "recover the true mode weights")
    ax[0].grid(alpha=.3, which="both")

    ebm = c["ebm_langevin_torch"]
    Ms2 = [int(m) for m in ebm]
    sps = [ebm[str(m)]["samples_per_sec"] for m in Ms2]
    ax[1].plot(Ms2, sps, "o-", c="darkgreen")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("# parallel Langevin chains M (batch)")
    ax[1].set_ylabel("samples / second")
    ax[1].set_title(f"C — batched Langevin throughput on {DEVICE}\n"
                    "(near-flat ms/step ⇒ parallel chains ≈ free)")
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "e3_parallel_chains.png", dpi=130)
    print(f"wrote {RESULTS / 'e3_parallel_chains.png'}  ({out['meta']['runtime_s']}s)")


if __name__ == "__main__":
    main()
