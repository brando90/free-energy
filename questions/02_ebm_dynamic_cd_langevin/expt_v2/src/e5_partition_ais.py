"""E5 — How intractable is Z, really? Enumeration blow-up + AIS (RQ1, Q6).

The note asks: if energy/the manifold is all we need, why fight the partition
function `Z(theta)` — and does it only *seem* intractable because we never
actually see it?

Two measurements:

  (A) Exact `log Z` by enumerating `2^{nv}` states, timed as `nv` grows: the
      cost is provably exponential, so beyond ~30-50 bits it is hopeless. This
      is the concrete, undeniable "intractable".
  (B) Annealed Importance Sampling (Neal 2001; Salakhutdinov & Murray 2008)
      estimates `log Z` with a tempered path uniform -> target; it tracks the
      exact value where enumeration is still feasible and keeps working past it.

Payoff (RQ1/Q6): `Z` is genuinely exponential to compute, but it is needed
*only to report normalized likelihood*. Sampling (Gibbs/Langevin), the energy
landscape, and score-matching training never touch `Z` (that is exactly what
E1-E4 do). So "we don't see Z" is right for *use* and wrong for *evaluation* —
and AIS is the "messy bypass" for the evaluation case.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import RBM, enumerate_binary

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def part_a_enumeration_cost() -> dict:
    """Time exact log Z by enumeration as nv grows (exponential blow-up)."""
    rows = []
    for nv in [8, 10, 12, 14, 16, 18, 20]:
        rbm = RBM.random(nv, 6, np.random.default_rng(nv), w_scale=1.0)
        t0 = time.time()
        _ = rbm.log_Z()
        dt = time.time() - t0
        rows.append({"nv": nv, "n_states": 2 ** nv, "sec": dt})
        print(f"  nv={nv:<3d} states={2**nv:>9d}  log_Z time={dt*1e3:8.2f} ms")
    # fit log10(sec) ~ a*nv + b over the larger sizes, extrapolate
    big = [r for r in rows if r["nv"] >= 12]
    nvs = np.array([r["nv"] for r in big], float)
    logs = np.log10(np.array([r["sec"] for r in big]))
    a, b = np.polyfit(nvs, logs, 1)
    extrap = {nv: float(10 ** (a * nv + b)) for nv in [30, 40, 50, 100]}
    print(f"  fit: sec ~ 10^({a:.3f}*nv + {b:.2f})  "
          f"=> nv=50: {extrap[50]:.1e}s, nv=100: {extrap[100]:.1e}s")
    return {"rows": rows, "fit_slope_per_bit": float(a),
            "extrapolated_sec": extrap}


def ais_logZ(rbm: RBM, n_particles: int, n_levels: int, n_mh: int,
             rng: np.random.Generator) -> float:
    """AIS estimate of log Z via the tempered-marginal path
    f_beta(v) = exp(-beta * F(v)), beta: 0 -> 1.  Z_0 = 2^{nv} (uniform)."""
    nv = rbm.nv
    betas = np.linspace(0.0, 1.0, n_levels + 1)
    v = rng.integers(0, 2, size=(n_particles, nv)).astype(np.float64)  # ~ f_0 unif
    logw = np.zeros(n_particles)
    F = rbm.free_energy(v)
    for t in range(1, len(betas)):
        b_prev, b_cur = betas[t - 1], betas[t]
        logw += -(b_cur - b_prev) * F                       # importance update
        # MH transitions leaving exp(-b_cur F) invariant: random bit flips
        for _ in range(n_mh):
            j = rng.integers(0, nv, size=n_particles)
            v_prop = v.copy()
            rows = np.arange(n_particles)
            v_prop[rows, j] = 1.0 - v_prop[rows, j]
            F_prop = rbm.free_energy(v_prop)
            accept = rng.random(n_particles) < np.exp(-b_cur * (F_prop - F))
            v[accept] = v_prop[accept]
            F[accept] = F_prop[accept]
    logZ0 = nv * np.log(2.0)
    m = logw.max()
    log_mean_w = m + np.log(np.mean(np.exp(logw - m)))
    return float(logZ0 + log_mean_w)


def part_b_ais(rng) -> dict:
    nv = 14
    rbm = RBM.random(nv, 6, np.random.default_rng(123), w_scale=1.5)
    exact = rbm.log_Z()
    print(f"  exact log_Z (nv={nv}) = {exact:.4f}")
    out = {"nv": nv, "exact_logZ": exact, "runs": []}
    for n_levels in [10, 50, 200, 1000]:
        ests = [ais_logZ(rbm, 200, n_levels, 1, np.random.default_rng(s))
                for s in range(5)]
        mean, std = float(np.mean(ests)), float(np.std(ests))
        out["runs"].append({"n_levels": n_levels, "mean": mean, "std": std,
                            "abs_err": abs(mean - exact)})
        print(f"  AIS levels={n_levels:<5d} logZ={mean:7.4f} +/- {std:.4f}  "
              f"|err|={abs(mean-exact):.4f}")
    return out


def main() -> None:
    t0 = time.time()
    print("=== A: exact log Z by enumeration — exponential blow-up ===")
    a = part_a_enumeration_cost()
    print("=== B: AIS estimate of log Z tracks exact (the messy bypass) ===")
    b = part_b_ais(np.random.default_rng(0))

    out = {"meta": {"runtime_s": round(time.time() - t0, 2)},
           "part_a_enumeration": a, "part_b_ais": b,
           "z_needed_for": {"normalized_likelihood": True,
                            "sampling_gibbs_langevin": False,
                            "energy_landscape_argmax": False,
                            "score_matching_training": False,
                            "contrastive_divergence_training": False}}
    (RESULTS / "e5_partition_ais.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'e5_partition_ais.json'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    rows = a["rows"]
    nvs = [r["nv"] for r in rows]
    secs = [r["sec"] for r in rows]
    ax[0].semilogy(nvs, secs, "o-", label="measured")
    xx = np.array([8, 50])
    ax[0].semilogy(xx, 10 ** (a["fit_slope_per_bit"] * xx
                   + np.log10(secs[-1]) - a["fit_slope_per_bit"] * nvs[-1]),
                   "--", c="gray", label="exp fit")
    ax[0].axhline(1.0, c="r", ls=":", lw=1, label="1 second")
    ax[0].set_xlabel("nv (visible units)"); ax[0].set_ylabel("time for exact log Z (s)")
    ax[0].set_title("A — exact Z is exponential in dimension\n"
                    f"(extrapolated nv=50 ≈ {a['extrapolated_sec'][50]:.0e}s)")
    ax[0].legend(); ax[0].grid(alpha=.3, which="both")

    runs = b["runs"]
    lv = [r["n_levels"] for r in runs]
    err = [r["abs_err"] for r in runs]
    ax[1].loglog(lv, err, "o-", c="purple")
    ax[1].set_xlabel("# AIS intermediate distributions")
    ax[1].set_ylabel("|AIS log Z − exact log Z|")
    ax[1].set_title(f"B — AIS converges to exact log Z\n(exact={b['exact_logZ']:.2f}, "
                    "the messy bypass for evaluation)")
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(RESULTS / "e5_partition_ais.png", dpi=130)
    print(f"wrote {RESULTS / 'e5_partition_ais.png'}  ({out['meta']['runtime_s']}s)")


if __name__ == "__main__":
    main()
