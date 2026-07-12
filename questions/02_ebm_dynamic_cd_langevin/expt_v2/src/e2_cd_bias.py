"""E2 — Why is short-run Contrastive Divergence "bad"? Measure the bias.

Answers Q6 (are we sampling the model? is `p0` constant?) and Q7 (is short-run
CD bad because of bias or variance?).

Testbed: a tiny RBM whose partition function and **exact** maximum-likelihood
gradient are computable by enumerating all `2^{nv}` visible states. We then
compare the exact gradient against CD-`k` and PCD estimators, decomposing the
error into bias, variance, and MSE.

Key design point (see header of `codex_crosscheck_spec.md`): CD bias *vanishes*
at the data-generating params, because a chain started at data is already at
equilibrium there. So we measure the gradient of an independent `eval` RBM on
data drawn from a different `data` RBM — i.e. the realistic mid-training regime
where model != data.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import RBM, cosine, flatten_grad

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

NV, NH = 14, 6
N_DATA = 4000
N_SEEDS = 800
BATCH = 128
K_LIST = [1, 2, 5, 10, 20, 50]
# Sharper weights => slower Gibbs mixing => short-run CD is meaningfully biased
# (so the bias->0-as-k-grows trend is visible above the MC noise floor).
W_SCALE = 1.5


def cd_k_negphase(eval_rbm: RBM, v0: np.ndarray, k: int, rng: np.random.Generator):
    """Run k Gibbs steps from v0 under eval_rbm; return final-state stats."""
    v = v0
    for _ in range(k):
        v, _ = eval_rbm.gibbs_step(v, rng)
    return eval_rbm.grad_stats(v)


def cd_k_estimator(eval_rbm: RBM, data: np.ndarray, k: int, rng: np.random.Generator):
    """One CD-k gradient sample: minibatch positive phase − k-step negative."""
    idx = rng.integers(0, len(data), size=BATCH)
    v0 = data[idx]
    pW, pb, pc = eval_rbm.grad_stats(v0)            # positive phase (data)
    nW, nb, nc = cd_k_negphase(eval_rbm, v0, k, rng)  # negative phase (chain)
    return flatten_grad(pW - nW, pb - nb, pc - nc)


def pcd_estimator_stream(eval_rbm: RBM, data: np.ndarray, n_samples: int,
                         rng: np.random.Generator, burn_in: int = 200):
    """PCD: persistent chains evolve 1 Gibbs step per call (amortized burn-in).

    Returns a list of gradient samples (positive phase = fresh data minibatch).
    """
    # init persistent chains from a data minibatch, then burn in
    idx = rng.integers(0, len(data), size=BATCH)
    chains = data[idx].copy()
    for _ in range(burn_in):
        chains, _ = eval_rbm.gibbs_step(chains, rng)
    out = []
    for _ in range(n_samples):
        chains, _ = eval_rbm.gibbs_step(chains, rng)
        nW, nb, nc = eval_rbm.grad_stats(chains)
        bidx = rng.integers(0, len(data), size=BATCH)
        pW, pb, pc = eval_rbm.grad_stats(data[bidx])
        out.append(flatten_grad(pW - nW, pb - nb, pc - nc))
    return np.array(out)


def summarize(samples: np.ndarray, g_exact: np.ndarray) -> dict:
    mean = samples.mean(0)
    bias_vec = mean - g_exact
    var = float(samples.var(0, ddof=1).sum())     # tr Cov
    bias_sq = float(bias_vec @ bias_vec)
    return {
        "bias_rel": float(np.linalg.norm(bias_vec) / np.linalg.norm(g_exact)),
        "cosine": cosine(mean, g_exact),
        "var": var,
        "bias_sq": bias_sq,
        "mse": bias_sq + var,
    }


def main() -> None:
    t0 = time.time()
    data_rbm = RBM.random(NV, NH, np.random.default_rng(0), w_scale=W_SCALE)
    eval_rbm = RBM.random(NV, NH, np.random.default_rng(1), w_scale=W_SCALE)
    data = data_rbm.sample_exact(N_DATA, np.random.default_rng(2))

    g_exact = eval_rbm.exact_grad(data)
    g_exact_norm = float(np.linalg.norm(g_exact))

    rng = np.random.default_rng(123)
    out: dict = {"meta": {
        "nv": NV, "nh": NH, "n_data": N_DATA, "n_seeds": N_SEEDS,
        "batch": BATCH, "k_list": K_LIST, "g_exact_norm": g_exact_norm,
        "note": "eval_rbm (seed1) measured on data from data_rbm (seed0)",
    }, "e2": {}}

    for k in K_LIST:
        samples = np.array([cd_k_estimator(eval_rbm, data, k, rng)
                            for _ in range(N_SEEDS)])
        out["e2"][str(k)] = summarize(samples, g_exact)
        s = out["e2"][str(k)]
        print(f"CD-{k:<3d}  bias_rel={s['bias_rel']:.4f}  cos={s['cosine']:.4f}"
              f"  var={s['var']:.3e}  mse={s['mse']:.3e}")

    pcd_samples = pcd_estimator_stream(eval_rbm, data, N_SEEDS, rng)
    out["e2"]["pcd"] = summarize(pcd_samples, g_exact)
    s = out["e2"]["pcd"]
    print(f"PCD    bias_rel={s['bias_rel']:.4f}  cos={s['cosine']:.4f}"
          f"  var={s['var']:.3e}  mse={s['mse']:.3e}")

    out["meta"]["runtime_s"] = round(time.time() - t0, 2)
    (RESULTS / "e2_cd_bias.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'e2_cd_bias.json'}  ({out['meta']['runtime_s']}s)")

    # plot: bias_rel and cosine vs k
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = K_LIST
    bias = [out["e2"][str(k)]["bias_rel"] for k in ks]
    cos = [out["e2"][str(k)]["cosine"] for k in ks]
    mse = [out["e2"][str(k)]["mse"] for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(ks, bias, "o-", label="CD-k relative bias")
    ax[0].axhline(out["e2"]["pcd"]["bias_rel"], ls="--", c="green",
                  label=f"PCD bias={out['e2']['pcd']['bias_rel']:.3f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("k (Gibbs steps)")
    ax[0].set_ylabel("relative bias  ‖E[ĝ]−g*‖/‖g*‖")
    ax[0].set_title("CD-k gradient bias falls with k"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax2 = ax[1]
    ax2.plot(ks, cos, "s-", c="purple", label="cos(E[ĝ], g*)")
    ax2.set_xscale("log"); ax2.set_xlabel("k (Gibbs steps)")
    ax2.set_ylabel("cosine to exact gradient")
    ax2.set_title("Direction aligns with the exact gradient as k grows")
    ax2.grid(alpha=.3); ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "e2_cd_bias.png", dpi=130)
    print(f"wrote {RESULTS / 'e2_cd_bias.png'}")


if __name__ == "__main__":
    main()
