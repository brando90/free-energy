"""E4 — The dynamic CD weighting scheme (Q8, Q9).

The note proposes weighting chain samples X^(t) along the trajectory with a
schedule alpha^(t) instead of using only the last sample. We test that idea on
the same exact-gradient RBM testbed as E2.

Two parts:
  A) gradient quality: bias / variance / MSE of each weighting schedule's
     negative phase vs the exact gradient (at the fixed eval_rbm point);
  B) end task: train an RBM from scratch with each schedule and report the
     EXACT negative log-likelihood (Z enumerable), proving whether the better
     gradient trains a better model.

Sharpened hypothesis (from E2): with data-initialized chains, *later* samples
are the better negatives (closer to the model), so late-weighted averaging
should keep CD-K's low bias while shaving variance -> lower gradient MSE than
last-only at equal compute. Uniform / early weighting averages in the
high-bias early samples and should be worse. (The note's "early=negative /
late=positive" wording is discussed in ANSWERS.md.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from common import RBM, flatten_grad

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

NV, NH = 14, 6
N_DATA = 4000
N_SEEDS = 800
BATCH = 128
K = 20
W_SCALE = 1.5


def schedules(k: int) -> dict[str, np.ndarray]:
    t = np.arange(1, k + 1)
    out = {
        "last": np.where(t == k, 1.0, 0.0),
        "uniform": np.ones(k),
        "geom_late": 0.7 ** (k - t),          # emphasize late steps
        "zipf_late": 1.0 / (k - t + 1.0),     # Zipf over distance-from-end
        "early": 0.7 ** t,                    # emphasize early steps
    }
    return {name: w / w.sum() for name, w in out.items()}


def trajectory_stats(eval_rbm: RBM, v0: np.ndarray, k: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Return (k, P) array of flattened neg-phase stats at each Gibbs step."""
    v = v0
    rows = []
    for _ in range(k):
        v, _ = eval_rbm.gibbs_step(v, rng)
        dW, db, dc = eval_rbm.grad_stats(v)
        rows.append(flatten_grad(dW, db, dc))
    return np.array(rows)


def part_a_gradient_quality(eval_rbm: RBM, data: np.ndarray, g_exact: np.ndarray,
                            rng: np.random.Generator) -> dict:
    W = schedules(K)
    per = {name: [] for name in W}
    g_exact_norm = float(np.linalg.norm(g_exact))
    for _ in range(N_SEEDS):
        idx = rng.integers(0, len(data), size=BATCH)
        v0 = data[idx]
        pos = flatten_grad(*eval_rbm.grad_stats(v0))
        traj = trajectory_stats(eval_rbm, v0, K, rng)   # (K, P)
        for name, w in W.items():
            neg = w @ traj                               # weighted negative phase
            per[name].append(pos - neg)
    out = {}
    for name in W:
        S = np.array(per[name])
        mean = S.mean(0)
        bias_vec = mean - g_exact
        var = float(S.var(0, ddof=1).sum())
        bias_sq = float(bias_vec @ bias_vec)
        out[name] = {
            "bias_rel": float(np.linalg.norm(bias_vec) / g_exact_norm),
            "var": var, "bias_sq": bias_sq, "mse": bias_sq + var,
        }
    return out


def train_rbm(data: np.ndarray, weight: np.ndarray, k: int, steps: int,
              lr: float, seed: int, eval_every: int):
    """Train an RBM from a small-weight init with a weighted-trajectory CD
    gradient; record exact NLL along the way."""
    rng = np.random.default_rng(seed)
    model = RBM(0.01 * rng.standard_normal((NV, NH)),
                np.zeros(NV), np.zeros(NH))
    curve = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            nll = float(-model.log_prob_visible(data).mean())
            curve.append((step, nll))
        idx = rng.integers(0, len(data), size=BATCH)
        v0 = data[idx]
        pW, pb, pc = model.grad_stats(v0)
        # weighted negative phase over the K-step trajectory
        v = v0
        nW = np.zeros_like(model.W); nb = np.zeros_like(model.b); nc = np.zeros_like(model.c)
        for t in range(k):
            v, _ = model.gibbs_step(v, rng)
            sW, sb, sc = model.grad_stats(v)
            nW += weight[t] * sW; nb += weight[t] * sb; nc += weight[t] * sc
        # ascend log-likelihood: grad = positive - negative
        model.W += lr * (pW - nW)
        model.b += lr * (pb - nb)
        model.c += lr * (pc - nc)
    return curve


def part_b_training(data: np.ndarray) -> dict:
    W = schedules(K)
    steps, lr, eval_every, seeds = 600, 0.05, 50, (10, 11)
    nll_star = None  # data-RBM's own NLL ~ irreducible; report best achieved
    out = {}
    for name, w in W.items():
        curves = [train_rbm(data, w, K, steps, lr, s, eval_every) for s in seeds]
        # average NLL across seeds at each checkpoint
        xs = [c[0] for c in curves[0]]
        ys = np.mean([[p[1] for p in c] for c in curves], axis=0)
        out[name] = {"steps": xs, "nll": [float(y) for y in ys],
                     "final_nll": float(ys[-1])}
    return out


def main() -> None:
    t0 = time.time()
    data_rbm = RBM.random(NV, NH, np.random.default_rng(0), w_scale=W_SCALE)
    eval_rbm = RBM.random(NV, NH, np.random.default_rng(1), w_scale=W_SCALE)
    data = data_rbm.sample_exact(N_DATA, np.random.default_rng(2))
    g_exact = eval_rbm.exact_grad(data)
    nll_data = float(-data_rbm.log_prob_visible(data).mean())

    rng = np.random.default_rng(123)
    part_a = part_a_gradient_quality(eval_rbm, data, g_exact, rng)
    print("=== Part A: gradient quality vs exact (K=20, equal compute) ===")
    for name, s in sorted(part_a.items(), key=lambda kv: kv[1]["mse"]):
        print(f"  {name:<10s} bias_rel={s['bias_rel']:.4f}  var={s['var']:.3e}"
              f"  mse={s['mse']:.4e}")

    part_b = part_b_training(data)
    print("\n=== Part B: exact final NLL after training (lower=better; "
          f"data-RBM NLL={nll_data:.3f}) ===")
    for name, s in sorted(part_b.items(), key=lambda kv: kv[1]["final_nll"]):
        print(f"  {name:<10s} final_NLL={s['final_nll']:.4f}")

    out = {"meta": {"nv": NV, "nh": NH, "K": K, "n_seeds": N_SEEDS,
                    "batch": BATCH, "w_scale": W_SCALE, "nll_data": nll_data,
                    "weights": {n: w.tolist() for n, w in schedules(K).items()}},
           "part_a_gradient": part_a, "part_b_training": part_b}
    out["meta"]["runtime_s"] = round(time.time() - t0, 2)
    (RESULTS / "e4_cd_weighting.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'e4_cd_weighting.json'}  ({out['meta']['runtime_s']}s)")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    names = list(part_a.keys())
    mses = [part_a[n]["mse"] for n in names]
    biases = [part_a[n]["bias_sq"] for n in names]
    vars = [part_a[n]["var"] for n in names]
    order = np.argsort(mses)
    names_o = [names[i] for i in order]
    x = np.arange(len(names_o))
    ax[0].bar(x, [biases[i] for i in order], label="bias²", color="tomato")
    ax[0].bar(x, [vars[i] for i in order], bottom=[biases[i] for i in order],
              label="variance", color="steelblue")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names_o, rotation=20)
    ax[0].set_ylabel("gradient MSE  (bias² + var)")
    ax[0].set_title("E4 Part A — trajectory-weighted CD gradient MSE\n(K=20, equal compute)")
    ax[0].legend()
    for name in part_b:
        c = part_b[name]
        ax[1].plot(c["steps"], c["nll"], marker="o", ms=3, label=name)
    ax[1].axhline(nll_data, ls="--", c="k", lw=1, label="data-RBM NLL")
    ax[1].set_xlabel("training step"); ax[1].set_ylabel("exact NLL (nats)")
    ax[1].set_title("E4 Part B — training with each schedule")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "e4_cd_weighting.png", dpi=130)
    print(f"wrote {RESULTS / 'e4_cd_weighting.png'}")


if __name__ == "__main__":
    main()
