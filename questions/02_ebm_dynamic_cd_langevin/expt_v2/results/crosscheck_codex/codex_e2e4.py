#!/usr/bin/env python3
"""Independent NumPy cross-check for E2 CD bias and E4 weighting."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


NV = 14
NH = 6
N_DATA = 4000
N_SEEDS = 800
BATCH = 128
W_SCALE = 1.5
BIAS_SCALE = 0.2
DATA_SEED = 0
EVAL_SEED = 1
CD_KS = (1, 2, 5, 10, 20, 50)
E4_K = 20
RHO = 0.7


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def enumerate_visible(nv: int) -> np.ndarray:
    ints = np.arange(1 << nv, dtype=np.uint32)
    bits = (ints[:, None] >> np.arange(nv, dtype=np.uint32)) & 1
    return bits.astype(np.float64)


def make_rbm(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "W": W_SCALE * rng.standard_normal((NV, NH)),
        "b": BIAS_SCALE * rng.standard_normal(NV),
        "c": BIAS_SCALE * rng.standard_normal(NH),
    }


def visible_log_weights(v: np.ndarray, rbm: dict[str, np.ndarray]) -> np.ndarray:
    W, b, c = rbm["W"], rbm["b"], rbm["c"]
    fields = c + v @ W
    return v @ b + np.logaddexp(0.0, fields).sum(axis=1)


def visible_probs(v: np.ndarray, rbm: dict[str, np.ndarray]) -> np.ndarray:
    logw = visible_log_weights(v, rbm)
    shift = logw.max()
    unnorm = np.exp(logw - shift)
    return unnorm / unnorm.sum()


def stats_mean(v: np.ndarray, rbm: dict[str, np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    W, c = rbm["W"], rbm["c"]
    vf = v.astype(np.float64, copy=False)
    hprob = sigmoid(c + vf @ W)

    if weights is None:
        inv_n = 1.0 / vf.shape[0]
        w_stat = (vf.T @ hprob) * inv_n
        b_stat = vf.mean(axis=0)
        c_stat = hprob.mean(axis=0)
    else:
        ww = weights.astype(np.float64, copy=False)
        weighted_h = hprob * ww[:, None]
        w_stat = vf.T @ weighted_h
        b_stat = ww @ vf
        c_stat = ww @ hprob

    return np.concatenate([w_stat.ravel(), b_stat, c_stat])


def gibbs_step(v: np.ndarray, rbm: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    W, b, c = rbm["W"], rbm["b"], rbm["c"]
    hprob = sigmoid(c + v @ W)
    h = rng.random(hprob.shape) < hprob
    vprob = sigmoid(b + h @ W.T)
    return rng.random(vprob.shape) < vprob


def schedules() -> dict[str, np.ndarray]:
    t = np.arange(1, E4_K + 1, dtype=np.float64)
    out = {
        "last": np.r_[np.zeros(E4_K - 1), 1.0],
        "uniform": np.ones(E4_K, dtype=np.float64),
        "geom_late": RHO ** (E4_K - t),
        "zipf_late": 1.0 / (E4_K - t + 1.0),
        "early": RHO**t,
    }
    return {name: w / w.sum() for name, w in out.items()}


def summarize(samples: np.ndarray, g_exact: np.ndarray, include_cosine: bool) -> dict[str, float]:
    mean = samples.mean(axis=0)
    bias = mean - g_exact
    centered = samples - mean
    var = float(np.mean(np.sum(centered * centered, axis=1)))
    bias_sq = float(np.dot(bias, bias))
    exact_norm = float(np.linalg.norm(g_exact))
    result = {
        "bias_rel": float(np.linalg.norm(bias) / exact_norm),
        "var": var,
        "mse": bias_sq + var,
    }
    if include_cosine:
        denom = float(np.linalg.norm(mean) * exact_norm)
        result["cosine"] = float(np.dot(mean, g_exact) / denom)
    return result


def format_seq(vals: list[float]) -> str:
    return ", ".join(f"{x:.6g}" for x in vals)


def make_verdict(metrics: dict[str, object]) -> str:
    e2 = metrics["e2"]
    e4 = metrics["e4"]

    cd_biases = [e2[str(k)]["bias_rel"] for k in CD_KS]
    cd_cosines = [e2[str(k)]["cosine"] for k in CD_KS]
    bias_mono = all(cd_biases[i + 1] <= cd_biases[i] + 1e-12 for i in range(len(cd_biases) - 1))
    cosine_to_one = cd_cosines[-1] > cd_cosines[0] and cd_cosines[-1] >= 0.99

    pcd_beats_cd1 = e2["pcd"]["bias_rel"] < e2["1"]["bias_rel"]

    e4_mse = {name: vals["mse"] for name, vals in e4.items()}
    best = min(e4_mse, key=e4_mse.get)
    late = ("last", "geom_late", "zipf_late")
    early = ("uniform", "early")
    late_beats = all(e4_mse[l] < e4_mse[e] for l in late for e in early)
    beat_both = [l for l in late if all(e4_mse[l] < e4_mse[e] for e in early)]
    miss_both = [l for l in late if l not in beat_both]
    beat_text = ", ".join(beat_both) if beat_both else "none"
    miss_text = ", ".join(miss_both) if miss_both else "none"
    miss_verb = "does" if len(miss_both) == 1 else "do"

    lines = [
        "# Codex E2/E4 cross-check verdict",
        f"n_seeds used: {N_SEEDS}.",
        f"Config: nv={NV}, nh={NH}, n_data={N_DATA}, batch={BATCH}, data_seed={DATA_SEED}, eval_seed={EVAL_SEED}, w_scale={W_SCALE}.",
        "E2 CD bias_rel by k [1,2,5,10,20,50]: " + format_seq(cd_biases) + ".",
        "E2 CD cosine by k [1,2,5,10,20,50]: " + format_seq(cd_cosines) + ".",
        f"CONFIRM (a): {'yes' if bias_mono and cosine_to_one else 'no'}; bias_rel is {'monotone decreasing' if bias_mono else 'not strictly monotone'} and cosine at k=50 is {cd_cosines[-1]:.6g}.",
        f"PCD bias_rel={e2['pcd']['bias_rel']:.6g}; CD-1 bias_rel={e2['1']['bias_rel']:.6g}.",
        f"CONFIRM (b): {'yes' if pcd_beats_cd1 else 'no'}; PCD bias is {'lower' if pcd_beats_cd1 else 'not lower'} than CD-1 bias.",
        "E4 MSE by schedule: " + ", ".join(f"{name}={e4_mse[name]:.6g}" for name in ("last", "uniform", "geom_late", "zipf_late", "early")) + ".",
        f"Best E4 schedule by gradient MSE: {best}.",
        f"CONFIRM (c): {'yes' if late_beats else 'no'}; {beat_text} beat both uniform and early, while {miss_text} {miss_verb} not.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    start = time.time()
    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "codex_e2e4.json"
    verdict_path = out_dir / "codex_verdict.md"

    all_v = enumerate_visible(NV)
    data_rbm = make_rbm(DATA_SEED)
    eval_rbm = make_rbm(EVAL_SEED)

    data_rng = np.random.default_rng(DATA_SEED)
    # Advance through parameter draws so data sampling is tied to the same seed stream.
    _ = W_SCALE * data_rng.standard_normal((NV, NH))
    _ = BIAS_SCALE * data_rng.standard_normal(NV)
    _ = BIAS_SCALE * data_rng.standard_normal(NH)
    p_data = visible_probs(all_v, data_rbm)
    data_idx = data_rng.choice(all_v.shape[0], size=N_DATA, replace=True, p=p_data)
    data = all_v[data_idx]

    p_eval = visible_probs(all_v, eval_rbm)
    positive_exact = stats_mean(data, eval_rbm)
    negative_exact = stats_mean(all_v, eval_rbm, weights=p_eval)
    g_exact = positive_exact - negative_exact
    dim = g_exact.size

    cd_samples = {str(k): np.empty((N_SEEDS, dim), dtype=np.float64) for k in CD_KS}
    e4_weights = schedules()
    e4_samples = {name: np.empty((N_SEEDS, dim), dtype=np.float64) for name in e4_weights}

    cd_k_set = set(CD_KS)
    for seed_i in range(N_SEEDS):
        rng = np.random.default_rng(100_000 + seed_i)
        batch_v = data[rng.integers(0, N_DATA, size=BATCH)]
        pos = stats_mean(batch_v, eval_rbm)
        v = batch_v.copy()
        trajectory_stats = []

        for t in range(1, max(CD_KS) + 1):
            v = gibbs_step(v, eval_rbm, rng)
            stat = None
            if t <= E4_K:
                stat = stats_mean(v, eval_rbm)
                trajectory_stats.append(stat)
            if t in cd_k_set:
                if stat is None:
                    stat = stats_mean(v, eval_rbm)
                cd_samples[str(t)][seed_i] = pos - stat

        traj = np.stack(trajectory_stats, axis=0)
        for name, w in e4_weights.items():
            e4_samples[name][seed_i] = pos - (w @ traj)

    pcd_samples = np.empty((N_SEEDS, dim), dtype=np.float64)
    init_rng = np.random.default_rng(200_000)
    pcd_v = data[init_rng.integers(0, N_DATA, size=BATCH)].copy()
    for seed_i in range(N_SEEDS):
        rng = np.random.default_rng(300_000 + seed_i)
        batch_v = data[rng.integers(0, N_DATA, size=BATCH)]
        pos = stats_mean(batch_v, eval_rbm)
        pcd_v = gibbs_step(pcd_v, eval_rbm, rng)
        pcd_samples[seed_i] = pos - stats_mean(pcd_v, eval_rbm)

    e2 = {str(k): summarize(cd_samples[str(k)], g_exact, include_cosine=True) for k in CD_KS}
    e2["pcd"] = summarize(pcd_samples, g_exact, include_cosine=True)
    e4 = {name: summarize(e4_samples[name], g_exact, include_cosine=False) for name in e4_weights}

    elapsed = time.time() - start
    metrics = {
        "e2": e2,
        "e4": e4,
        "meta": {
            "nv": NV,
            "nh": NH,
            "n_data": N_DATA,
            "n_seeds": N_SEEDS,
            "batch": BATCH,
            "w_scale": W_SCALE,
            "bias_scale": BIAS_SCALE,
            "data_seed": DATA_SEED,
            "eval_seed": EVAL_SEED,
            "cd_ks": list(CD_KS),
            "e4_k": E4_K,
            "rho": RHO,
            "gradient_dim": int(dim),
            "g_exact_norm": float(np.linalg.norm(g_exact)),
            "pcd_init": "data_minibatch_seed_200000",
            "estimator_seed_bases": {"cd_e4": 100_000, "pcd": 300_000},
            "runtime_sec": elapsed,
        },
    }

    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    verdict_path.write_text(make_verdict(metrics), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {verdict_path}")
    print(f"runtime_sec={elapsed:.3f}")


if __name__ == "__main__":
    main()
