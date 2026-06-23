"""Step 4: rho_k computation + figure.

Raw-geometry pilot metric (PLAN.md: exploratory, always normalized by the null):
  D_pert(L, k) = cosine distance( pert_state[k], gold_state[boundary_gold + k] )
  D_null(L, k) = mean over CORRECT null samples of the same quantity
  rho(L, k)    = D_pert / D_null      (<1 contraction, >1 excess divergence)

Offsets k counted from the end of the injected (correct|corrupted) sentence in each
sequence's own tokenization. Behavioral recovery reported alongside.
"""
import os, json, glob
import numpy as np
from common import LAYERS, RESULTS

HORIZON = 96
MIN_NULL = 2

def cos_dist(a, b):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8
    return 1.0 - num / den

def load(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    return {k.replace("layer_", ""): z[k] for k in z.files if k.startswith("layer_")}, meta

def main():
    runs = [json.loads(l) for l in open(os.path.join(RESULTS, "perturbed", "runs.jsonl"))]
    runs = [r for r in runs if "skip" not in r]
    stats = {p: {str(L): [] for L in LAYERS} for p in ["early", "mid", "late"]}
    behav = {p: {"n": 0, "recovered": 0} for p in ["early", "mid", "late"]}
    used, skipped_no_null = 0, 0

    for r in runs:
        safe = r["id"].replace("/", "_").replace("::", "__")
        p = r["point"]
        behav[p]["n"] += 1
        behav[p]["recovered"] += int(r["recovered"])

        gpath = os.path.join(RESULTS, "aligned_gold", safe + ".npz")
        ppath = os.path.join(RESULTS, "perturbed", "hs", f"{safe}__{p}.npz")
        if not (os.path.exists(gpath) and os.path.exists(ppath)):
            continue
        gold_hs, gmeta = load(gpath)
        pert_hs, _ = load(ppath)
        g_b = gmeta["boundaries"][p] - gmeta["ans_start"]   # boundary in gold answer coords

        nulls = []
        for npz in sorted(glob.glob(os.path.join(RESULTS, "null", "hs", f"{safe}__{p}__s*.npz"))):
            nh, nm = load(npz)
            if nm.get("correct"):
                nulls.append(nh)
        if len(nulls) < MIN_NULL:
            skipped_no_null += 1
            continue

        for L in LAYERS:
            Ls = str(L)
            g = gold_hs[Ls][g_b:].astype(np.float32)
            pt = pert_hs[Ls].astype(np.float32)
            K = min(HORIZON, len(g), len(pt))
            if K < 8:
                continue
            d_pert = cos_dist(pt[:K], g[:K])
            d_nulls = []
            for nh in nulls:
                nv = nh[Ls].astype(np.float32)
                Kn = min(K, len(nv))
                d = np.full(K, np.nan, dtype=np.float32)
                d[:Kn] = cos_dist(nv[:Kn], g[:Kn])
                d_nulls.append(d)
            d_null = np.nanmean(np.stack(d_nulls), axis=0)
            rho = np.full(K, np.nan, dtype=np.float32)
            valid = d_null > 1e-6
            rho[valid] = d_pert[valid] / d_null[valid]
            pad = np.full(HORIZON, np.nan, dtype=np.float32)
            pad[:K] = rho
            stats[p][Ls].append(pad)
        used += 1

    out = {"instances_used": used, "skipped_too_few_correct_nulls": skipped_no_null,
           "behavioral": {p: {**behav[p],
                              "recovery_rate": round(behav[p]["recovered"] / behav[p]["n"], 4)
                              if behav[p]["n"] else None} for p in behav},
           "rho_median_by_point_layer": {}}
    for p in stats:
        out["rho_median_by_point_layer"][p] = {}
        for Ls, arrs in stats[p].items():
            if not arrs:
                continue
            M = np.stack(arrs)
            med = np.nanmedian(M, axis=0)
            out["rho_median_by_point_layer"][p][Ls] = {
                "k8": float(np.nanmedian(med[:8])), "k32": float(np.nanmedian(med[8:32])),
                "k96": float(np.nanmedian(med[32:96])) if M.shape[1] >= 33 else None,
                "n": int(M.shape[0]),
            }
    with open(os.path.join(RESULTS, "stats.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out["behavioral"], indent=2))

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, p in zip(axes, ["early", "mid", "late"]):
        for Ls, arrs in stats[p].items():
            if not arrs:
                continue
            M = np.stack(arrs)
            med = np.nanmedian(M, axis=0)
            ax.plot(np.arange(HORIZON), med, label=f"layer {Ls} (n={M.shape[0]})")
        ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6)
        ax.set_title(f"{p} injection — recovery rate "
                     f"{out['behavioral'][p]['recovery_rate']}")
        ax.set_xlabel("tokens after injected sentence (k)")
        ax.set_ylim(0, 3)
    axes[0].set_ylabel(r"$\rho_k$ = D(pert, gold) / D(null, gold)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Latent recovery pilot — Qwen2.5-7B-Instruct on PrOntoQA "
                 r"($\rho_k<1$: contraction toward gold beyond natural spread)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "rho_vs_horizon.png"), dpi=150)
    print("wrote", os.path.join(RESULTS, "stats.json"), "and rho_vs_horizon.png")

if __name__ == "__main__":
    main()
