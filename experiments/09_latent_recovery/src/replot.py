"""Final paper figure: rho_func medians with bootstrap CI bands, log-y."""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RESULTS

d = json.load(open(os.path.join(RESULTS, "functional_rho.json")))
colors = {"wrong": "tab:red", "distractor": "tab:orange", "contradiction": "tab:purple",
          "paraphrase": "tab:green", "falsehood": "tab:brown", "negstep": "tab:blue"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
H = d["h_max"]
for ax, p in zip(axes, ["early", "mid", "late"]):
    for fam, fo in d["per_family"].items():
        if p not in fo["points"]:
            continue
        v = fo["points"][p]
        med = np.array([np.nan if x is None else x for x in v["rho_func_median"]], float)
        lo = np.array([np.nan if x is None else x for x in v["ci_lo"]], float)
        hi = np.array([np.nan if x is None else x for x in v["ci_hi"]], float)
        xs = np.arange(1, H + 1)
        ax.plot(xs, med, marker="o", ms=4, color=colors[fam], label=f"{fam} (n={v['n_instances']})")
        ax.fill_between(xs, lo, hi, color=colors[fam], alpha=0.15)
    ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"{p} injection")
    ax.set_xlabel("sentences after injection (h)")
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel(r"$\rho^{func}_h$ (log scale)")
axes[0].legend(fontsize=9)
fig.suptitle("Null-calibrated functional divergence from gold trajectory — Qwen2.5-7B-Instruct, PrOntoQA "
             r"(median, 95% bootstrap CI; $\rho=1$: indistinguishable from valid-trajectory spread)")
fig.tight_layout()
fig.savefig(os.path.join(RESULTS, "fig_rho_func_log.png"), dpi=170)
fig.savefig(os.path.join(RESULTS, "fig_rho_func_log.pdf"))
print("wrote fig_rho_func_log.{png,pdf}")
