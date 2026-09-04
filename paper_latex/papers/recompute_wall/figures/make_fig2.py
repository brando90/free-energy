#!/usr/bin/env python3
"""Regenerate fig2.pdf (13-model readable-vs-computed bars) for the recompute-wall
paper from results/verified_numbers.json ["fig2"]. Style matches make_figs.py
(HelveticaNeue, palette #2a78d6 readable / #eb6834 computed). Whiskers are the
registry's wlo/whi (Wilson 95%, one continuation per program per cell).
    python3 make_fig2.py [--outdir DIR] [--registry PATH] [--title TEXT]
"""
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
DARK, MID, LIGHT, BG = "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--registry", default=None)
ap.add_argument("--title", default="Readable errors get corrected as models improve; computed errors never do")
a = ap.parse_args()
reg_path = a.registry or os.path.join(a.outdir, "..", "results", "verified_numbers.json")
f2 = json.load(open(reg_path))["fig2"]
plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": DARK, "axes.edgecolor": LIGHT, "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})
labels, devs = f2["labels"], f2["developer"]
groups = {"Qwen": "open-weight", "OLMo": "open-weight", "Llama": "open-weight",
          "OpenAI": "OpenAI", "Anthropic": "Anthropic"}
# x positions with a gap between developer groups
xs, gname, x = [], [], 0.0
for i, d in enumerate(devs):
    g = groups[d]
    if i and g != gname[-1]: x += 1.0
    xs.append(x); gname.append(g); x += 1.0
fig, ax = plt.subplots(figsize=(10.0, 5.2))
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.30)
W = 0.38
for series, col, off in (("readable", BLUE, -W / 2), ("computed", ORANGE, W / 2)):
    d = f2[series]
    for i, xi in enumerate(xs):
        r = d["rate"][i]
        ax.bar(xi + off, r, width=W, color=col, zorder=3)
        ax.errorbar(xi + off, r, yerr=[[max(0, r - d["wlo"][i])], [max(0, d["whi"][i] - r)]],
                    color=MID, capsize=2.5, lw=1.2, zorder=4)
ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(LIGHT)
ax.set_yticks([0, .25, .5, .75, 1.0]); ax.set_yticklabels(["0", ".25", ".50", ".75", "1.0"], fontsize=12)
ax.grid(axis="y", color=LIGHT, lw=0.9, alpha=0.55); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.set_ylim(0, 1.06); ax.set_xlim(xs[0] - 0.8, xs[-1] + 0.8)
ax.set_ylabel("error absorbed  (rate)", fontsize=13)
ax.set_title(a.title, fontsize=14.5, pad=12)
# group labels under the axis
for g in dict.fromkeys(gname):
    gx = [xs[i] for i in range(len(xs)) if gname[i] == g]
    ax.text(sum(gx) / len(gx), -0.36, g, ha="center", va="top", fontsize=12, color=MID,
            transform=ax.get_xaxis_transform())
ax.text(xs[0] - 0.4, 0.93, "computed error", color=ORANGE, fontsize=12, fontweight="medium")
ax.text(xs[0] - 0.4, 0.85, "readable error", color=BLUE, fontsize=12, fontweight="medium")
fig.savefig(os.path.join(a.outdir, "fig2.pdf")); fig.savefig(os.path.join(a.outdir, "fig2.png"), dpi=160)
print("wrote fig2.pdf/fig2.png to", a.outdir)
