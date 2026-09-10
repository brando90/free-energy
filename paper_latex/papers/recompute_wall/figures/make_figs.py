#!/usr/bin/env python3
"""Regenerate fig3.pdf and fig4.pdf for the recompute-wall paper from
results/verified_numbers.json. Style matches the shipped 2026-07-28 figures
(HelveticaNeue/Menlo, palette #2a78d6/#eb6834/#1baf7a). Run from anywhere:
    python3 make_figs.py [--outdir DIR] [--registry PATH]
Error bands/bars are whatever the registry's wlo/whi carry (program-cluster
bootstrap 95% with Wilson fallback at degenerate ceiling cells — see
_sources.ci_policy_fig3_fig4).
"""
import json, os, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
DARK, MID, LIGHT, BG = "#52514e", "#898781", "#c3c2b7", "#fcfcfb"
PANEL = "#f3f2ec"

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument("--registry", default=None)
a = ap.parse_args()
reg_path = a.registry or os.path.join(a.outdir, "..", "results", "verified_numbers.json")
reg = json.load(open(reg_path))

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": DARK, "axes.edgecolor": LIGHT, "axes.labelcolor": DARK,
    "xtick.color": DARK, "ytick.color": DARK,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
})

YT = [0, .25, .50, .75, 1.0]
YL = ["0", ".25", ".50", ".75", "1.0"]

def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LIGHT)
    ax.set_yticks(YT); ax.set_yticklabels(YL, fontsize=13)
    ax.grid(axis="y", color=LIGHT, lw=0.9, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

# ---------------------------------------------------------------- fig3
f3 = reg["fig3"]
ks = f3["k"]
fig, ax = plt.subplots(figsize=(6.0, 5.2))
fig.subplots_adjust(left=0.13, right=0.80, top=0.90, bottom=0.13)
series = [("haiku", "Haiku 4.5", BLUE, 7.0), ("llama8b", "Llama 8B", ORANGE, 4.5),
          ("qwen7b", "Qwen 7B", GREEN, 4.5)]
for key, label, col, lw in series:
    d = f3[key]
    ax.fill_between(ks, d["wlo"], d["whi"], color=col, alpha=0.15, lw=0, zorder=2)
    if key == "qwen7b":
        # k=5 cell is below the pre-registered 50-program floor: draw it
        # unfilled on a dashed segment (reported, not scored).
        ax.plot(ks[:4], d["rate"][:4], color=col, lw=lw, zorder=3,
                marker="o", ms=8, mec="white", mew=1.5)
        ax.plot(ks[3:], d["rate"][3:], color=col, lw=lw, zorder=3,
                ls=(0, (4, 3)), marker="")
        ax.plot([ks[4]], [d["rate"][4]], color=col, zorder=4, marker="o",
                ms=8, mfc=BG, mec=col, mew=2.0, ls="")
    else:
        ax.plot(ks, d["rate"], color=col, lw=lw, zorder=3,
                marker="o", ms=8, mec="white", mew=1.5)
label_y = {"haiku": 1.0, "llama8b": 0.952, "qwen7b": 0.888}
for key, label, col, lw in series:
    ax.text(5.25, label_y[key], label, color=col, fontsize=14, va="center",
            fontweight="medium", clip_on=False)
ax.set_xlim(-0.25, 5.25); ax.set_ylim(-0.03, 1.06)
ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks], fontsize=13)
style_axes(ax)
ax.set_xlabel("verification depth  k  (operations to check)", fontsize=14)
ax.set_ylabel("error absorbed  (rate)", fontsize=14)
ax.set_title("A single check to verify flips absorption on", fontsize=16, pad=14)
ax.annotate("one operation\nis enough", xy=(0.42, 0.56), xytext=(0.85, 0.42),
            fontsize=13, color=DARK, va="top",
            arrowprops=dict(arrowstyle="-", color=MID, lw=1.2,
                            connectionstyle="arc3,rad=-0.3"))
fig.savefig(os.path.join(a.outdir, "fig3.pdf"))
fig.savefig(os.path.join(a.outdir, "fig3.png"), dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- fig4
f4 = reg["fig4"]
models = [("haiku", "Haiku 4.5"), ("llama8b", "Llama 8B"), ("qwen7b", "Qwen 7B")]
arm_cols = [BLUE, ORANGE, GREEN]
fig = plt.figure(figsize=(6.0, 6.7))
gs = fig.add_gridspec(2, 1, height_ratios=[2.05, 1.3], hspace=0.30,
                      left=0.12, right=0.95, top=0.92, bottom=0.03)
ax = fig.add_subplot(gs[0])
for gi, (key, label) in enumerate(models):
    d = f4[key]
    for ai in range(3):
        r = d["rate"][ai]
        ax.bar(gi + (ai - 1) * 0.26, r, width=0.24, color=arm_cols[ai], zorder=3)
        ax.errorbar(gi + (ai - 1) * 0.26, r,
                    yerr=[[r - d["wlo"][ai]], [d["whi"][ai] - r]],
                    color=MID, capsize=3, lw=1.4, zorder=4)
ax.set_xticks(range(3)); ax.set_xticklabels([m[1] for m in models], fontsize=14)
ax.set_ylim(0, 1.06)
style_axes(ax)
ax.set_ylabel("error absorbed  (rate)", fontsize=14)
ax.set_title("Showing the wrong arithmetic does not stop absorption",
             fontsize=15.5, pad=14)

axp = fig.add_subplot(gs[1]); axp.axis("off")
axp.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96,
              boxstyle="round,pad=0.012,rounding_size=0.025",
              transform=axp.transAxes, facecolor=PANEL,
              edgecolor="#e2e1da", lw=1.0, zorder=1))
ins = f4["inset_lines"]
axp.text(0.06, 0.88, "what the model is shown at k=1  (true value 90, planted 100):",
         transform=axp.transAxes, fontsize=11.5, color=MID, va="center")
rows = [("bare", BLUE), ("full", ORANGE), ("partial", GREEN)]
mono = ["Menlo", "Monaco", "DejaVu Sans Mono"]
for i, (arm, col) in enumerate(rows):
    y = 0.70 - i * 0.185
    axp.add_patch(plt.Rectangle((0.06, y - 0.045), 0.022, 0.09,
                  transform=axp.transAxes, facecolor=col, lw=0, zorder=2))
    axp.text(0.10, y, arm, transform=axp.transAxes, fontsize=12.5, color=col,
             va="center")
    axp.text(0.225, y, ins[arm], transform=axp.transAxes, fontsize=11.5,
             color="#3a3936", va="center", family=mono)
axp.text(0.06, 0.10, ins["note_caption"] if "note_caption" in ins else
         "'full' spells out 32 + 58 = 90 — yet the wrong 100 is still copied.",
         transform=axp.transAxes, fontsize=11.5, color=MID, va="center")
fig.savefig(os.path.join(a.outdir, "fig4.pdf"))
fig.savefig(os.path.join(a.outdir, "fig4.png"), dpi=160)
plt.close(fig)
print("wrote fig3.pdf/fig4.pdf (+png previews) to", a.outdir)
