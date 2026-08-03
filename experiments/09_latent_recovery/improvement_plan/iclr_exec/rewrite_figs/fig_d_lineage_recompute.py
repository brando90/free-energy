"""FIG D — Lineage split on re-read; universal wall on recompute.

Claim proven (C7): absorption of a RE-READABLE planted falsehood is training-recipe-
bound (Anthropic-2025/26 ~0.00; OpenAI + open-weights 0.34-0.58; the 2023 completions
model deepest) — a lab-level, not capability-level, split.  BUT absorption of a
RE-COMPUTABLE falsehood is universal: every model measured on the program-trace
recompute cells absorbs a 5-op chain at 1.00 — including the Anthropic models that
reject re-reads at 0.00.  "Re-read, never recompute."

Sources (descriptive/exploratory arm; scope stated in caption):
  LEFT re-read (echo-corrected global reuse) -> results/EXPH2_FRONTIER_FOLLOWUPS/EXPH2_REPORT.md
     (cross-experiment table) + results/EXPH_API_MODELS/EXPH_REPORT.md
       gpt-3.5-turbo-instruct 2023  0.571 (true completion)
       gpt-4o 2024                  0.340
       gpt-4.1 2025                 0.580
       gpt-5.1 2025 (reasoning off) 0.340
       Qwen2.5-7B 2024 (prefill,locked) 0.373
       claude-opus-4-1 2025         0.000
       claude-haiku-4-5 2025        0.000
       claude-sonnet-4-5 2025       0.000
       claude-sonnet-5 2026         0.010
       claude-opus-4-8 2026         0.000
  RIGHT recompute (EXPH2 regime2, deep_kc5 = 5-op recompute; measured on 3 models):
       Qwen2.5-7B 1.000 ; claude-haiku-4-5 1.000 ; claude-sonnet-4-5 1.000
       (re-read column for those 3 for the paired contrast: 0.373 / 0.000 / 0.000)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as S

S.apply_rc()

OPENAI = S.ORANGE
OPENW  = S.BLUE
ANTHRO = S.AQUA

# (label, year, lab_color, reread) — grouped by lab, chronological within
MODELS = [
    ("gpt-3.5-turbo-instruct", "2023", OPENAI, 0.571),
    ("gpt-4o",                 "2024", OPENAI, 0.340),
    ("gpt-4.1",                "2025", OPENAI, 0.580),
    ("gpt-5.1  (reasoning off)","2025", OPENAI, 0.340),
    ("Qwen2.5-7B",             "2024", OPENW,  0.373),
    ("claude-opus-4-1",        "2025", ANTHRO, 0.000),
    ("claude-haiku-4-5",       "2025", ANTHRO, 0.000),
    ("claude-sonnet-4-5",      "2025", ANTHRO, 0.000),
    ("claude-sonnet-5",        "2026", ANTHRO, 0.010),
    ("claude-opus-4-8",        "2026", ANTHRO, 0.000),
]

from matplotlib.patches import Patch

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 6.6),
                               gridspec_kw=dict(width_ratios=[1.7, 1.0], wspace=0.30))
fig.subplots_adjust(bottom=0.20, top=0.86)

# ---- LEFT: re-read absorption across the lineage (horizontal bars) ----------
S.despine(axL, left=False); axL.spines["left"].set_visible(False)
S.vgrid(axL)
n = len(MODELS)
ypos = np.arange(n)[::-1]  # first model at top
for (lab, yr, col, val), y in zip(MODELS, ypos):
    axL.barh(y, val, height=0.66, color=col, edgecolor=S.SURFACE, linewidth=1.2, zorder=3)
    axL.text(max(val, 0.0) + 0.014, y, f"{val:.3f}", va="center", fontsize=9.6,
             color=S.INK, fontweight="bold")
    axL.text(-0.016, y, f"{lab}  ({yr})", va="center", ha="right", fontsize=9.3, color=S.INK)

axL.set_yticks([])
axL.set_xlim(0, 0.70); axL.set_ylim(-0.7, n - 0.3)
axL.set_xlabel("Re-read absorption\n(echo-corrected reuse of a re-readable planted falsehood)")
axL.set_title("Re-reading a stated falsehood: a lab-level split",
              fontsize=12.5, fontweight="bold", color=S.INK, loc="left", pad=10)

# faint separators between lab groups
for sep in [ypos[3]-0.5, ypos[4]-0.5]:
    axL.axhline(sep, color=S.GRID, lw=1.0, zorder=1)

# lab legend in the empty lower-right of the left panel
lab_handles = [Patch(facecolor=OPENAI, label="OpenAI"),
               Patch(facecolor=OPENW,  label="open-weight (Qwen)"),
               Patch(facecolor=ANTHRO, label="Anthropic")]
axL.legend(handles=lab_handles, loc="center right", fontsize=9.5,
           bbox_to_anchor=(0.995, 0.28), title="lineage", title_fontsize=9,
           handlelength=1.2)

# ---- RIGHT: re-read vs recompute, the 3 models with both channels -----------
S.despine(axR); S.hgrid(axR)
pair_models = ["Qwen2.5-7B", "claude-\nhaiku-4-5", "claude-\nsonnet-4-5"]
reread    = [0.373, 0.000, 0.000]
recompute = [1.000, 1.000, 1.000]
xb = np.arange(3)
w = 0.36
axR.bar(xb - w/2, reread, w, color=S.BLUE, edgecolor=S.SURFACE, linewidth=1.2,
        zorder=3, label="Re-read a stated value")
axR.bar(xb + w/2, recompute, w, color=S.RED, edgecolor=S.SURFACE, linewidth=1.2,
        zorder=3, label="Recompute a 5-op value")
for xi, v in zip(xb - w/2, reread):
    axR.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9.6, color=S.BLUE,
             fontweight="bold")
for xi in xb + w/2:
    axR.text(xi, 1.00 + 0.02, "1.00", ha="center", fontsize=9.6, color=S.RED,
             fontweight="bold")

axR.set_xticks(xb); axR.set_xticklabels(pair_models, fontsize=9)
axR.set_ylim(0, 1.32); axR.set_xlim(-0.6, 2.6)
axR.set_ylabel("Absorption rate")
axR.legend(loc="upper center", fontsize=9.2, ncol=2, bbox_to_anchor=(0.5, 1.15),
           handlelength=1.3, columnspacing=1.2)
axR.set_title("Recomputing a value: a universal wall",
              fontsize=12.5, fontweight="bold", color=S.INK, loc="left", pad=10)

fig.suptitle("Absorption is training-recipe-bound on re-reads, but universal on recomputes",
             fontsize=14, fontweight="bold", color=S.INK, x=0.012, ha="left", y=0.98)
fig.text(0.012, 0.015,
         "Anthropic models reject re-reads at 0.00 yet absorb the recompute at 1.00.   "
         "Descriptive/exploratory arm (single seed, ~$31 API); recompute column measured on 3 models "
         "(EXPH2 regime2, deep_kc5); modes differ across rows (prefill / instruct / completion), "
         "linked via the format-robust echo-corrected-reuse anchor.",
         fontsize=8.0, color=S.MUTED, ha="left")

out = os.path.join(os.path.dirname(__file__) or ".", "fig_d_lineage_recompute.png")
S.save(fig, out)
