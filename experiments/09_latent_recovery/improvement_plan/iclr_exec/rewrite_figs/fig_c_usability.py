"""FIG C — Usability, not distance, governs absorption (and it reproduces in nature).

Claim proven (C3): a derivationally USABLE planted lie is silently reused at ~0.8
regardless of how close its refuting evidence sits (FLAT across distance — lead with
the flatness, not the asymmetry), while an INERT lie is reused at ~0.  The same
usable-vs-inert asymmetry reproduces in the model's OWN natural errors.

Sources (all Qwen2.5-7B):
  LEFT planted (EXPD derivational injection-dependence, positions pooled):
    results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md
      cat_false_usable:  d1 .863, d3 .768, dinf .787   (pooled 0.81 [0.78,0.83])
      cat_false_inert :  d1 .000, d3 .000, dinf .023
  RIGHT natural (per fresh entity-fact error, PrOntoQA, n=135):
    improvement_plan/natural_errors/NATURAL_ERRORS_REPORT.md  Section 5
      design-usable: 15/28  = 0.54  [0.36, 0.70]
      design-inert :  5/107 = 0.047 [0.02, 0.10]
      planted reference: usable 0.81, inert <=0.02
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as S

S.apply_rc()

USABLE = S.BLUE
INERT  = S.ORANGE

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.6, 5.5),
                               gridspec_kw=dict(width_ratios=[1.55, 1.0], wspace=0.28))

# ---- LEFT: planted, reuse vs distance --------------------------------------
S.despine(axL); S.hgrid(axL)
x = [0, 1, 2]
xlab = ["1", "3", "∞"]
u = [0.863, 0.768, 0.787]
inv = [0.000, 0.000, 0.023]

axL.fill_between(x, inv, u, color=S.BLUE, alpha=0.06, zorder=1)
axL.plot(x, u, "-o", color=USABLE, lw=2.4, ms=9, zorder=4, label="Usable lie")
axL.plot(x, inv, "-o", color=INERT, lw=2.4, ms=9, zorder=5,
         markerfacecolor=S.SURFACE, markeredgewidth=2.0, markeredgecolor=INERT,
         label="Inert lie")

axL.annotate("0.86", (0, .863), textcoords="offset points", xytext=(-2, 10),
             fontsize=10.5, color=USABLE, fontweight="bold")
axL.annotate("0.79", (2, .787), textcoords="offset points", xytext=(4, 8),
             fontsize=10.5, color=USABLE, fontweight="bold")
axL.text(1.0, 0.905, "reuse ≈ 0.8, flat in distance", fontsize=10, color=USABLE,
         ha="center", fontstyle="italic")
axL.annotate("≈ 0.00", (1, 0.0), textcoords="offset points", xytext=(-8, 8),
             fontsize=10.5, color=INERT, fontweight="bold")

axL.set_xticks(x); axL.set_xticklabels(xlab)
axL.set_xlim(-0.25, 2.25); axL.set_ylim(-0.03, 1.0)
axL.set_xlabel("Distance  d  to refuting evidence  (rule hops)")
axL.set_ylabel("Silent downstream reuse rate")
axL.legend(loc="center right", fontsize=10.5, handlelength=1.6)
axL.set_title("Planted errors  (EXPD)", fontsize=12, fontweight="bold",
              color=S.INK, loc="left", pad=10)

# ---- RIGHT: natural errors, usable vs inert with CIs -----------------------
S.despine(axR); S.hgrid(axR)
bx = [0, 1]
vals = [0.54, 0.047]
lo   = [0.36, 0.02]
hi   = [0.70, 0.10]
err = [[vals[i]-lo[i] for i in range(2)], [hi[i]-vals[i] for i in range(2)]]
cols = [USABLE, INERT]
bars = axR.bar(bx, vals, width=0.56, color=cols, zorder=3,
               edgecolor=S.SURFACE, linewidth=1.5)
axR.errorbar(bx, vals, yerr=err, fmt="none", ecolor=S.INK2, elinewidth=1.6,
             capsize=6, capthick=1.6, zorder=4)
for xi, v in zip(bx, vals):
    axR.text(xi, v + (0.075 if xi == 0 else 0.10), f"{v:.2f}", ha="center",
             fontsize=12, fontweight="bold", color=cols[xi])

# planted reference markers
axR.axhline(0.81, color=USABLE, lw=1.3, ls=(0, (4, 3)), alpha=0.8, zorder=2)
axR.text(1.34, 0.81, "planted usable 0.81", fontsize=8.4, color=USABLE,
         va="center", ha="right")
axR.axhline(0.01, color=INERT, lw=1.3, ls=(0, (4, 3)), alpha=0.8, zorder=2)
axR.text(1.34, 0.055, "planted inert ≤0.02", fontsize=8.4, color=INERT,
         va="center", ha="right")

axR.set_xticks(bx)
axR.set_xticklabels(["Usable", "Inert"])
axR.set_xlim(-0.6, 1.6); axR.set_ylim(0, 1.0)
axR.set_ylabel("Downstream reuse rate")
axR.set_title("Natural errors  (n = 135, PrOntoQA)", fontsize=12,
              fontweight="bold", color=S.INK, loc="left", pad=10)
axR.text(0.5, -0.165, "the model's OWN mistakes reproduce the ~11× asymmetry",
         transform=axR.transAxes, ha="center", fontsize=9, color=S.INK2,
         fontstyle="italic")

fig.suptitle("Absorption tracks derivational usability, not distance to refutation",
             fontsize=13.5, fontweight="bold", color=S.INK, x=0.012, ha="left", y=1.005)

out = os.path.join(os.path.dirname(__file__) or ".", "fig_c_usability_asymmetry.png")
S.save(fig, out)
