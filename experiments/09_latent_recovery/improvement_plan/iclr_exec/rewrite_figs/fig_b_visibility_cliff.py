"""FIG B — The visibility cliff.

Claim proven (C2, refined by E5): rejection of a planted error is a d=0 VISIBILITY
event, not distance-graded proof search.  The raw rejection rate (stated complement)
peaks at d=0 and collapses to a flat floor for d>=1.  The licensed-only overlay —
rejections backed by a genuine written closure-valid derivation — is ZERO at d=0 and
low-and-flat across d>=1: the big d=0 signal is 100% premise read-out ("seeing"),
not derivation ("checking").  Inset: the one real distance-checking spike (inert
lies checked when refutation is 1 cheap hop away) is genuine but rare and decays.

Sources (all Qwen2.5-7B):
  main aff_false_attr:
    raw stated-complement by d  -> results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md
       d0 .247, d1 .031, d2 .049, d3 .050, d5 .021
    licensed-only by d          -> exec_reports/e5_licensed_split_REPORT.md (v2, AUTHORITATIVE)
       d0 .000, d1 .027, d2 .020, d3 .043, d5 .014  (d0 = 111/111 premise-readout, 0 licensed)
  inset cat_false_inert licensed rate -> e5_licensed_split_REPORT.md v2
       d1 .187, d3 .025, dinf .000  (inert-d1 rejection is 96.6% licensed genuine 1-hop checking)
"""
import os
import matplotlib.pyplot as plt
import figstyle as S

S.apply_rc()

d_pos = [0, 1, 2, 3, 4]            # even spacing; label the last as d=5
d_lab = ["0", "1", "2", "3", "5"]
raw = [0.247, 0.031, 0.049, 0.050, 0.021]
lic = [0.000, 0.027, 0.020, 0.043, 0.014]

fig, ax = plt.subplots(figsize=(9.4, 6.0))
S.despine(ax)
S.hgrid(ax)

# shade the d=0 "read-out, not checking" gap
ax.fill_between([d_pos[0]-0.12, d_pos[0]+0.12], [0, 0], [raw[0], raw[0]],
                color=S.BLUE, alpha=0.08, zorder=1)

ax.plot(d_pos, raw, "-o", color=S.BLUE, lw=2.4, ms=9, zorder=4,
        label="Rejected  (stated complement)")
ax.plot(d_pos, lic, "-o", color=S.ORANGE, lw=2.4, ms=9, zorder=5,
        markerfacecolor=S.SURFACE, markeredgewidth=2.2, markeredgecolor=S.ORANGE,
        label="Genuine derivation  (licensed)")

# direct labels
ax.annotate("0.247", (d_pos[0], raw[0]), textcoords="offset points", xytext=(9, 3),
            fontsize=12, color=S.BLUE, fontweight="bold")
ax.annotate("0.000", (d_pos[0], lic[0]), textcoords="offset points", xytext=(9, -16),
            fontsize=12, color=S.ORANGE, fontweight="bold")
ax.annotate("0.031", (d_pos[1], raw[1]), textcoords="offset points", xytext=(0, 11),
            fontsize=9.5, color=S.BLUE)
ax.text(2.9, 0.072, "flat floor  ≈ 0.02–0.05", fontsize=9.5, color=S.INK2,
        ha="center", fontstyle="italic")

# the "seeing, not checking" bracket at d=0
ax.annotate("", xy=(0.28, 0.247), xytext=(0.28, 0.0),
            arrowprops=dict(arrowstyle="<->", color=S.MUTED, lw=1.3))
ax.text(0.55, 0.135,
        "the entire d = 0 signal is\npremise read-out\n(seeing, not checking)",
        fontsize=9.4, color=S.INK2, va="center")

ax.set_xticks(d_pos)
ax.set_xticklabels(d_lab)
ax.set_xlabel("Inferential distance  d  from planted error to its refuting evidence  (rule hops)")
ax.set_ylabel("Rejection rate  (fraction of continuations)")
ax.set_ylim(-0.012, 0.315)
ax.set_xlim(-0.35, 4.35)
ax.legend(loc="upper left", fontsize=10, handlelength=1.6,
          bbox_to_anchor=(0.235, 0.995))

ax.set_title("Models reject a planted error only when its contradiction is directly visible (d = 0)",
             fontsize=12.5, fontweight="bold", color=S.INK, loc="left", pad=12)

# ---- inset: the one genuine distance-check (inert lies, 1 cheap hop) --------
axin = ax.inset_axes([0.60, 0.42, 0.37, 0.40])
S.despine(axin)
S.hgrid(axin)
ipos = [0, 1, 2]
ilab = ["1", "3", "∞"]
inert_lic = [0.187, 0.025, 0.000]
axin.plot(ipos, inert_lic, "-o", color=S.AQUA, lw=2.0, ms=7, zorder=4)
axin.annotate("0.187", (0, 0.187), textcoords="offset points", xytext=(7, 1),
              fontsize=9.5, color=S.AQUA, fontweight="bold")
axin.set_xticks(ipos); axin.set_xticklabels(ilab)
axin.set_xlim(-0.3, 2.3); axin.set_ylim(-0.01, 0.235)
axin.tick_params(labelsize=8)
axin.set_title("genuine checking exists, but is rare:\ninert lies, licensed-derivation rate",
               fontsize=8.6, color=S.INK2, loc="left", pad=5)
axin.set_xlabel("distance d", fontsize=8, color=S.MUTED, labelpad=1)

out = os.path.join(os.path.dirname(__file__) or ".", "fig_b_visibility_cliff.png")
S.save(fig, out)
