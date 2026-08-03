"""FIG E — EXPB joint-outcome composition + judge-informed absorption.

Claim proven (instrument lesson, W3): a local certificate (the complement stated
directly in the prefix) does NOT convert absorption into clean recovery — it shifts
continuation STYLE wholesale into 'unparsed' (4.0% -> 31.7%).  The 32B judge (E7)
then reads 86/95 of the local-cert unparsed rows as EXPLICIT REJECTION of the plant,
not silent absorption.  That collapses S2's un-signable worst-case bound
(LOCAL-GLOBAL poisoning in [-0.167,+0.190]) to a real ~-0.10 reduction — but the
mechanism is d=0 visibility (the certificate is a directly-readable contradiction),
so EXPB corroborates the C-0 cliff rather than a distinct claim.  Appendix-bound.

Sources (Qwen2.5-7B only):
  LEFT composition (proportions of 300 rows/arm):
    exec_reports/expb_composition.csv  (=s2w3_expb_joint/expb_composition.csv)
      arm            valid  inj-dep parroted derailed unparsed
      no-cert        .377   .153    .367     .063     .040
      local-cert     .467   .027    .117     .073     .317
      irrelevant     .500   .240    .183     .040     .037
  judge split of unparsed (E7): exec_reports/e7_judge_RUN_REPORT.md Section 4
      local-cert 95 unparsed: 86 explicit-reject, 66 doubt, <=9 possibly absorbing
  RIGHT absorption (injection-dependent) accountings:
    S2 worst-case bounds -> exec_reports/s2w3_expb_joint_REPORT.md (b)
      no-cert [.1533,.1933]  local [.0267,.3433]  irrelevant [.2400,.2767]
    judge-informed points -> e7_judge_RUN_REPORT.md Section 4
      GLOBAL ~.153 ; LOCAL ~.050 ; IRRELEVANT ~.240
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as S

S.apply_rc()

# arms top->bottom
ARMS = ["no certificate\n(baseline)", "local certificate\n(complement stated)", "irrelevant\ncertificate"]
# segment order: valid, inj-dep(absorbed), parroted, derailed, unparsed
SEG = ["closure-valid", "injection-dependent\n(absorbed)", "parroted", "derailed", "unparsed"]
SEGCOL = [S.BLUE, S.ORANGE, S.AQUA, S.YELLOW, S.MAGENTA]
DATA = {
    "no certificate\n(baseline)":            [0.377, 0.153, 0.367, 0.063, 0.040],
    "local certificate\n(complement stated)":[0.467, 0.027, 0.117, 0.073, 0.317],
    "irrelevant\ncertificate":               [0.500, 0.240, 0.183, 0.040, 0.037],
}

from matplotlib.patches import Patch

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.6, 6.0),
                               gridspec_kw=dict(width_ratios=[1.85, 1.0], wspace=0.28))
fig.subplots_adjust(top=0.80, bottom=0.12)

# label ink per segment (white on dark fills, black on light fills)
LABINK = ["white", "white", "white", S.INK, S.INK]

# ---- LEFT: stacked horizontal composition ----------------------------------
S.despine(axL, left=False); axL.spines["left"].set_visible(False)
S.vgrid(axL)
yc = {"no certificate\n(baseline)": 2.2,
      "local certificate\n(complement stated)": 1.2,
      "irrelevant\ncertificate": 0.2}
barh = 0.5
for arm in ARMS:
    y = yc[arm]
    left = 0.0
    for j, (seg, col) in enumerate(zip(SEG, SEGCOL)):
        v = DATA[arm][j]
        axL.barh(y, v, left=left, height=barh, color=col,
                 edgecolor=S.SURFACE, linewidth=1.8, zorder=3)
        if v >= 0.035:
            axL.text(left + v/2, y, f"{v*100:.0f}%", ha="center", va="center",
                     fontsize=9.0, color=LABINK[j], fontweight="bold", zorder=4)
        left += v
    axL.text(-0.012, y, arm, ha="right", va="center", fontsize=9.8, color=S.INK)

axL.set_yticks([]); axL.set_xlim(0, 1.0); axL.set_ylim(-0.55, 3.35)
axL.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axL.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
axL.set_xlabel("Share of continuations  (n = 300 per arm)")

# segment legend (proper, horizontal, above the bars)
seg_handles = [Patch(facecolor=c, label=s.replace("\n", " ")) for s, c in zip(SEG, SEGCOL)]
axL.legend(handles=seg_handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
           ncol=5, fontsize=8.2, handlelength=1.1, columnspacing=1.1,
           handletextpad=0.5, borderpad=0.4)

# callout on local-cert unparsed — placed in the gap between baseline & local bars
un_start = sum(DATA["local certificate\n(complement stated)"][:4])
axL.annotate(
    "E7 32B judge reads 86 / 95 of these unparsed rows as\n"
    "EXPLICIT REJECTION of the plant  (≤9 possibly absorbing)",
    xy=(un_start + 0.317/2, 1.2 + barh/2),
    xytext=(0.30, 1.78),
    fontsize=8.6, color=S.INK, ha="left", va="center",
    arrowprops=dict(arrowstyle="-|>", color=S.MAGENTA, lw=1.6,
                    connectionstyle="arc3,rad=0.25"))

# ---- RIGHT: absorption accountings -----------------------------------------
S.despine(axR); S.hgrid(axR)
arm_short = ["baseline\n(no cert)", "local\ncert", "irrelevant\ncert"]
parsed  = [0.153, 0.027, 0.240]
wc_lo   = [0.1533, 0.0267, 0.2400]
wc_hi   = [0.1933, 0.3433, 0.2767]
judge   = [0.153, 0.050, 0.240]
xb = np.arange(3)

# worst-case envelope as a light vertical band
for xi, lo, hi in zip(xb, wc_lo, wc_hi):
    axR.add_patch(plt.Rectangle((xi-0.16, lo), 0.32, hi-lo, facecolor=S.MUTED,
                  alpha=0.16, edgecolor="none", zorder=1))
axR.plot([], [], color=S.MUTED, alpha=0.35, lw=8, label="S2 worst-case bound")
# parsed-only points
axR.scatter(xb-0.0, parsed, s=70, color=S.INK2, zorder=4, marker="o",
            label="parsed-only", edgecolor=S.SURFACE, linewidth=1.2)
# judge-informed points
axR.scatter(xb, judge, s=150, color=S.ORANGE, zorder=5, marker="D",
            edgecolor=S.SURFACE, linewidth=1.4, label="E7 judge-informed")
for xi, v in zip(xb, judge):
    axR.text(xi+0.17, v, f"{v:.2f}", va="center", fontsize=9.3, color=S.ORANGE,
             fontweight="bold")

# highlight the local-cert collapse
axR.annotate("worst-case spans 0\n→ judge fixes it low",
             xy=(1, 0.05), xytext=(1.05, 0.30), fontsize=8.2, color=S.INK2,
             arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.2),
             ha="left")

axR.set_xticks(xb); axR.set_xticklabels(arm_short, fontsize=9)
axR.set_xlim(-0.6, 2.7); axR.set_ylim(0, 0.40)
axR.set_ylabel("Absorption  (injection-dependent rate)")
axR.legend(loc="upper right", fontsize=8.3, handlelength=1.3)
axR.set_title("Judge tightens the absorption bound",
              fontsize=12, fontweight="bold", color=S.INK, loc="left", pad=10)

fig.suptitle("EXPB (appendix): the certificate breaks the parser, and the 'unparsed' tail is rejection, not absorption",
             fontsize=12.5, fontweight="bold", color=S.INK, x=0.012, ha="left", y=0.995)

out = os.path.join(os.path.dirname(__file__) or ".", "fig_e_expb_joint_outcome.png")
S.save(fig, out)
