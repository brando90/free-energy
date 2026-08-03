"""FIG A — Pre-registered confirmatory scoreboard (PLAN2 v1.1, Holm m=6).

Claim proven: the intuitive nearby-evidence-gradient account was registered as a
6-hypothesis Holm family BEFORE generation and only 1 of 6 survived; the two
distance/usability-gradient hypotheses (H1', H4) failed in the WRONG direction.

Sources (all Qwen2.5-7B):
  - EXPD slots H2', H3', H4 + C-0:
    results/EXPD_MATCHED_GRADIENT/EXPD_FULL_REPORT.md  (registered adjudications table)
    results/EXPD_MATCHED_GRADIENT/confirmatory_analysis.json
  - EXPE slots H1', H5', H6:
    results/EXPE_EVIDENCE_MOVER/hypothesis_verdicts.json
    (mirrored in EXPD confirmatory_analysis.json 'registered' block)
  - Holm assembly m=6, alpha=0.05, prereg commit 545b35db (2026-07-02T14:05:29-07:00)
  - H2' strict-DV survival: exec_reports/s1_strict_hopsound_REPORT.md (TOST p=8e-5)

Holm step-down (sorted p ascending), threshold alpha/(m-k+1):
  rank1 H2'  p=0.001202 < .05/6=.00833  -> PASS  (then continue)
  rank2 H3'  p=0.051333 > .05/5=.01000  -> FAIL  (stop; all later FAIL)
  rank3 H6   p=0.444942 > .05/4=.01250  -> FAIL
  rank4 H5'  p=0.628585 > .05/3=.01667  -> FAIL
  rank5 H1'  p=0.789711 > .05/2=.02500  -> FAIL
  rank6 H4   p=0.995740 > .05/1=.05000  -> FAIL
"""
import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import figstyle as S

S.apply_rc()

# Holm-ordered rows (rank 1 = smallest p at top)
ROWS = [
    dict(hid="H2′", suite="EXPD", verdict="PASS",
         pred="Affirmative vs negated polarity are equivalent\n(TOST, ±0.10 margin)",
         effect="diff +0.054   [+0.025, +0.083]",
         p=0.001202, thr="α/6 = .0083", dirtag=None, note="equivalent within ±0.10"),
    dict(hid="H3′", suite="EXPD", verdict="FAIL",
         pred="True-polarity equivalence AND\nneg-false rejected more than neg-true",
         effect="component (ii) at floor:  0.009 vs 0.000",
         p=0.051333, thr="α/5 = .0100", dirtag=None),
    dict(hid="H6", suite="EXPE", verdict="FAIL",
         pred="Moving refuting evidence within the problem\nlowers absorption",
         effect="diff −0.009  [−0.073, +0.055]",
         p=0.444942, thr="α/4 = .0125", dirtag=None),
    dict(hid="H5′", suite="EXPE", verdict="FAIL",
         pred="Genuine refuting evidence rejected more than\na frequency-matched non-refuting match",
         effect="vs FREQ  −0.004  [−0.059, +0.046]",
         p=0.628585, thr="α/3 = .0167", dirtag=None),
    dict(hid="H1′", suite="EXPE", verdict="FAIL",
         pred="Rejection falls as refuting evidence moves\naway (d = 1→2→3)",
         effect="slope +0.011/hop  (0.114→0.132→0.136)",
         p=0.789711, thr="α/2 = .0250", dirtag="wrong direction"),
    dict(hid="H4", suite="EXPD", verdict="FAIL",
         pred="Reuse of a usable lie rises with distance\nto its refutation",
         effect="slope −0.248 logit/hop",
         p=0.995740, thr="α/1 = .0500", dirtag="reversed — reuse peaks at d = 1"),
]

fig, ax = plt.subplots(figsize=(13.0, 7.6))
ax.set_xlim(0, 100)
n = len(ROWS)
ax.set_ylim(-1.6, n + 1.9)
ax.axis("off")

# column x anchors (0..100)
X_CHIP   = 1.5     # verdict chip left
X_HID    = 14.0    # hypothesis id / suite
X_PRED   = 22.5    # registered prediction
X_EFFECT = 54.0    # observed effect
X_P      = 84.5    # p vs threshold

row_h = 1.05

def row_y(i):
    return n - i - 0.5

# header band
ax.text(1.5, n + 1.25, "Pre-registered confirmatory family   (PLAN2 v1.1,  Holm  m = 6)",
        fontsize=16, fontweight="bold", color=S.INK, va="center")
ax.text(1.5, n + 0.62,
        "Registered 2026-07-02 before any generation (commit 545b35db).   "
        "1 of 6 hypotheses passed — the distance/usability gradients failed, two in the wrong direction.",
        fontsize=10.5, color=S.INK2, va="center")

# column headers
hy = n + 0.06
ax.text(X_HID - 2.5, hy, "HYPOTHESIS", fontsize=8.5, color=S.MUTED, va="center", fontweight="bold")
ax.text(X_PRED, hy, "REGISTERED PREDICTION", fontsize=8.5, color=S.MUTED, va="center", fontweight="bold")
ax.text(X_EFFECT, hy, "OBSERVED EFFECT  (95% CI)", fontsize=8.5, color=S.MUTED, va="center", fontweight="bold")
ax.text(X_P, hy, "p   vs   Holm α", fontsize=8.5, color=S.MUTED, va="center", fontweight="bold")
ax.plot([1.0, 99.0], [n + 0.32, n + 0.32], color=S.BASELINE, lw=1.2)

for i, r in enumerate(ROWS):
    y = row_y(i)
    passed = r["verdict"] == "PASS"
    band = "#eefaf0" if passed else ("#fbfbfa" if i % 2 else S.SURFACE)
    ax.add_patch(plt.Rectangle((1.0, y - row_h/2), 98.0, row_h, facecolor=band,
                               edgecolor="none", zorder=0))

    # verdict chip
    chip_col = S.GOOD if passed else S.CRITICAL
    glyph = "✓" if passed else "✗"
    chip = FancyBboxPatch((X_CHIP, y - 0.27), 10.0, 0.54,
                          boxstyle="round,pad=0.02,rounding_size=0.14",
                          facecolor=chip_col, edgecolor="none", zorder=2)
    ax.add_patch(chip)
    ax.text(X_CHIP + 5.0, y, f"{glyph}  {r['verdict']}", color="white", fontsize=11,
            fontweight="bold", ha="center", va="center", zorder=3)

    # hypothesis id + suite
    ax.text(X_HID, y + 0.13, r["hid"], fontsize=13.5, fontweight="bold",
            color=S.INK, va="center")
    ax.text(X_HID, y - 0.27, r["suite"], fontsize=8.5, color=S.MUTED, va="center")

    # prediction
    ax.text(X_PRED, y, r["pred"], fontsize=9.4, color=S.INK2, va="center")

    # effect
    ecol = S.INK if passed else S.INK2
    subline = r.get("dirtag") or r.get("note")
    ax.text(X_EFFECT, y + (0.12 if subline else 0.0), r["effect"], fontsize=9.6,
            color=ecol, va="center")
    if r.get("dirtag"):
        ax.text(X_EFFECT, y - 0.30, "↳ " + r["dirtag"], fontsize=8.4, color=S.CRITICAL,
                va="center", fontstyle="italic", fontweight="bold")
    elif r.get("note"):
        ax.text(X_EFFECT, y - 0.30, "↳ " + r["note"], fontsize=8.4, color=S.GOOD,
                va="center", fontstyle="italic", fontweight="bold")

    # p vs threshold
    ax.text(X_P, y + 0.13, f"{r['p']:.4f}".rstrip('0'), fontsize=11,
            color=S.INK if passed else S.INK2, va="center", fontweight="bold")
    ax.text(X_P, y - 0.27, r["thr"], fontsize=8.4, color=S.MUTED, va="center")

# footer note
ax.plot([1.0, 99.0], [-0.55, -0.55], color=S.BASELINE, lw=1.0)
ax.text(1.5, -1.05,
        "The single pass (H2′, polarity equivalence) also survives the strict hop-sound DV "
        "(TOST p = 8×10⁻⁵, within ±0.10).   H1′ and H4 are the registered distance / usability\n"
        "gradients — both failed, and their point estimates ran opposite to the registered direction.",
        fontsize=8.8, color=S.MUTED, va="center")

out = os.path.join(os.path.dirname(__file__) or ".", "fig_a_prereg_scoreboard.png")
S.save(fig, out)
