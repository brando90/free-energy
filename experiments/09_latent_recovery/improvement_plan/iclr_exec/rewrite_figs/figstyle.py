"""Shared style for the TMLR rewrite-kit draft figures.

Palette taken verbatim from the dataviz skill reference instance
(references/palette.md, light mode). These figures are LIGHT-mode only
(paper figures render on a white page), so we use the light column and
validate against the light surface #fcfcfb.

Categorical light slots (fixed order):
  1 blue   #2a78d6
  2 orange #eb6834
  3 aqua   #1baf7a
  4 yellow #eda100
  5 magenta#e87ba4
  6 green  #008300
  7 violet #4a3aa7
  8 red    #e34948
Status: good #0ca30c, critical #d03b3b, warning #fab219.
Ink: primary #0b0b0b, secondary #52514e, muted #898781, grid #e1e0d9,
     baseline #c3c2b7, surface #fcfcfb.

Relief rule (three light slots are sub-3:1 on white): every categorical
mark that carries meaning is direct-labeled.  DRAFT figures — not final art.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- palette ---------------------------------------------------------------
BLUE    = "#2a78d6"
ORANGE  = "#eb6834"
AQUA    = "#1baf7a"
YELLOW  = "#eda100"
MAGENTA = "#e87ba4"
GREEN   = "#008300"
VIOLET  = "#4a3aa7"
RED     = "#e34948"
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

GOOD     = "#0ca30c"
CRITICAL = "#d03b3b"
WARNING  = "#fab219"

SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK2      = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"

# ---- rcParams --------------------------------------------------------------
def apply_rc():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "axes.linewidth": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
    })

def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)

def hgrid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.xaxis.grid(False)

def vgrid(ax):
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.yaxis.grid(False)

def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18, facecolor=SURFACE)
    print("wrote", path)
