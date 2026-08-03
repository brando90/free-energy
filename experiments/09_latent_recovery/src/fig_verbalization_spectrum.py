"""Regenerate fig_verbalization_spectrum.pdf from result JSONs (reproducible).

Plots, for each model at mid 1-hop-falsehood (negstep) injection, verbalized doubt (x)
vs validated recovery (y). Shows the dissociation: doubt spans ~0 (silent) to .67 (R1)
while recovery stays in a narrow band.
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PAPER = os.path.join(BASE, "..", "..", "paper_latex", "papers", "latent_recovery")


def negstep_mid(resdir):
    d = json.load(open(os.path.join(BASE, resdir, "validated_summary_negstep.json")))
    c = d["mid"]
    n = c["n"]
    return c.get("acknowledged", 0) / n, c["valid_rederivation"] / n, n


def main():
    pts = []  # (label, doubt, recovery, color)
    pts.append(("Qwen-1.5B", *negstep_mid("results_1p5b"), "tab:gray"))
    pts.append(("Qwen-7B", *negstep_mid("results"), "tab:blue"))
    pts.append(("Qwen-32B", *negstep_mid("results_32b"), "tab:cyan"))
    pts.append(("OLMo-2-7B", *negstep_mid("results_olmo"), "tab:green"))
    pts.append(("Llama-3.1-8B", *negstep_mid("results_llama"), "tab:olive"))
    # R1: single negstep cell (visible-channel doubt = ack_visible)
    r1 = json.load(open(os.path.join(BASE, "results", "r1", "summary.json")))["negstep"]
    pts.append(("R1-Distill-7B", r1["ack_visible"], r1["valid_rederivation"], r1["n"], "tab:red"))

    # manual label offsets (x,y in data coords; ha) to declutter the silent cluster
    off = {"Qwen-1.5B": (0.03, -0.025, "left"), "OLMo-2-7B": (0.03, -0.045, "left"),
           "Llama-3.1-8B": (0.03, 0.028, "left"), "Qwen-7B": (0.015, -0.02, "left"),
           "Qwen-32B": (0.015, 0.0, "left"), "R1-Distill-7B": (-0.015, 0.0, "right")}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for label, doubt, rec, n, col in pts:
        ax.scatter(doubt, rec, s=90, color=col, zorder=3, edgecolor="k", linewidth=0.5)
        dx, dy, ha = off[label]
        ax.annotate(f"{label} (rec {rec:.2f}, n={n})", (doubt, rec),
                    xytext=(doubt + dx, rec + dy), ha=ha, va="center", fontsize=8.5,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="0.6")
                    if abs(dy) > 0.02 else None)
    ax.axhspan(0.53, 0.74, color="tab:blue", alpha=0.07, zorder=0)
    ax.set_xlabel("verbalized doubt (mid, 1-hop falsehood)")
    ax.set_ylabel("validated recovery")
    ax.set_xlim(-0.05, 0.8)
    ax.set_ylim(0.45, 0.82)
    ax.set_title("Verbalization is trained; recovery is intrinsic\n"
                 "doubt spans ~0 (silent) to .67 (R1); recovery stays in a narrow band",
                 fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(PAPER, f"fig_verbalization_spectrum.{ext}"), dpi=170)
    print("wrote fig_verbalization_spectrum.{pdf,png} with", len(pts), "models")
    for label, doubt, rec, n, _ in pts:
        print(f"  {label:16s} doubt={doubt:.3f} recovery={rec:.3f} n={n}")


if __name__ == "__main__":
    main()
