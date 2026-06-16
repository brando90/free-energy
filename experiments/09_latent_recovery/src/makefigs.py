"""Three paper figures from the result JSONs. Pure matplotlib, no GPU. v2 (label fixes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/latent_recovery"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
C = {"recovery": "#1b7837", "doubt": "#762a83", "b7": "#2166ac", "b32": "#d6604d"}

# ---------- FIG 1: dose-response ----------
k = [1,2,3,4,5]
rec7 = [.646,.562,.531,.594,.594]; dbt7 = [.392,.323,.231,.229,.246]
k32=[1,2,3]; rec32=[.699,.753,.710]; dbt32=[.237,.151,.161]
glob_rec, glob_dbt = .343, .000

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
for a, (y7, y32, gy, title, ylo, yhi) in zip(
        ax, [(dbt7,dbt32,glob_dbt,"Verbalized doubt",-.02,.45),
             (rec7,rec32,glob_rec,"Validated recovery",.25,.82)]):
    a.axvspan(2.7,5.5, color="grey", alpha=.07)
    a.text(4.1, ylo + .9*(yhi-ylo), "plateau", color="grey", fontsize=8.5, ha="center", style="italic")
    a.plot(k, y7, "o-", color=C["b7"], label="7B", lw=2)
    a.plot(k32, y32, "s--", color=C["b32"], label="32B", lw=2)
    a.plot([6.3], [gy], "*", color="k", ms=15)
    a.annotate("global-only\n(off-path lie)", (6.3, gy), xytext=(6.3, gy + .12*(yhi-ylo)*(1 if gy<.4 else -3)),
               fontsize=8, ha="center", color="k",
               arrowprops=dict(arrowstyle="-", color="grey", lw=.6))
    a.set_title(title); a.set_ylabel("rate"); a.set_ylim(ylo,yhi)
    a.set_xlabel("distance to contradicting evidence ($k$ hops)")
    a.set_xticks([1,2,3,4,5,6.3]); a.set_xticklabels(["1","2","3","4","5","$\\infty$"])
    a.legend(frameon=False, loc="center right")
fig.suptitle("Response decays with distance to the falsifying evidence, then plateaus; "
             "global-only lies collapse to silence", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_dose_response.pdf"); fig.savefig(f"{OUT}/fig_dose_response.png")
print("fig1 done")

# ---------- FIG 2: regime map ----------
tasks = ["PrOntoQA\n(redundant\nderivations)", "GSM8K\n(natural-language\nmath)", "Chained arith.\n(no redundancy)"]
rec = [.229, .088, .010]; absorb = [.600, .877, .990]
x = np.arange(3); w = .38
fig, ax = plt.subplots(figsize=(6.6, 4.2))
ax.bar(x-w/2, rec, w, color=C["recovery"], label="validated recovery")
ax.bar(x+w/2, absorb, w, color=C["b32"], label="error absorption (poisoning)")
for i,(r,a) in enumerate(zip(rec,absorb)):
    ax.text(i-w/2, r+.02, f"{r:.2f}", ha="center", fontsize=9)
    ax.text(i+w/2, a+.02, f"{a:.2f}", ha="center", fontsize=9)
ax.annotate("PrOntoQA shown with the globally-checkable (hard) family;\n"
            "locally-checkable lies there recover at .64",
            (0, .60), xytext=(0.35, .40), fontsize=7.8, color="#555",
            arrowprops=dict(arrowstyle="->", color="#999", lw=.7))
ax.set_xticks(x); ax.set_xticklabels(tasks, fontsize=9)
ax.set_ylabel("rate"); ax.set_ylim(0,1.08); ax.legend(frameon=False, loc="upper left")
ax.set_title("Task derivation structure sets the price of an absorbed error", fontsize=10.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_regime_map.pdf"); fig.savefig(f"{OUT}/fig_regime_map.png")
print("fig2 done")

# ---------- FIG 3: verbalization spectrum ----------
# (name, doubt, recovery, kind, label_dx, label_dy, ha)
models = [("Qwen-1.5B",.018,.527,"base", .016, -.024, "left"),
          ("OLMo-2-7B",.013,.527,"base", .016, .016, "left"),
          ("Llama-3.1-8B",.000,.533,"base", .016, -.005, "left"),
          ("Qwen-7B",.254,.638,"base", .015, .010, "left"),
          ("Qwen-32B",.129,.742,"base", .015, .010, "left"),
          ("R1-Distill-7B",.672,.689,"reasoning-RL", -.02, .012, "right")]
fig, ax = plt.subplots(figsize=(6.8,4.3))
ax.axhspan(.50,.76, color=C["recovery"], alpha=.07)
for name,d,r,kind,dx,dy,ha in models:
    col = C["doubt"] if kind=="reasoning-RL" else C["b7"]
    ax.scatter(d, r, s=95, color=col, zorder=3, edgecolor="k", linewidth=.5)
    ax.annotate(name, (d,r), xytext=(d+dx, r+dy), fontsize=8.5, ha=ha,
                arrowprops=dict(arrowstyle="-", color="#bbb", lw=.5) if abs(dy)>.018 else None)
ax.text(.74,.505,"recovery band (.53–.74)", color=C["recovery"], fontsize=8.5, va="bottom", ha="right")
ax.set_xlabel("verbalized doubt rate"); ax.set_ylabel("validated recovery rate")
ax.set_xlim(-.04,.78); ax.set_ylim(.45,.80)
ax.scatter([],[],color=C["b7"],label="base / instruct"); ax.scatter([],[],color=C["doubt"],label="reasoning-RL")
ax.legend(frameon=False, loc="upper center")
ax.set_title("Verbalization varies far more than recovery across models\n"
             "(negstep family, mid injection) — flagging is trained, recovery is intrinsic", fontsize=9.8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_verbalization_spectrum.pdf"); fig.savefig(f"{OUT}/fig_verbalization_spectrum.png")
print("fig3 done"); print("ALL FIGS WRITTEN")
