"""E5 v2 -- consolidated licensed/unlicensed/visible-unshown analysis.

Reuses the VALIDATED core classifier `classify_row` (and loaders) from
e5_classify.py unchanged. Adds:
  * per-cell table with 4-class counts + cluster-bootstrap 95% CIs, both as a
    fraction of stated-complement (SC) rows AND as a fraction of ALL rows;
  * designed_d vs measured_d cross-tab (quantifies d=0 contamination of the
    "distance" cells);
  * TWO headline framings for "fraction of one-hop-or-farther checking licensed":
      (B) designed-based: SC rows with designed_d != 0  (prior number; INCLUDES
          the non-derivable control arms FREQ/TEMPLATE/dinf -> all unlicensed);
      (A) measured-based: SC rows with measured_d finite AND >= 1 ("genuine
          checking actually possible" -- complement is world-derivable at >=1 hop);
    plus the pure-lexical control baseline (measured_d is None);
  * d=0-cliff table (absolute per-row rates by cell across d) to test whether the
    d=0 stated-complement cliff survives *within licensed-only* rejections.

Prints an anchor block first (must reproduce the published SC rates) before any
split, per the validation protocol.

Cluster unit = problem_id everywhere. Deterministic (fixed seed).
"""
import sys, os, json, time
from collections import defaultdict, Counter

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
SLUG = os.path.join(EXP, "improvement_plan", "iclr_exec", "e5_licensed_split")
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expe"))
sys.path.insert(0, SLUG)

from e5_classify import (classify_row, load_expe, load_expd,          # noqa: E402
                         cluster_bootstrap_rate, SEED, NBOOT)

OUT = os.path.join(SLUG, "v2")
KLASSES = ["licensed", "visible_unshown", "unlicensed", "d0_visible"]

PUB_ANCHORS = {  # (exp, cell) -> published stated-complement rate
    ("EXPD", "aff_false_attr_d0"): 0.247, ("EXPD", "aff_false_attr_d1"): 0.031,
    ("EXPD", "cat_false_inert_d1"): 0.193, ("EXPD", "cat_false_usable_d1"): 0.023,
    ("EXPE", "REFUTING_d1"): 0.114, ("EXPE", "FREQ_MATCHED_NONREFUTING"): 0.118,
}


def rnd(x, k=4):
    return round(x, k) if isinstance(x, float) else x


def ci_frac(rows, pred):
    """cluster-bootstrap point + 95% CI for mean of pred(r) over rows (unit denom)."""
    pt, lo, hi, num, den = cluster_bootstrap_rate(rows, lambda r: int(pred(r)), lambda r: 1)
    return [rnd(pt), rnd(lo), rnd(hi), num, den]


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    rows = load_expe() + load_expd()

    # ---- anchor reproduction (before any split) ----------------------------
    anchors = {}
    cell_all = defaultdict(list)
    for r in rows:
        cell_all[(r["exp"], r["cell"])].append(r)
    for key, pub in PUB_ANCHORS.items():
        allr = cell_all.get(key, [])
        n = len(allr); k = sum(1 for r in allr if r["stated"])
        anchors[f"{key[0]}:{key[1]}"] = {
            "n_all": n, "n_stated_complement": k,
            "sc_rate": rnd(k / n, 4) if n else None, "published": pub,
            "match": (abs(k / n - pub) < 0.0015) if n else None}

    # ---- classify every stated-complement row ------------------------------
    classified = []
    for r in rows:
        if not r["stated"]:
            continue
        c = classify_row(r["question"], r["entity"], r["planted"],
                         r["prefix_steps"], r["continuation"])
        r.update(c)
        classified.append(r)

    cls_index = defaultdict(list)
    for r in classified:
        cls_index[(r["exp"], r["cell"])].append(r)

    # ---- per-cell table ----------------------------------------------------
    per_cell = {}
    for key in sorted(cell_all):
        exp, cell = key
        allr = cell_all[key]
        crows = cls_index.get(key, [])
        n_all, n_sc = len(allr), len(crows)
        cnt = Counter(r["klass"] for r in crows)
        dd = allr[0]["designed_d"]
        rec = {"designed_d": dd, "n_all": n_all, "n_stated_complement": n_sc,
               "sc_rate_of_all": ci_frac(allr, lambda r: bool(r.get("stated"))),
               "counts": {kl: cnt.get(kl, 0) for kl in KLASSES},
               "class_frac_of_SC": {}, "class_rate_of_all": {},
               "unlicensed_reasons": dict(Counter(
                   r["reason"] for r in crows if r["klass"] == "unlicensed"))}
        for kl in KLASSES:
            rec["class_frac_of_SC"][kl] = (
                ci_frac(crows, lambda r, k=kl: r["klass"] == k) if n_sc else [None, None, None, 0, 0])
            rec["class_rate_of_all"][kl] = ci_frac(
                allr, lambda r, k=kl: bool(r.get("stated")) and r.get("klass") == k)
        # measured_d distribution among SC rows
        rec["measured_d_hist"] = dict(Counter(
            ("None" if r["measured_d"] is None else r["measured_d"]) for r in crows))
        per_cell[f"{exp}:{cell}"] = rec

    # ---- designed_d vs measured_d cross-tab (SC rows) ----------------------
    xtab = defaultdict(lambda: Counter())
    for r in classified:
        md = "None" if r["measured_d"] is None else r["measured_d"]
        xtab[str(r["designed_d"])][str(md)] += 1
    xtab = {k: dict(v) for k, v in xtab.items()}

    # ---- headline framings -------------------------------------------------
    def scope(pred, name):
        sub = [r for r in classified if pred(r)]
        cnt = Counter(r["klass"] for r in sub)
        return name, {
            "n_stated_complement": len(sub),
            "counts": {kl: cnt.get(kl, 0) for kl in KLASSES},
            "licensed_frac": ci_frac(sub, lambda r: r["klass"] == "licensed"),
            "licensed_lenient_frac": ci_frac(sub, lambda r: r["klass_lenient"] == "licensed"),
            "unlicensed_frac": ci_frac(sub, lambda r: r["klass"] == "unlicensed"),
        }
    headline = {}
    for nm, rec in [
        # (B) designed-based, prior number; includes non-derivable controls
        scope(lambda r: r["designed_d"] != 0, "B_designed_d>=1_incl_controls"),
        # (A) measured-based: complement genuinely world-derivable at >=1 hop
        scope(lambda r: r["measured_d"] is not None and r["measured_d"] >= 1,
              "A_measured_d>=1_derivable_only"),
        # control baseline: complement not derivable in world at all
        scope(lambda r: r["measured_d"] is None, "CONTROL_measured_d_None"),
        # EXPE refuting-only, EXPD-only, EXPE-only (designed>=1)
        scope(lambda r: r["exp"] == "EXPE" and r["cell"].startswith("REFUTING"),
              "EXPE_REFUTING_all"),
        scope(lambda r: r["exp"] == "EXPD" and r["designed_d"] != 0, "EXPD_designed_d>=1"),
        scope(lambda r: r["exp"] == "EXPE" and r["designed_d"] != 0, "EXPE_designed_d>=1"),
        # strictly-far genuine search: measured_d>=2 derivable
        scope(lambda r: r["measured_d"] is not None and r["measured_d"] >= 2,
              "measured_d>=2_derivable_only"),
    ]:
        headline[nm] = rec

    # ---- d=0 cliff within licensed-only ------------------------------------
    # For the attribute + refuting + cat ladders, show absolute per-row rates.
    cliff_ladders = {
        "EXPD_aff_false_attr": ["aff_false_attr_d0", "aff_false_attr_d1",
                                "aff_false_attr_d2", "aff_false_attr_d3", "aff_false_attr_d5"],
        "EXPD_cat_false_inert": ["cat_false_inert_d1", "cat_false_inert_d3", "cat_false_inert_dinf"],
        "EXPD_cat_false_usable": ["cat_false_usable_d1", "cat_false_usable_d3", "cat_false_usable_dinf"],
        "EXPE_REFUTING": ["REFUTING_d1", "REFUTING_d2", "REFUTING_d3"],
    }
    cliff = {}
    for lad, cells in cliff_ladders.items():
        exp = "EXPD" if lad.startswith("EXPD") else "EXPE"
        cliff[lad] = {}
        for cell in cells:
            allr = cell_all.get((exp, cell), [])
            if not allr:
                continue
            cliff[lad][cell] = {
                "designed_d": allr[0]["designed_d"], "n_all": len(allr),
                "sc_rate": ci_frac(allr, lambda r: bool(r.get("stated"))),
                "licensed_rate": ci_frac(allr, lambda r: bool(r.get("stated")) and r.get("klass") == "licensed"),
                "d0_visible_rate": ci_frac(allr, lambda r: bool(r.get("stated")) and r.get("klass") == "d0_visible"),
                "visible_unshown_rate": ci_frac(allr, lambda r: bool(r.get("stated")) and r.get("klass") == "visible_unshown"),
                "unlicensed_rate": ci_frac(allr, lambda r: bool(r.get("stated")) and r.get("klass") == "unlicensed"),
            }

    result = {
        "meta": {"nboot": NBOOT, "seed": SEED, "cluster_unit": "problem_id",
                 "n_rows_total": len(rows), "n_stated_complement": len(classified),
                 "classifier": "e5_classify.classify_row (validated; conservative on licensed)",
                 "elapsed_s": round(time.time() - t0, 1)},
        "anchors": anchors,
        "per_cell": per_cell,
        "designed_vs_measured_d_xtab": xtab,
        "headline": headline,
        "cliff_within_licensed": cliff,
    }
    with open(os.path.join(OUT, "e5_results_v2.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---- console summary ---------------------------------------------------
    print("=== ANCHOR reproduction (stated-complement rate vs published) ===")
    for k, a in anchors.items():
        print(f"  {k:34s} sc={a['sc_rate']}  pub={a['published']}  match={a['match']}")
    print(f"\ntotal rows={len(rows)}  stated-complement rows={len(classified)}")
    print("\n=== HEADLINE: licensed fraction of apparent >=1-hop checking ===")
    for nm, rec in headline.items():
        c = rec["counts"]; lf = rec["licensed_frac"]
        print(f"  {nm:34s} nSC={rec['n_stated_complement']:4d} "
              f"lic={c['licensed']:3d} vis={c['visible_unshown']:3d} d0={c['d0_visible']:3d} "
              f"unl={c['unlicensed']:3d} | lic_frac={lf[0]} [{lf[1]},{lf[2]}]")
    print("\n=== designed_d x measured_d (SC rows) ===")
    for dd in sorted(xtab, key=lambda x: (x == "inf", x)):
        print(f"  designed={dd:4s}: {xtab[dd]}")
    print("\n=== d=0 CLIFF within licensed-only (absolute per-row rates) ===")
    for lad, cells in cliff.items():
        print(f"  [{lad}]")
        for cell, c in cells.items():
            print(f"    {cell:24s} d={str(c['designed_d']):4s} nAll={c['n_all']:4d} "
                  f"sc={c['sc_rate'][0]:.4f} lic={c['licensed_rate'][0]:.4f} "
                  f"d0v={c['d0_visible_rate'][0]:.4f} vis={c['visible_unshown_rate'][0]:.4f} "
                  f"unl={c['unlicensed_rate'][0]:.4f}")
    print(f"\nwrote {os.path.join(OUT, 'e5_results_v2.json')}  ({result['meta']['elapsed_s']}s)")


if __name__ == "__main__":
    main()
