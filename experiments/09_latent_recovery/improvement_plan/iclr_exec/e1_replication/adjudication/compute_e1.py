"""E1 adjudication compute: B1/B2/B3 per model, reusing locked expd_confirmatory_analysis machinery.
Registered estimator: cluster unit = family_id, positions pooled, cluster-bootstrap CIs.
Significance p-values via the locked paired-family tests (one_sided_greater / tost).
Also computes an unpaired cluster-bootstrap of the point difference as a corroborating CI/p.
"""
import json, os, sys, math, random
from collections import defaultdict
import numpy as np

EXPD = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/expd"
sys.path.insert(0, EXPD)
import expd_confirmatory_analysis as C  # noqa

BASE = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e1_replication/results"
ANCHOR = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/results/EXPD_MATCHED_GRADIENT"

MODELS = [
    ("qwen1p5b", os.path.join(BASE, "qwen1p5b")),
    ("llama8b", os.path.join(BASE, "llama8b")),
    ("olmo7b", os.path.join(BASE, "olmo7b")),
    ("qwen32b", os.path.join(BASE, "qwen32b")),
    ("qwen7b_anchor", ANCHOR),
]

ALPHA = 0.05
HOLM_M = 3


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


def by_cond(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["condition"]].append(r)
    return d


def cell_n_by_pos(rows):
    """per (condition, position) n, for §8.2 classification."""
    d = defaultdict(int)
    for r in rows:
        d[(r["condition"], r["injection_position"])] += 1
    return d


def pooled_rate(rows, metric):
    if not rows:
        return None
    return float(np.mean([C.mval(r, metric) for r in rows]))


def unpaired_cluster_boot_diff(rows_a, rows_b, metric, seed=0, n_boot=2000):
    """Resample family_ids within each group; pooled_a - pooled_b. Returns (diff, ci95, p_one_sided_gt0)."""
    def group(rows):
        by = defaultdict(list)
        for r in rows:
            by[r["family_id"]].append(C.mval(r, metric))
        return by
    ba, bb = group(rows_a), group(rows_b)
    ida, idb = sorted(ba), sorted(bb)
    if not ida or not idb:
        return None, [None, None], None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        na = da = nb = db = 0
        for cid in (rng.choice(ida) for _ in ida):
            na += sum(ba[cid]); da += len(ba[cid])
        for cid in (rng.choice(idb) for _ in idb):
            nb += sum(bb[cid]); db += len(bb[cid])
        if da and db:
            diffs.append(na/da - nb/db)
    diffs = sorted(diffs)
    if not diffs:
        return None, [None, None], None
    pt = pooled_rate(rows_a, metric) - pooled_rate(rows_b, metric)
    lo = diffs[int(0.025*(len(diffs)-1))]
    hi = diffs[int(0.975*(len(diffs)-1))]
    p_gt = sum(1 for d in diffs if d <= 0) / len(diffs)  # one-sided H1: diff>0
    return rnd(pt), [rnd(lo), rnd(hi)], rnd(p_gt, 6)


def holm(pvals_named, alpha=ALPHA):
    """Standard Holm step-down. pvals_named: dict slot->p. Returns slot->{thresh, holm_reject}."""
    items = sorted(pvals_named.items(), key=lambda kv: (kv[1] if kv[1] is not None else 1.0))
    m = len(items)
    out = {}
    still = True
    for i, (slot, p) in enumerate(items):
        thr = alpha / (m - i)
        pp = p if p is not None else 1.0
        rej = still and (pp <= thr)
        if not rej:
            still = False
        out[slot] = {"p": p, "holm_thresh": rnd(thr, 6), "holm_reject": rej}
    return out


def classify_n(n):
    if n is None:
        return "NA"
    if n >= 150:
        return "full"
    if n >= 100:
        return "flag(100-149)"
    if n >= 50:
        return "CI-widened(50-99)"
    return "INVALID(<50)"


RESULTS = {}

for name, d in MODELS:
    rows, manifest = C.load_rows(d)
    cond = by_cond(rows)
    npos = cell_n_by_pos(rows)

    def cells(*names):
        out = []
        for nm in names:
            out += cond.get(nm, [])
        return out

    def min_pos_n(*names):
        vals = [npos.get(k) for k in npos if k[0] in names]
        return min(vals) if vals else 0

    # ---- B1: stated_complement d0 - d1 (aff_false_attr), one-sided >0, mag>=0.10
    a0 = cond.get("aff_false_attr_d0", [])
    a1 = cond.get("aff_false_attr_d1", [])
    ps_b1 = C.paired_family_stats(a0, a1, "stated_complement")
    os_b1 = C.one_sided_greater(ps_b1)
    pt_b1, ci_b1, pboot_b1 = unpaired_cluster_boot_diff(a0, a1, "stated_complement")
    b1 = {
        "rate_d0": rnd(ps_b1["rate_a"]), "rate_d1": rnd(ps_b1["rate_b"]),
        "point_diff_pooled": pt_b1, "boot_ci95": ci_b1, "boot_p_onesided": pboot_b1,
        "paired_mean_diff": rnd(os_b1["mean_diff"]), "paired_p_onesided": os_b1.get("p_raw"),
        "n_fam_paired": ps_b1["n_families_paired"], "n_d0": ps_b1["n_rows_a"], "n_d1": ps_b1["n_rows_b"],
        "min_cellpos_n": min_pos_n("aff_false_attr_d0", "aff_false_attr_d1"),
        "mag_gate_pass": (pt_b1 is not None and pt_b1 >= 0.10),
    }
    b1["cell_class"] = classify_n(b1["min_cellpos_n"])

    # ---- B2: inj_derivational usable(pooled d1,3,inf) - inert(pooled), one-sided>0, mag>=0.30
    us = cells("cat_false_usable_d1", "cat_false_usable_d3", "cat_false_usable_dinf")
    inr = cells("cat_false_inert_d1", "cat_false_inert_d3", "cat_false_inert_dinf")
    ps_b2 = C.paired_family_stats(us, inr, "inj_derivational")
    os_b2 = C.one_sided_greater(ps_b2)
    pt_b2, ci_b2, pboot_b2 = unpaired_cluster_boot_diff(us, inr, "inj_derivational")
    b2 = {
        "rate_usable": rnd(ps_b2["rate_a"]), "rate_inert": rnd(ps_b2["rate_b"]),
        "point_diff_pooled": pt_b2, "boot_ci95": ci_b2, "boot_p_onesided": pboot_b2,
        "paired_mean_diff": rnd(os_b2["mean_diff"]), "paired_p_onesided": os_b2.get("p_raw"),
        "n_fam_paired": ps_b2["n_families_paired"], "n_usable": ps_b2["n_rows_a"], "n_inert": ps_b2["n_rows_b"],
        "min_cellpos_n": min_pos_n("cat_false_usable_d1","cat_false_usable_d3","cat_false_usable_dinf",
                                   "cat_false_inert_d1","cat_false_inert_d3","cat_false_inert_dinf"),
        "mag_gate_pass": (pt_b2 is not None and pt_b2 >= 0.30),
    }
    b2["cell_class"] = classify_n(b2["min_cellpos_n"])

    # ---- B3: conjunctive TOST. (i) stated_complement aff d1 vs d5 +-0.10 ; (ii) inj_derivational usable d1 vs dinf +-0.15
    a5 = cond.get("aff_false_attr_d5", [])
    ps_b3i = C.paired_family_stats(a1, a5, "stated_complement")
    tost_i = C.tost(ps_b3i, margin=0.10)
    ud1 = cond.get("cat_false_usable_d1", [])
    udinf = cond.get("cat_false_usable_dinf", [])
    ps_b3ii = C.paired_family_stats(ud1, udinf, "inj_derivational")
    tost_ii = C.tost(ps_b3ii, margin=0.15)
    p_b3i = tost_i.get("p_raw"); p_b3ii = tost_ii.get("p_raw")
    slot_p_b3 = max([p for p in [p_b3i, p_b3ii] if p is not None], default=None)
    b3 = {
        "i_stated_d1": rnd(ps_b3i["rate_a"]), "i_stated_d5": rnd(ps_b3i["rate_b"]),
        "i_mean_diff": tost_i.get("mean_diff"), "i_ci90": tost_i.get("ci90"), "i_p_raw": p_b3i, "i_margin": 0.10,
        "ii_deriv_usable_d1": rnd(ps_b3ii["rate_a"]), "ii_deriv_usable_dinf": rnd(ps_b3ii["rate_b"]),
        "ii_mean_diff": tost_ii.get("mean_diff"), "ii_ci90": tost_ii.get("ci90"), "ii_p_raw": p_b3ii, "ii_margin": 0.15,
        "slot_p": slot_p_b3,
        "n_fam_i": ps_b3i["n_families_paired"], "n_fam_ii": ps_b3ii["n_families_paired"],
        "min_cellpos_n": min_pos_n("aff_false_attr_d1","aff_false_attr_d5","cat_false_usable_d1","cat_false_usable_dinf"),
    }
    b3["cell_class"] = classify_n(b3["min_cellpos_n"])

    # ---- Holm across 3 slots
    holm_res = holm({"B1": os_b1.get("p_raw"), "B2": os_b2.get("p_raw"), "B3": slot_p_b3})

    # ---- slot pass decisions (Holm-significant AND magnitude/margin AND not INVALID)
    def invalid(cls):
        return cls.startswith("INVALID")
    b1_pass = holm_res["B1"]["holm_reject"] and b1["mag_gate_pass"] and not invalid(b1["cell_class"])
    b2_pass = holm_res["B2"]["holm_reject"] and b2["mag_gate_pass"] and not invalid(b2["cell_class"])
    # B3 both TOST inside margin => both p < holm-alpha ; magnitude here = equivalence (p<thresh handles it)
    b3_both_sig = holm_res["B3"]["holm_reject"]  # slot_p = max, so reject means BOTH < thresh
    b3_pass = b3_both_sig and not invalid(b3["cell_class"])

    replicates = bool(b1_pass and b2_pass and b3_pass)

    RESULTS[name] = {
        "n_rows": len(rows), "n_fam_total": len({r["family_id"] for r in rows}),
        "B1": b1, "B2": b2, "B3": b3, "holm": holm_res,
        "B1_pass": b1_pass, "B2_pass": b2_pass, "B3_pass": b3_pass,
        "fully_replicates": replicates,
    }
    print(f"\n===== {name}  (n={len(rows)}) =====")
    print(f" B1 cliff: sc_d0={b1['rate_d0']} sc_d1={b1['rate_d1']} diff={b1['point_diff_pooled']} "
          f"bootCI={b1['boot_ci95']} paired_p={b1['paired_p_onesided']} holm_thr={holm_res['B1']['holm_thresh']} "
          f"mag>=.10={b1['mag_gate_pass']} cell={b1['cell_class']} -> PASS={b1_pass}")
    print(f" B2 usable-inert deriv: us={b2['rate_usable']} inert={b2['rate_inert']} diff={b2['point_diff_pooled']} "
          f"bootCI={b2['boot_ci95']} paired_p={b2['paired_p_onesided']} holm_thr={holm_res['B2']['holm_thresh']} "
          f"mag>=.30={b2['mag_gate_pass']} cell={b2['cell_class']} -> PASS={b2_pass}")
    print(f" B3i stated d1={b3['i_stated_d1']} d5={b3['i_stated_d5']} diff={b3['i_mean_diff']} ci90={b3['i_ci90']} p={b3['i_p_raw']} (m=.10)")
    print(f" B3ii deriv usable d1={b3['ii_deriv_usable_d1']} dinf={b3['ii_deriv_usable_dinf']} diff={b3['ii_mean_diff']} ci90={b3['ii_ci90']} p={b3['ii_p_raw']} (m=.15)")
    print(f" B3 slot_p={b3['slot_p']} holm_thr={holm_res['B3']['holm_thresh']} cell={b3['cell_class']} -> PASS={b3_pass}")
    print(f" >>> FULLY REPLICATES = {replicates}")

# program-level
scored = [m for m, _ in MODELS if m != "qwen7b_anchor"]
nrep = sum(RESULTS[m]["fully_replicates"] for m in scored)
percond = {}
for slot in ["B1_pass", "B2_pass", "B3_pass"]:
    percond[slot] = sum(RESULTS[m][slot] for m in scored)
print("\n########## PROGRAM LEVEL ##########")
print(f" fully-replicating models: {nrep}/4  (rule: >=3/4 AND floor 3)")
print(f" per-condition passes: {percond}")
print(f" PROGRAM REPLICATES = {nrep >= 3}")

json.dump(RESULTS, open("/tmp/e1_adjudication_results.json", "w"), indent=1)
print("\nwrote /tmp/e1_adjudication_results.json")
