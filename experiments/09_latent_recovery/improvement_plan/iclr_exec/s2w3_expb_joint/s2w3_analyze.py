#!/usr/bin/env python
"""S2 + W3 : EXPB joint-outcome analysis + strict-schema adaptation (0 GPU).

Deliverables:
  (a) full-composition table per arm (all 900 rows; unparsed & generation-failed
      as EXPLICIT outcomes; proportions of ALL rows, not the parsed subset).
  (b) monotone bounds: worst/best treatment of unparsed rows for the closure-valid
      and injection-dependence (poisoned) contrasts; which conclusions survive.
  (c) verify robust survivors: doubt contrast +0.370 and 111-0 discordant pairs.
  (d) strict-metric pass over the 900 rows (adapted stage0_strict tooling: imports
      the real SM.analyze/compose/summarize_family so numbers are directly
      comparable to STRICT_METRICS_REPORT.md).
  (e) figure-ready CSV (stacked composition per arm) + draft matplotlib PNG.

Schema adaptation (why EXPB was skipped by stage0_strict): load_expa requires the
manifest key `prefix_steps`, which EXPB lacks. EXPB stores the visible context as
`assistant_prefill_steps` = proof_prefix_steps (+ certificate) + [injected F]. The
faithful prefix (what the model conditioned on BEFORE the planted falsehood) is
`assistant_prefill_steps[:-1]`; this is condition-aware and seeds the certificate
into the validator state for the LOCAL/IRRELEVANT arms.

Self-validation (HARD RULE 5): every published anchor is asserted below; the script
raises if any drifts.
"""
import os, sys, json, math, datetime, csv, random
from collections import defaultdict, Counter

REPO = "/lfs/skampere2/0/eobbad/free-energy"
EXP = REPO + "/experiments/09_latent_recovery"
os.environ["LR_ROOT"] = EXP
sys.path.insert(0, EXP + "/improvement_plan/stage0_strict")
sys.path.insert(0, EXP + "/src")
import strict_metrics as SM   # the real, published strict tooling (M1-M4)
import validator as V         # noqa: F401 (imported for parity / availability)

OUT = EXP + "/improvement_plan/iclr_exec/s2w3_expb_joint"
EXPB = EXP + "/results/EXPB_LOCAL_CERT_FLIP"
CONDS = ["GLOBAL_BASELINE", "LOCAL_CERTIFICATE", "IRRELEVANT_CERTIFICATE_CONTROL"]
SHORT = {"GLOBAL_BASELINE": "no-cert", "LOCAL_CERTIFICATE": "local-cert",
         "IRRELEVANT_CERTIFICATE_CONTROL": "irrelevant-cert"}
CATS = ["valid_rederivation", "poisoned", "parroted", "derailed", "unparsed", "generation_failed"]
CATLBL = {"valid_rederivation": "closure_valid", "poisoned": "injection_dependent",
          "parroted": "parroted", "derailed": "derailed", "unparsed": "unparsed",
          "generation_failed": "generation_failed"}

man = {r["run_id"]: r for r in SM.read_jsonl(EXPB + "/manifest.jsonl")}
vrows = SM.read_jsonl(EXPB + "/validated_outputs.jsonl")
assert len(vrows) == 900 and len(man) == 900, (len(vrows), len(man))


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


# ============================================================ build row records
rows = []
prefill_ends_with_F = 0
for r in vrows:
    m = man[r["run_id"]]
    aps = m["assistant_prefill_steps"]
    if aps[-1] == m["injected_statement"]:
        prefill_ends_with_F += 1
    prefix = list(aps[:-1])   # condition-aware: includes certificate for cert arms
    gen_failed = bool(r.get("failed_generation"))
    cls_joint = "generation_failed" if gen_failed else r["class"]
    rows.append({
        "run_id": r["run_id"], "triple_id": r["triple_id"], "problem_id": r["problem_id"],
        "condition": r["condition"], "point": r["injection_position"],
        "class_stored": r["class"], "class_joint": cls_joint, "gen_failed": gen_failed,
        "doubt": bool(r["verbalized_doubt"]), "prefix": prefix,
        "injected": m["injected_statement"], "question": m["question"],
        "target": m["target"], "entity": m["entity"],
        "truth": r["injected_statement_truth_status"], "continuation": r.get("continuation", ""),
    })
assert prefill_ends_with_F == 900, prefill_ends_with_F  # schema invariant

by_arm = defaultdict(list)
for row in rows:
    by_arm[row["condition"]].append(row)
for c in CONDS:
    assert len(by_arm[c]) == 300, (c, len(by_arm[c]))

# ============================================================ (a) composition
comp = {}
for cond in CONDS:
    arm = by_arm[cond]
    cnt = Counter(row["class_joint"] for row in arm)
    assert sum(cnt.values()) == 300
    comp[cond] = {"n": 300, "counts": {c: cnt.get(c, 0) for c in CATS},
                  "props": {c: round(cnt.get(c, 0) / 300, 4) for c in CATS}}

# ---- ANCHOR CHECKS vs EXPB_REPORT.md / STAGE0_REPORT.md ----
A = {  # (valid, poisoned, unparsed, doubt) counts published
    "GLOBAL_BASELINE":               dict(valid=113, poisoned=46, unparsed=12, doubt=3),
    "LOCAL_CERTIFICATE":             dict(valid=140, poisoned=8,  unparsed=95, doubt=114),
    "IRRELEVANT_CERTIFICATE_CONTROL":dict(valid=150, poisoned=72, unparsed=11, doubt=7),
}
for cond, exp in A.items():
    c = comp[cond]["counts"]
    assert c["valid_rederivation"] == exp["valid"], (cond, "valid", c["valid_rederivation"], exp["valid"])
    assert c["poisoned"] == exp["poisoned"], (cond, "poisoned")
    assert c["unparsed"] == exp["unparsed"], (cond, "unparsed")
    dk = sum(1 for row in by_arm[cond] if row["doubt"])
    assert dk == exp["doubt"], (cond, "doubt", dk, exp["doubt"])

# ============================================================ (b) monotone bounds
def arm_metric(cond, cls):
    arm = by_arm[cond]
    n = len(arm)
    k = sum(1 for row in arm if row["class_joint"] == cls)
    u = sum(1 for row in arm if row["class_joint"] == "unparsed")
    return {"n": n, "k": k, "unparsed": u,
            "rate_worst": round(k / n, 4), "rate_best": round((k + u) / n, 4),
            "wilson_point": wilson(k, n)}

CLSMAP = {"closure_valid": "valid_rederivation", "injection_dependent": "poisoned"}
bounds = {m: {c: arm_metric(c, cls) for c in CONDS} for m, cls in CLSMAP.items()}

# ANCHOR: STAGE0 imputation bounds
assert (bounds["closure_valid"]["LOCAL_CERTIFICATE"]["rate_worst"],
        bounds["closure_valid"]["LOCAL_CERTIFICATE"]["rate_best"]) == (0.4667, 0.7833)
assert (bounds["closure_valid"]["GLOBAL_BASELINE"]["rate_worst"],
        bounds["closure_valid"]["GLOBAL_BASELINE"]["rate_best"]) == (0.3767, 0.4167)
assert (bounds["closure_valid"]["IRRELEVANT_CERTIFICATE_CONTROL"]["rate_worst"],
        bounds["closure_valid"]["IRRELEVANT_CERTIFICATE_CONTROL"]["rate_best"]) == (0.5, 0.5367)

def contrast_bounds(metric, a, b):
    """Full interval of Δ = rate(a) - rate(b) over ALL imputations of unparsed rows.
    Monotone: Δ minimized when a is smallest & b is largest; maximized vice-versa."""
    A_, B_ = bounds[metric][a], bounds[metric][b]
    return {"point_diff": round(A_["rate_worst"] - B_["rate_worst"], 4),
            "delta_min": round(A_["rate_worst"] - B_["rate_best"], 4),
            "delta_max": round(A_["rate_best"] - B_["rate_worst"], 4)}

contrasts = {}
for metric in CLSMAP:
    contrasts[metric] = {
        "LOCAL_vs_GLOBAL": contrast_bounds(metric, "LOCAL_CERTIFICATE", "GLOBAL_BASELINE"),
        "LOCAL_vs_IRRELEVANT": contrast_bounds(metric, "LOCAL_CERTIFICATE", "IRRELEVANT_CERTIFICATE_CONTROL"),
    }

# ============================================================ (c) survivors: doubt
tri = defaultdict(dict)
for row in rows:
    tri[row["triple_id"]][row["condition"]] = row
full_triples = [t for t, d in tri.items() if len(d) == 3]
assert len(full_triples) == 300, len(full_triples)

def mcnemar_discordant(metric_fn, a, b):
    bb = cc = cy = cn = 0
    for t in full_triples:
        va, vb = metric_fn(tri[t][a]), metric_fn(tri[t][b])
        if va and not vb: bb += 1
        elif vb and not va: cc += 1
        elif va and vb: cy += 1
        else: cn += 1
    return {"n_pairs": len(full_triples), "b_a_only": bb, "c_b_only": cc,
            "concordant_yes": cy, "concordant_no": cn,
            "net_diff": round((bb - cc) / len(full_triples), 4)}

def mcnemar_p(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)

doubt_fn = lambda row: row["doubt"]
valid_fn = lambda row: row["class_joint"] == "valid_rederivation"
pois_fn  = lambda row: row["class_joint"] == "poisoned"
unpar_fn = lambda row: row["class_joint"] == "unparsed"

surv = {}
for name, a, b in [("doubt_LOCAL_vs_GLOBAL", "LOCAL_CERTIFICATE", "GLOBAL_BASELINE"),
                   ("doubt_LOCAL_vs_IRRELEVANT", "LOCAL_CERTIFICATE", "IRRELEVANT_CERTIFICATE_CONTROL")]:
    d = mcnemar_discordant(doubt_fn, a, b)
    d["mcnemar_p_exact"] = mcnemar_p(d["b_a_only"], d["c_b_only"])
    surv[name] = d
# ANCHOR: doubt LOCAL vs GLOBAL discordant 111-0, net +0.37; vs IRRELEVANT 109-2, +0.3567
assert surv["doubt_LOCAL_vs_GLOBAL"]["b_a_only"] == 111 and surv["doubt_LOCAL_vs_GLOBAL"]["c_b_only"] == 0, surv["doubt_LOCAL_vs_GLOBAL"]
assert surv["doubt_LOCAL_vs_GLOBAL"]["net_diff"] == 0.37
assert surv["doubt_LOCAL_vs_IRRELEVANT"]["b_a_only"] == 109 and surv["doubt_LOCAL_vs_IRRELEVANT"]["c_b_only"] == 2, surv["doubt_LOCAL_vs_IRRELEVANT"]
assert surv["doubt_LOCAL_vs_IRRELEVANT"]["net_diff"] == 0.3567

surv["poisoned_LOCAL_vs_GLOBAL_paired"] = mcnemar_discordant(pois_fn, "LOCAL_CERTIFICATE", "GLOBAL_BASELINE")
surv["valid_LOCAL_vs_GLOBAL_paired"] = mcnemar_discordant(valid_fn, "LOCAL_CERTIFICATE", "GLOBAL_BASELINE")
surv["unparsed_LOCAL_vs_GLOBAL_paired"] = mcnemar_discordant(unpar_fn, "LOCAL_CERTIFICATE", "GLOBAL_BASELINE")

doubt_by_arm = {}
for cond in CONDS:
    arm = by_arm[cond]
    dk = sum(1 for row in arm if row["doubt"])
    unp = [row for row in arm if row["class_joint"] == "unparsed"]
    dk_unp = sum(1 for row in unp if row["doubt"])
    doubt_by_arm[cond] = {"n": len(arm), "doubt_k": dk, "doubt_rate": round(dk / len(arm), 4),
                          "unparsed_n": len(unp), "doubt_among_unparsed": dk_unp,
                          "wilson": wilson(dk, len(arm))}

def cluster_boot_contrast(a, b, fn, n_boot=2000, seed=0):
    by_prob = defaultdict(list)
    for t in full_triples:
        by_prob[tri[t][a]["problem_id"]].append(t)
    probs = sorted(by_prob)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = [rng.choice(probs) for _ in probs]
        na = ka = nb = kb = 0
        for pid in samp:
            for t in by_prob[pid]:
                na += 1; ka += int(bool(fn(tri[t][a])))
                nb += 1; kb += int(bool(fn(tri[t][b])))
        vals.append(ka / na - kb / nb)
    vals.sort()
    return (round(vals[int(0.025 * (len(vals) - 1))], 4), round(vals[int(0.975 * (len(vals) - 1))], 4))

surv["doubt_LOCAL_vs_GLOBAL"]["cluster_boot95"] = cluster_boot_contrast(
    "LOCAL_CERTIFICATE", "GLOBAL_BASELINE", doubt_fn)

# ============================================================ (d) strict pass
strict_families = {}
class_repro_agree = 0
class_mismatch = []
n_nonfailed = 0
strict_rows_out = []
for cond in CONDS:
    arm_rows = []
    for row in by_arm[cond]:
        if row["gen_failed"]:
            a = SM.failed_row(row["truth"] == "false", row["problem_id"], row["point"])
        else:
            n_nonfailed += 1
            a0 = SM.analyze(row["question"], row["prefix"], row["injected"],
                            row["continuation"], row["target"], row["entity"])
            if a0["class"] == row["class_stored"]:
                class_repro_agree += 1
            elif len(class_mismatch) < 20:
                class_mismatch.append({"run_id": row["run_id"], "cond": cond,
                                       "stored": row["class_stored"], "recomputed": a0["class"]})
            a = SM.compose({**a0, "is_false": row["truth"] == "false",
                            "cluster": row["problem_id"], "point": row["point"]})
        a.update({"family": SHORT[cond], "run_id": row["run_id"], "truth": row["truth"]})
        arm_rows.append(a)
        strict_rows_out.append({k: a.get(k) for k in (
            "run_id", "family", "point", "truth", "class", "closure_valid",
            "hop_sound_valid", "goal_jump0", "goal_jump_strict", "doubt", "targeted",
            "corrective", "echo", "derivational", "stated_reliance", "srr_task",
            "srr_plan", "used_any")})
    strict_families[SHORT[cond]] = SM.summarize_family(arm_rows)

# cross-check: strict pass closure_valid rate must equal the composition closure rate
for cond in CONDS:
    fam = strict_families[SHORT[cond]]["pooled"]
    assert fam["closure_valid"]["count"] == comp[cond]["counts"]["valid_rederivation"], \
        (cond, fam["closure_valid"]["count"], comp[cond]["counts"]["valid_rederivation"])
    assert fam["doubt"]["count"] == doubt_by_arm[cond]["doubt_k"], cond

# ============================================================ (e) CSV + PNG
csv_path = OUT + "/expb_composition.csv"
with open(csv_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["arm"] + [CATLBL[c] + "_prop" for c in CATS]
               + [CATLBL[c] + "_count" for c in CATS] + ["n"])
    for cond in CONDS:
        c = comp[cond]
        w.writerow([SHORT[cond]] + [c["props"][cc] for cc in CATS]
                   + [c["counts"][cc] for cc in CATS] + [c["n"]])

png_ok, png_err = False, None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    labels = [SHORT[c] for c in CONDS]
    colors = {"closure_valid": "#2c7fb8", "injection_dependent": "#d7301f",
              "parroted": "#fdae61", "derailed": "#8073ac", "unparsed": "#636363",
              "generation_failed": "#000000"}
    dark = {"closure_valid", "injection_dependent", "derailed", "unparsed", "generation_failed"}
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bottom = np.zeros(len(CONDS)); x = np.arange(len(CONDS))
    for cat in CATS:
        lbl = CATLBL[cat]
        vals = np.array([comp[c]["props"][cat] for c in CONDS])
        ax.bar(x, vals, bottom=bottom, label=lbl, color=colors[lbl], edgecolor="white", linewidth=0.5)
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v >= 0.03:
                ax.text(xi, b0 + v / 2, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if lbl in dark else "black")
        bottom += vals
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("proportion of ALL rows (n=300 / arm)")
    ax.set_ylim(0, 1.0)
    ax.set_title("EXPB joint-outcome composition (W3: unparsed as explicit outcome)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT + "/expb_composition.png", dpi=150, bbox_inches="tight")
    png_ok = True
except Exception as e:
    png_err = repr(e)

# ============================================================ dump JSON
result = {
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "n_rows": len(rows), "arms": CONDS, "n_per_arm": 300,
    "schema_adaptation": {
        "reason_skipped": "stage0_strict.load_expa requires manifest key 'prefix_steps'; "
                          "EXPB manifest lacks it (has proof_prefix_steps / assistant_prefill_steps).",
        "prefix_convention": "assistant_prefill_steps[:-1] (drops the injected F, which is passed "
                             "separately; condition-aware, so the certificate is seeded into the "
                             "validator prefix state for LOCAL/IRRELEVANT arms).",
        "prefill_ends_with_injected_F": f"{prefill_ends_with_F}/900",
    },
    "anchor_validation": {
        "composition_counts_match_EXPB_REPORT": True,
        "stage0_bounds_match": True,
        "doubt_discordant_111_0_match": True,
        "adapted_replay_reproduces_stored_class": f"{class_repro_agree}/{n_nonfailed} non-failed rows",
        "class_mismatches": class_mismatch,
        "generation_failed_rows": sum(1 for r in rows if r["gen_failed"]),
    },
    "a_composition": comp,
    "b_bounds": {"arm_level": bounds, "contrasts": contrasts},
    "c_survivors": {"doubt_by_arm": doubt_by_arm, "paired": surv},
    "png_ok": png_ok, "png_err": png_err, "csv": csv_path,
}
with open(OUT + "/expb_joint_results.json", "w") as fh:
    json.dump(result, fh, indent=2, default=str)
with open(OUT + "/strict_rows_expb.jsonl", "w") as fh:
    for r in strict_rows_out:
        fh.write(json.dumps(r, default=str) + "\n")
with open(OUT + "/strict_families_expb.json", "w") as fh:
    json.dump(strict_families, fh, indent=2, default=str)

# ============================================================ console summary
print(f"ANCHOR OK: composition==EXPB_REPORT, bounds==STAGE0, doubt discordant 111-0.")
print(f"ANCHOR: adapted replay reproduces stored class for {class_repro_agree}/{n_nonfailed} "
      f"non-failed rows; gen_failed={sum(1 for r in rows if r['gen_failed'])}; mismatches={len(class_mismatch)}")
print()
print("(a) COMPOSITION (proportion of all 300 rows/arm)")
print(f"{'outcome':<20}" + "".join(f"{SHORT[c]:>18}" for c in CONDS))
for cat in CATS:
    line = f"{CATLBL[cat]:<20}"
    for cond in CONDS:
        line += f"{comp[cond]['props'][cat]:>10.3f} ({comp[cond]['counts'][cat]:>3})"
    print(line)
print()
print("(b) MONOTONE BOUNDS on contrasts (unparsed treated as unknown outcome)")
for metric in CLSMAP:
    print(f"  metric={metric}")
    for cond in CONDS:
        bd = bounds[metric][cond]
        print(f"    {SHORT[cond]:<16} rate in [{bd['rate_worst']:.4f}, {bd['rate_best']:.4f}]  (k={bd['k']}, unparsed={bd['unparsed']})")
    for cn, cv in contrasts[metric].items():
        surv_txt = "SURVIVES (sign fixed)" if (cv["delta_min"] > 0 or cv["delta_max"] < 0) else "NOT robust (interval spans 0)"
        print(f"    {cn}: point={cv['point_diff']:+.4f}  delta in [{cv['delta_min']:+.4f}, {cv['delta_max']:+.4f}]  -> {surv_txt}")
print()
print("(c) SURVIVOR: doubt")
for cond in CONDS:
    d = doubt_by_arm[cond]
    print(f"    {SHORT[cond]:<16} doubt={d['doubt_rate']:.4f} ({d['doubt_k']}/{d['n']})  doubt_among_{d['unparsed_n']}_unparsed={d['doubt_among_unparsed']}")
dd = surv["doubt_LOCAL_vs_GLOBAL"]
print(f"    doubt LOCAL vs GLOBAL: net={dd['net_diff']:+.4f} discordant b(LOCAL-only)={dd['b_a_only']} c(GLOBAL-only)={dd['c_b_only']} "
      f"McNemar p={dd['mcnemar_p_exact']:.2e} boot95={dd['cluster_boot95']}")
di = surv["doubt_LOCAL_vs_IRRELEVANT"]
print(f"    doubt LOCAL vs IRRELEVANT: net={di['net_diff']:+.4f} discordant b={di['b_a_only']} c={di['c_b_only']} McNemar p={di['mcnemar_p_exact']:.2e}")
print(f"    poisoned LOCAL vs GLOBAL discordant: {surv['poisoned_LOCAL_vs_GLOBAL_paired']}")
print()
print("(d) STRICT PASS (pooled per arm; hop-sound & SRR via real stage0_strict)")
for cond in CONDS:
    fam = strict_families[SHORT[cond]]["pooled"]
    cv, hs = fam["closure_valid"], fam["hop_sound_valid"]
    m1 = fam["m1"]; srr = m1["srr_task"] if isinstance(m1, dict) else None
    der = fam["derivational"]; der_s = "n.m." if der == "n.m." else f"{der['rate']:.3f}({der['count']}/{der['n']})"
    print(f"    {SHORT[cond]:<16} n={fam['n']} closure_valid={cv['rate']:.3f}({cv['count']}) hop_sound={hs['rate']:.3f}({hs['count']}) "
          f"goal_jump0={fam['goal_jump0']['rate']:.3f} doubt={fam['doubt']['rate']:.3f} "
          f"SRR_task={srr['rate']:.3f}({srr['count']}/{srr['n']}) used_inj={fam['used_any']['rate']:.3f} "
          f"echo={fam['echo']['rate']:.3f} deriv={der_s}")
print()
print("PNG:", "written" if png_ok else f"FAILED {png_err}")
print("wrote:", OUT + "/{expb_joint_results.json,strict_families_expb.json,strict_rows_expb.jsonl,expb_composition.csv,expb_composition.png}")
