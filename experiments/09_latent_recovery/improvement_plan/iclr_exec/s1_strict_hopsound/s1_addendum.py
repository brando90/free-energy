#!/usr/bin/env python
"""S1 addendum: per-d (d1-only, d3-only) paired TOST for H2' under closure vs
hop_sound, to confirm the pooled equivalence isn't masking a within-distance
reversal. Also a tighter-margin (+-0.05) equivalence check on the pooled DV."""
import os, sys, json
EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
os.environ.setdefault("LR_ROOT", EXP)
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expd"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "stage0_strict"))
sys.path.insert(0, os.path.join(EXP, "src"))
import expd_confirmatory_analysis as CA
import strict_metrics as SM
OUT = os.path.join(EXP, "improvement_plan", "iclr_exec", "s1_strict_hopsound")

def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]

rows, man = CA.load_rows(os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT"))
cohort = read_jsonl(os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT", "gold_cohort.jsonl"))
joint, prefix_same = CA.attr_cohort_maps(cohort)
CA.annotate_cohort(rows, joint, prefix_same)
for r in rows:
    m = man.get(r["run_id"])
    if r.get("failed_generation"):
        r["_hop_sound_valid"] = 0
    else:
        a = SM.analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                       r.get("continuation", ""), m["target"], m["entity"])
        r["_hop_sound_valid"] = int(bool(r.get("valid_recovery")) and bool(a["hop_sound_all"]))

orig = CA.mval
def mval2(r, metric):
    if metric == "hop_sound_valid":
        return int(bool(r.get("_hop_sound_valid")))
    return orig(r, metric)
CA.mval = mval2

def tost_run(a_cells, b_cells, metric, margin):
    ra = CA.cells_rows(rows, a_cells, cohort_only=True)
    rb = CA.cells_rows(rows, b_cells, cohort_only=True)
    ps = CA.paired_family_stats(ra, rb, metric)
    t = CA.tost(ps, margin=margin)
    return {"metric": metric, "margin": margin, "diff": t["mean_diff"], "ci95": t["ci95"],
            "p_raw": t["p_raw"], "pass_0.05": bool(t["p_raw"] is not None and t["p_raw"] < 0.05),
            "pass_holm": bool(t["p_raw"] is not None and t["p_raw"] < 0.05/6),
            "aff": round(ps["rate_a"], 4), "neg": round(ps["rate_b"], 4),
            "fams": ps["n_families_paired"], "rows": [ps["n_rows_a"], ps["n_rows_b"]]}

res = {}
for label, ac, bc in [("pooled_d1_d3", ["aff_false_attr_d1", "aff_false_attr_d3"], ["neg_false_attr_d1", "neg_false_attr_d3"]),
                      ("d1_only", ["aff_false_attr_d1"], ["neg_false_attr_d1"]),
                      ("d3_only", ["aff_false_attr_d3"], ["neg_false_attr_d3"])]:
    res[label] = {"closure_valid": tost_run(ac, bc, "closure_valid", 0.10),
                  "hop_sound_valid": tost_run(ac, bc, "hop_sound_valid", 0.10)}
# tighter margin on pooled hop-sound
res["pooled_tighter_margin_0.05"] = {"hop_sound_valid": tost_run(
    ["aff_false_attr_d1", "aff_false_attr_d3"], ["neg_false_attr_d1", "neg_false_attr_d3"],
    "hop_sound_valid", 0.05)}
CA.mval = orig
json.dump(res, open(os.path.join(OUT, "s1_h2prime_sensitivity.json"), "w"), indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
