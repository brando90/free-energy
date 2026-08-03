#!/usr/bin/env python3
import json, os
SLUG = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/s7_solved_fix_rowcount"
b = json.load(open(os.path.join(SLUG, "out_baseline", "summary.json")))
f = json.load(open(os.path.join(SLUG, "out_fixed", "summary.json")))

def fabU(seg):
    return seg["n_rule_fabricated"] - seg["fab_rule_true_entailed_implication"]

def row(name, bv, fv):
    flag = "" if bv == fv else "   <<< CHANGED"
    print("%-52s baseline=%-14s fixed=%-14s%s" % (name, str(bv), str(fv), flag))

print("=== HEADLINE-NUMBER DIFF (baseline -> fixed) ===")
for lab in ["combined", "prontoqa", "expd_synthetic"]:
    B, F = b[lab], f[lab]
    print("-- %s --" % lab)
    row("  n_fresh_entity_errors", B["n_fresh_entity_errors"], F["n_fresh_entity_errors"])
    row("  n_failed_rollouts_auditable", B["n_failed_rollouts_auditable"], F["n_failed_rollouts_auditable"])
    row("  n_rule_fabricated(raw,incl entailed-true)", B["n_rule_fabricated"], F["n_rule_fabricated"])
    row("  fabricated_rules_UNENTAILED(report num)", fabU(B), fabU(F))
    row("  n_unparsed_entity", B["n_unparsed_entity"], F["n_unparsed_entity"])
    row("  n_off_entity", B["n_off_entity"], F["n_off_entity"])
    row("  used_deriv_given_design_usable", tuple(B["used_deriv_given_design_usable"]), tuple(F["used_deriv_given_design_usable"]))
    row("  used_deriv_given_design_inert", tuple(B["used_deriv_given_design_inert"]), tuple(F["used_deriv_given_design_inert"]))
    row("  truth.false_cwa_unentailed", B["truth"].get("false_cwa_unentailed"), F["truth"].get("false_cwa_unentailed"))

row("control_false_positive", tuple(b["control_false_positive"]), tuple(f["control_false_positive"]))
row("N anomalies (revalidate-valid)", len(b["revalidates_valid_but_pipeline_failed"]), len(f["revalidates_valid_but_pipeline_failed"]))

print("\n=== derived rates (PrOntoQA) ===")
B, F = b["prontoqa"], f["prontoqa"]
bu, fu = B["used_deriv_given_design_usable"], F["used_deriv_given_design_usable"]
bi, fi = B["used_deriv_given_design_inert"], F["used_deriv_given_design_inert"]
print("usable rate: base %d/%d=%.3f  fixed %d/%d=%.3f" % (bu[0], bu[1], bu[0]/bu[1], fu[0], fu[1], fu[0]/fu[1]))
print("inert  rate: base %d/%d=%.3f  fixed %d/%d=%.3f" % (bi[0], bi[1], bi[0]/bi[1], fi[0], fi[1], fi[0]/fi[1]))
bn, fn = B["n_fresh_entity_errors"], F["n_fresh_entity_errors"]
bmod, fmod = B["truth"].get("false_cwa_unentailed"), F["truth"].get("false_cwa_unentailed")
print("modal cell: base %d/%d=%.3f  fixed %d/%d=%.3f" % (bmod, bn, bmod/bn, fmod, fn, fmod/fn))

print("\n=== which rollouts reclassified (baseline anomalies) ===")
for a in b["revalidates_valid_but_pipeline_failed"]:
    print("  ", a["source"], a["problem_id"], "solved=", a["solved"])
print("fixed anomaly list (should be empty):", f["revalidates_valid_but_pipeline_failed"])

# rollout-level taxonomy change (denominators the report cites: 107/225 etc.)
print("\n=== PrOntoQA rollout taxonomy (denominator 225 -> ?) ===")
print("baseline:", json.dumps(b["prontoqa"]["rollout_taxonomy"], sort_keys=True))
print("fixed   :", json.dumps(f["prontoqa"]["rollout_taxonomy"], sort_keys=True))
print("baseline official_class:", json.dumps(b["prontoqa"]["official_class"], sort_keys=True))
print("fixed    official_class:", json.dumps(f["prontoqa"]["official_class"], sort_keys=True))
