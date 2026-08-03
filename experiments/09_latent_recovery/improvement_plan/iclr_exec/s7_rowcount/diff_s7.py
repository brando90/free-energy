#!/usr/bin/env python3
"""Validate baseline reproduction vs published NATURAL_ERRORS summary, then diff fixed vs baseline."""
import json, hashlib, os

NAT = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/natural_errors"
SLUG = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/s7_rowcount"

pub = json.load(open(f"{NAT}/summary.json"))
base = json.load(open(f"{SLUG}/out_baseline/summary.json"))
fix = json.load(open(f"{SLUG}/out_fixed/summary.json"))

def hl(s):
    return dict(
        n_comb=s["combined"]["n_fresh_entity_errors"],
        n_pronto=s["prontoqa"]["n_fresh_entity_errors"],
        n_expd=s["expd_synthetic"]["n_fresh_entity_errors"],
        modal_false_cwa=s["prontoqa"]["truth"].get("false_cwa_unentailed"),
        usable=s["prontoqa"]["used_deriv_given_design_usable"],
        inert=s["prontoqa"]["used_deriv_given_design_inert"],
        ctrl_fp=s["control_false_positive"],
        n_anomalies=len(s["revalidates_valid_but_pipeline_failed"]),
        n_fab_comb=s["combined"]["n_rule_fabricated"],
        n_fab_pronto=s["prontoqa"]["n_rule_fabricated"],
        n_fab_expd=s["expd_synthetic"]["n_rule_fabricated"],
        n_failed_roll_comb=s["combined"]["n_failed_rollouts_auditable"],
        n_failed_roll_pronto=s["prontoqa"]["n_failed_rollouts_auditable"],
        n_failed_roll_expd=s["expd_synthetic"]["n_failed_rollouts_auditable"],
        taxonomy_pronto=s["prontoqa"]["rollout_taxonomy"],
    )

print("=== PUBLISHED vs BASELINE (reproduction check) ===")
p, b = hl(pub), hl(base)
for k in p:
    same = "OK " if p[k] == b[k] else "DIFF!!"
    print(f"  [{same}] {k}: pub={p[k]}  base={b[k]}")

print("\n=== BASELINE vs FIXED (the S7 change) ===")
f = hl(fix)
for k in b:
    same = "same" if b[k] == f[k] else "CHANGED"
    print(f"  [{same:8}] {k}: base={b[k]}  fixed={f[k]}")

# anomalies list (the reclassified rollouts)
print("\n=== anomalies in baseline (reclassify failed->valid under fix) ===")
for a in base["revalidates_valid_but_pipeline_failed"]:
    print("   ", a)
print("count:", len(base["revalidates_valid_but_pipeline_failed"]))

# byte-identity of the 236 entity_fact error rows across baseline & fixed
def entity_hash(path):
    rows = [l for l in open(path) if '"kind": "entity_fact"' in l]
    h = hashlib.sha256("".join(sorted(rows)).encode()).hexdigest()
    return len(rows), h
nb, hb = entity_hash(f"{SLUG}/out_baseline/natural_errors.jsonl")
nf, hf = entity_hash(f"{SLUG}/out_fixed/natural_errors.jsonl")
print(f"\nentity_fact rows: baseline n={nb} sha={hb[:16]} | fixed n={nf} sha={hf[:16]} | identical={hb==hf}")

# rollout-level diff: which problem_ids present in baseline roll set but not fixed
def rollset(path):
    return {(json.loads(l)["source"], json.loads(l)["problem_id"]) for l in open(path)}
rb = rollset(f"{SLUG}/out_baseline/natural_rollouts.jsonl")
rf = rollset(f"{SLUG}/out_fixed/natural_rollouts.jsonl")
print(f"\nnatural_rollouts: baseline={len(rb)} fixed={len(rf)} removed={len(rb-rf)} added={len(rf-rb)}")
print("removed (failed->valid):")
for x in sorted(rb-rf): print("   ", x)

# control cohort sizes
print(f"\ncontrol cohort: baseline valid={base['control_false_positive'][1]} fixed valid={fix['control_false_positive'][1]}")
