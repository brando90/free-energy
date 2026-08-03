#!/usr/bin/env python
"""Probe: confirm imports + analyze() reproduces stored EXPD closure class on a slice."""
import os, sys, json
EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
os.environ["LR_ROOT"] = EXP
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expd"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "stage0_strict"))
sys.path.insert(0, os.path.join(EXP, "src"))

import expd_confirmatory_analysis as CA
import strict_metrics as SM
print("imports OK")

OUT = os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT")
# load first N validated rows + manifest lookup
man = {}
with open(os.path.join(OUT, "manifest.jsonl")) as fh:
    for line in fh:
        r = json.loads(line)
        man[r["run_id"]] = r
print("manifest rows:", len(man))

n = 0
agree = 0
cell_stored = {}
cell_recomp = {}
mism = []
with open(os.path.join(OUT, "validated_outputs.jsonl")) as fh:
    for line in fh:
        r = json.loads(line)
        if r.get("failed_generation"):
            continue
        m = man.get(r["run_id"])
        if not m:
            continue
        a = SM.analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                       r.get("continuation", ""), m["target"], m["entity"])
        stored_valid = int(bool(r.get("valid_recovery")))
        recomp_valid = int(a["class"] == "valid_rederivation")
        cond = r["condition"]
        cell_stored.setdefault(cond, [0, 0]); cell_stored[cond][0] += stored_valid; cell_stored[cond][1] += 1
        cell_recomp.setdefault(cond, [0, 0]); cell_recomp[cond][0] += recomp_valid; cell_recomp[cond][1] += 1
        if stored_valid == recomp_valid:
            agree += 1
        elif len(mism) < 8:
            mism.append({"run_id": r["run_id"], "cond": cond, "stored_class": r.get("class"),
                         "recomp_class": a["class"], "stored_valid": stored_valid,
                         "hop_sound_all": a["hop_sound_all"]})
        n += 1
        if n >= 4000:
            break

print(f"\nrows checked: {n}  closure-class agreement: {agree}/{n} = {agree/n:.4f}")
print("\nsample mismatches:")
for x in mism:
    print(" ", x)
print("\nper-cell stored closure-valid vs recomputed (first slice, for anchor check):")
for c in sorted(cell_stored):
    s = cell_stored[c]; rc = cell_recomp[c]
    print(f"  {c:24s} stored {s[0]/s[1]:.4f} ({s[0]}/{s[1]})  recomp {rc[0]/rc[1]:.4f}")
