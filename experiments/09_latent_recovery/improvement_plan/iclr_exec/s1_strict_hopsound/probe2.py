#!/usr/bin/env python
"""S1 probe: validate SM.analyze adaptation to EXPD/EXPE + reproduce anchors.
Read-only. Imports the PUBLISHED cluster estimators (CA, SM, validator)."""
import os, sys, json
from collections import defaultdict

EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
os.environ.setdefault("LR_ROOT", EXP)
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expd"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "stage0_strict"))
sys.path.insert(0, os.path.join(EXP, "src"))
import expd_confirmatory_analysis as CA   # published H2' estimator
import strict_metrics as SM               # published hop-sound metric
import validator as V

def read_jsonl(p):
    with open(p) as fh:
        return [json.loads(l) for l in fh if l.strip()]

EXPD = os.path.join(EXP, "results", "EXPD_MATCHED_GRADIENT")
EXPE = os.path.join(EXP, "results", "EXPE_EVIDENCE_MOVER")

# ---------- EXPD: recompute-class agreement + hop-sound + cell closure anchor ----------
print("== EXPD ==")
man = {r["run_id"]: r for r in read_jsonl(os.path.join(EXPD, "manifest.jsonl"))}
agree = tot = 0
cell_cv = defaultdict(lambda: [0, 0])       # stored valid_recovery
cell_hs = defaultdict(lambda: [0, 0])       # hop_sound_valid (recomputed)
cell_cv_recomp = defaultdict(lambda: [0, 0])  # recomputed valid class
mism = []
for r in read_jsonl(os.path.join(EXPD, "validated_outputs.jsonl")):
    m = man.get(r["run_id"])
    if not m:
        continue
    cond = r["condition"]
    stored_valid = int(bool(r.get("valid_recovery")))
    cell_cv[cond][0] += stored_valid; cell_cv[cond][1] += 1
    if r.get("failed_generation"):
        recomp_valid = 0; hop_all = 0
    else:
        a = SM.analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                       r.get("continuation", ""), m["target"], m["entity"])
        recomp_valid = int(a["class"] == "valid_rederivation")
        hop_all = int(bool(a["hop_sound_all"]))
    cell_cv_recomp[cond][0] += recomp_valid; cell_cv_recomp[cond][1] += 1
    hs = int(stored_valid == 1 and hop_all == 1)
    cell_hs[cond][0] += hs; cell_hs[cond][1] += 1
    if stored_valid == recomp_valid:
        agree += 1
    elif len(mism) < 6:
        mism.append({"run_id": r["run_id"], "cond": cond, "stored_class": r.get("class"),
                     "recomp_valid": recomp_valid, "stored_valid": stored_valid})
    tot += 1
print(f"EXPD closure recompute agreement: {agree}/{tot} = {agree/tot:.5f}")
print("mismatches sample:", mism[:6])
print("\nEXPD per-cell (published closure-valid anchor in parens):")
anchors = {"aff_false_attr_d0": 0.9222, "aff_false_attr_d1": 0.9222, "aff_false_attr_d3": 0.9456,
           "neg_false_attr_d1": 0.9044, "neg_false_attr_d3": 0.8552,
           "cat_false_usable_d1": 0.0933, "cat_false_inert_d1": 0.6233,
           "benign_paraphrase": 0.9978, "true_interruption": 0.9178}
for c in sorted(cell_cv):
    s = cell_cv[c]; rc = cell_cv_recomp[c]; h = cell_hs[c]
    anc = anchors.get(c)
    flag = ""
    if anc is not None:
        flag = f"  [pub {anc:.4f} {'OK' if abs(s[0]/s[1]-anc)<0.0005 else 'MISMATCH'}]"
    print(f"  {c:22s} storedCV {s[0]/s[1]:.4f}  recompCV {rc[0]/rc[1]:.4f}  hopSound {h[0]/h[1]:.4f}{flag}")

# ---------- EXPE: recompute-class agreement ----------
print("\n== EXPE ==")
mane = {r["run_id"]: r for r in read_jsonl(os.path.join(EXPE, "manifest.jsonl"))}
agree = tot = 0
cell_cv = defaultdict(lambda: [0, 0]); cell_hs = defaultdict(lambda: [0, 0])
for r in read_jsonl(os.path.join(EXPE, "validated_outputs.jsonl")):
    m = mane.get(r["run_id"])
    if not m:
        continue
    cell = "%s@%s" % (r.get("arm", r.get("condition")), r["injection_position"])
    stored_valid = int(bool(r.get("valid_recovery")))
    cell_cv[cell][0] += stored_valid; cell_cv[cell][1] += 1
    if r.get("failed_generation"):
        recomp_valid = 0; hop_all = 0
    else:
        a = SM.analyze(m["question"], m.get("prefix_steps", []), m["injected_statement"],
                       r.get("continuation", ""), m["target"], m["entity"])
        recomp_valid = int(a["class"] == "valid_rederivation")
        hop_all = int(bool(a["hop_sound_all"]))
    hs = int(stored_valid == 1 and hop_all == 1)
    cell_hs[cell][0] += hs; cell_hs[cell][1] += 1
    if stored_valid == recomp_valid:
        agree += 1
    tot += 1
print(f"EXPE closure recompute agreement: {agree}/{tot} = {agree/tot:.5f}")
# EXPE published closure-valid anchors (from EXPE_FULL_REPORT descriptives)
epub = {"REFUTING_d1@mid": 0.3636, "BASELINE_FILLER@mid": 0.3591, "FREQ_MATCHED_NONREFUTING@mid": 0.35,
        "TEMPLATE_IRRELEVANT@mid": 0.4045, "REFUTING_d2@mid": 0.3864, "REFUTING_d3@mid": 0.3591}
print("EXPE per-cell closure-valid vs published + hop-sound:")
for c in sorted(cell_cv):
    s = cell_cv[c]; h = cell_hs[c]; anc = epub.get(c)
    flag = f"  [pub {anc:.4f} {'OK' if abs(s[0]/s[1]-anc)<0.001 else 'CHK'}]" if anc else ""
    print(f"  {c:32s} storedCV {s[0]/s[1]:.4f}  hopSound {h[0]/h[1]:.4f}{flag}")
