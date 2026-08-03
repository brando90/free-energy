"""E10 TOPUP superset generator (batch2).

Generates NEW EXPG program-trace worlds with the SAME KNOBS as the original
regime2 batch1 (knob_hash 507786) but NON-OVERLAPPING seed bases, so batch1 and
batch2 are a clean disjoint superset: identical generative distribution, distinct
program_ids (seed-suffixed). Batch1 (the original 250 in regime2/programs.jsonl)
is left untouched and remains a subset of the union.

Only the three shapes carrying the FOUR KEY CELLS are generated:
  kr1 -> adjacent_contradiction, opfree_kr1   kc1 -> onehop_kc1   kc5 -> deep_kc5
(kr8 / benign_paraphrase / true_interruption are not key cells; omitted to save budget.)
"""
import json, os, sys, hashlib
from collections import Counter
EXP="/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
EXPG=os.path.join(EXP,"improvement_plan","expg")
sys.path.insert(0, EXPG)
import gen_programs as gp

OUT_DIR=os.path.join(EXP,"improvement_plan","iclr_exec","recompute_wall","e10_topup_fable")
os.makedirs(OUT_DIR, exist_ok=True)

# batch1 (regime2) seed ranges, per shape (measured): kr1 100000..100083,
# kc1 300000..300078, kc5 400001..400077. batch2 bases sit safely beyond.
N_PER_SHAPE=120
SHAPE_BASE={"kr1":100100, "kc1":300100, "kc5":400100}

# batch1 program_ids for the disjointness assertion
b1=[json.loads(l) for l in open(os.path.join(EXP,"results","EXPH2_FRONTIER_FOLLOWUPS","regime2","programs.jsonl"))]
b1_ids={r["program_id"] for r in b1}
b1_kh={r["knob_hash"] for r in b1}

allprogs=[]; funnels={}
for shape, base in SHAPE_BASE.items():
    progs, funnel = gp.generate_shape_programs(shape, N_PER_SHAPE, base)
    allprogs.extend(progs)
    funnels[shape]=funnel

kh={p["knob_hash"] for p in allprogs}
assert kh==b1_kh, f"knob_hash mismatch batch2={kh} batch1={b1_kh}"  # same generative distribution
b2_ids={p["program_id"] for p in allprogs}
assert b2_ids.isdisjoint(b1_ids), "batch2 program_ids overlap batch1!"

outp=os.path.join(OUT_DIR,"programs_batch2.jsonl")
with open(outp,"w") as fh:
    for p in allprogs:
        fh.write(json.dumps(p)+"\n")

fp=hashlib.sha256(json.dumps(sorted(b2_ids)).encode()).hexdigest()[:24]
prov={
 "batch":"batch2 (E10 TOPUP superset supplement)",
 "generator":"improvement_plan/expg/gen_programs.py generate_shape_programs (verbatim; no edits)",
 "knob_hash":list(kh),
 "knob_hash_matches_batch1":kh==b1_kh,
 "n_per_shape":N_PER_SHAPE,
 "shapes":dict(Counter(p["shape"] for p in allprogs)),
 "seed_bases":SHAPE_BASE,
 "seed_ranges":{s:[min(p["seed"] for p in allprogs if p["shape"]==s),
                   max(p["seed"] for p in allprogs if p["shape"]==s)] for s in SHAPE_BASE},
 "batch2_program_id_fingerprint":fp,
 "disjoint_from_batch1":True,
 "batch1_source":"results/EXPH2_FRONTIER_FOLLOWUPS/regime2/programs.jsonl (250 programs; the original, kept as a subset of the union)",
 "note":"Only kr1/kc1/kc5 generated (the shapes carrying the four KEY cells). "
        "batch1+batch2 form the superset; batches are distinguished by disjoint seed "
        "ranges under an identical knob_hash. Pooling is over the union per (model,cell).",
 "generator_funnels":funnels,
}
json.dump(prov, open(os.path.join(OUT_DIR,"worlds_provenance_batch2.json"),"w"), indent=2)
print("batch2 worlds:", len(allprogs), "fp=", fp, "shapes=", prov["shapes"], "kh=", kh)
print("seed_ranges=", prov["seed_ranges"])
