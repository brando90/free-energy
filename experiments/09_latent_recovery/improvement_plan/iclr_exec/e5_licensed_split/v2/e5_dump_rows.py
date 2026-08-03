"""Emit fresh per-row classification (CURRENT classifier) to v2/classified_rows_v2.jsonl,
and per-cell strict-vs-lenient licensed counts. No bootstrap; fast."""
import sys, os, json
from collections import defaultdict, Counter
EXP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery"
SLUG = os.path.join(EXP, "improvement_plan", "iclr_exec", "e5_licensed_split")
sys.path.insert(0, os.path.join(EXP, "src"))
sys.path.insert(0, os.path.join(EXP, "improvement_plan", "expe"))
sys.path.insert(0, SLUG)
from e5_classify import classify_row, load_expe, load_expd
OUT = os.path.join(SLUG, "v2")

rows = load_expe() + load_expd()
classified = []
for r in rows:
    if not r["stated"]:
        continue
    c = classify_row(r["question"], r["entity"], r["planted"], r["prefix_steps"], r["continuation"])
    r.update(c)
    classified.append(r)

with open(os.path.join(OUT, "classified_rows_v2.jsonl"), "w") as f:
    for r in classified:
        rec = {k: r[k] for k in ("exp", "cell", "position", "designed_d", "problem_id",
               "run_id", "measured_d", "WD", "sw_strict", "sw_closure", "sw_primary",
               "sw_lenient", "rule_written_any", "wrote_advancing_intermediate",
               "klass", "klass_lenient", "reason", "comp_pred")}
        f.write(json.dumps(rec, sort_keys=True) + "\n")

# per-cell strict vs lenient licensed
strict = defaultdict(Counter); lenient = defaultdict(Counter)
for r in classified:
    strict[(r["exp"], r["cell"])][r["klass"]] += 1
    lenient[(r["exp"], r["cell"])][r["klass_lenient"]] += 1
print("cell | nSC | lic_strict | lic_lenient (delta)")
for k in sorted(strict):
    s = strict[k]; l = lenient[k]
    nsc = sum(s.values())
    ls = s.get("licensed", 0); ll = l.get("licensed", 0)
    if nsc:
        print(f"{k[0]}:{k[1]:26s} nSC={nsc:3d}  lic_strict={ls:3d}  lic_lenient={ll:3d}  (+{ll-ls})")
print("wrote", os.path.join(OUT, "classified_rows_v2.jsonl"), "n=", len(classified))
