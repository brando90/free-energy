import json, csv, sys, re
csv.field_size_limit(10**8)
R="/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results"
jp=f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_gpt-4-0314.json"
cp=f"{R}/gsm8k/test_adjusted_position-calc_perturbation-add1_annotated.csv"
d=json.load(open(jp))
print("JSON entries:", len(d))
ks=list(d.keys())
for k in ks[:2]:
    v=d[k]
    print("KEY:", repr(k[:120])); print("TYPE:", type(v), len(v))
    print("V0 (first 300):", repr(v[0][:300]))
    print("V0 (last 120):", repr(v[0][-120:]))
    print("V1:", repr(v[1]))
    print("---")
rows=list(csv.DictReader(open(cp, encoding="utf-8")))
print("CSV rows:", len(rows))
print("CSV cols:", rows[0].keys())
from collections import Counter
for col in ["Model Name","Prompt Style","Error Type","Correct?","Recovery Behavior"]:
    print(col, Counter(r.get(col) for r in rows))
