#!/usr/bin/env python3
"""Pin down what differs between the two annotated variants per awps problem,
and write a compact corrected-rates CSV."""
import csv, re
csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/audit_extraction"
MODEL = "gpt-4-0314"
MARKER = "A: Let's think step by step."

def norm(s): return s.replace("\r\n", "\n").replace("\r", "\n")

for ds, stem, pert in (("awps","MultiArith","add1"), ("asdiv","ASDiv","add1"), ("gsm8k","test","add1"), ("svamp","SVAMP","add1")):
    path = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("Model Name") == MODEL and r.get("Prompt Style") == "sbs"]
    byq = {}
    for r in rows:
        byq.setdefault(norm(r["Question"]), []).append(r)
    # group by problem
    probs = {}
    for q in byq:
        mi = q.find(MARKER)
        probs.setdefault(q[3:mi].strip(), []).append(q)
    multi = {p: qs for p, qs in probs.items() if len(qs) > 1}
    if not multi:
        print(f"{ds}/{pert}: no multi-prefix problems"); continue
    p, qs = next(iter(multi.items()))
    a, b = qs[0], qs[1]
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]: i += 1
    print(f"{ds}/{pert}: {len(multi)} problems with 2 Question variants")
    print(f"  lens {len(a)} vs {len(b)}, first divergence at {i}")
    print(f"  A: ...{a[max(0,i-40):i+50]!r}")
    print(f"  B: ...{b[max(0,i-40):i+50]!r}")
    ra, rb = byq[a][0], byq[b][0]
    fa, fb = norm(ra["Full Prompt"]), norm(rb["Full Prompt"])
    # continuation comparison: strip the question part
    ca, cb = fa[len(a):], fb[len(b):]
    print(f"  continuations identical: {ca == cb}; Answers identical: {ra['Answer'] == rb['Answer']}")
    ann = [( (byq[q][0].get('Correct?') or '').strip(), (byq[q][0].get('Recovery Behavior') or '').strip()) for q in qs]
    print(f"  annotations: {ann}")
    # how many of the multi pairs have identical continuation+answer
    same = 0; tot = 0
    for pp, qq in multi.items():
        if len(qq) != 2: continue
        r1, r2 = byq[qq[0]][0], byq[qq[1]][0]
        f1, f2 = norm(r1["Full Prompt"])[len(qq[0]):], norm(r2["Full Prompt"])[len(qq[1]):]
        tot += 1
        if f1 == f2 and r1["Answer"] == r2["Answer"]: same += 1
    print(f"  pairs with identical continuation AND Answer: {same}/{tot}")
    print()
