#!/usr/bin/env python3
"""Check near-duplicate structure in gsm8k add1 (decode fact 12 said 595 gpt-4 rows,
267 dup pairs w/ identical Full Prompt, but exact-Question dedupe found 0 dups)."""
import csv
from collections import Counter, defaultdict
csv.field_size_limit(10**9)
P = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results/gsm8k/test_adjusted_position-calc_perturbation-add1_annotated.csv"
rows = list(csv.DictReader(open(P, encoding='utf-8', newline='')))
print("rows:", len(rows))
bym = Counter(r["Model Name"] for r in rows)
print("by model:", bym)
g4 = [r for r in rows if r["Model Name"] == "gpt-4-0314"]
print("gpt-4 rows:", len(g4))
fp = Counter(r["Full Prompt"] for r in g4)
q = Counter(r["Question"] for r in g4)
bare = Counter(r["Question"].split("\n\nA:")[0] for r in g4)
print("gpt-4 unique Full Prompt:", len(fp), " dup FP groups:", sum(1 for v in fp.values() if v > 1))
print("gpt-4 unique Question:", len(q), " dup Q groups:", sum(1 for v in q.values() if v > 1))
print("gpt-4 unique bare problem:", len(bare), " dup bare groups:", sum(1 for v in bare.values() if v > 1))
# find one bare-dup pair and show how their Question strings differ
byb = defaultdict(list)
for i, r in enumerate(g4):
    byb[r["Question"].split("\n\nA:")[0]].append(i)
shown = 0
for b, idxs in byb.items():
    if len(idxs) > 1 and shown < 3:
        i1, i2 = idxs[0], idxs[1]
        q1, q2 = g4[i1]["Question"], g4[i2]["Question"]
        print("\n--- bare-dup pair ---")
        print("Q equal:", q1 == q2, "| FP equal:", g4[i1]["Full Prompt"] == g4[i2]["Full Prompt"],
              "| Ans equal:", g4[i1]["Answer"] == g4[i2]["Answer"],
              "| Correct?:", repr(g4[i1]["Correct?"]), repr(g4[i2]["Correct?"]),
              "| ErrType:", repr(g4[i1]["Error Type"]), repr(g4[i2]["Error Type"]))
        if q1 != q2:
            # first divergence
            for j, (a, b2) in enumerate(zip(q1, q2)):
                if a != b2:
                    print(f"first Q divergence at char {j}: {q1[max(0,j-40):j+40]!r} VS {q2[max(0,j-40):j+40]!r}")
                    break
            else:
                print(f"one is prefix of other: len {len(q1)} vs {len(q2)}; tail: {q1[len(q2):][:80]!r} {q2[len(q1):][:80]!r}")
        shown += 1
# annotation status of dup pairs
both, one, zero = 0, 0, 0
for b, idxs in byb.items():
    if len(idxs) > 1:
        ann = sum(1 for i in idxs if g4[i]["Correct?"].strip() != "")
        if ann == len(idxs): both += 1
        elif ann > 0: one += 1
        else: zero += 1
print("\nbare-dup groups: all-annotated:", both, "partially:", one, "none:", zero)
print("unannotated gpt-4 rows:", sum(1 for r in g4 if not r["Correct?"].strip()))
# do unannotated rows sit in dup groups?
undup = sum(1 for b, idxs in byb.items() if len(idxs) > 1
            for i in idxs if not g4[i]["Correct?"].strip())
print("unannotated gpt-4 rows inside bare-dup groups:", undup)
