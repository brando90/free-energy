#!/usr/bin/env python3
import json, re
rows = json.load(open("/lfs/skampere2/0/eobbad/scratch/yee_audit/verify0/flagged_boundary_check.json"))
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
def canon(x):
    s = str(x).replace(",", "")
    if s.endswith(".0"): s = s[:-2]
    return s
ow = amb = 0
for r in rows:
    tgt = canon(float(r["target"].replace(",", "").replace("$", "")))
    pm = NUM.findall(r["prefix_tail"])
    head = r["cont_head"].lstrip()
    cm = NUM.match(head)
    flagged = False
    if pm and cm:
        p = canon(pm[-1].replace(",", ""))
        c = cm.group(0).replace(",", "")
        if len(c) < len(p) and p[: len(p) - len(c)] + c == tgt:
            ow += 1; flagged = True
            print(f"DIGIT-OVERWRITE {r['ds']}/{r['pert']} target={tgt} prefixnum={p} contstart={c!r} :: {r['prefix_tail'][-30:]!r} + {r['cont_head'][:40]!r}")
        elif p + c == tgt:
            ow += 1; flagged = True
            print(f"DIGIT-APPEND    {r['ds']}/{r['pert']} target={tgt} prefixnum={p} contstart={c!r} :: {r['prefix_tail'][-30:]!r} + {r['cont_head'][:40]!r}")
    if not flagged and head[:1].isdigit():
        amb += 1
        print(f"cont-starts-num {r['ds']}/{r['pert']} target={tgt} :: {r['prefix_tail'][-30:]!r} + {r['cont_head'][:40]!r}")
print(f"\ntotal flagged={len(rows)} digit-overwrite/append={ow} other-cont-starts-with-number={amb}")
