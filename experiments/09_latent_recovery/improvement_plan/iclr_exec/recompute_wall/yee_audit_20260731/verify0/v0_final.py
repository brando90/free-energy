#!/usr/bin/env python3
"""verify0 final: adversarial sensitivity — reclassify ALL digit-overwrite /
cont-starts-with-number flagged rows as continuation recoveries and recompute
headline stats; plus error-carried and bare-number sub-claims."""
import json, re
rows = json.load(open("/lfs/skampere2/0/eobbad/scratch/yee_audit/verify0/flagged_boundary_check.json"))
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
def canon(x):
    s = str(x).replace(",", "")
    if s.endswith(".0"): s = s[:-2]
    return s
# pooled totals from v0_rederive.out (independent run)
base = {"random": dict(n=966, recov=748, flag=59),
        "add1": dict(n=985, recov=755, flag=73),
        "add101": dict(n=983, recov=956, flag=13)}
susp = {"random": 0, "add1": 0, "add101": 0}
carried = {"random": 0, "add1": 0, "add101": 0}
bare_start = 0; bare_full = 0
for r in rows:
    pert = r["pert"]
    head = r["cont_head"].lstrip()
    if head[:1].isdigit():
        susp[pert] += 1
    # error carried to conclusion: last number of continuation equals prefix's last number
    pm = NUM.findall(r["prefix_tail"]); cm = NUM.findall(r["cont_tail"])
    if pm and cm and canon(pm[-1]) == canon(cm[-1]):
        carried[pert] += 1
    a = (r["extracted"] or "").strip()
    if a[:1].isdigit() or a[:1] == "$": bare_start += 1
    if re.fullmatch(r"[\$]?-?[\d,]+(?:\.\d+)?\.?", a): bare_full += 1
print("MAX-ADVERSARIAL sensitivity (reclassify every flagged row whose continuation starts with a digit):")
for p, b in base.items():
    f2 = b["flag"] - susp[p]
    print(f"  {p}: flagged {b['flag']} -> {f2}; extraction-only {100*b['flag']/b['recov']:.1f}% -> {100*f2/b['recov']:.1f}% of recoveries; "
          f"corrected_gen {100*(b['recov']-b['flag'])/b['n']:.1f}% -> {100*(b['recov']-f2)/b['n']:.1f}%")
print("\nerror-carried-to-conclusion (cont last num == prefix perturbed num), flagged rows:")
tot = sum(carried.values())
print(f"  {carried} total {tot}/{len(rows)} = {100*tot/len(rows):.0f}%  (their claim: 102/146 on random+add1)")
print(f"\nextraction Answer starts with digit/$: {bare_start}/{len(rows)}; strictly bare number: {bare_full}/{len(rows)} (their claim 155/160)")
