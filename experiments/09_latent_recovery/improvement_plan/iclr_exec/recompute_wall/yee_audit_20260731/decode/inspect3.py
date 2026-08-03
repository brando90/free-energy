import json, csv, os, glob
from collections import Counter, defaultdict
csv.field_size_limit(10**8)
R="/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results"

def norm(s): return (s or "").replace("\r\n","\n").replace("\r","\n")

# 1. Global RB / Correct? enumeration across all annotated CSVs + inventory
inv=[]
rb_all=Counter(); corr_all=Counter(); rb_by_corr=Counter(); et_all=Counter()
for f in sorted(glob.glob(f"{R}/*/*_annotated.csv")):
    rows=list(csv.DictReader(open(f, encoding="utf-8")))
    models=Counter(r["Model Name"] for r in rows)
    anno=Counter((r["Model Name"]) for r in rows if r.get("Correct?"))
    inv.append((os.path.relpath(f,R), len(rows), dict(models), dict(anno)))
    for r in rows:
        rb_all[r.get("Recovery Behavior","")]+=1
        corr_all[r.get("Correct?","")]+=1
        rb_by_corr[(r.get("Correct?",""), r.get("Recovery Behavior",""))]+=1
        et_all[r.get("Error Type","")]+=1
print("== RB values global =="); [print(repr(k),v) for k,v in rb_all.most_common()]
print("== Correct? values global =="); [print(repr(k),v) for k,v in corr_all.most_common()]
print("== Error Type values global =="); [print(repr(k),v) for k,v in et_all.most_common()]
print("== Correct? x RB (only RB nonblank or Correct nonblank) ==")
for k,v in sorted(rb_by_corr.items()):
    if k[0] or k[1]: print(k,v)
print()
print("== Inventory (file, total rows, rows per model, annotated rows per model) ==")
for line in inv: print(line)
