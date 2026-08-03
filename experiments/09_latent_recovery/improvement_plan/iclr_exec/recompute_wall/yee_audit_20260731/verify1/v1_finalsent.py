#!/usr/bin/env python3
"""For wrong->RIGHT overrides where target appears in continuation:
is the target in the FINAL sentence (parser reading the conclusion)?
Also recompute corrected per-dataset solver-evidence bound = target-ABSENT overrides
plus target-in-cont-but-not-final-sentence overrides."""
import csv, re
csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
MODEL = "gpt-4-0314"; SUFFIX = " Therefore, the answer (arabic numerals) is"
MARKER = "A: Let's think step by step."
DATASETS = {"gsm8k":"test","asdiv":"ASDiv","awps":"MultiArith","svamp":"SVAMP"}
PERTS = ["random","add1","add101"]
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def norm(s): return s.replace("\r\n","\n").replace("\r","\n")
def nums(t):
    out=[]
    for m in NUM_RE.finditer(t):
        try: out.append(float(m.group(0).replace(",","")))
        except ValueError: pass
    return out
def close(a,b): return abs(a-b)<1e-6*max(1.0,abs(a),abs(b))
def load(ds,stem,pert):
    path=f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
    with open(path,newline="",encoding="utf-8-sig") as f:
        rows=[r for r in csv.DictReader(f) if r.get("Model Name")==MODEL and r.get("Prompt Style")=="sbs"]
    byq={}
    for r in rows:
        q=norm(r["Question"]); cur=byq.get(q)
        if cur is None or ((r.get("Correct?") or "").strip() and not (cur.get("Correct?") or "").strip()):
            byq[q]=r
    return byq
def final_sentence(t):
    t=t.strip()
    parts=re.split(r"(?<=[.!?])\s+|\n+", t)
    parts=[p for p in parts if p.strip()]
    return parts[-1] if parts else t

print(f"{'ds':6} {'n':>5} {'w2r':>4} {'absent':>6} {'in-final-sent':>13} {'mid-text-only':>13}  corrected-solver-bound")
for ds,stem in DATASETS.items():
    n=w2r=absent=infinal=midonly=0
    for pert in PERTS:
        for q,r in load(ds,stem,pert).items():
            if (r.get("Error Type") or "").strip()!="calculation": continue
            if (r.get("Correct?") or "").strip() not in ("True","False"): continue
            try: target=float((r["Target Answer"] or "").replace(",",""))
            except ValueError: continue
            fp=norm(r["Full Prompt"]); cont=fp[len(q):]
            if cont.endswith(SUFFIX): cont=cont[:-len(SUFFIX)]
            cont=cont.strip()
            cnums=nums(cont); anums=nums(r.get("Answer") or "")
            if not cnums or not anums: continue
            n+=1
            if close(anums[0],cnums[-1]): continue
            if close(anums[0],target) and not close(cnums[-1],target):
                w2r+=1
                if not any(close(x,target) for x in cnums): absent+=1
                elif any(close(x,target) for x in nums(final_sentence(cont))): infinal+=1
                else: midonly+=1
    solver=absent+midonly
    print(f"{ds:6} {n:>5} {w2r:>4} {absent:>6} {infinal:>13} {midonly:>13}  {solver} ({100*solver/n:.1f}pp) vs claimed {100*w2r/n:.1f}pp")
