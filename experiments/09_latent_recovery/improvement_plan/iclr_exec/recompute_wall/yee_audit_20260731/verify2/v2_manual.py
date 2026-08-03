#!/usr/bin/env python3
import csv, glob, re, random
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
CUE = "Therefore, the answer (arabic numerals) is"

def numbers_in(text):
    out = set()
    for m in re.finditer(r'-?\$?(\d[\d,]*\.?\d*)', text):
        s = m.group(1).replace(',', '').rstrip('.')
        try: out.add(float(s))
        except ValueError: pass
    return out

def parse_target(t):
    try: return float(str(t).replace(',', '').strip())
    except ValueError: return None

absent, present = [], []
for ds in ['asdiv','awps','gsm8k','svamp']:
    for f in sorted(glob.glob(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv")):
        for r in csv.DictReader(open(f)):
            if r['Correct?']!='True' or r['Recovery Behavior'].strip()!='complete hallucination': continue
            q, fp = r['Question'], r['Full Prompt']
            cont = fp[len(q):] if fp.startswith(q) else ''
            i = cont.rfind(CUE)
            if i != -1: cont = cont[:i]
            tgt = parse_target(r['Target Answer'])
            rec = (ds, r['Target Answer'], r['Answer'][:30], r['Question'][-80:], cont[:300])
            if tgt is not None and tgt in numbers_in(cont): present.append(rec)
            else: absent.append(rec)

random.seed(42)
print(f"### 30 random ABSENT (of {len(absent)}) — expect: continuation concludes WITHOUT target ###")
for ds,t,a,qt,c in random.sample(absent, 30):
    print(f"[{ds}] tgt={t} ext_ans={a!r}\n  prefix_end: ...{qt!r}\n  CONT: {c!r}\n")
print(f"### 10 random PRESENT (of {len(present)}) — expect: target somewhere in continuation ###")
for ds,t,a,qt,c in random.sample(present, 10):
    print(f"[{ds}] tgt={t} ext_ans={a!r}\n  CONT: {c!r}\n")
