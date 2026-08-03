#!/usr/bin/env python3
import csv, glob, re, collections
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

datasets = ['asdiv', 'awps', 'gsm8k', 'svamp']
print("Per (dataset, perturbation-file, model): N_CH_correct  / absent")
for ds in datasets:
    for f in sorted(glob.glob(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv")):
        rows = [r for r in csv.DictReader(open(f))
                if r['Correct?']=='True' and r['Recovery Behavior'].strip()=='complete hallucination']
        by_model = collections.Counter(r['Model Name'] for r in rows)
        stats = {}
        for r in rows:
            q, fp = r['Question'], r['Full Prompt']
            cont = fp[len(q):] if fp.startswith(q) else ''
            i = cont.rfind(CUE)
            if i != -1: cont = cont[:i]
            tgt = parse_target(r['Target Answer'])
            absent = not (tgt is not None and tgt in numbers_in(cont))
            m = r['Model Name']
            stats.setdefault(m, [0,0])
            stats[m][0] += 1
            stats[m][1] += absent
        short = f.split('position-calc_')[1].replace('_annotated.csv','')
        for m,(n,a) in sorted(stats.items()):
            print(f"{ds:6s} {short:35s} {m:32s} N={n:4d} absent={a:4d} ({a/n*100:.0f}%)")
