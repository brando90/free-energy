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

def cont_of(r):
    q, fp = r['Question'], r['Full Prompt']
    cont = fp[len(q):] if fp.startswith(q) else ''
    i = cont.rfind(CUE)
    return cont[:i] if i != -1 else cont

datasets = ['asdiv', 'awps', 'gsm8k', 'svamp']

def run(label, rowfilter, filepat="*position-calc_*_annotated.csv"):
    print(f"--- filter: {label} ---")
    tot = collections.Counter()
    for ds in datasets:
        rows = []
        for f in sorted(glob.glob(f"{REPO}/results/{ds}/{filepat}")):
            rows += [r for r in csv.DictReader(open(f)) if rowfilter(r)]
        sub = [r for r in rows if r['Correct?']=='True' and r['Recovery Behavior'].strip()=='complete hallucination']
        a = 0
        for r in sub:
            tgt = parse_target(r['Target Answer'])
            if not (tgt is not None and tgt in numbers_in(cont_of(r))): a += 1
        n = len(sub)
        print(f"{ds}: N={n} absent={a} ({a/n*100 if n else 0:.0f}%)")
        tot['n'] += n; tot['a'] += a
    print(f"pooled: {tot['a']}/{tot['n']}")

run("ErrorType==calculation (all models)", lambda r: r['Error Type'].strip()=='calculation')
run("ErrorType==calculation AND gpt-4-0314", lambda r: r['Error Type'].strip()=='calculation' and r['Model Name']=='gpt-4-0314')
run("gpt-4-0314 only (any Error Type)", lambda r: r['Model Name']=='gpt-4-0314')

# Now: among target-ABSENT cases (all models, position-calc), how often does the
# continuation's final stated value equal the ERRONEOUS value at the end of the prefix?
print("--- perturbed-value-carried-forward check among ABSENT cases ---")
carried = tot_abs = 0
for ds in datasets:
    for f in sorted(glob.glob(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv")):
        for r in csv.DictReader(open(f)):
            if r['Correct?']!='True' or r['Recovery Behavior'].strip()!='complete hallucination': continue
            tgt = parse_target(r['Target Answer'])
            cont = cont_of(r)
            if tgt is not None and tgt in numbers_in(cont): continue
            tot_abs += 1
            # erroneous value = last number in Question prefix
            mq = re.findall(r'(\d[\d,]*\.?\d*)', r['Question'])
            if not mq: continue
            try: errval = float(mq[-1].replace(',', '').rstrip('.'))
            except ValueError: continue
            if errval in numbers_in(cont): carried += 1
print(f"absent cases: {tot_abs}, erroneous prefix value present in continuation: {carried} ({carried/tot_abs*100:.0f}%)")
