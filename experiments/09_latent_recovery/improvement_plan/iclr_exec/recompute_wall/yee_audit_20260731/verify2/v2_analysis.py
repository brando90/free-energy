#!/usr/bin/env python3
"""verify2: independent re-derivation of the extraction-contamination finding.

For rows annotated Correct?=True + Recovery Behavior='complete hallucination'
in position-calc files, check whether the TARGET answer appears anywhere in the
model's CONTINUATION (Full Prompt minus Question prefix minus extraction cue).
If absent, the 'recovery' happened in the extraction (2nd) call, not the CoT.
"""
import csv, glob, re, collections, sys, random

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

def continuation(row):
    fp, q = row['Full Prompt'], row['Question']
    cont = fp[len(q):] if fp.startswith(q) else None
    if cont is None:
        # fallback: try whitespace-normalized prefix match
        return None
    # strip trailing extraction cue
    idx = cont.rfind(CUE)
    if idx != -1:
        cont = cont[:idx]
    return cont

def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        for row in csv.DictReader(open(f)):
            row['_file'] = f
            rows.append(row)
    return rows

datasets = ['asdiv', 'awps', 'gsm8k', 'svamp']
print("=== A. Reproduce subset: position-calc files, Correct?=True, RB='complete hallucination' ===")
grand = collections.Counter()
examples = []
for ds in datasets:
    rows = load(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv")
    sub = [r for r in rows if r['Correct?'] == 'True'
           and r['Recovery Behavior'].strip() == 'complete hallucination']
    n_no_prefix = 0
    absent = present = 0
    carried = 0  # perturbed/errored value appears in continuation conclusion? (approx: target absent AND answer stated != target)
    for r in sub:
        cont = continuation(r)
        if cont is None:
            n_no_prefix += 1
            continue
        tgt = parse_target(r['Target Answer'])
        nums = numbers_in(cont)
        # also string-level check to be generous
        tgt_str_variants = set()
        if tgt is not None:
            tgt_str_variants = {('%g' % tgt), ('%.1f' % tgt)}
        str_hit = any(v in cont for v in tgt_str_variants)
        if tgt is not None and (tgt in nums or str_hit):
            present += 1
        else:
            absent += 1
            examples.append((ds, r))
    n = len(sub)
    print(f"{ds}: N={n} target_ABSENT_in_continuation={absent} ({(absent/n*100 if n else 0):.1f}%) "
          f"present={present} prefix_mismatch={n_no_prefix}")
    grand['N'] += n; grand['absent'] += absent; grand['present'] += present; grand['nopfx'] += n_no_prefix
print(f"POOLED: N={grand['N']} absent={grand['absent']} ({grand['absent']/max(grand['N'],1)*100:.1f}%)")

print()
print("=== B. Model-name breakdown of subset (is it all gpt-4-0314?) ===")
for ds in datasets:
    rows = load(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv")
    sub = [r for r in rows if r['Correct?'] == 'True' and r['Recovery Behavior'].strip() == 'complete hallucination']
    print(ds, collections.Counter(r['Model Name'] for r in sub).most_common())

print()
print("=== C. Sanity: Correct?=True consistency — extracted Answer vs Target ===")
mismatch = 0; tot = 0
for ds in datasets:
    for r in load(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv"):
        if r['Correct?'] != 'True': continue
        tot += 1
        tgt = parse_target(r['Target Answer'])
        ans_nums = numbers_in(r['Answer'])
        if tgt is None or tgt not in ans_nums: mismatch += 1
print(f"Correct?=True rows: {tot}, extracted-answer-vs-target mismatch: {mismatch}")

print()
print("=== D. Faithful categories cleanliness (target present in continuation) ===")
for cat in ['states correct value', 'directly redoes calculation', 'explicitly identifies error']:
    pres = tot2 = 0
    for ds in datasets:
        for r in load(f"{REPO}/results/{ds}/*position-calc_*_annotated.csv"):
            if r['Correct?'] != 'True' or r['Recovery Behavior'].strip() != cat: continue
            cont = continuation(r)
            if cont is None: continue
            tgt = parse_target(r['Target Answer'])
            if tgt is not None and tgt in numbers_in(cont): pres += 1
            tot2 += 1
    print(f"{cat}: target present {pres}/{tot2}")

print()
print("=== E. Random sample of 12 target-ABSENT cases for manual inspection ===")
random.seed(7)
for ds, r in random.sample(examples, min(12, len(examples))):
    cont = continuation(r)
    print("-----", ds, "| target:", r['Target Answer'], "| extracted Answer:", repr(r['Answer'][:40]))
    print("Q-prefix tail:", repr(r['Question'][-120:]))
    print("CONTINUATION:", repr(cont[:400]))
