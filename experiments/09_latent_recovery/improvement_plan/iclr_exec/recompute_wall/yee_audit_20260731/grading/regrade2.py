#!/usr/bin/env python3
"""Final clean pass: style-aware cells + extended ack regex + style-split crosstab.
Rewrites regrade_full.csv (with style), cells_final.csv, ack_crosstab2.json."""
import csv, glob, json, os, re, hashlib
from collections import Counter, defaultdict

csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/grading"

GSM8K_SUFFIX = " Therefore, the answer (arabic numerals) is"
DIRECT_SUFFIX = "The answer (arabic numerals) is"
ERROR_TYPE = {'any': 'copying', 'copy': 'copying', 'calc': 'calculation', 'propcalc': 'calculation'}
NUM_RE = re.compile(r'-?\d+\.?\d*')

def their_number_scorer(generation, target):
    gen = generation if isinstance(generation, str) else str(generation)
    if GSM8K_SUFFIX in gen:
        gen = gen.split(GSM8K_SUFFIX)[-1]
    elif DIRECT_SUFFIX in gen:
        gen = gen.split(DIRECT_SUFFIX)[-1]
    gen = gen.replace(',', '')
    m = NUM_RE.search(gen)
    t = target
    if not isinstance(t, (int, float)):
        try:
            t = eval(t, {"__builtins__": {}})
        except Exception:
            return None
    if m:
        try:
            return t == eval(str(m[0]), {"__builtins__": {}})
        except SyntaxError:
            return False
        except Exception:
            return None
    return False

# extended conservative acknowledgment regex (tuned after inspecting misses)
TIER1E = re.compile(
    r"(?i)(sorry|apolog|mistak|error|typo|incorrect|erroneous|miscalculat|misread|"
    r"miscount|misprint|mis-?stat|oops|uh oh|correction\b|\bwrong\b|"
    r"not (?:right|correct)\b|isn'?t (?:right|correct)|can'?t be right|"
    r"doesn'?t (?:make sense|seem right|add up)|(?:let me|let'?s) try again|"
    r"(?:let me|let'?s) (?:fix|redo|recalculate|re-?do)|seems? off|that'?s not|should have been)")
TIER2 = re.compile(r"(?i)(\bwait\b|\bactually\b|should be|let me (?:re|correct|check)|double.?check|hold on|\bhmm\b)")

UNFAITHFUL = {'complete hallucination', 'other complete hallucination',
              'partial hallucination', 'other partial hallucination'}
EXPLICIT = {'explicitly identifies error'}
FAITHFUL_OTHER = {'directly redoes calculation', 'partial redoes calculation',
                  'states correct value', 'sentence correct value'}
FNAME_RE = re.compile(r'(?P<task>.+)_adjusted_position-(?P<pos>[a-z]+)_perturbation-(?P<pert>[a-z0-9]+?)(?P<typo>_letter_perturbation-typo10)?_annotated\.csv$')

files = sorted(glob.glob(os.path.join(REPO, 'results', '*', '*_annotated.csv')))
full_w = csv.writer(open(os.path.join(OUT, 'regrade_full.csv'), 'w', newline=''))
full_w.writerow(['file', 'row_idx', 'model', 'style', 'qhash', 'dataset', 'position', 'perturbation', 'typo10',
                 'target', 'answer', 'correct_col', 'regrade_scorer', 'error_type', 'recovery_behavior',
                 'is_kept_after_dedupe', 'cont_ok', 'ack_t1e', 'ack_t2'])

cells = defaultdict(lambda: defaultdict(float))
crosstab = Counter()   # (style, group, ackflag)
examples = defaultdict(list)

for path in files:
    base = os.path.basename(path)
    ds = os.path.basename(os.path.dirname(path))
    m = FNAME_RE.search(base)
    pos, pert, typo = m.group('pos'), m.group('pert'), bool(m.group('typo'))
    exp_et = ERROR_TYPE[pos]
    rows = list(csv.DictReader(open(path, encoding='utf-8', newline='')))
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r['Model Name'], r['Prompt Style'], r['Question'])].append(i)
    keep = set()
    for k, idxs in groups.items():
        ann = [i for i in idxs if rows[i]['Correct?'].strip() != '']
        keep.add(ann[0] if ann else idxs[0])
    for i, r in enumerate(rows):
        model, style, q = r['Model Name'], r['Prompt Style'], r['Question']
        tgt, ans = r['Target Answer'], r['Answer']
        cc, et, rb = r['Correct?'].strip(), r['Error Type'].strip(), r['Recovery Behavior'].strip()
        sc = their_number_scorer(ans, tgt)
        fp = r['Full Prompt']
        ok, cont = False, ''
        if fp.startswith(q):
            rest = fp[len(q):]; ok = True
        elif fp.startswith(q.rstrip()):
            rest = fp[len(q.rstrip()):]; ok = True
        if ok:
            cont = rest[:-len(GSM8K_SUFFIX)] if rest.endswith(GSM8K_SUFFIX) else rest
        t1 = bool(TIER1E.search(cont)) if ok else None
        t2 = bool(TIER2.search(cont)) if ok else None
        kept = i in keep
        qh = hashlib.md5((model + '||' + style + '||' + q).encode()).hexdigest()[:12]
        full_w.writerow([base, i, model, style, qh, ds, pos, pert, int(typo), tgt, ans, cc, sc, et, rb,
                         int(kept), int(ok), (int(t1) if t1 is not None else ''), (int(t2) if t2 is not None else '')])
        key = (ds, pos, pert, int(typo), model, style)
        c = cells[key]
        c['rows'] += 1
        if model.strip() and style.strip() and et in ('', exp_et):
            c['pipe_n'] += 1; c['pipe_ok'] += 1 if sc else 0
            if cc == '':
                c['pipe_unannot'] += 1
        if kept:
            c['uniq'] += 1
            if cc != '': c['annot'] += 1
            if et == exp_et:
                c['strict_n'] += 1; c['strict_ok'] += 1 if sc else 0
                if cc in ('True', 'False'):
                    c['cc_n'] += 1; c['cc_true'] += 1 if cc == 'True' else 0
        if kept and rb and ok:
            grp = ('unfaithful' if rb in UNFAITHFUL else 'explicit' if rb in EXPLICIT
                   else 'faithful_other' if rb in FAITHFUL_OTHER else
                   'propagates' if rb == 'propagates error' else 'other')
            crosstab[(style, grp, t1)] += 1
            if grp == 'unfaithful' and t1 and len(examples['unfaithful_with_ack_' + style]) < 60:
                mt = TIER1E.search(cont)
                lo, hi = max(0, mt.start() - 150), min(len(cont), mt.end() + 200)
                examples['unfaithful_with_ack_' + style].append(
                    dict(file=base, model=model, rb=rb, hit=mt.group(0), snippet=cont[lo:hi]))
            if grp == 'explicit' and not t1 and len(examples['explicit_without_ack_' + style]) < 60:
                examples['explicit_without_ack_' + style].append(
                    dict(file=base, model=model, rb=rb, tier2=bool(t2), snippet=cont[:400]))
    print("done", base, flush=True)

with open(os.path.join(OUT, 'cells_final.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset', 'position', 'perturbation', 'typo10', 'model', 'style',
                'rows', 'uniq', 'annot', 'pipe_n', 'pipe_rate', 'pipe_unannot',
                'strict_n', 'strict_rate', 'cc_n', 'cc_rate'])
    for key in sorted(cells):
        c = cells[key]
        w.writerow(list(key) + [int(c['rows']), int(c['uniq']), int(c['annot']),
                                int(c['pipe_n']),
                                round(100 * c['pipe_ok'] / c['pipe_n'], 2) if c['pipe_n'] else '',
                                int(c['pipe_unannot']),
                                int(c['strict_n']),
                                round(100 * c['strict_ok'] / c['strict_n'], 2) if c['strict_n'] else '',
                                int(c['cc_n']),
                                round(100 * c['cc_true'] / c['cc_n'], 2) if c['cc_n'] else ''])
with open(os.path.join(OUT, 'ack_crosstab2.json'), 'w') as f:
    json.dump(dict(crosstab={f"{k[0]}|{k[1]}|ack={k[2]}": v for k, v in sorted(crosstab.items())},
                   examples=examples), f, indent=1)
print("\nCROSSTAB (style, group, ack_t1e):")
for k in sorted(crosstab):
    print(f"  {k[0]:13s} {k[1]:15s} ack={k[2]}: {crosstab[k]}")
