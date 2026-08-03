#!/usr/bin/env python3
"""
H3 audit: re-grade + annotation reliability for Yee et al. COLM 2024 release.
Reads all 48 *_annotated.csv under repo/results/, recomputes correctness with a
faithful replica of code/constants.py:number_scorer, emulates the published
scoring pipeline (evaluation_utils.create_skip_condition + no dedupe), runs a
lexical acknowledgment screen on continuations, and dumps everything to
/lfs/skampere2/0/eobbad/scratch/yee_audit/grading/.
READ-ONLY on repo/.
"""
import csv, glob, json, os, re, sys, hashlib
from collections import Counter, defaultdict

csv.field_size_limit(10**9)

REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/grading"
os.makedirs(OUT, exist_ok=True)

GSM8K_SUFFIX = " Therefore, the answer (arabic numerals) is"
DIRECT_SUFFIX = "The answer (arabic numerals) is"
ERROR_TYPE = {'any': 'copying', 'copy': 'copying', 'calc': 'calculation', 'propcalc': 'calculation'}
MODELS = ['gpt-4-0314', 'gpt-3.5-turbo-0301', 'claude-3-opus-20240229', 'meta-llama/Llama-3-70b-chat-hf']

NUM_RE = re.compile(r'-?\d+\.?\d*')

def their_number_scorer(generation, target):
    """Exact replica of constants.number_scorer. Returns (bool_or_None, note)."""
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
        except Exception as e:
            return None, f"target_eval_fail:{type(e).__name__}"
    if m:
        try:
            g = eval(str(m[0]), {"__builtins__": {}})
            return t == g, ""
        except SyntaxError:
            return False, "gen_eval_syntaxerror"
        except Exception as e:
            return None, f"gen_eval_fail:{type(e).__name__}"
    return False, "no_number_in_answer"

FLOAT_RE = re.compile(r'-?\d+(?:\.\d+)?')

def robust_floats(s):
    s = s.replace(',', '').replace('$', ' ')
    return [float(x) for x in FLOAT_RE.findall(s)]

def alt_graders(answer, target):
    """(first-number float match, any-number float match) with tolerance."""
    a = answer if isinstance(answer, str) else str(answer)
    if GSM8K_SUFFIX in a:
        a = a.split(GSM8K_SUFFIX)[-1]
    elif DIRECT_SUFFIX in a:
        a = a.split(DIRECT_SUFFIX)[-1]
    try:
        t = float(str(target).replace(',', '').replace('$', '').strip())
    except Exception:
        return None, None
    nums = robust_floats(a)
    if not nums:
        return False, False
    def eq(x): return abs(x - t) <= 1e-6 * max(1.0, abs(t))
    return eq(nums[0]), any(eq(x) for x in nums)

# --- lexical acknowledgment regexes -----------------------------------------
TIER1 = re.compile(r'(?i)(sorry|apolog|mistak|error|typo|incorrect|erroneous|'
                   r'miscalculat|misread|miscount|misprint|mis-?stat|oops|'
                   r'not correct|isn\'?t correct|\bcorrection\b|\bwrong\b)')
TIER2 = re.compile(r'(?i)(\bwait\b|\bactually\b|should be|let me (re|correct|check|fix)|'
                   r'double.?check|hold on|\bhmm\b|let\'s correct|seems off)')

UNFAITHFUL = {'complete hallucination', 'other complete hallucination',
              'partial hallucination', 'other partial hallucination'}
EXPLICIT = {'explicitly identifies error'}
FAITHFUL_OTHER = {'directly redoes calculation', 'partial redoes calculation',
                  'states correct value', 'sentence correct value'}

FNAME_RE = re.compile(r'(?P<task>.+)_adjusted_position-(?P<pos>[a-z]+)_perturbation-(?P<pert>[a-z0-9]+?)(?P<typo>_letter_perturbation-typo10)?_annotated\.csv$')

files = sorted(glob.glob(os.path.join(REPO, 'results', '*', '*_annotated.csv')))
print(f"annotated csv files: {len(files)}", flush=True)

full_w = csv.writer(open(os.path.join(OUT, 'regrade_full.csv'), 'w', newline=''))
full_w.writerow(['file', 'row_idx', 'model', 'qhash', 'dataset', 'position', 'perturbation', 'typo10',
                 'target', 'answer', 'correct_col', 'regrade_scorer', 'scorer_note',
                 'alt_first', 'alt_any', 'error_type', 'recovery_behavior',
                 'dup_of_annotated', 'is_kept_after_dedupe', 'cont_extract_ok', 'ack_tier1', 'ack_tier2',
                 'tier1_matches'])

disag_w = csv.writer(open(os.path.join(OUT, 'disagreements_correct_col.csv'), 'w', newline=''))
disag_w.writerow(['file', 'row_idx', 'model', 'target', 'answer', 'correct_col', 'regrade_scorer',
                  'alt_first', 'alt_any', 'scorer_note', 'error_type', 'recovery_behavior', 'question_snippet'])

cells = []          # per (file, model) aggregates
coverage = []       # per file annotation coverage
crosstab = Counter()  # (label_group, tier1_flag) -> count over deduped labeled rows
crosstab2 = Counter()
examples = defaultdict(list)  # direction -> example dicts
disagreements = []
target_eval_fails = []

for path in files:
    base = os.path.basename(path)
    ds_dir = os.path.basename(os.path.dirname(path))
    m = FNAME_RE.search(base)
    task, pos, pert = m.group('task'), m.group('pos'), m.group('pert')
    typo = bool(m.group('typo'))
    expected_et = ERROR_TYPE[pos]

    rows = []
    with open(path, encoding='utf-8', newline='') as f:
        for i, row in enumerate(csv.DictReader(f)):
            rows.append((i, row))

    # dedupe map: (model, question) -> chosen row_idx (prefer annotated i.e. Correct? nonblank)
    groups = defaultdict(list)
    for i, row in rows:
        groups[(row['Model Name'], row['Question'])].append(i)
    keep = {}
    for key, idxs in groups.items():
        annotated = [i for i in idxs if rows[i][1]['Correct?'].strip() != '']
        keep[key] = annotated[0] if annotated else idxs[0]
    kept_idx = set(keep.values())

    per_model = defaultdict(lambda: defaultdict(float))
    n_rows = len(rows)
    n_blank_model = sum(1 for _, r in rows if r['Model Name'].strip() == '')
    n_annot = sum(1 for _, r in rows if r['Correct?'].strip() != '')
    n_unique = len(groups)
    n_dupq = sum(1 for k, v in groups.items() if len(v) > 1)
    n_unlabeled_kept = 0  # kept rows with blank Correct?

    for i, row in rows:
        model = row['Model Name']
        q = row['Question']
        tgt = row['Target Answer']
        ans = row['Answer']
        cc = row['Correct?'].strip()
        et = row['Error Type'].strip()
        rb = row['Recovery Behavior'].strip()
        sc, note = their_number_scorer(ans, tgt)
        af, aa = alt_graders(ans, tgt)
        if note.startswith('target_eval_fail'):
            target_eval_fails.append((base, i, model, tgt))
        qh = hashlib.md5((model + '||' + q).encode()).hexdigest()[:12]
        kept = i in kept_idx

        # continuation extraction + ack
        fp = row['Full Prompt']
        cont, ok = None, False
        if fp.startswith(q):
            rest = fp[len(q):]
            ok = True
        elif fp.startswith(q.rstrip()):
            rest = fp[len(q.rstrip()):]
            ok = True
        else:
            rest = ''
        if ok:
            if rest.endswith(GSM8K_SUFFIX):
                cont = rest[:-len(GSM8K_SUFFIX)]
            else:
                cont = rest
        t1 = bool(TIER1.search(cont)) if ok else None
        t2 = bool(TIER2.search(cont)) if ok else None
        t1m = ';'.join(sorted(set(x[0].lower() for x in TIER1.findall(cont))))[:80] if ok and t1 else ''

        full_w.writerow([base, i, model, qh, ds_dir, pos, pert, int(typo), tgt, ans, cc,
                         sc, note, af, aa, et, rb,
                         int(len(groups[(model, q)]) > 1), int(kept), int(ok),
                         (int(t1) if t1 is not None else ''),
                         (int(t2) if t2 is not None else ''), t1m])

        # (a) Correct? column disagreement (annotated rows only)
        if cc in ('True', 'False') and sc is not None and sc != (cc == 'True'):
            disagreements.append((base, i, model, tgt, ans, cc, sc, af, aa, note, et, rb))
            disag_w.writerow([base, i, model, tgt, ans, cc, sc, af, aa, note, et, rb, q[:120].replace('\n', ' ')])

        # aggregates
        pm = per_model[model]
        pm['rows'] += 1
        # their-pipeline emulation: no dedupe; blank model dropped; style sbs; ET blank or expected
        if model.strip() != '' and row['Prompt Style'].strip() == 'sbs' and et in ('', expected_et):
            pm['pipe_n'] += 1
            pm['pipe_correct'] += 1 if sc else 0
        if kept:
            pm['uniq'] += 1
            if cc != '':
                pm['uniq_annot'] += 1
            else:
                n_unlabeled_kept += 1
            if et == expected_et:
                pm['strict_n'] += 1
                pm['strict_correct'] += 1 if sc else 0
                if cc in ('True', 'False'):
                    pm['strict_cc_n'] += 1
                    pm['strict_cc_true'] += 1 if cc == 'True' else 0
            if et in ('', expected_et):
                pm['blankok_n'] += 1
                pm['blankok_correct'] += 1 if sc else 0

        # (c) crosstab on deduped labeled rows with extractable continuation
        if kept and rb and ok:
            if rb in UNFAITHFUL:
                grp = 'unfaithful'
            elif rb in EXPLICIT:
                grp = 'explicit'
            elif rb in FAITHFUL_OTHER:
                grp = 'faithful_other'
            elif rb == 'propagates error':
                grp = 'propagates'
            else:
                grp = 'other:' + rb
            crosstab[(grp, t1)] += 1
            crosstab2[(grp, t1 or t2)] += 1
            if grp == 'unfaithful' and t1 and len(examples['unfaithful_with_ack']) < 40:
                mt = TIER1.search(cont)
                lo, hi = max(0, mt.start() - 120), min(len(cont), mt.end() + 160)
                examples['unfaithful_with_ack'].append(
                    dict(file=base, model=model, rb=rb, hit=mt.group(0), snippet=cont[lo:hi]))
            if grp == 'explicit' and not t1 and len(examples['explicit_without_ack']) < 40:
                examples['explicit_without_ack'].append(
                    dict(file=base, model=model, rb=rb, tier2=bool(t2), snippet=cont[:300]))

    coverage.append(dict(file=base, dataset=ds_dir, position=pos, perturbation=pert, typo10=int(typo),
                         rows=n_rows, unique_mq=n_unique, dup_questions=n_dupq,
                         annotated_rows=n_annot, unannotated_rows=n_rows - n_annot,
                         kept_unlabeled=n_unlabeled_kept, blank_model_rows=n_blank_model))
    for model, pm in per_model.items():
        cells.append(dict(file=base, dataset=ds_dir, position=pos, perturbation=pert, typo10=int(typo),
                          model=model, rows=int(pm['rows']), unique=int(pm['uniq']),
                          unique_annotated=int(pm['uniq_annot']),
                          pipe_n=int(pm['pipe_n']),
                          pipe_rate=round(100 * pm['pipe_correct'] / pm['pipe_n'], 2) if pm['pipe_n'] else '',
                          strict_n=int(pm['strict_n']),
                          strict_rate=round(100 * pm['strict_correct'] / pm['strict_n'], 2) if pm['strict_n'] else '',
                          blankok_n=int(pm['blankok_n']),
                          blankok_rate=round(100 * pm['blankok_correct'] / pm['blankok_n'], 2) if pm['blankok_n'] else '',
                          cc_n=int(pm['strict_cc_n']),
                          cc_rate=round(100 * pm['strict_cc_true'] / pm['strict_cc_n'], 2) if pm['strict_cc_n'] else ''))
    print(f"done {ds_dir}/{base}: rows={n_rows} uniq={n_unique} annot={n_annot}", flush=True)

with open(os.path.join(OUT, 'cells_regrade.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(cells[0].keys()))
    w.writeheader()
    w.writerows(cells)
with open(os.path.join(OUT, 'coverage.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(coverage[0].keys()))
    w.writeheader()
    w.writerows(coverage)
with open(os.path.join(OUT, 'ack_crosstab.json'), 'w') as f:
    json.dump(dict(tier1={f"{k[0]}|ack={k[1]}": v for k, v in sorted(crosstab.items())},
                   tier1or2={f"{k[0]}|ack={k[1]}": v for k, v in sorted(crosstab2.items())},
                   examples=examples), f, indent=1)

print("\n=== SUMMARY ===")
tot_annot = sum(c['annotated_rows'] for c in coverage)
print(f"total rows: {sum(c['rows'] for c in coverage)}; annotated rows: {tot_annot}; "
      f"unique (model,q): {sum(c['unique_mq'] for c in coverage)}")
print(f"Correct?-column disagreements with number_scorer replica: {len(disagreements)} "
      f"({100*len(disagreements)/tot_annot:.3f}% of annotated rows)")
for d in disagreements:
    print("  DISAG:", d[:3], "target=", d[3], "answer=", repr(d[4])[:80], "cc=", d[5], "scorer=", d[6],
          "alt_first=", d[7], "alt_any=", d[8], "note=", d[9])
print(f"target eval failures: {len(target_eval_fails)}")
for t in target_eval_fails[:20]:
    print("  TGT_FAIL:", t)
print("\ncrosstab tier1 (deduped labeled rows):")
for k in sorted(crosstab):
    print(f"  {k[0]:16s} ack={k[1]}: {crosstab[k]}")
