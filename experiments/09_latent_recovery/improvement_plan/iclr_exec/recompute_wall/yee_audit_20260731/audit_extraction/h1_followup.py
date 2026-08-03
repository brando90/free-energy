#!/usr/bin/env python3
"""Follow-up: decompose extraction-only recoveries, strict channel test,
JSON spot-verify, awps duplication probe, scorer disagreements, extraction-answer length."""
import csv, json, os, re
csv.field_size_limit(10**9)

REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/audit_extraction"
MODEL = "gpt-4-0314"
SUFFIX = " Therefore, the answer (arabic numerals) is"
MARKER = "A: Let's think step by step."
DATASETS = {"gsm8k": "test", "asdiv": "ASDiv", "awps": "MultiArith", "svamp": "SVAMP"}
PERTS = ["random", "add1", "add101"]
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def norm(s): return s.replace("\r\n", "\n").replace("\r", "\n")

def parse_nums(t):
    out = []
    for m in NUM_RE.finditer(t):
        try: out.append(float(m.group(0).replace(",", "")))
        except ValueError: pass
    return out

def close(a, b): return abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))

def load_rows(ds, stem, pert):
    path = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("Model Name") == MODEL and r.get("Prompt Style") == "sbs"]
    byq = {}
    for r in rows:
        q = norm(r["Question"])
        cur = byq.get(q)
        if cur is None or ((r.get("Correct?") or "").strip() and not (cur.get("Correct?") or "").strip()):
            byq[q] = r
    return byq

def dissect(q, r):
    fp = norm(r["Full Prompt"])
    mi = q.find(MARKER)
    problem = q[3:mi].strip()
    prefix = q[mi + len(MARKER):]
    cont = fp[len(q):]
    if cont.endswith(SUFFIX): cont = cont[:-len(SUFFIX)]
    cont_body = cont[1:] if cont.startswith(" ") else cont
    return problem, prefix, cont_body

def last_num_str(text):
    ms = list(NUM_RE.finditer(text))
    return ms[-1].group(0) if ms else None

print("="*100)
print("A) SPOT-VERIFY prefix reconstruction vs stimuli JSON (3 rows per dataset, add1)")
print("="*100)
for ds, stem in DATASETS.items():
    jpath = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-add1_gpt-4-0314.json"
    with open(jpath) as f: stim = json.load(f)
    byq = load_rows(ds, stem, "add1")
    checked = matched = 0
    for q, r in byq.items():
        if checked >= 3: break
        problem, prefix, cont = dissect(q, r)
        if problem in stim:
            checked += 1
            jpref = stim[problem][0]
            ok = prefix.strip() == jpref.strip()
            matched += ok
            if not ok:
                print(f"  MISMATCH {ds}: csv_prefix_tail={prefix.strip()[-60:]!r} json_tail={jpref.strip()[-60:]!r}")
    print(f"  {ds}: {matched}/{checked} exact prefix matches vs JSON")
print()

print("="*100)
print("B) JSON key counts vs unique CSV questions/problems (duplication probe)")
print("="*100)
for ds, stem in DATASETS.items():
    for pert in PERTS:
        jpath = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_gpt-4-0314.json"
        nj = len(json.load(open(jpath))) if os.path.exists(jpath) else -1
        byq = load_rows(ds, stem, pert)
        probs = {}
        for q in byq:
            mi = q.find(MARKER)
            p = q[3:mi].strip()
            probs.setdefault(p, []).append(q)
        multi = {p: qs for p, qs in probs.items() if len(qs) > 1}
        n_ann = sum(1 for r in byq.values() if (r.get("Correct?") or "").strip())
        print(f"  {ds}/{pert}: json_keys={nj} uniq_questions={len(byq)} uniq_problems={len(probs)} "
              f"problems_with_multiple_prefixes={len(multi)} annotated_rows={n_ann}")
        if multi and pert == "add1":
            p, qs = next(iter(multi.items()))
            for q in qs[:2]:
                _, pref, _ = dissect(q, byq[q])
                ann = (byq[q].get("Correct?") or "").strip()
                print(f"      ex prefix tail={pref.strip()[-70:]!r} annotated={ann!r}")
print()

print("="*100)
print("C) MAIN TABLE with strict + generous channel tests and artifact decomposition")
print("="*100)
rows_out = []
hdr = ("ds/pert", "n", "recov", "rate", "strictOK", "strict_rate", "genOK", "gen_rate",
       "xo_total", "xo_nonum", "xo_splice", "xo_errfwd", "xo_other")
print(("{:<14}"+"{:>8}"*12).format(*hdr))
pooled = {"n":0,"recov":0,"strict":0,"gen":0,"xo":0,"xo_nonum":0,"xo_splice":0,"xo_errfwd":0,"xo_other":0}
pooled_r1 = dict(pooled)  # random+add1 only
examples_splice = []
xo_answer_long = 0; xo_answer_eq = 0; xo_n = 0
for ds, stem in DATASETS.items():
    for pert in PERTS:
        byq = load_rows(ds, stem, pert)
        n = recov = strictok = genok = xo = xo_nonum = xo_splice = xo_errfwd = xo_other = 0
        for q, r in byq.items():
            et = (r.get("Error Type") or "").strip()
            cor = (r.get("Correct?") or "").strip()
            if et != "calculation" or cor not in ("True", "False"): continue
            n += 1
            if cor != "True": continue
            recov += 1
            problem, prefix, cont = dissect(q, r)
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            cnums = parse_nums(cont)
            pstr = last_num_str(prefix)
            perturbed = float(pstr.replace(",", "")) if pstr else None
            has_t = any(close(x, target) for x in cnums)
            last_t = bool(cnums) and close(cnums[-1], target)
            if last_t: strictok += 1
            if has_t: genok += 1
            if not has_t:
                xo += 1; xo_n += 1
                ans = (r.get("Answer") or "").strip()
                if len(ans) > 20 or "=" in ans: xo_answer_long += 1
                # splice: perturbed number string + leading fragment of cont parses to target
                lead = re.match(r"[\d,\.]{1,12}", cont)
                spliced = False
                if pstr and lead:
                    cand = (pstr + lead.group(0)).rstrip(".,")
                    try: spliced = close(float(cand.replace(",", "")), target)
                    except ValueError: spliced = False
                if not cnums:
                    xo_nonum += 1
                elif spliced:
                    xo_splice += 1
                    if len(examples_splice) < 8:
                        examples_splice.append((ds, pert, target, pstr, cont[:70]))
                elif perturbed is not None and any(close(x, perturbed) for x in cnums):
                    xo_errfwd += 1
                else:
                    xo_other += 1
        def pc(a, b): return f"{100*a/b:.1f}" if b else "-"
        print(("{:<14}"+"{:>8}"*12).format(f"{ds}/{pert}", n, recov, pc(recov, n),
              strictok, pc(strictok, n), genok, pc(genok, n),
              xo, xo_nonum, xo_splice, xo_errfwd, xo_other))
        for k, v in zip(("n","recov","strict","gen","xo","xo_nonum","xo_splice","xo_errfwd","xo_other"),
                        (n,recov,strictok,genok,xo,xo_nonum,xo_splice,xo_errfwd,xo_other)):
            pooled[k] += v
            if pert in ("random", "add1"): pooled_r1[k] += v
print()
for name, P in (("ALL pooled", pooled), ("random+add1 pooled", pooled_r1)):
    print(f"  {name}: n={P['n']} recov={P['recov']} ({100*P['recov']/P['n']:.1f}%) "
          f"strict={P['strict']} ({100*P['strict']/P['n']:.1f}%) gen={P['gen']} ({100*P['gen']/P['n']:.1f}%) "
          f"extraction-only={P['xo']} ({100*P['xo']/P['n']:.1f}% of n; {100*P['xo']/P['recov']:.1f}% of recoveries) "
          f"[no-num={P['xo_nonum']} splice={P['xo_splice']} err-fwd={P['xo_errfwd']} other={P['xo_other']}]")
print()
print(f"  Among extraction-only recoveries: Answer field long/contains '=' (extraction call visibly "
      f"reasons/computes): {xo_answer_long}/{xo_n}")
print()
print("  SPLICE examples (perturbed str + cont lead -> target):")
for e in examples_splice:
    print(f"    {e[0]}/{e[1]} target={e[2]} prefix_num={e[3]!r} cont_head={e[4]!r}")
print()

print("="*100)
print("D) Correct? vs scorer disagreements (list all, calc rows)")
print("="*100)
for ds, stem in DATASETS.items():
    for pert in PERTS:
        byq = load_rows(ds, stem, pert)
        for q, r in byq.items():
            cor = (r.get("Correct?") or "").strip()
            if cor not in ("True", "False"): continue
            if (r.get("Error Type") or "").strip() != "calculation": continue
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            anums = parse_nums(r.get("Answer") or "")
            sc = bool(anums) and close(anums[0], target)
            if sc != (cor == "True"):
                print(f"  {ds}/{pert}: Correct?={cor} scorer={sc} target={target} "
                      f"Answer={ (r.get('Answer') or '')[:80]!r} rb={(r.get('Recovery Behavior') or '').strip()!r}")
