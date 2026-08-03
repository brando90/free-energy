#!/usr/bin/env python3
"""verify1: adversarial re-derivation of the extraction-override asymmetry.

Independent parsing choices where possible; then stress-test the proxy:
for each wrong->RIGHT override, is the target present ANYWHERE in the
continuation (parser could be legitimately reading a stated answer) vs
ABSENT (extraction pass genuinely produced a new number = solver)?
Also dump a random sample of >=30 override rows for manual inspection.
"""
import csv, re, random, sys
csv.field_size_limit(10**9)

REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
MODEL = "gpt-4-0314"
SUFFIX = " Therefore, the answer (arabic numerals) is"
MARKER = "A: Let's think step by step."
DATASETS = {"gsm8k": "test", "asdiv": "ASDiv", "awps": "MultiArith", "svamp": "SVAMP"}
PERTS = ["random", "add1", "add101"]
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def norm(s): return s.replace("\r\n", "\n").replace("\r", "\n")

def nums(t):
    out = []
    for m in NUM_RE.finditer(t):
        try: out.append(float(m.group(0).replace(",", "")))
        except ValueError: pass
    return out

def close(a, b): return abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))

def load(ds, stem, pert):
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

def continuation(q, r):
    fp = norm(r["Full Prompt"])
    cont = fp[len(q):]
    if cont.endswith(SUFFIX): cont = cont[:-len(SUFFIX)]
    return cont.strip()

overrides = []   # collected wrong->RIGHT rows for inspection
grand = {}
for ds, stem in DATASETS.items():
    n = agree = w2r = r2w = other = 0
    w2r_target_in_cont = w2r_target_absent = 0
    r2w_rows = []
    for pert in PERTS:
        for q, r in load(ds, stem, pert).items():
            if (r.get("Error Type") or "").strip() != "calculation": continue
            if (r.get("Correct?") or "").strip() not in ("True", "False"): continue
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            cont = continuation(q, r)
            cnums = nums(cont)
            anums = nums(r.get("Answer") or "")
            if not cnums or not anums: continue
            n += 1
            ext = anums[0]; last = cnums[-1]
            if close(ext, last): agree += 1
            elif close(ext, target) and not close(last, target):
                w2r += 1
                present = any(close(x, target) for x in cnums)
                if present: w2r_target_in_cont += 1
                else: w2r_target_absent += 1
                overrides.append((ds, pert, target, last, ext, present, cont,
                                  (r.get("Answer") or "").strip(),
                                  (r.get("Recovery Behavior") or "").strip()))
            elif close(last, target) and not close(ext, target):
                r2w += 1; r2w_rows.append((ds, pert, target, last, ext, cont[:200]))
            else: other += 1
    grand[ds] = (n, agree, w2r, r2w, other, w2r_target_in_cont, w2r_target_absent)
    print(f"{ds}: n={n} agree={agree} ({100*agree/n:.1f}%) "
          f"wrong->RIGHT={w2r} ({100*w2r/n:.1f}%) right->WRONG={r2w} ({100*r2w/n:.1f}%) other={other}")
    print(f"    of wrong->RIGHT: target ALSO appears in continuation text: {w2r_target_in_cont} "
          f"({100*w2r_target_in_cont/max(1,w2r):.0f}%), target ABSENT from continuation: {w2r_target_absent} "
          f"({100*w2r_target_absent/max(1,w2r):.0f}% = {100*w2r_target_absent/n:.1f}pp of rows)")
    for row in r2w_rows[:2]:
        print(f"    r2w example: {row[0]}/{row[1]} target={row[2]} last={row[3]} ext={row[4]}")

print()
tot_n = sum(v[0] for v in grand.values())
tot_w2r = sum(v[2] for v in grand.values())
tot_r2w = sum(v[3] for v in grand.values())
tot_absent = sum(v[6] for v in grand.values())
print(f"POOLED: n={tot_n} wrong->RIGHT={tot_w2r} ({100*tot_w2r/tot_n:.1f}%) "
      f"right->WRONG={tot_r2w} ({100*tot_r2w/tot_n:.1f}%) ratio={tot_w2r/max(1,tot_r2w):.0f}x")
print(f"POOLED wrong->RIGHT with target ABSENT from continuation (pure solver evidence): "
      f"{tot_absent} ({100*tot_absent/tot_n:.1f}pp of rows)")

print("\n" + "="*100)
print("MANUAL-INSPECTION SAMPLE: 34 random wrong->RIGHT overrides (seed 7)")
print("="*100)
random.seed(7)
for ds, pert, target, last, ext, present, cont, ans, rb in random.sample(overrides, min(34, len(overrides))):
    tail = cont[-260:].replace("\n", " | ")
    print(f"--- {ds}/{pert} target={target} cont_last={last} ext={ext} target_in_cont={present} rb={rb!r}")
    print(f"    CONT TAIL: ...{tail!r}")
    print(f"    ANSWER   : {ans!r}")
