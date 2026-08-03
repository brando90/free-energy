#!/usr/bin/env python3
"""Final checks: awps dual-prefix diff, per-dataset hallucination x channel,
full err-fwd examples, extraction override rate in both directions."""
import csv, json, re
csv.field_size_limit(10**9)

REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
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

def load_rows(ds, stem, pert, dedupe=True):
    path = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("Model Name") == MODEL and r.get("Prompt Style") == "sbs"]
    if not dedupe: return rows
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
    return problem, prefix, (cont[1:] if cont.startswith(" ") else cont)

print("="*100)
print("1) AWPS add1: diff the two prefix variants of one problem")
print("="*100)
byq = load_rows("awps", "MultiArith", "add1")
probs = {}
for q in byq:
    mi = q.find(MARKER)
    probs.setdefault(q[3:mi].strip(), []).append(q)
p, qs = next((p, qs) for p, qs in probs.items() if len(qs) > 1)
a, b = sorted(qs, key=len)
_, pa, ca = dissect(a, byq[a])
_, pb, cb = dissect(b, byq[b])
i = 0
while i < min(len(pa), len(pb)) and pa[i] == pb[i]: i += 1
print(f"  problem: {p[:80]!r}")
print(f"  prefixA len={len(pa)} prefixB len={len(pb)} first divergence at char {i}")
print(f"  A around div: {pa[max(0,i-50):i+60]!r}")
print(f"  B around div: {pb[max(0,i-50):i+60]!r}")
ann_a = (byq[a].get("Correct?") or "").strip(); ann_b = (byq[b].get("Correct?") or "").strip()
et_a = (byq[a].get("Error Type") or "").strip(); et_b = (byq[b].get("Error Type") or "").strip()
print(f"  A: annotated={ann_a!r} error_type={et_a!r}  B: annotated={ann_b!r} error_type={et_b!r}")
# also compare against JSON stimulus
stim = json.load(open(f"{REPO}/results/awps/MultiArith_adjusted_position-calc_perturbation-add1_gpt-4-0314.json"))
if p in stim:
    jp = norm(stim[p][0]).strip()
    print(f"  JSON prefix matches A: {jp == pa.strip()}  matches B: {jp == pb.strip()}")
# count how often the two variants disagree on Correct?
n_pairs = n_disagree = 0
for pp, qq in probs.items():
    if len(qq) == 2:
        c1 = (byq[qq[0]].get("Correct?") or "").strip()
        c2 = (byq[qq[1]].get("Correct?") or "").strip()
        if c1 in ("True","False") and c2 in ("True","False"):
            n_pairs += 1
            if c1 != c2: n_disagree += 1
print(f"  pairs both annotated: {n_pairs}, Correct? disagrees between variants: {n_disagree}")
print()

print("="*100)
print("2) Per-dataset: 'complete hallucination' recoveries lacking target in continuation")
print("="*100)
for ds, stem in DATASETS.items():
    tot = no_t = errfwd = 0
    for pert in PERTS:
        for q, r in load_rows(ds, stem, pert).items():
            if (r.get("Error Type") or "").strip() != "calculation": continue
            if (r.get("Correct?") or "").strip() != "True": continue
            rb = (r.get("Recovery Behavior") or "").strip()
            if "complete hallucination" not in rb: continue
            tot += 1
            _, prefix, cont = dissect(q, r)
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            cnums = parse_nums(cont)
            if not any(close(x, target) for x in cnums):
                no_t += 1
                pms = list(NUM_RE.finditer(prefix))
                if pms:
                    pv = float(pms[-1].group(0).replace(",", ""))
                    if any(close(x, pv) for x in cnums): errfwd += 1
    print(f"  {ds}: complete-hallucination recoveries={tot}, target ABSENT from continuation={no_t} "
          f"({100*no_t/tot:.0f}%), of which perturbed value carried forward={errfwd}")
print()

print("="*100)
print("3) FULL examples: err-fwd extraction-only recoveries (continuation concludes wrong, extraction right)")
print("="*100)
shown = 0
for ds, stem in (("asdiv","ASDiv"), ("svamp","SVAMP")):
    for pert in ("random","add1"):
        for q, r in load_rows(ds, stem, pert).items():
            if shown >= 6: break
            if (r.get("Error Type") or "").strip() != "calculation": continue
            if (r.get("Correct?") or "").strip() != "True": continue
            _, prefix, cont = dissect(q, r)
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            cnums = parse_nums(cont)
            if any(close(x, target) for x in cnums): continue
            pms = list(NUM_RE.finditer(prefix))
            if not pms: continue
            pv = float(pms[-1].group(0).replace(",", ""))
            if not any(close(x, pv) for x in cnums): continue
            shown += 1
            print(f"--- {ds}/{pert} target={target} perturbed={pv} rb={(r.get('Recovery Behavior') or '').strip()!r}")
            print(f"    PREFIX TAIL : ...{prefix.strip()[-90:]!r}")
            print(f"    CONTINUATION: {cont!r}")
            print(f"    EXTRACTION  : {(r.get('Answer') or '').strip()!r}")
print()

print("="*100)
print("4) Extraction override rate, BOTH directions (all annotated calc rows with a numeric cont)")
print("="*100)
for ds, stem in DATASETS.items():
    n = agree = 0; over_to_right = 0; over_to_wrong = 0; over_other = 0
    for pert in PERTS:
        for q, r in load_rows(ds, stem, pert).items():
            if (r.get("Error Type") or "").strip() != "calculation": continue
            if (r.get("Correct?") or "").strip() not in ("True","False"): continue
            _, prefix, cont = dissect(q, r)
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            cnums = parse_nums(cont)
            anums = parse_nums(r.get("Answer") or "")
            if not cnums or not anums: continue
            n += 1
            if close(anums[0], cnums[-1]): agree += 1
            else:
                if close(anums[0], target) and not close(cnums[-1], target): over_to_right += 1
                elif close(cnums[-1], target) and not close(anums[0], target): over_to_wrong += 1
                else: over_other += 1
    print(f"  {ds}: n={n} extraction==cont_last {agree} ({100*agree/n:.1f}%); "
          f"overrides: wrong->RIGHT {over_to_right} ({100*over_to_right/n:.1f}%), "
          f"right->WRONG {over_to_wrong} ({100*over_to_wrong/n:.1f}%), other {over_other}")
