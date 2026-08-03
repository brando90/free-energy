#!/usr/bin/env python3
"""Headline table deduped at PROBLEM level (collapses whitespace-variant double rows).
Also writes corrected_rates.csv."""
import csv, re
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

rows_csv = []
print(("{:<14}"+"{:>7}"*9).format("ds/pert","n","recov","rate","gen","genrate","strict","strate","xonly","xorate"))
pooled = {}
for ds, stem in DATASETS.items():
    for pert in PERTS:
        path = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
        with open(path, newline="", encoding="utf-8-sig") as f:
            raw = [r for r in csv.DictReader(f)
                   if r.get("Model Name") == MODEL and r.get("Prompt Style") == "sbs"]
        byp = {}
        for r in raw:
            q = norm(r["Question"])
            mi = q.find(MARKER)
            if mi < 0: continue
            p = q[3:mi].strip()
            cur = byp.get(p)
            if cur is None or ((r.get("Correct?") or "").strip() and not (cur[1].get("Correct?") or "").strip()):
                byp[p] = (q, r)
        n = recov = gen = strict = xonly = 0
        for p, (q, r) in byp.items():
            if (r.get("Error Type") or "").strip() != "calculation": continue
            if (r.get("Correct?") or "").strip() not in ("True","False"): continue
            n += 1
            if (r.get("Correct?") or "").strip() != "True": continue
            recov += 1
            fp = norm(r["Full Prompt"])
            mi = q.find(MARKER)
            cont = fp[len(q):]
            if cont.endswith(SUFFIX): cont = cont[:-len(SUFFIX)]
            cont = cont[1:] if cont.startswith(" ") else cont
            try: target = float((r["Target Answer"] or "").replace(",", ""))
            except ValueError: continue
            cnums = parse_nums(cont)
            has_t = any(close(x, target) for x in cnums)
            last_t = bool(cnums) and close(cnums[-1], target)
            if has_t: gen += 1
            if last_t: strict += 1
            if not has_t: xonly += 1
        def pc(a,b): return f"{100*a/b:.1f}" if b else "-"
        print(("{:<14}"+"{:>7}"*9).format(f"{ds}/{pert}", n, recov, pc(recov,n),
              gen, pc(gen,n), strict, pc(strict,n), xonly, pc(xonly,n)))
        rows_csv.append(dict(dataset=ds, perturbation=pert, n=n, recovered=recov,
            published_style_rate=round(100*recov/n,1) if n else None,
            cont_channel_generous=gen, corrected_rate_generous=round(100*gen/n,1) if n else None,
            cont_channel_strict=strict, corrected_rate_strict=round(100*strict/n,1) if n else None,
            extraction_only=xonly, extraction_only_pct_of_recoveries=round(100*xonly/recov,1) if recov else None))
        key = pert
        d = pooled.setdefault(key, dict(n=0,recov=0,gen=0,strict=0,xonly=0))
        d["n"] += n; d["recov"] += recov; d["gen"] += gen; d["strict"] += strict; d["xonly"] += xonly
print()
for pert, d in pooled.items():
    print(f"pooled {pert}: n={d['n']} published-style {100*d['recov']/d['n']:.1f}% "
          f"corrected(gen) {100*d['gen']/d['n']:.1f}% corrected(strict) {100*d['strict']/d['n']:.1f}% "
          f"extraction-only {d['xonly']} = {100*d['xonly']/d['recov']:.1f}% of recoveries")
with open(f"{OUT}/corrected_rates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
    w.writeheader(); w.writerows(rows_csv)
print(f"\nwrote {OUT}/corrected_rates.csv")
