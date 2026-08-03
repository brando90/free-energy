#!/usr/bin/env python3
"""verify0 follow-up: for every flagged (extraction-only) row, inspect the
prefix/continuation boundary to detect false flags where the target numeral is
formed across the junction (prefix's last digits + continuation's leading digits),
and print prefix tail + continuation head for manual reading."""
import csv, re, json
csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/verify0"
MODEL = "gpt-4-0314"
SUFFIX = "Therefore, the answer (arabic numerals) is"
MARKER = "A: Let's think step by step."
DATASETS = {"gsm8k": "test", "asdiv": "ASDiv", "awps": "MultiArith", "svamp": "SVAMP"}
PERTS = ["random", "add1", "add101"]
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def norm(s): return (s or "").replace("\r\n", "\n").replace("\r", "\n")
def close(a, b): return abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))
def nums(t):
    out = []
    for m in NUM_RE.finditer(t):
        try: out.append(float(m.group(0).replace(",", "")))
        except ValueError: pass
    return out

rows_out = []
counts = {}
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
            key = re.sub(r"\s+", " ", q[:mi]).strip()
            prev = byp.get(key)
            ann = (r.get("Correct?") or "").strip() in ("True", "False")
            if prev is None or (ann and (prev.get("Correct?") or "").strip() not in ("True","False")):
                byp[key] = r
        for r in byp.values():
            if (r.get("Error Type") or "").strip() != "calculation": continue
            if (r.get("Correct?") or "").strip() != "True": continue
            q, fp = norm(r["Question"]), norm(r["Full Prompt"])
            if not fp.startswith(q): continue
            cont = fp[len(q):]
            i = cont.rfind(SUFFIX)
            if i >= 0: cont = cont[:i]
            try: target = float((r["Target Answer"] or "").replace(",", "").replace("$",""))
            except ValueError: continue
            if any(close(x, target) for x in nums(cont)): continue  # not flagged
            # boundary test: target numeral formed across the junction?
            joined = q[-14:] + cont[:40]
            boundary = any(close(x, target) for x in nums(joined)) and not any(close(x, target) for x in nums(cont[:40]))
            # also: continuation starts with a digit (glued to prefix number)?
            starts_digit = bool(cont) and cont[0].isdigit()
            cat = "BOUNDARY" if boundary else ("STARTS_DIGIT" if starts_digit else "clean")
            counts[cat] = counts.get(cat, 0) + 1
            counts[(pert, cat)] = counts.get((pert, cat), 0) + 1
            rows_out.append(dict(ds=ds, pert=pert, target=r["Target Answer"],
                extracted=(r.get("Answer") or "")[:60], cat=cat,
                prefix_tail=q[-70:], cont_head=cont[:150], cont_tail=cont[-150:]))
print("flagged total:", len(rows_out))
for k, v in sorted(counts.items(), key=str): print(k, v)
print()
for r in rows_out:
    if r["cat"] != "clean":
        print(f"--- {r['ds']}/{r['pert']} target={r['target']} extracted={r['extracted']!r} [{r['cat']}]")
        print("  PREFIX_TAIL:", repr(r["prefix_tail"]))
        print("  CONT_HEAD :", repr(r["cont_head"]))
with open(f"{OUT}/flagged_boundary_check.json", "w") as f:
    json.dump(rows_out, f, indent=1)
print(f"wrote {OUT}/flagged_boundary_check.json")
