#!/usr/bin/env python3
"""verify0: independent adversarial re-derivation of the 'extraction pass performs
the recovery' finding. Written from scratch; does NOT reuse audit_extraction code paths.

Checks:
 A. alignment sanity: Full Prompt must start with Question and end with suffix
 B. per dataset/pert: n (calc-error, annotated True/False), recovered, and among
    recovered: target absent from continuation numerically (flag), target absent
    even allowing ENGLISH WORD FORM of the number (flag_strict_words),
    continuation's last number == target
 C. bare-number extraction outputs among flagged rows
 D. dump a random sample of 40 flagged rows for manual eyeball
"""
import csv, re, json, random
csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/verify0"
MODEL = "gpt-4-0314"
SUFFIX = "Therefore, the answer (arabic numerals) is"
MARKER = "A: Let's think step by step."
DATASETS = {"gsm8k": "test", "asdiv": "ASDiv", "awps": "MultiArith", "svamp": "SVAMP"}
PERTS = ["random", "add1", "add101"]
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
def int_to_words(n):
    # english words for 0..9999, enough for these datasets
    if n < 0: return None
    n = int(n)
    if n < 20: return ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return TENS[t] + ("" if o == 0 else "-" + ONES[o])
    if n < 1000:
        h, r = divmod(n, 100)
        s = ONES[h] + " hundred"
        return s if r == 0 else s + " " + int_to_words(r)
    if n < 10000:
        th, r = divmod(n, 1000)
        s = ONES[th] + " thousand"
        return s if r == 0 else s + " " + int_to_words(r)
    return None

def norm(s): return (s or "").replace("\r\n", "\n").replace("\r", "\n")
def close(a, b): return abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))
def nums(t):
    out = []
    for m in NUM_RE.finditer(t):
        try: out.append(float(m.group(0).replace(",", "")))
        except ValueError: pass
    return out

random.seed(0)
flagged_all = []
align_bad = 0
table = []
pooled = {}
for ds, stem in DATASETS.items():
    for pert in PERTS:
        path = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
        with open(path, newline="", encoding="utf-8-sig") as f:
            raw = [r for r in csv.DictReader(f)
                   if r.get("Model Name") == MODEL and r.get("Prompt Style") == "sbs"]
        # my own dedup: by problem text (before CoT marker), prefer annotated row
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
        n = recov = flag = flagw = lastok = bare = 0
        for r in byp.values():
            if (r.get("Error Type") or "").strip() != "calculation": continue
            corr = (r.get("Correct?") or "").strip()
            if corr not in ("True", "False"): continue
            n += 1
            if corr != "True": continue
            recov += 1
            q, fp = norm(r["Question"]), norm(r["Full Prompt"])
            if not fp.startswith(q):
                align_bad += 1
                continue
            cont = fp[len(q):]
            i = cont.rfind(SUFFIX)
            if i >= 0: cont = cont[:i]
            try: target = float((r["Target Answer"] or "").replace(",", "").replace("$",""))
            except ValueError: continue
            cn = nums(cont)
            has_num = any(close(x, target) for x in cn)
            has_word = False
            if target == int(target):
                w = int_to_words(target)
                if w and re.search(r"\b" + re.escape(w) + r"\b", cont.lower().replace("-"," ").replace("  "," ")):
                    has_word = True
                # also plain hyphenless / hyphen variants
                if w and "-" in w and w.replace("-", " ") in cont.lower():
                    has_word = True
            if cn and close(cn[-1], target): lastok += 1
            if not has_num:
                flag += 1
                if not has_word: flagw += 1
                ans = (r.get("Answer") or "").strip()
                if re.fullmatch(r"-?[\d,\.\$]+", ans): bare += 1
                flagged_all.append(dict(ds=ds, pert=pert, target=r["Target Answer"],
                    answer=ans, word_form_present=has_word,
                    cont_tail=cont[-320:], question_head=q[:160]))
        table.append((ds, pert, n, recov, flag, flagw, lastok, bare))
        d = pooled.setdefault(pert, [0]*6)
        for i, v in enumerate((n, recov, flag, flagw, lastok, bare)): d[i] += v

print(f"alignment failures (FullPrompt !startswith Question): {align_bad}")
print(f"{'ds/pert':<15}{'n':>6}{'recov':>7}{'rate%':>7}{'xonly':>7}{'xonly_w':>8}{'lastok':>7}{'bare':>6}  corr_gen%  corr_strictlast%")
for ds, pert, n, recov, flag, flagw, lastok, bare in table:
    pr = lambda a,b: f"{100*a/b:.1f}" if b else "-"
    print(f"{ds+'/'+pert:<15}{n:>6}{recov:>7}{pr(recov,n):>7}{flag:>7}{flagw:>8}{lastok:>7}{bare:>6}"
          f"  {pr(recov-flag,n):>8}  {pr(lastok,n):>8}")
print()
for pert, (n, recov, flag, flagw, lastok, bare) in pooled.items():
    print(f"POOLED {pert}: n={n} recovered={recov} ({100*recov/n:.1f}%) "
          f"extraction-only(num)={flag} ({100*flag/recov:.1f}% of recoveries) "
          f"extraction-only(num+words)={flagw} ({100*flagw/recov:.1f}%) "
          f"corrected_gen={100*(recov-flag)/n:.1f}% corrected_strict_last={100*lastok/n:.1f}% "
          f"bare-number extractions among flagged={bare}/{flag}")

sample = random.sample(flagged_all, min(40, len(flagged_all)))
with open(f"{OUT}/flagged_sample40.json", "w") as f:
    json.dump(sample, f, indent=1)
with open(f"{OUT}/flagged_all.json", "w") as f:
    json.dump(flagged_all, f, indent=1)
print(f"\ntotal flagged rows: {len(flagged_all)}; wrote sample40 + all to {OUT}")
