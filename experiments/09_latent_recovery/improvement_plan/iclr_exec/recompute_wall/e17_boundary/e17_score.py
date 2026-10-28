#!/usr/bin/env python3
"""E17 scorer: same concluding-number parse as yee_memo/score.py; outcomes recovered/propagated/other/no_answer;
restart + malformed annotations; paired numeral-vs-statement analysis with exact McNemar. Writes results.json + results.txt."""
import json, math, re, os, collections
from fractions import Fraction
OUT = "/lfs/skampere2/0/eobbad/scratch/e17_boundary"; NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?"); ANS_RE = re.compile(r"answer[^0-9]{0,80}(-?\d[\d,]*(?:\.\d+)?)", re.I)
MARKER = "A: Let's think step by step."
def val(s):
    try: return float(str(s).replace(",", "").rstrip("."))
    except ValueError: return None
def close(a, b): return a is not None and b is not None and abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))
def concluding(text, q_nums):
    ms = list(ANS_RE.finditer(text))
    if ms: return val(ms[-1].group(1))
    for s in reversed(re.split(r"(?<=[.!?\n])", text)):
        nums = [x for x in (val(t) for t in NUM_RE.findall(s)) if x is not None]
        if not nums: continue
        ne = [x for x in nums if not any(close(x, q) for q in q_nums)]; return ne[-1] if ne else nums[-1]
    return None
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return ((c - h) / d, (c + h) / d)
def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c); p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n; return min(1.0, 2 * p)
items = {(i["phash"], i["numbering"]): i for i in json.load(open(OUT + "/e17_items.json"))}
rows = {}
for line in open(OUT + "/responses.jsonl"):
    d = json.loads(line)
    if d.get("error"): continue
    key = (d["phash"], d["numbering"], d["arm"])
    if key in rows: continue
    it = items[(d["phash"], d["numbering"])]; text = d["text"] or ""
    q = it["prompts"][d["arm"]]; qtext = q[:q.find(MARKER)]; q_nums = [x for x in (val(t) for t in NUM_RE.findall(qtext)) if x is not None]
    tgt = val(it["target"]); prop = None
    try: prop = float(Fraction(it["propagated"])) if it["propagated"] is not None else None
    except Exception: prop = val(it["propagated"])
    cm = concluding(text, q_nums)
    outcome = "recovered" if close(cm, tgt) else "propagated" if (close(cm, prop) and d["arm"].startswith("corr")) else "no_answer" if cm is None else "other"
    # restart = the continuation re-solves from the problem statement rather than continuing the chain:
    # it re-derives the site value (writes "<a> <op> <b> =" for the site's own operands) or opens with a fresh-solve cue
    cot = q[q.find(MARKER) + len(MARKER):]
    site_eq = re.findall(r"(\d[\d,]*)\s*[-+*/x]\s*(\d[\d,]*)\s*=\s*$", cot.split(it["suffix"])[0] if it["suffix"] and cot.endswith(it["suffix"]) else cot)
    rederives_site = bool(site_eq and re.search(re.escape(site_eq[-1][0]) + r"\s*[-+*/x]\s*" + re.escape(site_eq[-1][1]) + r"\s*=", text))
    fresh_cue = bool(re.match(r"\s*(?:First|Let's|Let us|To (?:find|solve|determine)|We (?:need|start|begin)|Step 1)", text))
    tnums = [x for x in (val(t) for t in NUM_RE.findall(text)) if x is not None]
    rows[key] = {"outcome": outcome, "restart": rederives_site or fresh_cue, "rederives_site": rederives_site, "fresh_cue": fresh_cue,
                 "mentions_sitetrue": any(close(x, val(it["site_true"])) for x in tnums), "mentions_planted": any(close(x, val(it["planted"])) for x in tnums),
                 "malformed": d.get("finish_reason") == "length" or cm is None, "text_head": text[:80]}
res = {"n_items": len(items)}; lines = []
def P(s): lines.append(s); print(s)
for numbering in ("ORIG", "RENUM", "POOLED"):
    keys = [k for k in items if numbering == "POOLED" or k[1] == numbering]
    P(f"\n=== {numbering}: {len(keys)} items ===")
    blk = {}
    for arm in ("corr_numeral", "corr_statement", "clean_numeral", "clean_statement"):
        rs = [rows[(ph, nb, arm)] for ph, nb in keys if (ph, nb, arm) in rows]; n = len(rs)
        c = collections.Counter(r["outcome"] for r in rs); rec = c["recovered"]; lo, hi = wilson(rec, n)
        blk[arm] = {"n": n, "recovered": rec, "recovered_rate": round(rec / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)], "propagated": c["propagated"], "other": c["other"], "no_answer": c["no_answer"],
                    "restart": sum(r["restart"] for r in rs), "rederives_site": sum(r["rederives_site"] for r in rs), "fresh_cue": sum(r["fresh_cue"] for r in rs), "mentions_sitetrue": sum(r["mentions_sitetrue"] for r in rs), "mentions_planted": sum(r["mentions_planted"] for r in rs), "malformed": sum(r["malformed"] for r in rs),
                    "recovered_nonrestart_rate": round(sum(1 for r in rs if r["outcome"] == "recovered" and not r["restart"]) / max(1, sum(1 for r in rs if not r["restart"])), 4)}
        P(f"{arm:16s} n={n:3d} recovered {rec:3d} ({100*rec/max(n,1):5.1f}% [{100*lo:4.1f},{100*hi:4.1f}]) propagated {c['propagated']:3d} other {c['other']:3d} no_answer {c['no_answer']:3d} | restart {blk[arm]['restart']:3d} (rederive {blk[arm]['rederives_site']:2d} cue {blk[arm]['fresh_cue']:2d}) true-val-mentioned {blk[arm]['mentions_sitetrue']:3d} planted-mentioned {blk[arm]['mentions_planted']:3d} malformed {blk[arm]['malformed']:3d}")
    for pair, label in ((("corr_numeral", "corr_statement"), "corrupted"), (("clean_numeral", "clean_statement"), "clean")):
        a_, b_ = pair; tab = collections.Counter()
        for ph, nb in keys:
            if (ph, nb, a_) in rows and (ph, nb, b_) in rows:
                tab[(rows[(ph, nb, a_)]["outcome"] == "recovered", rows[(ph, nb, b_)]["outcome"] == "recovered")] += 1
        b = tab[(True, False)]; c = tab[(False, True)]; pval = mcnemar_exact(b, c); nn = sum(tab.values())
        diff = (tab[(True, True)] + b - (tab[(True, True)] + c)) / max(nn, 1)
        blk[f"paired_{label}"] = {"n_pairs": nn, "both_recovered": tab[(True, True)], "numeral_only": b, "statement_only": c, "neither": tab[(False, False)], "recovery_diff_numeral_minus_statement": round(diff, 4), "mcnemar_exact_p": round(pval, 4)}
        P(f"paired {label:9s}: n={nn} both={tab[(True,True)]} numeral-only={b} statement-only={c} neither={tab[(False,False)]} | recovery numeral-statement = {100*diff:+.1f} pts, McNemar p={pval:.4f}")
    res[numbering] = blk
json.dump(res, open(OUT + "/results.json", "w"), indent=1); open(OUT + "/results.txt", "w").write("\n".join(lines))
