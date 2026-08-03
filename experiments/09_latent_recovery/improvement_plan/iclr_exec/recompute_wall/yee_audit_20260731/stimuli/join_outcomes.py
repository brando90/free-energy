#!/usr/bin/env python3
"""
Join per-stimulus audit table to the released annotated CSVs (gpt-4-0314,
perturbation-random) to quantify how each stimulus-validity issue moves the
published recovery rates. Also measures <<..>>-annotation truncation artifacts.
"""
import csv, json, os, re, sys
from collections import Counter, defaultdict
sys.path.insert(0, "/lfs/skampere2/0/eobbad/scratch/yee_audit/stimuli")

csv.field_size_limit(10**9)
REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/stimuli"
MODEL = "gpt-4-0314"
COT = " Let's think step by step."
DATASETS = {"gsm8k": "test", "asdiv": "ASDiv", "awps": "MultiArith", "svamp": "SVAMP"}
POSITIONS = ["copy", "calc", "propcalc"]
EXPECTED_ET = {"copy": "copying", "calc": "calculation", "propcalc": "calculation"}

def norm(s):
    # collapse all whitespace runs: JSON keys can carry trailing spaces inside
    # the question that the annotated CSV Question column lacks
    return " ".join(s.split())

# ---- load annotated CSVs, dedup keeping the annotated copy ----
ann = {}  # (ds,pos) -> {normalized Question: row}
for ds, task in DATASETS.items():
    for pos in POSITIONS:
        path = os.path.join(REPO, "results", ds,
                            "%s_adjusted_position-%s_perturbation-random_annotated.csv" % (task, pos))
        d = {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("Model Name") != MODEL:
                    continue
                q = norm(row["Question"])
                prev = d.get(q)
                if prev is None or (not prev.get("Correct?", "").strip() and row.get("Correct?", "").strip()):
                    d[q] = row
        ann[(ds, pos)] = d

# ---- load stimuli JSONs to rebuild the exact pass-1 input strings ----
stim_key = {}  # (ds,pos,question_head) -> (key, prefix)
for ds, task in DATASETS.items():
    for pos in POSITIONS:
        jp = os.path.join(REPO, "results", ds,
                          "%s_adjusted_position-%s_perturbation-random_%s.json" % (task, pos, MODEL))
        for k, v in json.load(open(jp, encoding="utf-8")).items():
            stim_key[(ds, pos, k[:80].replace("\n", " "))] = (k, v[0])

# ---- join ----
audit_rows = list(csv.DictReader(open(os.path.join(OUT, "stimulus_audit_gpt4_random.csv"), encoding="utf-8")))
joined = 0
annotated = 0
for r in audit_rows:
    ds, pos = r["dataset"], r["position"]
    k, prefix = stim_key[(ds, pos, r["question_head"])]
    q1 = norm("Q: " + k + "\n\nA: Let's think step by step. " + prefix)
    row = ann[(ds, pos)].get(q1)
    if row is None:
        # fallback: match on prefix tail within Question
        tail = norm(prefix)[-80:]
        cands = [rr for qq, rr in ann[(ds, pos)].items() if qq.endswith(tail)]
        row = cands[0] if len(cands) == 1 else None
    if row is not None:
        joined += 1
        r["ann_error_type"] = row.get("Error Type", "").strip()
        r["ann_correct"] = row.get("Correct?", "").strip()
        r["ann_behavior"] = row.get("Recovery Behavior", "").strip()
        if r["ann_correct"]:
            annotated += 1
    else:
        r["ann_error_type"] = r["ann_correct"] = r["ann_behavior"] = "NOJOIN"
    # << annotation truncation flags
    p = prefix.replace("\r\n", "\n")
    last_open = p.rfind("<<")
    r["ends_inside_annot"] = last_open != -1 and p.rfind(">>") < last_open

outp = os.path.join(OUT, "stimulus_audit_joined.csv")
cols = list(audit_rows[0].keys())
with open(outp, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(audit_rows)
print("WROTE", outp, "joined %d/%d annotated(Correct? set) %d" % (joined, len(audit_rows), annotated))

def rate(rs):
    n = len(rs)
    c = sum(1 for r in rs if r["ann_correct"] == "True")
    return "%d/%d=%.1f%%" % (c, n, 100.0 * c / n) if n else "0/0"

print("\n================ RECOVERY (Correct?=True) BY STRATA ================")
for pos in POSITIONS:
    sub = [r for r in audit_rows if r["position"] == pos and r["status"] == "ok"
           and r["ann_correct"] in ("True", "False")]
    pub = [r for r in sub if r["ann_error_type"] == EXPECTED_ET[pos]]
    print("\n### %s: annotated n=%d; published-denominator (Error Type=%s) n=%d" %
          (pos, len(sub), EXPECTED_ET[pos], len(pub)))
    print("  recovery, all annotated: %s | published-denominator: %s" % (rate(sub), rate(pub)))
    for ds in DATASETS:
        dpub = [r for r in pub if r["dataset"] == ds]
        dall = [r for r in sub if r["dataset"] == ds]
        print("    %-6s all %-14s pubdenom %s" % (ds, rate(dall), rate(dpub)))
    for fld in ["loadbearing", "collision_any", "pert_eq_target", "orig_eq_target",
                "ends_inside_annot", "site_zone"]:
        groups = defaultdict(list)
        for r in pub:
            groups[r[fld]].append(r)
        parts = ["%s:%s" % (k, rate(v)) for k, v in sorted(groups.items())]
        print("  by %s (pubdenom): %s" % (fld, "  ".join(parts)))
    et = Counter(r["ann_error_type"] for r in sub)
    print("  annotator Error Type dist (all annotated):", dict(et))

print("\n================ THE 36 AWPS/CALC CODE-INCONSISTENT ITEMS ================")
bad = [r for r in audit_rows if r["cand_match"] == "False"]
print("Error Type dist:", dict(Counter(r["ann_error_type"] for r in bad)))
print("Correct? dist:", dict(Counter(r["ann_correct"] for r in bad)))
inpub = [r for r in bad if r["ann_error_type"] == "calculation" and r["ann_correct"] in ("True", "False")]
print("in published calc denominator: n=%d recovery %s" % (len(inpub), rate(inpub)))
print("Recovery Behavior:", dict(Counter(r["ann_behavior"] for r in bad if r["ann_behavior"])))

print("\n================ ANNOTATION-TRUNCATION ARTIFACT (prefix ends inside <<..>>) ================")
for pos in POSITIONS:
    for ds in DATASETS:
        sub = [r for r in audit_rows if r["position"] == pos and r["dataset"] == ds and r["status"] == "ok"]
        e = sum(1 for r in sub if r["ends_inside_annot"])
        if e:
            print("  %s/%s: %d/%d=%.1f%%" % (ds, pos, e, len(sub), 100.0 * e / len(sub)))

print("\n================ ERROR TYPE vs MY MECHANICAL SITE CLASS ================")
# For calc/propcalc published denominators: how many sites are question-value leaks
for pos in ["calc", "propcalc"]:
    pub = [r for r in audit_rows if r["position"] == pos and r["status"] == "ok"
           and r["ann_error_type"] == "calculation" and r["ann_correct"] in ("True", "False")]
    dig = [r for r in pub if r["calc_val_in_question_digits"] == "True"]
    wrd = [r for r in pub if r["calc_val_in_question_words"] == "True"]
    print("%s pubdenom n=%d: question-digit-leak n=%d (recovery %s), word-leak n=%d (recovery %s)"
          % (pos, len(pub), len(dig), rate(dig), len(wrd), rate(wrd)))

# copy condition: first-occurrence source vs recovery
pubc = [r for r in audit_rows if r["position"] == "copy" and r["status"] == "ok"
        and r["ann_error_type"] == "copying" and r["ann_correct"] in ("True", "False")]
for src in ("question", "cot"):
    sub = [r for r in pubc if r["copy_first_occ_src"] == src]
    print("copy pubdenom first-occ=%s: recovery %s" % (src, rate(sub)))
