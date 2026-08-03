#!/usr/bin/env python3
"""
H2 stimulus-validity audit for Yee et al. COLM 2024 release.
Conditions: gpt-4-0314, perturbation-random, positions copy/calc/propcalc, all 4 datasets.

Checks:
 (a) delta applied vs spec (random => nonzero integer in [-3,3]); |delta| distribution
 (b) collision: perturbed value equals another quantity in question or prior CoT; also == target
 (c) load-bearingness: parse equations in ORIGINAL CoT, naively propagate perturbed value,
     check whether final answer mechanically changes
 (d) calc position correctness: perturbed-site original value present in question digits
     (code invariant, expect 0) or as number-WORD in question (leak)

Method: exact alignment of released stimulus prefix against the released original CoT
(baseline CSV Full Prompt), plus bug-for-bug replication of intervention()'s candidate
enumeration (step-number masking, value-keyed occurrence grouping) as a cross-check.
"""
import csv, json, re, os, sys
from collections import Counter, defaultdict

csv.field_size_limit(10**9)

REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/stimuli"
COT_PROMPT = " Let's think step by step."
SUFFIX = " Therefore, the answer (arabic numerals) is"
MODEL = "gpt-4-0314"
POSITIONS = ["copy", "calc", "propcalc"]
DATASETS = {  # dataset dir -> (taskname, baseline csv)
    "gsm8k": ("test", "test.csv"),
    "asdiv": ("ASDiv", "ASDiv.csv"),
    "awps": ("MultiArith", "MultiArith.csv"),
    "svamp": ("SVAMP", "SVAMP.csv"),
}

NUM_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d*)?")  # exactly their regex
TRAIL_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d*)?\.?$")
FIRSTNUM_RE = re.compile(r"-?\d+\.?\d*")

def tokval(tok):
    t = tok.replace(",", "").rstrip(".").lstrip("$")
    if t in ("", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None

def canon(v):
    if v is None:
        return None
    if abs(v - round(v)) < 1e-9:
        return int(round(v))
    return round(v, 9)

def parse_target(s):
    m = FIRSTNUM_RE.search(str(s).replace(",", ""))
    return canon(float(m.group())) if m else None

# ---------- replication of intervention() candidate enumeration ----------
def remove_steps(reasoning, reasoning_start):
    rns = reasoning
    step_start = reasoning_start
    for step in range(1, 10000):
        pat = re.compile(r"(\s|^)%d\.\s" % step)
        m = pat.search(rns)
        if m:
            repl = "@" * len(str(step))
            new_step_start = m.span()[0]
            rns = rns[:step_start] + pat.sub(" " + repl + ". ", rns[step_start:], count=1)
            step_start = new_step_start
        else:
            break
    return rns

def enumerate_candidates(reasoning, position):
    rs = reasoning.index(COT_PROMPT) + len(COT_PROMPT) if COT_PROMPT in reasoning else 0
    rns = remove_steps(reasoning, rs)
    len_ok = (len(rns) == len(reasoning))
    re_end = len(rns)
    matches = list(NUM_RE.finditer(rns))
    occ = {}
    for m in matches:
        g = float(m.group().replace(",", ""))
        if int(g) == g:
            g = int(g)
        occ.setdefault(g, []).append(m)
    if position == "calc":
        cands = [[ms[0]] for ms in occ.values()
                 if ms[0].start() >= rs and ms[0].start() <= re_end]
    elif position == "copy":
        flat = [m for ms in occ.values() if len(ms) > 1 for m in ms[1:]]
        cands = [[m] for m in flat if m.start() >= rs and m.start() <= re_end]
    elif position == "propcalc":
        cands = [ms[:2] for ms in occ.values()
                 if len(ms) > 1 and ms[0].start() >= rs and ms[0].start() <= re_end]
    else:
        cands = []
    return rs, rns, cands, occ, len_ok

# ---------- number-word leak detection ----------
WORDS = {"zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,
 "eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,
 "fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
 "thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,
 "hundred":100,"thousand":1000,"dozen":12,"twice":2,"double":2,"thrice":3,"triple":3,
 "half":0.5,"quarter":0.25}
TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,
        "eighty":80,"ninety":90}
ONES = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}

def question_word_values(qtext):
    q = qtext.lower()
    vals = set()
    for t, tv in TENS.items():
        for o, ov in ONES.items():
            if re.search(r"\b%s[-\s]%s\b" % (t, o), q):
                vals.add(tv + ov)
    for w, wv in WORDS.items():
        if re.search(r"\b%s\b" % w, q):
            vals.add(wv)
    return set(canon(float(v)) for v in vals)

# ---------- equation parsing + naive propagation ----------
OPTOK = r"\$?\d+(?:,\d{3})*(?:\.\d+)?%?"
EQ_RE = re.compile(r"(%s)((?:\s*[-+*/x×÷]\s*%s)+)\s*=\s*(%s)" % (OPTOK, OPTOK, OPTOK))
OPSPLIT_RE = re.compile(r"\s*([-+*/x×÷])\s*(%s)" % OPTOK)

def parse_equations(text):
    """Return list of dicts: {start,end,ops:[(val,(s,e))],expr,rhs,rhs_span,verified}.
    Scanning resumes at RHS start so running equalities chain (a+b=c*d=e)."""
    eqs, pos = [], 0
    while True:
        m = EQ_RE.search(text, pos)
        if not m:
            break
        lhs_first, ops_str, rhs_tok = m.group(1), m.group(2), m.group(3)
        ops = [(tokval(lhs_first), (m.start(1), m.end(1)))]
        expr_parts = [str(tokval(lhs_first))]
        base = m.start(2)
        for om in OPSPLIT_RE.finditer(ops_str):
            op = om.group(1)
            op = {"x": "*", "×": "*", "÷": "/"}.get(op, op)
            v = tokval(om.group(2))
            ops.append((v, (base + om.start(2), base + om.end(2))))
            expr_parts.append(op)
            expr_parts.append(str(v))
        rhs = tokval(rhs_tok)
        verified = False
        lhs_val = None
        if all(o[0] is not None for o in ops) and rhs is not None:
            try:
                lhs_val = eval("".join(expr_parts))
                verified = abs(lhs_val - rhs) <= max(1e-6, 1e-6 * abs(rhs))
            except (ZeroDivisionError, SyntaxError):
                pass
        eqs.append({"start": m.start(), "end": m.end(), "ops": ops,
                    "expr_ops": [p for p in expr_parts if p in "+-*/"],
                    "rhs": rhs, "rhs_span": (m.start(3), m.end(3)), "verified": verified})
        pos = m.start(3)  # allow RHS to start the next equation
        if pos <= m.start():
            pos = m.end()
    return eqs

def recompute(ops_vals, expr_ops):
    parts = [str(ops_vals[0])]
    for i, op in enumerate(expr_ops):
        parts.append(op)
        parts.append(str(ops_vals[i + 1]))
    try:
        return eval("".join(parts))
    except (ZeroDivisionError, SyntaxError):
        return None

def propagate(eqs, site_start, site_end, v_orig, v_pert, target, seed_by_value):
    """Naive propagation. seed_by_value=True (calc/propcalc/copy-at-rhs):
    mapping {v_orig: v_pert}; eq containing site as RHS is NOT recomputed.
    copy at operand: positional replacement in containing eq, then value mapping."""
    veqs = [e for e in eqs if e["verified"]]
    if not veqs:
        return "not_analyzable", None, 0
    rhs_vals = set(canon(e["rhs"]) for e in veqs)
    if canon(target) not in rhs_vals:
        return "not_analyzable", None, len(veqs)
    mapping = {}
    if seed_by_value:
        mapping[canon(v_orig)] = v_pert
    n_recomputed = 0
    for e in veqs:
        if e["end"] <= site_start:
            continue  # entirely before the error site
        contains = (e["start"] <= site_start < e["end"])
        if contains:
            in_rhs = e["rhs_span"][0] <= site_start < e["rhs_span"][1]
            if in_rhs:
                # site is this eq's result: stimulus asserts v_pert; do not recompute
                mapping[canon(e["rhs"])] = v_pert
                continue
            # site is an operand: positional replacement + any value mapping
            new_vals = []
            for (ov, (s, epos)) in e["ops"]:
                if s <= site_start < epos:
                    new_vals.append(v_pert)
                else:
                    new_vals.append(mapping.get(canon(ov), ov))
        else:
            new_vals = [mapping.get(canon(ov), ov) for (ov, _) in e["ops"]]
            if all(canon(a) == canon(b[0]) for a, b in zip(new_vals, e["ops"])):
                continue  # nothing substituted
        new_rhs = recompute(new_vals, e["expr_ops"])
        n_recomputed += 1
        if new_rhs is not None and canon(new_rhs) != canon(e["rhs"]):
            mapping[canon(e["rhs"])] = new_rhs
    final = mapping.get(canon(target), target)
    return ("changed" if canon(final) != canon(target) else "unchanged"), canon(final), len(veqs)

# ---------- load baseline ----------
def load_baseline(ds, csvname):
    path = os.path.join(REPO, "results", ds, csvname)
    rows = {}
    dup = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Model Name"] != MODEL or row["Prompt Style"] != "sbs":
                continue
            q = row["Question"]
            if q in rows:
                dup += 1
                continue
            rows[q] = row
    return rows, dup

# ---------- main per-stimulus audit ----------
def audit_stimulus(ds, position, key, prefix, target_str, baserow, out):
    rec = {"dataset": ds, "position": position, "question_head": key[:80].replace("\n", " "),
           "status": "", "site_abs": "", "orig_tok": "", "pert_tok": "",
           "orig_val": "", "pert_val": "", "delta": "", "delta_in_spec": "",
           "exact_reconstruction": "", "cand_match": "", "site2_ok": "",
           "collision_question": "", "collision_prior_cot": "", "collision_any": "",
           "pert_eq_target": "", "orig_eq_target": "", "copy_first_occ_src": "",
           "site_is_eq_rhs_char": "", "calc_val_in_question_digits": "",
           "calc_val_in_question_words": "", "n_eqs": 0, "n_eqs_verified": 0,
           "site_zone": "", "loadbearing": "", "propagated_final": "",
           "target": "", "notes": ""}
    out.append(rec)
    target = parse_target(target_str)
    rec["target"] = target
    if baserow is None:
        rec["status"] = "no_baseline_row"
        return rec
    # normalize line endings: released JSON prefixes carry \r\n where csv-parsed
    # baseline fields carry \n (Excel/Sheets round-trip artifact, decode fact)
    reasoning = baserow["Full Prompt"].replace("\r\n", "\n")
    P = prefix.replace("\r\n", "\n")
    if COT_PROMPT not in reasoning:
        rec["status"] = "no_cot_prompt"
        return rec
    rs_idx = reasoning.index(COT_PROMPT) + len(COT_PROMPT)
    R_full = reasoning[rs_idx:]
    if R_full.endswith(SUFFIX):
        R_full = R_full[: -len(SUFFIX)]
    lead = len(R_full) - len(R_full.lstrip())
    Rs = R_full.lstrip()

    # find first divergence
    d = None
    for i in range(min(len(P), len(Rs))):
        if P[i] != Rs[i]:
            d = i
            break
    if d is None:
        rec["status"] = "no_divergence"
        rec["notes"] = "prefix matches original verbatim for its whole length"
        return rec

    # site token in original at divergence
    site_m = None
    for m in NUM_RE.finditer(Rs):
        if m.start() <= d < m.end():
            site_m = m
            break
        if m.start() > d:
            break
    if site_m is None:
        rec["status"] = "diverge_outside_number"
        rec["notes"] = "orig@d=%r pert@d=%r ctx=%r" % (Rs[d:d+12], P[d:d+12], Rs[max(0,d-30):d+12])
        return rec

    s0 = site_m.start()
    orig_tok = site_m.group()
    if position in ("calc", "copy"):
        pert_tok = P[s0:]
        if not TRAIL_NUM_RE.match(pert_tok.strip()):
            rec["status"] = "prefix_not_ending_at_number"
            rec["notes"] = "tail=%r" % pert_tok[-30:]
            return rec
        exact = (P[:s0] == Rs[:s0])
    else:  # propcalc: first perturbed occurrence at s0, prefix continues to 2nd
        pm = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d*)?").match(P, s0)
        if pm is None:
            rec["status"] = "propcalc_site1_unparsed"
            return rec
        pert_tok = pm.group()
        # scan for second divergence
        i, j = pm.end(), site_m.end()
        d2 = None
        while i < len(P) and j < len(Rs):
            if P[i] != Rs[j]:
                d2 = (i, j)
                break
            i += 1
            j += 1
        if d2 is None:
            if i >= len(P):
                rec["status"] = "propcalc_no_second_divergence"
                rec["notes"] = "prefix ends before 2nd site diverges"
                return rec
            rec["status"] = "propcalc_orig_exhausted"
            return rec
        i2, j2 = d2
        site2_m = None
        for m in NUM_RE.finditer(Rs, max(0, j2 - 25)):
            if m.start() <= j2 < m.end():
                site2_m = m
                break
            if m.start() > j2:
                break
        pert2 = P[i2 - (j2 - site2_m.start()):] if site2_m else None
        site2_ok = (site2_m is not None and pert2 is not None
                    and TRAIL_NUM_RE.match(pert2.strip()) is not None
                    and tokval(pert2) == tokval(pert_tok)
                    and canon(tokval(site2_m.group())) == canon(tokval(orig_tok)))
        rec["site2_ok"] = site2_ok
        exact = (P[:s0] == Rs[:s0])

    ov, pv = tokval(orig_tok), tokval(pert_tok)
    if ov is None or pv is None:
        rec["status"] = "token_value_unparsed"
        return rec
    delta = pv - ov
    rec.update(status="ok", site_abs=s0, orig_tok=orig_tok, pert_tok=pert_tok.strip(),
               orig_val=canon(ov), pert_val=canon(pv), delta=round(delta, 6),
               delta_in_spec=(abs(delta - round(delta)) < 1e-6 and 1 <= abs(round(delta)) <= 3),
               exact_reconstruction=exact)

    # cross-check vs replicated candidate enumeration (abs coords: rs_idx offset + lead)
    rs2, rns, cands, occ, len_ok = enumerate_candidates(reasoning, position)
    site_abs_in_reasoning = rs_idx + lead + s0
    cand_starts = set(c[0].start() for c in cands)
    rec["cand_match"] = (site_abs_in_reasoning in cand_starts) if len_ok else "len_mismatch"

    # (b) collisions
    qvals = set(canon(tokval(m.group())) for m in NUM_RE.finditer(key))
    qvals.discard(None)
    prior_vals = set(canon(tokval(m.group())) for m in NUM_RE.finditer(Rs[:s0]))
    prior_vals.discard(None)
    cq = canon(pv) in qvals
    cp = canon(pv) in prior_vals
    rec.update(collision_question=cq, collision_prior_cot=cp, collision_any=(cq or cp),
               pert_eq_target=(target is not None and canon(pv) == target),
               orig_eq_target=(target is not None and canon(ov) == target))

    # copy: where does the value's first occurrence live (question vs CoT)?
    if position == "copy":
        g = canon(ov)
        first = None
        for m in NUM_RE.finditer(reasoning):
            if canon(tokval(m.group())) == g:
                first = m
                break
        if first is not None:
            rec["copy_first_occ_src"] = "question" if first.start() < rs_idx else "cot"

    # site char context: is it an equation RHS position?
    k = s0 - 1
    while k >= 0 and Rs[k] in " $ ":
        k -= 1
    rec["site_is_eq_rhs_char"] = (k >= 0 and Rs[k] in "=>")

    # (d) calc leak
    if position in ("calc", "propcalc"):
        rec["calc_val_in_question_digits"] = canon(ov) in qvals
        rec["calc_val_in_question_words"] = canon(ov) in question_word_values(key)

    # (c) load-bearing
    eqs = parse_equations(Rs)
    veqs = [e for e in eqs if e["verified"]]
    rec["n_eqs"] = len(eqs)
    rec["n_eqs_verified"] = len(veqs)
    in_eq = None
    for e in veqs:
        if e["start"] <= s0 < e["end"]:
            in_eq = e
            break
    if in_eq is not None:
        in_rhs = in_eq["rhs_span"][0] <= s0 < in_eq["rhs_span"][1]
        rec["site_zone"] = "eq_rhs" if in_rhs else "eq_operand"
    elif veqs and s0 >= max(e["end"] for e in veqs):
        rec["site_zone"] = "after_last_eq"
    elif veqs:
        rec["site_zone"] = "prose_mid"
    else:
        rec["site_zone"] = "no_verified_eqs"
    if target is None:
        rec["loadbearing"] = "no_target"
    else:
        seed_by_value = position in ("calc", "propcalc") or rec["site_zone"] == "eq_rhs"
        resmode, final, nv = propagate(eqs, s0, s0 + len(orig_tok), ov, pv, target, seed_by_value)
        rec["loadbearing"] = resmode
        rec["propagated_final"] = final if final is not None else ""
    return rec

def main():
    os.makedirs(OUT, exist_ok=True)
    all_recs = []
    file_stats = []
    for ds, (task, basecsv) in DATASETS.items():
        baserows, dup = load_baseline(ds, basecsv)
        for pos in POSITIONS:
            jp = os.path.join(REPO, "results", ds,
                              "%s_adjusted_position-%s_perturbation-random_%s.json" % (task, pos, MODEL))
            stim = json.load(open(jp, encoding="utf-8"))
            matched = 0
            for key, val in stim.items():
                prefix, tgt = val[0], val[1]
                row = baserows.get(key)
                if row is not None:
                    matched += 1
                audit_stimulus(ds, pos, key, prefix, tgt, row, all_recs)
            file_stats.append((ds, pos, len(stim), matched, dup))
    # write per-stimulus table
    outcsv = os.path.join(OUT, "stimulus_audit_gpt4_random.csv")
    cols = list(all_recs[0].keys())
    with open(outcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_recs:
            w.writerow(r)
    print("WROTE", outcsv, len(all_recs), "rows")
    print("\n=== FILE STATS (dataset, position, n_stimuli, matched_baseline, baseline_dup_rows) ===")
    for t in file_stats:
        print(t)
    # aggregates
    def agg(recs, label):
        n = len(recs)
        ok = [r for r in recs if r["status"] == "ok"]
        print("\n--- %s: n=%d ok=%d statuses=%s" % (label, n, len(ok), dict(Counter(r["status"] for r in recs))))
        if not ok:
            return
        spec = sum(1 for r in ok if r["delta_in_spec"] is True)
        exact = sum(1 for r in ok if r["exact_reconstruction"] is True)
        cand = Counter(str(r["cand_match"]) for r in ok)
        print("  delta_in_spec %d/%d  exact_recon %d/%d  cand_match %s" % (spec, len(ok), exact, len(ok), dict(cand)))
        print("  delta dist:", dict(sorted(Counter(r["delta"] for r in ok).items(), key=lambda kv: kv[0])))
        for fld in ["collision_question", "collision_prior_cot", "collision_any",
                    "pert_eq_target", "orig_eq_target", "site_is_eq_rhs_char"]:
            c = sum(1 for r in ok if r[fld] is True)
            print("  %s: %d/%d = %.1f%%" % (fld, c, len(ok), 100.0 * c / len(ok)))
        lb = Counter(r["loadbearing"] for r in ok)
        print("  loadbearing:", dict(lb))
        anz = [r for r in ok if r["loadbearing"] in ("changed", "unchanged")]
        if anz:
            unch = sum(1 for r in anz if r["loadbearing"] == "unchanged")
            print("  UNCHANGED among analyzable: %d/%d = %.1f%%" % (unch, len(anz), 100.0 * unch / len(anz)))
        print("  site_zone:", dict(Counter(r["site_zone"] for r in ok)))
        calcish = [r for r in ok if r["position"] in ("calc", "propcalc")]
        if calcish:
            dig = sum(1 for r in calcish if r["calc_val_in_question_digits"] is True)
            wrd = sum(1 for r in calcish if r["calc_val_in_question_words"] is True)
            print("  calc leak digits: %d/%d  words: %d/%d = %.1f%%"
                  % (dig, len(calcish), wrd, len(calcish), 100.0 * wrd / len(calcish)))
        csrc = Counter(r["copy_first_occ_src"] for r in ok if r["copy_first_occ_src"])
        if csrc:
            print("  copy first-occurrence source:", dict(csrc))
        s2 = Counter(str(r["site2_ok"]) for r in ok if r["position"] == "propcalc")
        if s2:
            print("  propcalc site2_ok:", dict(s2))

    for pos in POSITIONS:
        agg([r for r in all_recs if r["position"] == pos], "POOLED %s" % pos)
    for ds in DATASETS:
        for pos in POSITIONS:
            agg([r for r in all_recs if r["dataset"] == ds and r["position"] == pos], "%s/%s" % (ds, pos))

    # delta uniformity test (pooled, all ok)
    ok = [r for r in all_recs if r["status"] == "ok" and r["delta_in_spec"] is True]
    cnt = Counter(int(round(r["delta"])) for r in ok)
    obs = [cnt.get(k, 0) for k in (-3, -2, -1, 1, 2, 3)]
    try:
        from scipy.stats import chisquare
        stat, p = chisquare(obs)
        print("\nDelta uniformity over {-3..3}\\{0}: obs=%s chi2=%.2f p=%.4f" % (obs, stat, p))
    except ImportError:
        print("\nDelta uniformity: obs=", obs)

if __name__ == "__main__":
    main()
