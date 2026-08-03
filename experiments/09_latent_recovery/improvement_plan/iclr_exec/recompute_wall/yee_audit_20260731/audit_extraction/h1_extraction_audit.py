#!/usr/bin/env python3
"""H1 audit: does the answer-extraction second pass do the recovering?

Focus: gpt-4-0314, position-calc, perturbation in {random, add1, add101}, all 4 datasets.
For each deduped annotated row, reconstruct (prefix, continuation, extracted answer),
classify the continuation channel mechanically, and cross-tab against Correct? and
Recovery Behavior.
"""
import csv, json, os, re, sys
csv.field_size_limit(10**9)

REPO = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo"
OUT = "/lfs/skampere2/0/eobbad/scratch/yee_audit/audit_extraction"
MODEL = "gpt-4-0314"
SUFFIX = " Therefore, the answer (arabic numerals) is"
MARKER = "A: Let's think step by step."

DATASETS = {
    "gsm8k": "test",
    "asdiv": "ASDiv",
    "awps": "MultiArith",
    "svamp": "SVAMP",
}
PERTS = ["random", "add1", "add101"]

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

def norm(s):
    return s.replace("\r\n", "\n").replace("\r", "\n")

def parse_nums(text):
    out = []
    for m in NUM_RE.finditer(text):
        t = m.group(0).replace(",", "")
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out

def close(a, b):
    return abs(a - b) < 1e-6 * max(1.0, abs(a), abs(b))

def load_baseline(path):
    """(bare problem) -> original CoT for MODEL rows."""
    base = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Model Name"] != MODEL or row.get("Prompt Style") != "sbs":
                continue
            fp = norm(row["Full Prompt"])
            i = fp.find(MARKER)
            if i < 0:
                continue
            cot = fp[i + len(MARKER):]
            if cot.endswith(SUFFIX):
                cot = cot[: -len(SUFFIX)]
            base[norm(row["Question"]).strip()] = cot
    return base

def orig_value_at_site(prefix, orig_cot):
    """Perturbation edits digits in place; text before the perturbed number matches
    the original CoT. Find divergence, back up to the number token containing it,
    parse the original number there. Returns (orig_val, ok)."""
    p, o = prefix, orig_cot
    n = min(len(p), len(o))
    i = 0
    while i < n and p[i] == o[i]:
        i += 1
    if i == 0:
        return None, False
    # back up to start of the digit token containing/preceding position i
    j = i
    while j > 0 and (o[j-1].isdigit() or o[j-1] in ".,"):
        j -= 1
    m = NUM_RE.match(o, j)
    if not m:
        # divergence may be right before a digit in o
        m = NUM_RE.match(o, i)
        if not m:
            return None, False
    try:
        return float(m.group(0).replace(",", "")), True
    except ValueError:
        return None, False

def main():
    os.makedirs(OUT, exist_ok=True)
    recs = []
    stats_diag = []
    for ds, stem in DATASETS.items():
        base = load_baseline(f"{REPO}/results/{ds}/{stem}.csv")
        for pert in PERTS:
            path = f"{REPO}/results/{ds}/{stem}_adjusted_position-calc_perturbation-{pert}_annotated.csv"
            if not os.path.exists(path):
                stats_diag.append(f"MISSING {path}")
                continue
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = [r for r in csv.DictReader(f)
                        if r.get("Model Name") == MODEL and r.get("Prompt Style") == "sbs"]
            # dedupe by Question, keep annotated (Correct? non-blank) row
            byq = {}
            for r in rows:
                q = norm(r["Question"])
                cur = byq.get(q)
                if cur is None:
                    byq[q] = r
                else:
                    ann_new = (r.get("Correct?") or "").strip() != ""
                    ann_old = (cur.get("Correct?") or "").strip() != ""
                    if ann_new and not ann_old:
                        byq[q] = r
            n_raw, n_uniq = len(rows), len(byq)
            n_parse_fail = 0
            for q, r in byq.items():
                fp = norm(r["Full Prompt"])
                if not q.startswith("Q: "):
                    n_parse_fail += 1
                    continue
                mi = q.find(MARKER)
                if mi < 0:
                    n_parse_fail += 1
                    continue
                problem = q[3:mi].strip()
                prefix = q[mi + len(MARKER):]
                # continuation = Full Prompt minus Question minus suffix
                if not fp.startswith(q):
                    n_parse_fail += 1
                    continue
                cont = fp[len(q):]
                had_suffix = cont.endswith(SUFFIX)
                if had_suffix:
                    cont = cont[: -len(SUFFIX)]
                # first-call join is q + ' ' + continuation
                joined_space = cont.startswith(" ")
                cont_body = cont[1:] if joined_space else cont
                cont_stripped = cont_body.strip()

                target_raw = (r.get("Target Answer") or "").strip()
                try:
                    target = float(target_raw.replace(",", ""))
                except ValueError:
                    target = None

                pnums = parse_nums(prefix)
                perturbed = pnums[-1] if pnums else None
                cnums = parse_nums(cont_body)
                answer_nums = parse_nums(r.get("Answer") or "")

                orig_val, orig_ok = (None, False)
                ocot = base.get(problem)
                if ocot is not None:
                    orig_val, orig_ok = orig_value_at_site(prefix, ocot)

                has_target = target is not None and any(close(x, target) for x in cnums)
                last_num = cnums[-1] if cnums else None
                last_is_target = (last_num is not None and target is not None
                                  and close(last_num, target))
                has_perturbed = perturbed is not None and any(close(x, perturbed) for x in cnums)
                has_original = (orig_val is not None
                                and any(close(x, orig_val) for x in cnums))
                empty = len(cont_stripped) == 0
                trivial = len(cont_stripped) <= 10
                no_nums = len(cnums) == 0
                starts_digit = bool(cont_body[:1].isdigit() or
                                    (cont_body[:1] == "." and cont_body[1:2].isdigit()))
                # scorer replication: first number in Answer == target
                scorer_correct = (target is not None and len(answer_nums) > 0
                                  and close(answer_nums[0], target))
                correct = (r.get("Correct?") or "").strip()
                recs.append({
                    "dataset": ds, "perturbation": pert,
                    "problem_head": problem[:60].replace("\n", " "),
                    "target": target_raw,
                    "perturbed_val": perturbed, "orig_val": orig_val,
                    "orig_align_ok": orig_ok,
                    "cont_len": len(cont_stripped), "cont_n_nums": len(cnums),
                    "cont_empty": empty, "cont_trivial": trivial,
                    "cont_no_nums": no_nums,
                    "cont_starts_digit": starts_digit,
                    "cont_has_target": has_target,
                    "cont_last_num_is_target": last_is_target,
                    "cont_has_perturbed": has_perturbed,
                    "cont_has_original": has_original,
                    "had_suffix": had_suffix,
                    "answer": (r.get("Answer") or "").strip()[:60],
                    "scorer_correct": scorer_correct,
                    "correct": correct,
                    "error_type": (r.get("Error Type") or "").strip(),
                    "recovery_behavior": (r.get("Recovery Behavior") or "").strip(),
                    "cont_head": cont_body[:120].replace("\n", " | "),
                    "cont_tail": cont_body[-120:].replace("\n", " | "),
                })
            stats_diag.append(
                f"{ds}/{pert}: raw_gpt4_rows={n_raw} unique={n_uniq} parse_fail={n_parse_fail}")

    # write per-record CSV
    outcsv = f"{OUT}/calc_gpt4_continuation_classification.csv"
    with open(outcsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    # ---------- summaries ----------
    def pct(a, b):
        return f"{100.0*a/b:.1f}%" if b else "n/a"

    print("DIAG:")
    for d in stats_diag:
        print(" ", d)
    print()

    print("="*100)
    print("PER (dataset, perturbation), gpt-4-0314, position-calc, DENOMINATOR = deduped")
    print("annotated rows with Error Type == 'calculation' (the published denominator).")
    print("="*100)
    header = ("ds/pert", "n", "recov", "rate", "cont_has_tgt", "extr_only",
              "extr_rate", "corr_rate", "empty&corr", "err_fwd&corr")
    print(("{:<14}{:>5}{:>7}{:>8}{:>13}{:>10}{:>10}{:>10}{:>11}{:>13}").format(*header))
    agg = {}
    for ds in DATASETS:
        for pert in PERTS:
            sub = [r for r in recs if r["dataset"] == ds and r["perturbation"] == pert
                   and r["error_type"] == "calculation" and r["correct"] in ("True", "False")]
            n = len(sub)
            cor = [r for r in sub if r["correct"] == "True"]
            nc = len(cor)
            has_t = [r for r in cor if r["cont_has_target"]]
            extr_only = [r for r in cor if not r["cont_has_target"]]
            empty_cor = [r for r in cor if r["cont_empty"]]
            errfwd = [r for r in extr_only if r["cont_has_perturbed"]]
            agg[(ds, pert)] = (n, nc, len(has_t), len(extr_only), len(empty_cor), len(errfwd))
            print(("{:<14}{:>5}{:>7}{:>8}{:>13}{:>10}{:>10}{:>10}{:>11}{:>13}").format(
                f"{ds}/{pert}", n, nc, pct(nc, n), len(has_t), len(extr_only),
                pct(len(extr_only), n), pct(len(has_t), n), len(empty_cor), len(errfwd)))
    print()
    print("Columns: recov = Correct?=True count; rate = published-style recovery rate;")
    print("cont_has_tgt = correct AND target value appears in continuation;")
    print("extr_only = correct but target NOWHERE in continuation (extraction-pass recovery);")
    print("corr_rate = corrected recovery rate counting only continuation-channel recoveries;")
    print("empty&corr = correct with EMPTY continuation; err_fwd&corr = extraction-only recoveries")
    print("whose continuation still contains the perturbed value (error carried forward).")
    print()

    print("="*100)
    print("RECOVERY BEHAVIOR cross-tab (correct rows only, Error Type=calculation, all ds/pert pooled)")
    print("="*100)
    from collections import Counter, defaultdict
    tab = defaultdict(Counter)
    for r in recs:
        if r["error_type"] != "calculation" or r["correct"] != "True":
            continue
        rb = r["recovery_behavior"] or "(blank)"
        ch = ("has_target" if r["cont_has_target"] else
              ("empty" if r["cont_empty"] else
               ("no_nums" if r["cont_no_nums"] else "nums_no_target")))
        tab[rb][ch] += 1
        tab[rb]["_total"] += 1
        if r["cont_has_perturbed"] and not r["cont_has_target"]:
            tab[rb]["err_fwd_no_tgt"] += 1
    print("{:<32}{:>7}{:>12}{:>8}{:>9}{:>16}{:>16}".format(
        "Recovery Behavior", "total", "has_target", "empty", "no_nums",
        "nums_no_target", "err_fwd_no_tgt"))
    for rb, c in sorted(tab.items(), key=lambda kv: -kv[1]["_total"]):
        print("{:<32}{:>7}{:>12}{:>8}{:>9}{:>16}{:>16}".format(
            rb[:32], c["_total"], c["has_target"], c["empty"], c["no_nums"],
            c["nums_no_target"], c["err_fwd_no_tgt"]))
    print()

    print("="*100)
    print("SCORER/annotation agreement + misc diagnostics (all deduped calc rows)")
    print("="*100)
    sub = [r for r in recs if r["correct"] in ("True", "False")]
    agree = sum(1 for r in sub if (r["correct"] == "True") == r["scorer_correct"])
    print(f"Correct? vs number_scorer(Answer,Target) agreement: {agree}/{len(sub)}")
    sd = [r for r in sub if r["cont_starts_digit"]]
    print(f"continuation starts with digit (boundary artifact flag): {len(sd)}/{len(sub)}")
    nosfx = [r for r in recs if not r["had_suffix"]]
    print(f"rows whose Full Prompt lacked the extraction suffix: {len(nosfx)}/{len(recs)}")
    align = [r for r in recs if r["orig_align_ok"]]
    print(f"rows with original-value alignment vs baseline CoT: {len(align)}/{len(recs)}")
    print()

    print("="*100)
    print("EXAMPLES: correct but target nowhere in continuation (extraction-pass recovery candidates)")
    print("="*100)
    shown = 0
    for r in recs:
        if r["error_type"] == "calculation" and r["correct"] == "True" \
           and not r["cont_has_target"] and shown < 12:
            print(f"--- {r['dataset']}/{r['perturbation']} target={r['target']} "
                  f"perturbed={r['perturbed_val']} orig={r['orig_val']} rb={r['recovery_behavior']!r}")
            print(f"    cont_len={r['cont_len']} nums={r['cont_n_nums']} "
                  f"empty={r['cont_empty']} answer={r['answer']!r}")
            print(f"    HEAD: {r['cont_head']!r}")
            print(f"    TAIL: {r['cont_tail']!r}")
            shown += 1
    print(f"\nPer-record CSV written: {outcsv} ({len(recs)} rows)")

if __name__ == "__main__":
    main()
