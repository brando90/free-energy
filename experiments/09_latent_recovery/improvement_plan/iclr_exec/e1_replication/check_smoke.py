#!/usr/bin/env python3
"""E1 smoke acceptance checks. Fails (rc!=0) if the pipeline did not exercise
gold->plant->continue->validate cleanly on the 1.5B model."""
import argparse
import json
import os
import sys
from collections import Counter


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    o = a.out
    cohort = read_jsonl(os.path.join(o, "gold_cohort.jsonl"))
    validated = read_jsonl(os.path.join(o, "validated_outputs.jsonl"))

    problems = []
    ok = True

    gen = [c for c in cohort if c.get("gold_generated")]
    elig = [c for c in cohort if c.get("eligible")]
    n_gen, n_elig = len(gen), len(elig)
    attrition = 1 - (n_elig / n_gen) if n_gen else 1.0
    print("gold: generated=%d eligible=%d attrition=%.3f" % (n_gen, n_elig, attrition))
    if n_gen == 0:
        problems.append("no gold rollouts generated"); ok = False
    if n_elig == 0:
        problems.append("gold attrition total: 0 eligible (pipeline unusable)"); ok = False
    if n_gen and attrition <= 0.0:
        # nonzero attrition expected for a weak model; 0 is suspicious but not fatal
        print("WARN: gold attrition is exactly 0 (unexpected for 1.5B, not fatal)")

    if not validated:
        problems.append("no validated perturbed rows"); ok = False
    else:
        cls = Counter(r.get("class") for r in validated)
        unparsed = sum(1 for r in validated if r.get("class") in
                       ("unparsed", "parse_failure") or r.get("unparsed"))
        print("validated rows=%d class_dist=%s" % (len(validated), dict(cls)))
        # validator must PARSE most rows (format compatibility check)
        parsed_frac = 1 - (unparsed / len(validated))
        print("parsed_fraction=%.3f" % parsed_frac)
        need = {"valid_recovery", "class", "doubt"}
        missing = [k for k in need if k not in validated[0]]
        if missing:
            problems.append("validator output missing fields: %s" % missing); ok = False
        if parsed_frac < 0.5:
            problems.append("validator parsed <50%% of 1.5B outputs (template issue)"); ok = False
        # stated-complement DV must be computable
        sc = [r for r in validated if "stated_complement" in r or "closure_validation" in r]
        print("rows with stated_complement/closure fields=%d" % len(sc))

    print("\nSMOKE:", "PASS" if ok else "FAIL")
    for p in problems:
        print("  -", p)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
