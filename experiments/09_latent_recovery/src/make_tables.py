"""Emit LaTeX table rows + key stats for paper v2 directly from validated summaries."""
import os, json, math
from common import RESULTS

def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval

FAMS = [("benign paraphrase", "_paraphrase"), ("true interruption (wrong-cat)", ""),
        ("distractor rule", "_distractor"), ("contradiction (estab.)", "_contradiction"),
        ("falsehood (off-path cat.)", "_falsehood"), ("falsehood (negated step)", "_negstep")]

def main():
    rows, store = [], {}
    for label, suf in FAMS:
        p_ = os.path.join(RESULTS, f"validated_summary{suf}.json")
        if not os.path.exists(p_):
            continue
        d = json.load(open(p_))
        for p in ["early", "mid", "late"]:
            c = d[p]
            n = c["n"]
            k = c.get("valid_rederivation", 0)
            lo, hi = wilson(k, n)
            store[(label, p)] = (k, n, c)
            rows.append("%s & %s & %d & %.2f [%.2f,%.2f] & %.2f & %.2f & %.2f \\\\" % (
                label, p, n, k / n, lo, hi,
                c.get("poisoned", 0) / n, c.get("parroted", 0) / n,
                c.get("acknowledged", 0) / n))
    print("% family & injection & n & valid [CI] & poisoned & parroted & doubt")
    for r in rows:
        print(r)
    print()
    # key contrasts
    def get(label, p):
        k, n, _ = store[(label, p)]
        return k, n
    for p in ["early", "mid", "late"]:
        for a, b in [("falsehood (negated step)", "benign paraphrase"),
                     ("falsehood (negated step)", "contradiction (estab.)"),
                     ("falsehood (negated step)", "true interruption (wrong-cat)")]:
            if (a, p) in store and (b, p) in store:
                z, pv = two_prop_z(*get(a, p), *get(b, p))
                print(f"% {p}: {a} vs {b}: z={z:.2f} p={pv:.5f}")
    # pooled poisoning for negstep
    if ("falsehood (negated step)", "early") in store:
        kp = sum(store[("falsehood (negated step)", p)][2].get("poisoned", 0) for p in ["early", "mid", "late"])
        np_ = sum(store[("falsehood (negated step)", p)][1] for p in ["early", "mid", "late"])
        lo, hi = wilson(kp, np_)
        print(f"% negstep pooled poisoned: {kp}/{np_} = {kp/np_:.3f} [{lo:.3f},{hi:.3f}]")

if __name__ == "__main__":
    main()
