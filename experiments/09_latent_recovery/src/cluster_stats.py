"""Problem-clustered logistic robustness checks for the workshop paper.

This script is deliberately dependency-free: the environment used for paper edits may
not have numpy/statsmodels, but the statistic we need is small enough to fit directly.
It reports fixed-effect logistic odds ratios with problem-cluster bootstrap intervals.
"""
import json
import math
import os
import random
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "results", "cluster_stats.json")

FAMILIES = [
    ("benign", "validated_paraphrase.jsonl"),
    ("off_path", "validated.jsonl"),
    ("distractor", "validated_distractor.jsonl"),
    ("contradiction", "validated_contradiction.jsonl"),
    ("one_hop_false", "validated_negstep.jsonl"),
    ("global_false", "validated_falsehood.jsonl"),
]

FEATURES = [
    "mid",
    "late",
    "off_path",
    "distractor",
    "contradiction",
    "one_hop_false",
    "global_false",
]


def read_rows():
    rows = []
    for family, name in FAMILIES:
        path = os.path.join(BASE, "results", name)
        with open(path) as fh:
            for line in fh:
                r = json.loads(line)
                rows.append(
                    {
                        "id": r["id"],
                        "point": r["point"],
                        "family": family,
                        "valid": int(r["class"] == "valid_rederivation"),
                        "doubt": int(r.get("acknowledged", False)),
                        "injection_dependent": int(r["class"] == "poisoned"),
                    }
                )
    return rows


def xvec(r):
    return [
        1.0,
        float(r["point"] == "mid"),
        float(r["point"] == "late"),
        float(r["family"] == "off_path"),
        float(r["family"] == "distractor"),
        float(r["family"] == "contradiction"),
        float(r["family"] == "one_hop_false"),
        float(r["family"] == "global_false"),
    ]


def solve_linear(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda i: abs(m[i][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular matrix")
        if piv != col:
            m[col], m[piv] = m[piv], m[col]
        div = m[col][col]
        for j in range(col, n + 1):
            m[col][j] /= div
        for i in range(n):
            if i == col:
                continue
            fac = m[i][col]
            if fac == 0:
                continue
            for j in range(col, n + 1):
                m[i][j] -= fac * m[col][j]
    return [m[i][n] for i in range(n)]


def sigmoid(z):
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def fit_logit(rows, outcome, cluster_weights=None, ridge=1e-5, max_iter=60):
    p = 1 + len(FEATURES)
    beta = [0.0] * p
    for _ in range(max_iter):
        grad = [0.0] * p
        hess = [[0.0] * p for _ in range(p)]
        for r in rows:
            w = 1.0 if cluster_weights is None else float(cluster_weights.get(r["id"], 0))
            if w <= 0:
                continue
            x = xvec(r)
            mu = sigmoid(sum(beta[j] * x[j] for j in range(p)))
            y = r[outcome]
            for j in range(p):
                grad[j] += w * x[j] * (y - mu)
            s = w * mu * (1.0 - mu)
            for j in range(p):
                xj = x[j]
                if xj == 0:
                    continue
                for k in range(p):
                    hess[j][k] += s * xj * x[k]
        for j in range(p):
            hess[j][j] += ridge
        step = solve_linear(hess, grad)
        for j in range(p):
            beta[j] += step[j]
        if max(abs(s) for s in step) < 1e-7:
            break
    return beta


def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def summarize_logit(rows, outcome, boot=400, seed=0):
    rng = random.Random(seed)
    ids = sorted({r["id"] for r in rows})
    beta = fit_logit(rows, outcome)
    boots = {name: [] for name in FEATURES}
    for _ in range(boot):
        counts = Counter(rng.choice(ids) for _ in ids)
        try:
            b = fit_logit(rows, outcome, counts)
        except (ValueError, OverflowError):
            continue
        for i, name in enumerate(FEATURES, start=1):
            boots[name].append(b[i])
    out = {}
    for i, name in enumerate(FEATURES, start=1):
        ci = (percentile(boots[name], 0.025), percentile(boots[name], 0.975))
        out[name] = {
            "log_or": round(beta[i], 3),
            "or": round(math.exp(beta[i]), 3),
            "ci95_or": [round(math.exp(ci[0]), 3), round(math.exp(ci[1]), 3)],
            "boot_n": len(boots[name]),
        }
    return out


def summarize_rates(rows):
    cells = {}
    for fam in [f[0] for f in FAMILIES]:
        sub = [r for r in rows if r["family"] == fam]
        n = len(sub)
        cells[fam] = {
            "n": n,
            "valid": round(sum(r["valid"] for r in sub) / n, 3),
            "doubt": round(sum(r["doubt"] for r in sub) / n, 3),
            "injection_dependent": round(sum(r["injection_dependent"] for r in sub) / n, 3),
        }
    return cells


def main():
    rows = read_rows()
    out = {
        "note": (
            "Fixed-effect logistic models over Qwen2.5-7B validation rows. "
            "Reference condition is benign paraphrase at early injection; intervals "
            "are problem-cluster bootstraps over problem ids."
        ),
        "rates_pooled": summarize_rates(rows),
        "valid_rederivation": summarize_logit(rows, "valid"),
        "verbalized_doubt": summarize_logit(rows, "doubt"),
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
