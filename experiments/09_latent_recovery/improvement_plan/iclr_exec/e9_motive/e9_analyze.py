"""E9 analysis: the registered contrasts (E9_REGISTRATION section 5).

Primary   : paired cluster bootstrap of  licensed_strict(inert) - licensed_strict(usable)
            at LOW verification; reject H0 iff one-sided 95% lower bound (BCa + percentile
            both reported) > MES = 0.05.  Anchor to beat = 0.164.
Null (5.3): equivalence/upper-bound -- 'artifact' only if the 95% CI EXCLUDES 0.164.
            Otherwise a lower bound <= MES with CI admitting an effect > MES => INCONCLUSIVE.
S1 (Holm) : reuse usable - inert > 0 (directional).
S2 (Holm) : |unlicensed inert - usable| within +/-0.05 (TOST equivalence).
S3        : usefulness x verification interaction (motive: gap survives high verification;
            compliance-interference: gap shrinks). In-family only if G-D2 = in-family.
Also      : surprise-conditioned gap (2.2), effort-conditioned gap (2.3), active-path
            overlap by arm (2.4), power curve recompute (5.4) -> power_curve.json.

Denominators: PRIMARY = engaged rollouts (4.4). ITT sensitivity = all rollouts.

Usage: python e9_analyze.py --run-dir <dir> [--gd2-in-family] [--itt]
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

SEED = 20260724
NBOOT = 5000
MES = 0.05
ANCHOR = 0.164


def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------ per-pair rate assembly

def pair_rates(rows, dv, verification="low", itt=False):
    """Return {world_id: {'usable': rate, 'inert': rate, 'n_usable':, 'n_inert':}}.
    Denominator = engaged rollouts unless itt (then all rollouts)."""
    agg = defaultdict(lambda: {"usable": [0, 0], "inert": [0, 0]})
    for r in rows:
        if verification is not None and r.get("verification") != verification:
            continue
        if not itt and not r.get("engaged"):
            continue
        arm = r["arm"]
        if arm not in ("usable", "inert"):
            continue
        agg[r["world_id"]][arm][0] += int(r.get(dv, 0))
        agg[r["world_id"]][arm][1] += 1
    out = {}
    for wid, a in agg.items():
        if a["usable"][1] == 0 or a["inert"][1] == 0:
            continue  # pair needs both arms with >=1 eligible rollout
        out[wid] = {
            "usable": a["usable"][0] / a["usable"][1],
            "inert": a["inert"][0] / a["inert"][1],
            "n_usable": a["usable"][1], "n_inert": a["inert"][1],
        }
    return out


# --------------------------------------------------------- cluster bootstrap

def _boot_diffs(diffs, nboot, seed):
    rng = random.Random(seed)
    K = len(diffs)
    boots = []
    for _ in range(nboot):
        s = sum(diffs[rng.randrange(K)] for _ in range(K))
        boots.append(s / K)
    boots.sort()
    return boots


def paired_bootstrap(pairs, key_hi="inert", key_lo="usable", nboot=NBOOT, seed=SEED):
    """One-sided lower bound of mean(hi - lo) over pairs, percentile + BCa."""
    diffs = [p[key_hi] - p[key_lo] for p in pairs.values()]
    if not diffs:
        return None
    point = sum(diffs) / len(diffs)
    boots = _boot_diffs(diffs, nboot, seed)
    lo_pct = boots[int(0.05 * len(boots))]                # one-sided 95% lower
    hi_pct = boots[int(0.95 * len(boots)) - 1]
    # BCa bias-correction
    n_less = sum(1 for b in boots if b < point)
    z0 = _invnorm((n_less + 0.5) / len(boots)) if 0 < n_less < len(boots) else 0.0
    jack = []
    d = list(diffs)
    for i in range(len(d)):
        m = (sum(d) - d[i]) / (len(d) - 1) if len(d) > 1 else d[i]
        jack.append(m)
    jbar = sum(jack) / len(jack)
    num = sum((jbar - x) ** 3 for x in jack)
    den = 6 * (sum((jbar - x) ** 2 for x in jack) ** 1.5) or 1e-12
    acc = num / den
    def bca_q(alpha):
        za = _invnorm(alpha)
        adj = z0 + (z0 + za) / (1 - acc * (z0 + za))
        return _norm_cdf(adj)
    q_lo = bca_q(0.05)
    idx_lo = min(len(boots) - 1, max(0, int(q_lo * len(boots))))
    return {
        "n_pairs": len(diffs), "point": point,
        "lower_95_pct": lo_pct, "upper_95_pct": hi_pct,
        "lower_95_bca": boots[idx_lo],
        "reject_H0_at_MES": lo_pct > MES,
        "reject_H0_at_MES_bca": boots[idx_lo] > MES,
        "n_informative": sum(1 for x in diffs if x != 0),
    }


def equivalence_vs_anchor(pairs, key_hi="inert", key_lo="usable", nboot=NBOOT, seed=SEED, anchor=ANCHOR):
    """Null branch (5.3): 'artifact' licensed only if the 95% CI EXCLUDES the anchor
    (upper < anchor). Returns the two-sided CI + verdict scaffold."""
    diffs = [p[key_hi] - p[key_lo] for p in pairs.values()]
    if not diffs:
        return None
    boots = _boot_diffs(diffs, nboot, seed)
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    return {
        "ci95_two_sided": [lo, hi],
        "excludes_anchor_0164": hi < anchor,           # licenses 'artifact' (branch b)
        "ci_admits_effect_above_MES": hi > MES,        # underpowered => INCONCLUSIVE
    }


def tost(pairs, dv_hi="inert", dv_lo="usable", margin=0.05, nboot=NBOOT, seed=SEED):
    """S2 equivalence: |hi - lo| within +/- margin iff CI within (-margin, margin)."""
    diffs = [p[dv_hi] - p[dv_lo] for p in pairs.values()]
    if not diffs:
        return None
    boots = _boot_diffs(diffs, nboot, seed)
    lo = boots[int(0.05 * len(boots))]
    hi = boots[int(0.95 * len(boots)) - 1]
    return {"ci90": [lo, hi], "equivalent": (lo > -margin and hi < margin), "margin": margin}


# ------------------------------------------------------------- normal helpers

def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _invnorm(p):
    # Acklam's rational approximation
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


# ------------------------------------------------------------- power recompute

def power_curve(pairs, deltas=(0.164, 0.12, 0.10, 0.08), ns=(150, 300, 450),
                nsim=150, nboot=250, seed=SEED):
    """5.4: recompute power against the ACTUAL decision rule (bootstrap lower
    bound > MES) by resampling the piloted per-pair inert/usable rates, shifting
    the inert arm to induce each target delta. Writes an indicative curve."""
    if not pairs:
        return {"note": "no pilot pairs yet; run generation first"}
    usable = [p["usable"] for p in pairs.values()]
    inert = [p["inert"] for p in pairs.values()]
    base_delta = (sum(inert) / len(inert)) - (sum(usable) / len(usable))
    rng = random.Random(seed)
    curve = {}
    for delta in deltas:
        shift = delta - base_delta
        for n in ns:
            rejects = 0
            for _ in range(nsim):
                sim = []
                for _ in range(n):
                    j = rng.randrange(len(usable))
                    sim.append((inert[j] + shift) - usable[j])
                bt = sorted(sum(sim[rng.randrange(n)] for _ in range(n)) / n for _ in range(nboot))
                if bt[int(0.05 * len(bt))] > MES:
                    rejects += 1
            curve["delta=%.3f,n=%d" % (delta, n)] = round(rejects / nsim, 3)
    curve["base_delta_in_pilot"] = round(base_delta, 4)
    return curve


# ------------------------------------------------------------------- driver

def summarize(run_dir, gd2_in_family=False, itt=False):
    rows = read_jsonl(os.path.join(run_dir, "scored_rows.jsonl"))
    res = {"meta": {"nboot": NBOOT, "seed": SEED, "MES": MES, "anchor": ANCHOR,
                    "denominator": "ITT_all_rollouts" if itt else "engaged_rollouts",
                    "gd2_in_family": gd2_in_family}}
    lic_low = pair_rates(rows, "licensed_strict", "low", itt)
    res["primary_licensed_gap_low_verification"] = paired_bootstrap(lic_low)
    res["null_equivalence_vs_anchor"] = equivalence_vs_anchor(lic_low)
    res["S1_reuse_gap"] = paired_bootstrap(pair_rates(rows, "reuse_poisoned", "low", itt),
                                           key_hi="usable", key_lo="inert")
    res["S2_unlicensed_TOST"] = tost(pair_rates(rows, "unlicensed_rejection", "low", itt))
    # S3 verification interaction (gap_low - gap_high)
    lic_high = pair_rates(rows, "licensed_strict", "high", itt)
    common = set(lic_low) & set(lic_high)
    if common:
        gl = [lic_low[w]["inert"] - lic_low[w]["usable"] for w in common]
        gh = [lic_high[w]["inert"] - lic_high[w]["usable"] for w in common]
        inter = {w: {"usable": lic_high[w]["usable"] + lic_low[w]["usable"],
                     "inert": lic_high[w]["inert"] + lic_low[w]["inert"]} for w in common}
        # interaction = mean(gap_low) - mean(gap_high); >0 => gap shrinks under high verif
        res["S3_verification_interaction"] = {
            "gap_low": sum(gl) / len(gl), "gap_high": sum(gh) / len(gh),
            "interaction_low_minus_high": (sum(gl) / len(gl)) - (sum(gh) / len(gh)),
            "in_confirmatory_family": gd2_in_family, "n_pairs": len(common),
            "reading": "gap SURVIVES high verif => motive; gap SHRINKS => compliance-interference",
        }
    # descriptive: active-path overlap + realized effort by arm (2.3/2.4)
    by_arm = defaultdict(lambda: {"overlap": [], "nsteps": [], "surprisal": []})
    for r in rows:
        if r.get("verification") != "low":
            continue
        by_arm[r["arm"]]["overlap"].append(r.get("active_path_overlap", 0))
        by_arm[r["arm"]]["nsteps"].append(r.get("realized_n_steps", 0))
        if r.get("plant_surprisal") is not None:
            by_arm[r["arm"]]["surprisal"].append(r["plant_surprisal"])
    res["descriptive_by_arm"] = {
        arm: {"mean_active_path_overlap": round(sum(v["overlap"]) / len(v["overlap"]), 4) if v["overlap"] else None,
              "mean_realized_n_steps": round(sum(v["nsteps"]) / len(v["nsteps"]), 3) if v["nsteps"] else None,
              "mean_plant_surprisal": round(sum(v["surprisal"]) / len(v["surprisal"]), 4) if v["surprisal"] else None}
        for arm, v in by_arm.items()}
    res["power_curve"] = power_curve(lic_low)
    with open(os.path.join(run_dir, "e9_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(run_dir, "power_curve.json"), "w") as f:
        json.dump(res["power_curve"], f, indent=2)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gd2-in-family", action="store_true")
    ap.add_argument("--itt", action="store_true")
    args = ap.parse_args(argv)
    res = summarize(args.run_dir, args.gd2_in_family, args.itt)
    print(json.dumps({k: res[k] for k in
                      ("primary_licensed_gap_low_verification", "null_equivalence_vs_anchor",
                       "S1_reuse_gap", "S2_unlicensed_TOST")
                      if res.get(k) is not None}, indent=2))


if __name__ == "__main__":
    main()
