"""EXPE full-run registered adjudications (PLAN2 v1.1 section B: H1', H5', H6).

Reads the completed results/EXPE_EVIDENCE_MOVER/ directory (validated outputs,
manifest, judge_v2 verdicts, runner summary) and writes:

  hypothesis_verdicts.json   {test_id, statistic, p_raw, estimate, ci95, ns, ...}
  EXPE_FULL_REPORT.md        full-schema report (adjudications + descriptives)

DV registry (v1.1 section B):
  PRIMARY   stated_complement_of_falsehood  (strict stated-complement)
  SECONDARY judge explicit_rejection        (doubt_judge_v2, 32B)
  DESCRIPTIVE verbalized doubt (lexical regex + judge doubt), closure-valid,
              derivational inj-dep (validator 'poisoned'), echo ('parroted'),
              unparsed, re-solve (MC-6), vocab tiers (C.1)

Tests:
  H1'  Cochran-Armitage trend, stated-complement over d in {1,2,3} (mid),
       decreasing; clustering on problem handled by a within-problem
       permutation test of the trend statistic (d-labels permuted within
       problem; 20,000 draws); classical unclustered CA Z reported alongside.
  H5'  conjunction, McNemar within problem (mid): stated-complement
       REFUTING_d1 > FREQ_MATCHED_NONREFUTING and > TEMPLATE_IRRELEVANT;
       exact one-sided binomial on discordant pairs; slot p = max(components)
       (intersection-union, MOD-7 convention).
  H6   derivational inj-dep REFUTING_d1 < BASELINE_FILLER; exact one-sided
       McNemar within problem. Primary scope = mid (one pair per problem);
       early/late and pooled reported as sensitivity.

Holm is NOT applied here: raw p-values feed the coordinator's m=6 family
(these three slots + EXPD's H2'/H3'/H4).
"""
import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

B_PERM = 20000
B_BOOT = 4000
SEED = 0


def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def binom_sf_onesided(k, n):
    """P(X >= k), X ~ Bin(n, 0.5); exact."""
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n))


def percentile(vals, q):
    vals = sorted(vals)
    if not vals:
        return None
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def rate(rows, key):
    vals = [key(r) for r in rows]
    vals = [v for v in vals if v is not None]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


# ---------------------------------------------------------------- DV getters

def dv_stated(r):
    return int(bool(r.get("stated_complement_of_falsehood")))


def dv_injdep(r):
    v = r.get("injection_dependent")
    return None if v is None else int(bool(v))


def make_dv_judge(judge_by_rid):
    def dv(r):
        v = judge_by_rid.get(r["run_id"])
        return None if v is None else int(bool(v.get("explicit_rejection")))
    return dv


# ---------------------------------------------------------------- H1' trend

def ca_trend_clustered(rows_by_d, dv, seed=SEED, n_perm=B_PERM, n_boot=B_BOOT):
    """rows_by_d: {1: rows, 2: rows, 3: rows} (mid, one row/problem/d).
    Returns dict with slope, classical CA Z, permutation p (one-sided,
    decreasing), cluster-bootstrap CI on the slope, per-d rates."""
    ds = sorted(rows_by_d)
    per_problem = defaultdict(dict)
    for d in ds:
        for r in rows_by_d[d]:
            v = dv(r)
            if v is not None:
                per_problem[r["problem_id"]][d] = v
    complete = {pid: yd for pid, yd in per_problem.items() if len(yd) == len(ds)}
    pids = sorted(complete)
    n_prob = len(pids)

    # pooled slope: cov(d, y) / var(d) over all rows of complete problems
    pts = [(d, complete[pid][d]) for pid in pids for d in ds]
    dbar = sum(d for d, _ in pts) / len(pts)
    ybar = sum(y for _, y in pts) / len(pts)
    sdd = sum((d - dbar) ** 2 for d, _ in pts)
    slope = sum((d - dbar) * (y - ybar) for d, y in pts) / sdd

    # classical (unclustered) Cochran-Armitage Z, scores = d
    n_d = {d: len([1 for pid in pids]) for d in ds}
    r_d = {d: sum(complete[pid][d] for pid in pids) for d in ds}
    N = sum(n_d.values())
    R = sum(r_d.values())
    pbar = R / N
    T = sum(d * r_d[d] for d in ds) - R * sum(d * n_d[d] for d in ds) / N
    var = pbar * (1 - pbar) * (sum(d * d * n_d[d] for d in ds)
                               - (sum(d * n_d[d] for d in ds)) ** 2 / N)
    z_ca = T / math.sqrt(var) if var > 0 else 0.0
    p_ca_unclustered = 0.5 * math.erfc(-z_ca / math.sqrt(2))  # P(Z <= z_ca)

    # within-problem permutation of d labels; statistic = weighted trend sum
    w = {d: d - dbar for d in ds}
    obs = sum(w[d] * complete[pid][d] for pid in pids for d in ds)
    rng = random.Random(seed)
    triples = [[complete[pid][d] for d in ds] for pid in pids]
    wlist = [w[d] for d in ds]
    perms = []
    import itertools
    all_perm = list(itertools.permutations(range(len(ds))))
    le = 0
    for _ in range(n_perm):
        t = 0.0
        for tri in triples:
            pm = all_perm[rng.randrange(len(all_perm))]
            t += sum(wlist[j] * tri[pm[j]] for j in range(len(ds)))
        if t <= obs + 1e-12:
            le += 1
    p_perm = (1 + le) / (n_perm + 1)

    # cluster bootstrap CI on the slope (resample problems)
    boots = []
    for _ in range(n_boot):
        sample = [pids[rng.randrange(n_prob)] for _ in range(n_prob)]
        pts_b = [(d, complete[pid][d]) for pid in sample for d in ds]
        db = sum(d for d, _ in pts_b) / len(pts_b)
        yb = sum(y for _, y in pts_b) / len(pts_b)
        sdd_b = sum((d - db) ** 2 for d, _ in pts_b)
        if sdd_b == 0:
            continue
        boots.append(sum((d - db) * (y - yb) for d, y in pts_b) / sdd_b)
    return {
        "n_problems_complete": n_prob,
        "n_rows": N,
        "rates_by_d": {str(d): round(r_d[d] / n_d[d], 4) for d in ds},
        "counts_by_d": {str(d): [r_d[d], n_d[d]] for d in ds},
        "slope_per_hop": round(slope, 5),
        "slope_ci95_cluster_bootstrap": [round(percentile(boots, 0.025), 5),
                                         round(percentile(boots, 0.975), 5)],
        "ca_z_unclustered": round(z_ca, 4),
        "ca_p_unclustered_one_sided_decreasing": float(f"{p_ca_unclustered:.6g}"),
        "perm_stat_observed": round(obs, 4),
        "p_perm_one_sided_decreasing": float(f"{p_perm:.6g}"),
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------- McNemar

def mcnemar_onesided(rows, arm_hi, arm_lo, dv, direction, positions=None,
                     seed=SEED, n_boot=B_BOOT):
    """Exact one-sided McNemar within problem on paired (arm_hi, arm_lo) rows.
    direction='hi_gt_lo': H1 says dv(arm_hi) > dv(arm_lo).
    direction='hi_lt_lo': H1 says dv(arm_hi) < dv(arm_lo).
    positions: restrict to these injection positions (None = all)."""
    by_key = defaultdict(dict)
    for r in rows:
        if positions and r["injection_position"] not in positions:
            continue
        if r["arm"] in (arm_hi, arm_lo):
            by_key[(r["problem_id"], r["injection_position"])][r["arm"]] = r
    pairs = []
    for (pid, pos), d in sorted(by_key.items()):
        if arm_hi in d and arm_lo in d:
            a, b = dv(d[arm_hi]), dv(d[arm_lo])
            if a is not None and b is not None:
                pairs.append((pid, a, b))
    n = len(pairs)
    if n == 0:
        return {"n_pairs": 0}
    hi1lo0 = sum(1 for _, a, b in pairs if a == 1 and b == 0)
    hi0lo1 = sum(1 for _, a, b in pairs if a == 0 and b == 1)
    if direction == "hi_gt_lo":
        p = binom_sf_onesided(hi1lo0, hi1lo0 + hi0lo1)
    else:
        p = binom_sf_onesided(hi0lo1, hi1lo0 + hi0lo1)
    diff = sum(a - b for _, a, b in pairs) / n
    # cluster bootstrap over problems for the paired difference
    by_pid = defaultdict(list)
    for pid, a, b in pairs:
        by_pid[pid].append(a - b)
    pids = sorted(by_pid)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        vals = []
        for pid in [pids[rng.randrange(len(pids))] for _ in pids]:
            vals.extend(by_pid[pid])
        boots.append(sum(vals) / len(vals))
    return {
        "n_pairs": n, "n_problems": len(pids),
        "rate_hi": round(sum(a for _, a, _ in pairs) / n, 4),
        "rate_lo": round(sum(b for _, _, b in pairs) / n, 4),
        "diff_hi_minus_lo": round(diff, 4),
        "diff_ci95_cluster_bootstrap": [round(percentile(boots, 0.025), 4),
                                        round(percentile(boots, 0.975), 4)],
        "discordant_hi1_lo0": hi1lo0, "discordant_hi0_lo1": hi0lo1,
        "p_exact_one_sided": float(f"{p:.6g}"),
        "direction": direction,
    }


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu-seconds-generate", type=float, default=None)
    ap.add_argument("--gpu-seconds-judge", type=float, default=None)
    args = ap.parse_args()
    OUT = args.out_dir

    rows = read_jsonl(os.path.join(OUT, "validated_outputs.jsonl"))
    manifest = read_jsonl(os.path.join(OUT, "manifest.jsonl"))
    meta = json.load(open(os.path.join(OUT, "run_metadata.json")))
    summary = json.load(open(os.path.join(OUT, "summary_tables.json")))
    judge_path = os.path.join(OUT, "judge_v2_full_verdicts.jsonl")
    judge = read_jsonl(judge_path) if os.path.exists(judge_path) else []
    judge_by_rid = {v["run_id"]: v for v in judge}
    judge_summary_path = os.path.join(OUT, "judge_v2_full_summary.json")
    judge_summary = (json.load(open(judge_summary_path))
                     if os.path.exists(judge_summary_path) else {})

    dv_judge = make_dv_judge(judge_by_rid)
    mid = [r for r in rows if r["injection_position"] == "mid"]

    # ---------------- H1'
    rows_by_d = {d: [r for r in mid if r["arm"] == f"REFUTING_d{d}"] for d in (1, 2, 3)}
    h1_primary = ca_trend_clustered(rows_by_d, dv_stated)
    h1_secondary = ca_trend_clustered(rows_by_d, dv_judge) if judge else None

    # ---------------- H5'
    h5a = mcnemar_onesided(rows, "REFUTING_d1", "FREQ_MATCHED_NONREFUTING",
                           dv_stated, "hi_gt_lo", positions={"mid"})
    h5b = mcnemar_onesided(rows, "REFUTING_d1", "TEMPLATE_IRRELEVANT",
                           dv_stated, "hi_gt_lo", positions={"mid"})
    h5_p = max(h5a["p_exact_one_sided"], h5b["p_exact_one_sided"])
    h5a_j = h5b_j = None
    if judge:
        h5a_j = mcnemar_onesided(rows, "REFUTING_d1", "FREQ_MATCHED_NONREFUTING",
                                 dv_judge, "hi_gt_lo", positions={"mid"})
        h5b_j = mcnemar_onesided(rows, "REFUTING_d1", "TEMPLATE_IRRELEVANT",
                                 dv_judge, "hi_gt_lo", positions={"mid"})

    # ---------------- H6
    h6_mid = mcnemar_onesided(rows, "REFUTING_d1", "BASELINE_FILLER",
                              dv_injdep, "hi_lt_lo", positions={"mid"})
    h6_early = mcnemar_onesided(rows, "REFUTING_d1", "BASELINE_FILLER",
                                dv_injdep, "hi_lt_lo", positions={"early"})
    h6_late = mcnemar_onesided(rows, "REFUTING_d1", "BASELINE_FILLER",
                               dv_injdep, "hi_lt_lo", positions={"late"})
    h6_pooled = mcnemar_onesided(rows, "REFUTING_d1", "BASELINE_FILLER",
                                 dv_injdep, "hi_lt_lo", positions=None)
    h6_j = (mcnemar_onesided(rows, "REFUTING_d1", "BASELINE_FILLER",
                             dv_judge, "hi_lt_lo", positions={"mid"})
            if judge else None)

    verdicts = [
        {
            "test_id": "H1_prime_stated_complement_trend_d123",
            "suite": "EXPE", "dv": "strict stated-complement (primary)",
            "position_scope": "mid",
            "test": ("Cochran-Armitage trend over d in {1,2,3}, decreasing; "
                     "clustered on problem via within-problem permutation "
                     f"({h1_primary['n_perm']} draws)"),
            "statistic": {"perm_trend_stat": h1_primary["perm_stat_observed"],
                          "ca_z_unclustered": h1_primary["ca_z_unclustered"]},
            "p_raw": h1_primary["p_perm_one_sided_decreasing"],
            "p_unclustered_reference": h1_primary["ca_p_unclustered_one_sided_decreasing"],
            "estimate": {"slope_per_hop": h1_primary["slope_per_hop"],
                         "rates_by_d": h1_primary["rates_by_d"]},
            "ci95": h1_primary["slope_ci95_cluster_bootstrap"],
            "ns": {"problems": h1_primary["n_problems_complete"],
                   "rows": h1_primary["n_rows"]},
            "detail": h1_primary,
        },
        {
            "test_id": "H5_prime_derivability_vs_lexical_conjunction",
            "suite": "EXPE", "dv": "strict stated-complement (primary)",
            "position_scope": "mid",
            "test": ("conjunction (intersection-union): stated-complement "
                     "REFUTING_d1 > FREQ_MATCHED_NONREFUTING and > "
                     "TEMPLATE_IRRELEVANT; exact one-sided McNemar within problem; "
                     "slot p = max(component p)"),
            "statistic": {
                "vs_FREQ_discordant": [h5a["discordant_hi1_lo0"], h5a["discordant_hi0_lo1"]],
                "vs_TEMPLATE_discordant": [h5b["discordant_hi1_lo0"], h5b["discordant_hi0_lo1"]]},
            "p_raw": float(f"{h5_p:.6g}"),
            "components": {
                "vs_FREQ_MATCHED_NONREFUTING": h5a,
                "vs_TEMPLATE_IRRELEVANT": h5b},
            "estimate": {"diff_vs_FREQ": h5a["diff_hi_minus_lo"],
                         "diff_vs_TEMPLATE": h5b["diff_hi_minus_lo"]},
            "ci95": {"vs_FREQ": h5a["diff_ci95_cluster_bootstrap"],
                     "vs_TEMPLATE": h5b["diff_ci95_cluster_bootstrap"]},
            "ns": {"pairs_vs_FREQ": h5a["n_pairs"], "pairs_vs_TEMPLATE": h5b["n_pairs"]},
        },
        {
            "test_id": "H6_within_problem_evidence_flip_injdep",
            "suite": "EXPE", "dv": "derivational injection-dependence (validator 'poisoned')",
            "position_scope": "mid (primary; early/late/pooled sensitivity in detail)",
            "test": ("derivational inj-dep REFUTING_d1 < BASELINE_FILLER; exact "
                     "one-sided McNemar within problem"),
            "statistic": {"discordant_R1_F0": h6_mid["discordant_hi1_lo0"],
                          "discordant_R0_F1": h6_mid["discordant_hi0_lo1"]},
            "p_raw": h6_mid["p_exact_one_sided"],
            "estimate": {"diff_R_minus_F": h6_mid["diff_hi_minus_lo"],
                         "rate_REFUTING_d1": h6_mid["rate_hi"],
                         "rate_BASELINE_FILLER": h6_mid["rate_lo"]},
            "ci95": h6_mid["diff_ci95_cluster_bootstrap"],
            "ns": {"pairs_mid": h6_mid["n_pairs"]},
            "sensitivity": {"early": h6_early, "late": h6_late, "pooled": h6_pooled},
        },
    ]
    secondary = []
    if judge:
        secondary = [
            {"test_id": "H1_prime_SECONDARY_judge_explicit_rejection",
             "dv": "judge explicit-rejection (registered secondary; ~0.15-0.20 "
                   "false-positive floor on controls per v1.1)",
             "p_raw": h1_secondary["p_perm_one_sided_decreasing"],
             "estimate": {"slope_per_hop": h1_secondary["slope_per_hop"],
                          "rates_by_d": h1_secondary["rates_by_d"]},
             "ci95": h1_secondary["slope_ci95_cluster_bootstrap"],
             "ns": {"problems": h1_secondary["n_problems_complete"]},
             "detail": h1_secondary},
            {"test_id": "H5_prime_SECONDARY_judge_explicit_rejection",
             "p_raw": float(f"{max(h5a_j['p_exact_one_sided'], h5b_j['p_exact_one_sided']):.6g}"),
             "components": {"vs_FREQ": h5a_j, "vs_TEMPLATE": h5b_j}},
            {"test_id": "H6_SECONDARY_judge_explicit_rejection",
             "p_raw": h6_j["p_exact_one_sided"], "detail": h6_j},
        ]

    # ---------------- descriptives per arm x position
    tier_by_key = {}
    for m in manifest:
        tier_by_key[(m["problem_id"], m["injection_position"])] = m["vocab_tier"]
    arms = meta.get("arms")
    positions = ["early", "mid", "late"]
    cells = {}
    for arm in arms:
        for pos in positions:
            sub = [r for r in rows if r["arm"] == arm and r["injection_position"] == pos]
            if not sub:
                continue
            n = len(sub)
            cls = Counter(r.get("class") for r in sub)
            cell = {
                "n": n,
                "stated_complement": round(rate(sub, dv_stated)[0], 4),
                "closure_valid": round(cls.get("valid_rederivation", 0) / n, 4),
                "inj_dep_derivational": (
                    lambda t: round(t[0], 4) if t[0] is not None else None)(rate(sub, dv_injdep)),
                "echo_parroted": round(cls.get("parroted", 0) / n, 4),
                "derailed": round(cls.get("derailed", 0) / n, 4),
                "unparsed": round(cls.get("unparsed", 0) / n, 4),
                "lexical_doubt": round(sum(bool(r.get("verbalized_doubt")) for r in sub) / n, 4),
                "resolve_rate_mod11": round(sum(
                    bool(r.get("gold_revalidates_under_augmentation")) for r in sub) / n, 4),
                "six_class": {k: cls.get(k, 0) for k in
                              ("valid_rederivation", "poisoned", "parroted",
                               "derailed", "unparsed")},
                "vocab_tiers": dict(Counter(
                    tier_by_key.get((r["problem_id"], pos)) for r in sub)),
            }
            cell["six_class"]["other"] = n - sum(cell["six_class"].values())
            if judge:
                jd = [judge_by_rid.get(r["run_id"]) for r in sub]
                jd = [v for v in jd if v]
                if jd:
                    cell["judge_explicit_rejection"] = round(
                        sum(bool(v["explicit_rejection"]) for v in jd) / len(jd), 4)
                    cell["judge_doubt"] = round(
                        sum(bool(v["judge_doubt"]) for v in jd) / len(jd), 4)
                    cell["judge_n"] = len(jd)
            cells[f"{arm}|{pos}"] = cell

    out = {
        "created_at": meta.get("created_at"),
        "prereg_evidence": meta.get("prereg_evidence"),
        "plan": "PLAN2.md v1.1 section B (m=6 Holm family assembled by coordinator; "
                "raw p-values only here)",
        "model": meta.get("model"), "model_revision": meta.get("model_revision"),
        "backend": meta.get("backend"),
        "cohort_source": meta.get("cohort_source"),
        "position_arms": meta.get("position_arms"),
        "registered": verdicts,
        "secondary_judge_dv": secondary,
        "descriptive_cells": cells,
        "judge_summary_cells": judge_summary.get("cells"),
        "gpu_seconds": {"generate": args.gpu_seconds_generate,
                        "judge": args.gpu_seconds_judge},
    }
    vp = os.path.join(OUT, "hypothesis_verdicts.json")
    with open(vp, "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=False)
    print(f"wrote {vp}")

    # ---------------- report
    L = []
    L.append("# EXPE_EVIDENCE_MOVER — full confirmatory run report")
    L.append("")
    L.append(f"- Pre-registration: {meta.get('prereg_evidence')}")
    L.append(f"- Model: {meta.get('model')} @ {meta.get('model_revision')} "
             f"(backend {meta.get('backend')}, greedy, max_new="
             f"{meta.get('decoding', {}).get('max_new_tokens')})")
    L.append(f"- Cohort: {meta.get('cohort_source')} "
             "(results/EXPE_GOLD_EXPANSION/cohort.jsonl, PLAN2 v1.1 C.1; the "
             "runner was patched to read it via --cohort expanded; --cohort "
             "legacy preserves the smoke-era path)")
    L.append(f"- Grid (v1.1 C.5): {json.dumps(meta.get('position_arms'))}")
    L.append(f"- GPU seconds: generate={args.gpu_seconds_generate}, "
             f"judge(32B)={args.gpu_seconds_judge}")
    L.append("")
    L.append("DV registry (v1.1 B): PRIMARY = strict stated-complement; "
             "SECONDARY = judge explicit-rejection (32B, doubt_judge_v2); "
             "verbalized doubt (lexical + judge) DESCRIPTIVE ONLY.")
    L.append("")
    L.append("Protocol disclosure (MOD-11): own-trace generated under the "
             "unaugmented question; all arms replay it under an augmented "
             "question -- equally off-policy across arms. Re-solve rate (MC-6) "
             "reported per cell below.")
    L.append("")
    L.append("## Registered adjudications (raw p; Holm m=6 assembled by coordinator)")
    L.append("")
    for v in verdicts:
        L.append(f"### {v['test_id']}")
        L.append("")
        L.append(f"- Test: {v['test']}")
        L.append(f"- DV: {v['dv']}; scope: {v['position_scope']}")
        L.append(f"- statistic: {json.dumps(v['statistic'])}")
        L.append(f"- p_raw (one-sided): **{v['p_raw']}**")
        L.append(f"- estimate: {json.dumps(v['estimate'])}")
        L.append(f"- ci95: {json.dumps(v['ci95'])}")
        L.append(f"- ns: {json.dumps(v['ns'])}")
        if "components" in v:
            for cname, c in v["components"].items():
                L.append(f"  - component {cname}: rate_hi={c['rate_hi']} "
                         f"rate_lo={c['rate_lo']} diff={c['diff_hi_minus_lo']} "
                         f"CI={c['diff_ci95_cluster_bootstrap']} "
                         f"discordant={c['discordant_hi1_lo0']}/{c['discordant_hi0_lo1']} "
                         f"p={c['p_exact_one_sided']} (n={c['n_pairs']})")
        if "sensitivity" in v:
            for sname, c in v["sensitivity"].items():
                if c.get("n_pairs"):
                    L.append(f"  - sensitivity {sname}: diff={c['diff_hi_minus_lo']} "
                             f"CI={c['diff_ci95_cluster_bootstrap']} "
                             f"p={c['p_exact_one_sided']} (n={c['n_pairs']})")
        L.append("")
    if secondary:
        L.append("## Secondary DV (judge explicit-rejection, 32B)")
        L.append("")
        for v in secondary:
            L.append(f"- {v['test_id']}: p_raw={v['p_raw']} "
                     f"{json.dumps(v.get('estimate', ''))[:200]}")
        L.append("")
    L.append("## Descriptives per arm x position")
    L.append("")
    L.append("| cell | n | stated-comp | judge-reject | closure-valid | inj-dep(deriv) "
             "| echo | judge-doubt | lex-doubt | unparsed | re-solve | tiers |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells, key=lambda k: (k.split("|")[1], k.split("|")[0])):
        c = cells[key]
        L.append(f"| {key} | {c['n']} | {c['stated_complement']} | "
                 f"{c.get('judge_explicit_rejection', '-')} | {c['closure_valid']} | "
                 f"{c['inj_dep_derivational']} | {c['echo_parroted']} | "
                 f"{c.get('judge_doubt', '-')} | {c['lexical_doubt']} | "
                 f"{c['unparsed']} | {c['resolve_rate_mod11']} | "
                 f"{json.dumps(c['vocab_tiers'])} |")
    L.append("")
    L.append("## Six-class multinomial per cell (unparsed visible, house rule)")
    L.append("")
    for key in sorted(cells, key=lambda k: (k.split("|")[1], k.split("|")[0])):
        L.append(f"- {key}: {json.dumps(cells[key]['six_class'])}")
    L.append("")
    L.append("## Integrity and audits (runner summary)")
    L.append("")
    for k, v in summary.get("integrity", {}).items():
        L.append(f"- [{'x' if v else ' '}] {k}")
    L.append(f"- sanity: {json.dumps(summary.get('sanity_checks', {}))[:800]}")
    L.append(f"- availability: "
             f"{json.dumps(summary.get('availability', {}).get('paired_availability_constraint'))}")
    L.append("")
    L.append("## Deviations and notes")
    L.append("")
    L.append("- late-position availability is 172 < 220: the full available "
             "late cohort was taken (v1.1 C.5 'full available cohort' rule); "
             "early had 239 available, 220 selected by seeded shuffle.")
    L.append("- Eligibility still requires all six arms constructible under one "
             "shared F at every position (paired-availability constraint held "
             "constant); only the GENERATED arm set is restricted at early/late "
             "per v1.1 C.5.")
    L.append("- MC-1 (parse rates): unparsed differs across compared cells at mid "
             "(REFUTING_d1 0.0545 vs REFUTING_d2/d3 0.0182, FREQ 0.0409, "
             "TEMPLATE 0.0045, FILLER 0.0091). The primary stated-complement DV "
             "is a string-level check computed on EVERY generated row "
             "(parse-independent), and the judge pass covered all rows, so no "
             "parse-conditioned imputation is required; unparsed rows are "
             "retained in all denominators (house rule) and visible per cell "
             "in the six-class table.")
    L.append("- Registered outcome branch (v1.1 section B, carried from 5.1): "
             "H5' failed specifically in the pre-stated confound direction -- "
             "FREQ ~= REFUTING on the primary DV (diff -0.0045, "
             "CI [-0.0591, +0.0455]) while REFUTING >> TEMPLATE (p ~ 3e-8). "
             "This is the 'professor's confound account SUPPORTED (lexical "
             "priming)' condition on the EXPE side; the family-level call "
             "belongs to the coordinator once EXPD's H2'/H3'/H4 slots land.")
    L.append("- Qualitative check on the DV (disclosed): sampled FREQ-arm "
             "stated-complement rows reach the complement via UNLICENSED "
             "chains (e.g. treating 'Every brimpus is not a gorpus' as if it "
             "refuted 'Stella is a gorpus'), whereas REFUTING_d1 rows derive it "
             "through the licensed added rule -- consistent with lexical/"
             "template priming producing the same surface statement.")
    L.append("- Holm (m=6) is assembled by the coordinator over "
             "{H1', H2', H3', H4, H5', H6}; only raw p-values are reported here.")
    rp = os.path.join(OUT, "EXPE_FULL_REPORT.md")
    with open(rp, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {rp}")


if __name__ == "__main__":
    main()
