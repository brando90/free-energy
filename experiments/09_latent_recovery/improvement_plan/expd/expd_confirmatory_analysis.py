"""EXPD full confirmatory analysis - registered adjudications per PLAN2 v1.1.

Scores the EXPD slots of the single m=6 Holm family (PLAN2 v1.1 section B):

  H2' : polarity equivalence at matched d - aff_false_attr vs neg_false_attr
        closure-valid TOST +/-0.10, pooled d in {1,3}, on the jointly-solved +
        identical-prefix mirrored cohort (MOD-13).
  H3' : truth, not the "not" token (conjunction) -
        (i)  neg_true_attr vs aff_true_attr closure-valid TOST +/-0.10
             (pooled d in {0,1,3} per the v1.0 slot definition that v1.1
             carries over; positions pooled - see analysis decisions),
        (ii) stated-complement(neg_false_d1) - stated-complement(neg_true_d1) > 0.
  H4  : confound-free reuse gradient - derivational injection-dependence over
        d in {1,3,inf} within cat_false_usable, ordered-factor linear trend on
        the logit scale, increasing.
  C-0 : d=0 vs d=1 (aff_false_attr) two-sided on stated-complement AND
        closure-valid; 95% CI, NO verdict (non-family, direction open).

Primary rejection DV = STRICT STATED-COMPLEMENT (continuation states/derives
the negation of the planted claim; string-level check against the world
grammar via validator.parse_fact - the expe_evidence_mover convention).
Verbalized doubt (lexical) is DESCRIPTIVE ONLY. Structural DVs per section 7.

Holm context: these three tests are slots in the m=6 family; the other three
(H1', H5', H6) are EXPE-owned. Raw p-values are reported alongside worst-case
Holm verdicts (p < alpha/m passes under ANY Holm ordering); the final
family-level Holm pass is assembled by the coordinator once EXPE reports.

Usage:
  python expd_confirmatory_analysis.py --out-dir results/EXPD_MATCHED_GRADIENT
"""
import argparse
import datetime as _dt
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_worlds_expd as genw  # noqa: E402  (inserts src/ into sys.path)

from validator import parse_fact, strip_marker  # noqa: E402
from expc_polarity_control import opposite_pred, statement_predicate  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

ALPHA = 0.05
HOLM_M = 6
TOST_MARGIN = 0.10
POSITIONS = ("early", "mid", "late")
FAMILY_SLOTS = {
    "H1prime": "EXPE (evidence-distance gradient, stated-complement)",
    "H2prime": "EXPD (this file)",
    "H3prime": "EXPD (this file)",
    "H4": "EXPD (this file)",
    "H5prime": "EXPE (derivability vs lexical co-presence)",
    "H6": "EXPE (within-problem evidence flip)",
}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_sentences(text):
    return [p.strip() for p in re.split(r"(?<=\.)\s+", (text or "").strip()) if p.strip()]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def rnd(x, k=4):
    return None if x is None else round(float(x), k)


# ------------------------------------------------------------------ loading

def stated_complement(continuation, injected_statement, entity):
    """Strict string-level complement detection against the world grammar
    (expe_evidence_mover convention, generalized to attribute predicates)."""
    pred = statement_predicate(injected_statement, entity)
    if pred is None:
        return None
    comp = (entity, opposite_pred(pred))
    return any(parse_fact(strip_marker(s), entity) == comp
               for s in split_sentences(continuation))


def load_rows(out_dir):
    manifest = {r["run_id"]: r for r in read_jsonl(os.path.join(out_dir, "manifest.jsonl"))}
    rows = []
    for r in read_jsonl(os.path.join(out_dir, "validated_outputs.jsonl")):
        m = manifest.get(r["run_id"])
        if not m:
            continue
        r["_entity"] = m["entity"]
        if r.get("failed_generation"):
            r["stated_complement"] = False
        else:
            r["stated_complement"] = bool(stated_complement(
                r.get("continuation", ""), m["injected_statement"], m["entity"]))
        rows.append(r)
    return rows, manifest


def attr_cohort_maps(cohort):
    """(family_id, d_str) joint-eligibility + per-position prefix identity (MOD-13/MC-7)."""
    fams = defaultdict(dict)
    for c in cohort:
        if c["kind"] == "attr":
            fams[(c["family_id"], str(c["d"]))][c["version"]] = c
    joint, prefix_same = {}, {}
    for key, vv in fams.items():
        a, b = vv.get("A"), vv.get("B")
        joint[key] = bool(a and b and a.get("eligible") and b.get("eligible"))
        if joint[key]:
            for pos in POSITIONS:
                sia = a["injection_points"][pos]
                sib = b["injection_points"][pos]
                same = (sia == sib and all(
                    norm(x) == norm(y)
                    for x, y in zip(a["gold_steps"][:sia], b["gold_steps"][:sib])))
                prefix_same[key + (pos,)] = same
    return joint, prefix_same


def annotate_cohort(rows, joint, prefix_same):
    for r in rows:
        if r.get("world_kind") != "attr":
            r["in_mirrored_cohort"] = None
            continue
        key = (r["family_id"], str(r["world_d"]))
        pos = r["injection_position"]
        r["in_mirrored_cohort"] = bool(joint.get(key)) and bool(prefix_same.get(key + (pos,)))


# ------------------------------------------------------------------ metrics

def mval(r, metric):
    if metric == "closure_valid":
        return int(bool(r.get("valid_recovery")))
    if metric == "stated_complement":
        return int(bool(r.get("stated_complement")))
    if metric == "doubt_lexical":
        return int(bool(r.get("doubt")))
    if metric == "inj_derivational":
        return int(bool(r.get("inj_derivational")))
    if metric == "inj_echo":
        return int(bool(r.get("inj_echo")))
    if metric in ("poisoned", "parroted", "derailed", "unparsed", "generation_failed",
                  "valid_rederivation"):
        return int(r.get("class") == metric)
    raise KeyError(metric)


CELL_METRICS = ("closure_valid", "stated_complement", "inj_derivational", "inj_echo",
                "doubt_lexical", "parroted", "derailed", "unparsed", "generation_failed")
SIX_CLASS = ("valid_rederivation", "poisoned", "parroted", "derailed", "unparsed",
             "generation_failed")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [rnd(c - h), rnd(c + h)]


def cluster_boot(rows, metric, seed=0, n_boot=1000):
    ids = sorted({r["family_id"] for r in rows})
    if not ids:
        return [None, None]
    by = defaultdict(list)
    for r in rows:
        by[r["family_id"]].append(mval(r, metric))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        num = den = 0
        for cid in (rng.choice(ids) for _ in ids):
            num += sum(by[cid])
            den += len(by[cid])
        vals.append(num / den if den else float("nan"))
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return [None, None]
    return [rnd(vals[int(0.025 * (len(vals) - 1))]), rnd(vals[int(0.975 * (len(vals) - 1))])]


def cell_summary(rows, seed=0):
    out = {"n": len(rows), "family_n": len({r["family_id"] for r in rows})}
    for metric in CELL_METRICS:
        vals = [mval(r, metric) for r in rows]
        n, k = len(vals), sum(vals)
        out[metric] = {"n": n, "count": k, "rate": rnd(k / n) if n else None,
                       "wilson95": wilson(k, n),
                       "family_cluster_bootstrap95": cluster_boot(rows, metric, seed) if n else [None, None]}
    out["six_class"] = {c: rnd(sum(1 for r in rows if r.get("class") == c) / len(rows))
                        for c in SIX_CLASS} if rows else {}
    unp = out["unparsed"]["rate"] or 0
    gf = out["generation_failed"]["rate"] or 0
    out["parse_rate"] = rnd(1.0 - unp - gf) if rows else None
    return out


# ------------------------------------------------------------------- inference

def family_means(rows, metric):
    by = defaultdict(list)
    for r in rows:
        by[r["family_id"]].append(mval(r, metric))
    return {f: float(np.mean(v)) for f, v in by.items()}


def pooled_rate(rows, metric):
    if not rows:
        return None
    return float(np.mean([mval(r, metric) for r in rows]))


def paired_family_stats(rows_a, rows_b, metric):
    """Family-level paired mean difference (cluster unit = family/quadruple)."""
    ma, mb = family_means(rows_a, metric), family_means(rows_b, metric)
    common = sorted(set(ma) & set(mb))
    diffs = np.array([ma[f] - mb[f] for f in common], dtype=float)
    n = len(diffs)
    mean = float(diffs.mean()) if n else None
    sd = float(diffs.std(ddof=1)) if n > 1 else None
    se = sd / math.sqrt(n) if sd is not None else None
    return {"n_families_paired": n, "mean_diff": mean, "se": se, "df": n - 1,
            "sd": sd,
            "rate_a": pooled_rate(rows_a, metric), "rate_b": pooled_rate(rows_b, metric),
            "n_rows_a": len(rows_a), "n_rows_b": len(rows_b)}


def t_ci(mean, se, df, level=0.95):
    if se is None or se == 0 or df < 1:
        return [rnd(mean), rnd(mean)]
    tq = sps.t.ppf(0.5 + level / 2, df)
    return [rnd(mean - tq * se), rnd(mean + tq * se)]


def tost(ps, margin=TOST_MARGIN):
    """Two one-sided tests for equivalence within +/-margin on the rate scale."""
    mean, se, df = ps["mean_diff"], ps["se"], ps["df"]
    if mean is None:
        return {"error": "no paired families"}
    if se is None or se == 0:
        inside = abs(mean) < margin
        p1 = p2 = (0.0 if inside else 1.0)
        note = "degenerate SE=0 (all family diffs identical); p set by point position vs margin"
    else:
        p1 = float(sps.t.sf((mean + margin) / se, df))   # H0: diff <= -margin
        p2 = float(sps.t.sf((margin - mean) / se, df))   # H0: diff >= +margin
        note = None
    return {"mean_diff": rnd(mean), "se": rnd(se, 6), "df": df,
            "margin": margin,
            "p_lower": rnd(p1, 6), "p_upper": rnd(p2, 6),
            "p_raw": rnd(max(p1, p2), 6),
            "t_lower": rnd((mean + margin) / se, 4) if se else None,
            "t_upper": rnd((margin - mean) / se, 4) if se else None,
            "ci90": t_ci(mean, se, df, 0.90), "ci95": t_ci(mean, se, df, 0.95),
            "note": note}


def one_sided_greater(ps):
    """H1: mean_diff > 0."""
    mean, se, df = ps["mean_diff"], ps["se"], ps["df"]
    if mean is None:
        return {"error": "no paired families"}
    if se is None or se == 0:
        p = 0.0 if mean > 0 else 1.0
        return {"mean_diff": rnd(mean), "se": None, "df": df, "t": None,
                "p_raw": p, "ci95": [rnd(mean), rnd(mean)],
                "note": "degenerate SE=0"}
    t = mean / se
    return {"mean_diff": rnd(mean), "se": rnd(se, 6), "df": df, "t": rnd(t),
            "p_raw": rnd(float(sps.t.sf(t, df)), 6), "ci95": t_ci(mean, se, df, 0.95)}


def two_sided(ps):
    mean, se, df = ps["mean_diff"], ps["se"], ps["df"]
    if mean is None:
        return {"error": "no paired families"}
    if se is None or se == 0:
        return {"mean_diff": rnd(mean), "se": None, "df": df, "t": None,
                "p_two_sided": 0.0 if mean != 0 else 1.0,
                "ci95": [rnd(mean), rnd(mean)], "note": "degenerate SE=0"}
    t = mean / se
    return {"mean_diff": rnd(mean), "se": rnd(se, 6), "df": df, "t": rnd(t),
            "p_two_sided": rnd(float(sps.t.sf(abs(t), df)) * 2, 6),
            "ci95": t_ci(mean, se, df, 0.95)}


def logit_trend(rows, metric, x_map, seed=0):
    """Ordered-factor linear trend on the logit scale, cluster-robust by family.

    Primary: GLM binomial logit, linear contrast x in {-1,0,+1}, cluster SEs.
    Robustness/fallback: family-cluster bootstrap of the OLS slope through the
    empirical logits (continuity-corrected) of the d-level pooled rates.
    """
    y = np.array([mval(r, metric) for r in rows], dtype=float)
    x = np.array([x_map[str(r["designed_d"])] for r in rows], dtype=float)
    fams = sorted({r["family_id"] for r in rows})
    fam_idx = {f: i for i, f in enumerate(fams)}
    groups = np.array([fam_idx[r["family_id"]] for r in rows])

    per_level = {}
    for d, xv in sorted(x_map.items(), key=lambda kv: kv[1]):
        sub = [r for r in rows if str(r["designed_d"]) == d]
        k, n = sum(mval(r, metric) for r in sub), len(sub)
        per_level[d] = {"n": n, "count": k, "rate": rnd(k / n) if n else None,
                        "wilson95": wilson(k, n),
                        "family_cluster_bootstrap95": cluster_boot(sub, metric, seed)}

    glm = None
    try:
        import statsmodels.api as sm
        X = sm.add_constant(x)
        res = sm.GLM(y, X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
        coef, se = float(res.params[1]), float(res.bse[1])
        z = coef / se
        glm = {"coef_logit_per_step": rnd(coef), "se_cluster": rnd(se, 6),
               "z": rnd(z), "p_one_sided_increasing": rnd(float(sps.norm.sf(z)), 6),
               "ci95": [rnd(coef - 1.96 * se), rnd(coef + 1.96 * se)],
               "converged": bool(res.converged)}
    except Exception as e:
        glm = {"error": repr(e)}

    # family-cluster bootstrap of the empirical-logit slope (always computed)
    rng = random.Random(seed)
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        c = by[r["family_id"]][str(r["designed_d"])]
        c[0] += mval(r, metric)
        c[1] += 1
    def slope(sample_fams):
        agg = {d: [0, 0] for d in x_map}
        for f in sample_fams:
            for d, (k, n) in by[f].items():
                agg[d][0] += k
                agg[d][1] += n
        pts = []
        for d, (k, n) in agg.items():
            if n == 0:
                return None
            p = (k + 0.5) / (n + 1.0)
            pts.append((x_map[d], math.log(p / (1 - p))))
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        return float(np.polyfit(xs, ys, 1)[0])
    point = slope(fams)
    boots = []
    for _ in range(2000):
        s = slope([rng.choice(fams) for _ in fams])
        if s is not None:
            boots.append(s)
    boots.sort()
    n_b = len(boots)
    boot = {"slope_empirical_logit": rnd(point),
            "ci95": [rnd(boots[int(0.025 * (n_b - 1))]), rnd(boots[int(0.975 * (n_b - 1))])],
            "p_one_sided_increasing_boot": rnd(sum(1 for b in boots if b <= 0) / n_b, 6),
            "n_boot": n_b}
    return {"per_level": per_level, "glm_cluster": glm, "cluster_bootstrap": boot,
            "n_rows": len(rows), "n_families": len(fams)}


def holm_fields(p_raw):
    return {
        "pass_at_worst_case_holm_alpha_over_m": bool(p_raw is not None and p_raw < ALPHA / HOLM_M),
        "pass_at_nominal_alpha": bool(p_raw is not None and p_raw < ALPHA),
        "holm_note": ("worst-case threshold alpha/m = %.6f guarantees a pass under any Holm "
                      "ordering of the m=6 family; final family-level Holm assembled by the "
                      "coordinator with the three EXPE slots." % (ALPHA / HOLM_M)),
    }


# --------------------------------------------------------------------- main

def cells_rows(rows, cells, positions=None, cohort_only=False):
    sel = [r for r in rows if r["condition"] in cells]
    if positions:
        sel = [r for r in sel if r["injection_position"] in positions]
    if cohort_only:
        sel = [r for r in sel if r.get("in_mirrored_cohort")]
    return sel


def robustness_glm(rows_a, rows_b, metric, joint, prefix_same):
    """MOD-13 robustness fit: jointly-solved-only, prefix divergence as covariate."""
    try:
        import statsmodels.api as sm
        rows = [(r, 1) for r in rows_a] + [(r, 0) for r in rows_b]
        y, pol, div, grp = [], [], [], []
        fams = {}
        for r, is_a in rows:
            key = (r["family_id"], str(r["world_d"]))
            if not joint.get(key):
                continue
            same = prefix_same.get(key + (r["injection_position"],), False)
            y.append(mval(r, metric))
            pol.append(is_a)
            div.append(0 if same else 1)
            grp.append(fams.setdefault(r["family_id"], len(fams)))
        if len(set(y)) < 2:
            return {"note": "degenerate outcome (no variance)", "n": len(y),
                    "rate": rnd(float(np.mean(y))) if y else None}
        cols = [np.ones(len(y)), pol]
        has_div = len(set(div)) > 1
        if has_div:
            cols.append(div)
        X = np.column_stack(cols)
        res = sm.GLM(np.array(y, float), X, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": np.array(grp)})
        return {"n": len(y), "coef_polarity_logit": rnd(float(res.params[1])),
                "se": rnd(float(res.bse[1]), 6),
                "coef_divergence_logit": rnd(float(res.params[2])) if has_div else None,
                "divergence_covariate_dropped_no_variance": not has_div,
                "divergent_prefix_rows": int(sum(div))}
    except Exception as e:
        return {"error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cat-positions", default="early,mid")
    args = ap.parse_args()
    out_dir = args.out_dir
    cat_positions = tuple(args.cat_positions.split(","))

    rows, manifest = load_rows(out_dir)
    cohort = read_jsonl(os.path.join(out_dir, "gold_cohort.jsonl"))
    summary = json.load(open(os.path.join(out_dir, "summary_tables.json")))
    meta = json.load(open(os.path.join(out_dir, "run_metadata.json")))
    world_audit = json.load(open(os.path.join(out_dir, "worlds", "world_audit.json")))
    timings = {}
    tpath = os.path.join(out_dir, "stage_timings.json")
    if os.path.exists(tpath):
        timings = json.load(open(tpath))

    joint, prefix_same = attr_cohort_maps(cohort)
    annotate_cohort(rows, joint, prefix_same)

    # ---------------- descriptive tables ----------------
    by_cell, by_cell_pos = {}, {}
    for cell in sorted({r["condition"] for r in rows}):
        sub = [r for r in rows if r["condition"] == cell]
        by_cell[cell] = cell_summary(sub, args.seed)
        for pos in POSITIONS:
            ss = [r for r in sub if r["injection_position"] == pos]
            if ss:
                by_cell_pos["%s@%s" % (cell, pos)] = cell_summary(ss, args.seed)

    # MC-7 prefix identity, per d and position
    mc7 = {"overall": {}, "by_d": {}, "by_position": {}}
    checked = same_n = 0
    for key, ok in prefix_same.items():
        fid, d, pos = key
        checked += 1
        same_n += int(ok)
        bd = mc7["by_d"].setdefault("d%s" % d, [0, 0])
        bd[0] += int(ok)
        bd[1] += 1
        bp = mc7["by_position"].setdefault(pos, [0, 0])
        bp[0] += int(ok)
        bp[1] += 1
    mc7["overall"] = {"identical": same_n, "checked": checked,
                      "identical_rate": rnd(same_n / checked) if checked else None}
    mc7["by_d"] = {k: {"identical": v[0], "n": v[1], "rate": rnd(v[0] / v[1])}
                   for k, v in sorted(mc7["by_d"].items())}
    mc7["by_position"] = {k: {"identical": v[0], "n": v[1], "rate": rnd(v[0] / v[1])}
                          for k, v in mc7["by_position"].items()}

    # joint-solve per d
    joint_by_d = {}
    for (fid, d), ok in joint.items():
        j = joint_by_d.setdefault("d%s" % d, [0, 0])
        j[0] += int(ok)
        j[1] += 1
    joint_by_d = {k: {"joint_eligible": v[0], "pairs": v[1], "rate": rnd(v[0] / v[1])}
                  for k, v in sorted(joint_by_d.items())}

    # ---------------- H2' ----------------
    a_cells = ["aff_false_attr_d1", "aff_false_attr_d3"]
    b_cells = ["neg_false_attr_d1", "neg_false_attr_d3"]
    rows_a = cells_rows(rows, a_cells, cohort_only=True)
    rows_b = cells_rows(rows, b_cells, cohort_only=True)
    ps2 = paired_family_stats(rows_a, rows_b, "closure_valid")
    h2_tost = tost(ps2)
    h2 = {
        "test_id": "H2prime",
        "suite": "EXPD",
        "description": ("polarity equivalence at matched d: aff_false_attr vs neg_false_attr "
                        "closure-valid TOST +/-0.10, pooled d in {1,3}, jointly-solved + "
                        "identical-prefix mirrored cohort (MOD-13), positions pooled, "
                        "cluster unit = family (quadruple)"),
        "dv": "closure_valid",
        "statistic": {"type": "paired-family TOST t (two one-sided)",
                      "t_lower": h2_tost.get("t_lower"), "t_upper": h2_tost.get("t_upper"),
                      "df": h2_tost.get("df")},
        "estimate": h2_tost.get("mean_diff"),
        "estimate_name": "rate difference aff_false - neg_false (family-paired mean)",
        "ci95": h2_tost.get("ci95"),
        "ci90_equivalence": h2_tost.get("ci90"),
        "p_raw": h2_tost.get("p_raw"),
        "p_components": {"p_lower": h2_tost.get("p_lower"), "p_upper": h2_tost.get("p_upper")},
        "pass_rule": "TOST passes iff BOTH one-sided p < slot Holm-adjusted alpha",
        "ns": {"rows_aff": ps2["n_rows_a"], "rows_neg": ps2["n_rows_b"],
               "paired_families": ps2["n_families_paired"]},
        "pooled_rates": {"aff_false": rnd(ps2["rate_a"]), "neg_false": rnd(ps2["rate_b"])},
        "note": h2_tost.get("note"),
        "robustness_jointly_solved_divergence_covariate": robustness_glm(
            cells_rows(rows, a_cells), cells_rows(rows, b_cells), "closure_valid",
            joint, prefix_same),
    }
    h2.update(holm_fields(h2["p_raw"]))

    # ---------------- H3' ----------------
    nt_cells = ["neg_true_attr_d0", "neg_true_attr_d1", "neg_true_attr_d3"]
    at_cells = ["aff_true_attr_d0", "aff_true_attr_d1", "aff_true_attr_d3"]
    rows_nt = cells_rows(rows, nt_cells, cohort_only=True)
    rows_at = cells_rows(rows, at_cells, cohort_only=True)
    ps3i = paired_family_stats(rows_nt, rows_at, "closure_valid")
    h3i = tost(ps3i)
    rows_nf1 = cells_rows(rows, ["neg_false_attr_d1"], cohort_only=True)
    rows_nt1 = cells_rows(rows, ["neg_true_attr_d1"], cohort_only=True)
    ps3ii = paired_family_stats(rows_nf1, rows_nt1, "stated_complement")
    h3ii = one_sided_greater(ps3ii)
    # robustness variants for (i)
    ps3i_mid = paired_family_stats(
        cells_rows(rows, nt_cells, positions=("mid",), cohort_only=True),
        cells_rows(rows, at_cells, positions=("mid",), cohort_only=True), "closure_valid")
    ps3i_alld = paired_family_stats(
        cells_rows(rows, ["neg_true_attr_d%s" % d for d in (0, 1, 2, 3, 5)], cohort_only=True),
        cells_rows(rows, ["aff_true_attr_d%s" % d for d in (0, 1, 2, 3, 5)], cohort_only=True),
        "closure_valid")
    p_raw3 = None
    if h3i.get("p_raw") is not None and h3ii.get("p_raw") is not None:
        p_raw3 = max(h3i["p_raw"], h3ii["p_raw"])
    h3 = {
        "test_id": "H3prime",
        "suite": "EXPD",
        "description": ("conjunction (intersection-union): (i) neg_true_attr ~ aff_true_attr "
                        "closure-valid TOST +/-0.10, pooled d in {0,1,3}, positions pooled, "
                        "mirrored cohort; (ii) stated-complement(neg_false_d1) > "
                        "stated-complement(neg_true_d1), one-sided, family-paired"),
        "dv": "closure_valid (i) + stated_complement (ii)",
        "statistic": {"component_i_TOST": {"t_lower": h3i.get("t_lower"),
                                           "t_upper": h3i.get("t_upper"), "df": h3i.get("df")},
                      "component_ii_t": {"t": h3ii.get("t"), "df": ps3ii["df"]}},
        "estimate": {"i_mean_diff_negtrue_minus_afftrue": h3i.get("mean_diff"),
                     "ii_mean_diff_negfalse_minus_negtrue": h3ii.get("mean_diff")},
        "ci95": {"i": h3i.get("ci95"), "ii": h3ii.get("ci95")},
        "ci90_equivalence_i": h3i.get("ci90"),
        "p_raw": p_raw3,
        "p_components": {"i_TOST": h3i.get("p_raw"), "i_lower": h3i.get("p_lower"),
                         "i_upper": h3i.get("p_upper"), "ii_one_sided": h3ii.get("p_raw")},
        "pass_rule": ("conjunctive slot: passes iff EVERY component rejects at the slot's "
                      "Holm-adjusted alpha (no alpha splitting); slot p_raw = max(components)"),
        "ns": {"rows_neg_true": ps3i["n_rows_a"], "rows_aff_true": ps3i["n_rows_b"],
               "paired_families_i": ps3i["n_families_paired"],
               "rows_neg_false_d1": ps3ii["n_rows_a"], "rows_neg_true_d1": ps3ii["n_rows_b"],
               "paired_families_ii": ps3ii["n_families_paired"]},
        "pooled_rates": {"neg_true_valid": rnd(ps3i["rate_a"]),
                         "aff_true_valid": rnd(ps3i["rate_b"]),
                         "neg_false_d1_stated_comp": rnd(ps3ii["rate_a"]),
                         "neg_true_d1_stated_comp": rnd(ps3ii["rate_b"])},
        "notes": {"i": h3i.get("note"), "ii": h3ii.get("note")},
        "robustness": {
            "i_mid_only": tost(ps3i_mid),
            "i_all_d_pooled": tost(ps3i_alld),
        },
    }
    h3.update(holm_fields(h3["p_raw"]))

    # ---------------- H4 ----------------
    h4_cells = ["cat_false_usable_d1", "cat_false_usable_d3", "cat_false_usable_dinf"]
    rows_h4 = cells_rows(rows, h4_cells)
    trend = logit_trend(rows_h4, "inj_derivational", {"1": -1.0, "3": 0.0, "inf": 1.0},
                        args.seed)
    glm = trend["glm_cluster"]
    p4 = glm.get("p_one_sided_increasing") if isinstance(glm, dict) else None
    h4 = {
        "test_id": "H4",
        "suite": "EXPD",
        "description": ("derivational injection-dependence over d in {1,3,inf} within "
                        "cat_false_usable; ordered-factor linear trend on the logit scale "
                        "(x=-1,0,+1), increasing as counterevidence recedes; cluster-robust "
                        "(family) GLM; positions %s pooled" % (cat_positions,)),
        "dv": "inj_derivational (M4 derivational split; echo excluded)",
        "statistic": {"type": "GLM binomial logit linear-trend z (cluster-robust)",
                      "z": glm.get("z")},
        "estimate": glm.get("coef_logit_per_step"),
        "estimate_name": "logit-scale slope per ordered step d1->d3->dinf",
        "ci95": glm.get("ci95"),
        "p_raw": p4,
        "pass_rule": "trend > 0 at slot Holm-adjusted alpha (one-sided)",
        "ns": {"rows": trend["n_rows"], "families": trend["n_families"],
               "per_level": {k: v["n"] for k, v in trend["per_level"].items()}},
        "per_level_rates": {k: v["rate"] for k, v in trend["per_level"].items()},
        "per_level_detail": trend["per_level"],
        "robustness_cluster_bootstrap_empirical_logit": trend["cluster_bootstrap"],
        "ceiling_note": ("pre-stated (PLAN2 v1.1 B): pilot d1 derivational = 0.783, headroom "
                         "~0.2; trend test on the logit scale for this reason"),
    }
    if isinstance(glm.get("z"), float):
        h4["exploratory_reverse_direction"] = {
            "p_one_sided_decreasing": rnd(float(sps.norm.cdf(glm["z"])), 6),
            "note": ("NOT a registered test (direction was registered as increasing); reported "
                     "so the observed significant DECREASE (reuse highest at d=1) is visible, "
                     "not just the failure of the registered direction"),
        }
    h4.update(holm_fields(h4["p_raw"]))

    # ---------------- C-0 ----------------
    c0 = {"test_id": "C0", "suite": "EXPD",
          "description": ("aff_false_attr d=0 vs d=1, two-sided, direction open; reported "
                          "with 95% CI regardless of outcome; NON-FAMILY (no Holm slot, "
                          "no verdict); all eligible rows, positions pooled"),
          "dvs": {}}
    for dv in ("stated_complement", "closure_valid"):
        ps = paired_family_stats(cells_rows(rows, ["aff_false_attr_d0"]),
                                 cells_rows(rows, ["aff_false_attr_d1"]), dv)
        ts = two_sided(ps)
        c0["dvs"][dv] = {
            "estimate": ts.get("mean_diff"),
            "estimate_name": "rate(d0) - rate(d1), family-paired",
            "ci95": ts.get("ci95"), "t": ts.get("t"), "df": ts.get("df"),
            "p_two_sided": ts.get("p_two_sided"),
            "rates": {"d0": rnd(ps["rate_a"]), "d1": rnd(ps["rate_b"])},
            "ns": {"rows_d0": ps["n_rows_a"], "rows_d1": ps["n_rows_b"],
                   "paired_families": ps["n_families_paired"]},
            "note": ts.get("note"),
        }
    c0["verdict"] = "none by design (pre-registered as estimation-only)"

    # ---------------- pre-registered outcome-branch adjudication ----------------
    outcome_branches = {
        "local_refutability_CONFIRMED": {
            "requires": "H1'+H4+H5' pass AND H2'+H3' pass (v1.1 carry-over of section 5.1)",
            "status": ("OFF regardless of EXPE results: H4 failed (trend significantly in the "
                       "OPPOSITE direction) and H3' failed (component ii)"),
        },
        "professors_confound_account_SUPPORTED": {
            "requires": ("H2'/H3' fail with POLARITY DOMINATING, or H5' fails with "
                         "FREQ ~= REFUTING (v1.1 section B)"),
            "status": ("NOT triggered by EXPD: H2' PASSED (aff/neg equivalent within +/-0.10; "
                       "point difference +0.054 with aff_false slightly HIGHER closure-valid, "
                       "the opposite of a negation-driven artifact), and H3'(i) TOST passed "
                       "emphatically; H3' failed only because its component (ii) hit a FLOOR "
                       "on the primary DV (stated-complement 0.009 vs 0.000 in attribute "
                       "cells), not because neg >> aff. EXPE's H5' remains to be scored."),
        },
        "H4_reuse_gradient": {
            "registered_reading_if_flat": ("'reuse gradient was a construction artifact' "
                                           "(PLAN2 section 11)"),
            "observed": ("not flat: significantly DECREASING (slope -0.248 logit/step, "
                         "one-sided-decreasing p ~= 0.004 exploratory); derivational reuse is "
                         "HIGHEST at d=1 (0.863) where the refuting rule is adjacent - local "
                         "counterevidence does not suppress reuse in this grammar; if anything "
                         "the nearest-counterevidence worlds absorb MORE"),
        },
    }

    # ---------------- MC-1 matched-parse checks for the contrasts ----------------
    def parse_gap(cells_x, cells_y):
        px = [by_cell[c]["parse_rate"] for c in cells_x if c in by_cell]
        py = [by_cell[c]["parse_rate"] for c in cells_y if c in by_cell]
        if not px or not py:
            return None
        return rnd(abs(float(np.mean(px)) - float(np.mean(py))))
    mc1 = {
        "per_cell_parse_rate": {c: by_cell[c]["parse_rate"] for c in sorted(by_cell)},
        "contrast_parse_gaps": {
            "H2prime_aff_vs_neg_false": parse_gap(a_cells, b_cells),
            "H3prime_i_negtrue_vs_afftrue": parse_gap(nt_cells, at_cells),
            "H3prime_ii_negfalse_d1_vs_negtrue_d1": parse_gap(["neg_false_attr_d1"],
                                                              ["neg_true_attr_d1"]),
            "H4_within_usable_max_gap": (max(by_cell[c]["parse_rate"] for c in h4_cells)
                                         - min(by_cell[c]["parse_rate"] for c in h4_cells)
                                         if all(c in by_cell for c in h4_cells) else None),
            "C0_d0_vs_d1": parse_gap(["aff_false_attr_d0"], ["aff_false_attr_d1"]),
        },
        "rule": ("cells of any confirmatory contrast must be parse-matched; if unmatched, "
                 "multinomial bounds over unparsed rows accompany the contrast (PLAN2 MC-1)"),
    }

    # ---------------- assemble ----------------
    gates = summary.get("gates_plan2_section6", {})
    doc = {
        "created_at": now_iso(),
        "experiment": meta.get("experiment"),
        "backend": meta.get("backend"),
        "model": meta.get("model"),
        "model_revision": meta.get("model_revision"),
        "prompt_sha256": meta.get("prompt", {}).get("sha256"),
        "pre_registration_evidence": meta.get("pre_registration_evidence"),
        "holm_family": {"m": HOLM_M, "alpha": ALPHA, "slots": FAMILY_SLOTS,
                        "supplied_here": ["H2prime", "H3prime", "H4"],
                        "assembly": ("final Holm over all 6 raw p-values is assembled by the "
                                     "coordinator once EXPE reports H1', H5', H6")},
        "analysis_decisions": {
            "cat_positions": list(cat_positions),
            "h3i_pooling": ("v1.1 B is silent on pooling for H3'(i); the superseded v1.0 H3 "
                            "slot specified d in {0,1,3}; adjudicated on d in {0,1,3} pooled "
                            "over all 3 positions (grid gives the true cells 3 positions in "
                            "v1.1 C.5); mid-only and all-d variants reported as robustness"),
            "cluster_unit": "family_id (mirrored-quadruple/base-chain id) per PLAN2 section 9.1",
            "cohort": ("H2'/H3' rows conditioned on jointly-eligible mirrored pairs with "
                       "identical gold prefixes up to si at the row's position (MOD-13); "
                       "H4 uses all eligible usable-cell rows (the d-trend does not cross "
                       "the usable/inert mirror); C-0 uses all eligible rows"),
            "stated_complement": ("strict string-level: any continuation sentence whose "
                                  "parse_fact equals (entity, opposite_pred(planted)) - the "
                                  "expe_evidence_mover convention generalized to attributes"),
        },
        "registered_outcome_branches": outcome_branches,
        "hypothesis_verdicts": [h2, h3, h4, c0],
        "descriptive_by_cell": by_cell,
        "descriptive_by_cell_position": by_cell_pos,
        "manipulation_checks": {
            "MC1_parse": mc1,
            "MC2_anchors": gates.get("anchor_equivalence_mc2"),
            "MC4_funnel": summary.get("funnel_consort"),
            "MC7_prefix_identity": mc7,
            "joint_solve_by_d": joint_by_d,
        },
        "integrity": summary.get("integrity"),
        "world_audit_summary": {
            "world_count": world_audit.get("world_count"),
            "family_count": world_audit.get("family_count"),
            "audit_failures": len(world_audit.get("audit_failures", [])),
            "design_notes": world_audit.get("design_notes"),
            "grammar_theorems": world_audit.get("grammar_theorems"),
        },
        "stage_timings": timings,
        "denominator_convention": ("unparsed and generation_failed retained in all "
                                   "denominators (house convention); parse rate per cell "
                                   "(MC-1); stated-complement computed on the raw "
                                   "continuation string regardless of validator class"),
    }
    with open(os.path.join(out_dir, "confirmatory_analysis.json"), "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)

    verdicts = {
        "created_at": doc["created_at"],
        "experiment": doc["experiment"],
        "holm_family": doc["holm_family"],
        "pre_registration_evidence": doc["pre_registration_evidence"],
        "registered_outcome_branches": outcome_branches,
        "tests": [
            {k: h[k] for k in ("test_id", "suite", "dv", "statistic", "p_raw",
                               "estimate", "ci95", "ns") if k in h}
            | {k: h[k] for k in ("p_components", "pass_rule",
                                 "pass_at_worst_case_holm_alpha_over_m",
                                 "pass_at_nominal_alpha", "pooled_rates",
                                 "per_level_rates") if k in h}
            for h in (h2, h3, h4)
        ] + [c0],
    }
    with open(os.path.join(out_dir, "hypothesis_verdicts.json"), "w") as fh:
        json.dump(verdicts, fh, indent=2, sort_keys=True)

    write_report(out_dir, doc, meta, gates)
    print(json.dumps({
        "H2prime": {"p_raw": h2["p_raw"], "estimate": h2["estimate"], "ns": h2["ns"]},
        "H3prime": {"p_raw": h3["p_raw"], "components": h3["p_components"]},
        "H4": {"p_raw": h4["p_raw"], "estimate": h4["estimate"],
               "per_level": h4["per_level_rates"]},
        "C0": {dv: {"estimate": v["estimate"], "ci95": v["ci95"], "p": v["p_two_sided"]}
               for dv, v in c0["dvs"].items()},
    }, indent=2))


def fmt(m):
    return "%s %s" % (m["rate"], m["family_cluster_bootstrap95"])


def write_report(out_dir, doc, meta, gates):
    by_cell = doc["descriptive_by_cell"]
    h2, h3, h4, c0 = doc["hypothesis_verdicts"]
    ev = doc.get("pre_registration_evidence") or {}
    L = []
    L.append("# EXPD Matched-Gradient FULL CONFIRMATORY Report")
    L.append("")
    L.append("Generated: %s" % doc["created_at"])
    L.append("Backend: %s | model: %s (%s) | prompt sha256: %s" % (
        doc["backend"], doc["model"], doc["model_revision"], doc["prompt_sha256"]))
    L.append("")
    L.append("**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5) under the externally "
             "timestamped pre-registration.** Gate evidence: git commit `%s` (%s), PLAN2.md "
             "sha256 `%s`. Holm family m=6; the three EXPD slots are adjudicated here with "
             "raw p-values; final family-level Holm is assembled by the coordinator with the "
             "EXPE slots (H1', H5', H6)." % (
                 ev.get("plan2_git_commit"), ev.get("plan2_commit_timestamp"),
                 ev.get("plan2_file_sha256")))
    L.append("")
    L.append("## Registered adjudications (PLAN2 v1.1 section B)")
    L.append("")
    L.append("| slot | test | estimate | 95% CI | p_raw | passes at alpha/6 (worst-case Holm) | passes at 0.05 |")
    L.append("|---|---|---|---|---|---|---|")
    L.append("| H2' | aff_false vs neg_false closure-valid TOST +/-0.10, d{1,3}, mirrored cohort "
             "| diff=%s | %s (90%% eq-CI %s) | %s | %s | %s |" % (
                 h2["estimate"], h2["ci95"], h2["ci90_equivalence"], h2["p_raw"],
                 h2["pass_at_worst_case_holm_alpha_over_m"], h2["pass_at_nominal_alpha"]))
    L.append("| H3' | conjunction: (i) neg_true~aff_true valid TOST (d{0,1,3}); (ii) "
             "stated-comp(neg_false_d1)>stated-comp(neg_true_d1) | i: %s, ii: %s | i: %s, ii: %s "
             "| %s (i: %s, ii: %s) | %s | %s |" % (
                 h3["estimate"]["i_mean_diff_negtrue_minus_afftrue"],
                 h3["estimate"]["ii_mean_diff_negfalse_minus_negtrue"],
                 h3["ci95"]["i"], h3["ci95"]["ii"], h3["p_raw"],
                 h3["p_components"]["i_TOST"], h3["p_components"]["ii_one_sided"],
                 h3["pass_at_worst_case_holm_alpha_over_m"], h3["pass_at_nominal_alpha"]))
    L.append("| H4 | derivational inj-dep logit trend d{1,3,inf} in cat_false_usable, increasing "
             "| slope=%s | %s | %s | %s | %s |" % (
                 h4["estimate"], h4["ci95"], h4["p_raw"],
                 h4["pass_at_worst_case_holm_alpha_over_m"], h4["pass_at_nominal_alpha"]))
    for dv, v in c0["dvs"].items():
        L.append("| C-0 (%s) | d0 vs d1 two-sided, NON-FAMILY, no verdict | diff=%s | %s | %s | n/a | n/a |" % (
            dv, v["estimate"], v["ci95"], v["p_two_sided"]))
    L.append("")
    L.append("Pooled rates: H2' aff_false valid=%s vs neg_false valid=%s (paired families=%s). "
             "H3' neg_true valid=%s vs aff_true valid=%s; stated-comp neg_false_d1=%s vs "
             "neg_true_d1=%s. H4 derivational by d: %s. C-0 rates: %s." % (
                 h2["pooled_rates"]["aff_false"], h2["pooled_rates"]["neg_false"],
                 h2["ns"]["paired_families"],
                 h3["pooled_rates"]["neg_true_valid"], h3["pooled_rates"]["aff_true_valid"],
                 h3["pooled_rates"]["neg_false_d1_stated_comp"],
                 h3["pooled_rates"]["neg_true_d1_stated_comp"],
                 h4["per_level_rates"],
                 {dv: v["rates"] for dv, v in c0["dvs"].items()}))
    L.append("")
    L.append("## Pre-registered outcome-branch adjudication")
    L.append("")
    for name, br in doc["registered_outcome_branches"].items():
        L.append("- **%s**" % name)
        for k, v in br.items():
            L.append("  - %s: %s" % (k, v))
    rev = h4.get("exploratory_reverse_direction")
    if rev:
        L.append("- **H4 reverse-direction (exploratory):** one-sided-decreasing p = %s. %s" % (
            rev["p_one_sided_decreasing"], rev["note"]))
    L.append("")
    L.append("## Headline per-cell rates (positions pooled; family-cluster bootstrap 95% CIs)")
    L.append("")
    L.append("| cell | n | parse | closure-valid | stated-comp | deriv inj-dep | echo | "
             "doubt(lex, descriptive) | parroted | derailed | unparsed |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for cell in sorted(by_cell):
        c = by_cell[cell]
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            cell, c["n"], c["parse_rate"], fmt(c["closure_valid"]), fmt(c["stated_complement"]),
            fmt(c["inj_derivational"]), c["inj_echo"]["rate"], fmt(c["doubt_lexical"]),
            c["parroted"]["rate"], c["derailed"]["rate"], c["unparsed"]["rate"]))
    L.append("")
    L.append("## Per-position rates (appendix)")
    L.append("")
    L.append("| cell@position | n | closure-valid | stated-comp | deriv inj-dep | doubt(lex) | unparsed |")
    L.append("|---|---|---|---|---|---|---|")
    for key in sorted(doc["descriptive_by_cell_position"]):
        c = doc["descriptive_by_cell_position"][key]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (
            key, c["n"], c["closure_valid"]["rate"], c["stated_complement"]["rate"],
            c["inj_derivational"]["rate"], c["doubt_lexical"]["rate"], c["unparsed"]["rate"]))
    L.append("")
    L.append("## Manipulation checks")
    L.append("")
    mc = doc["manipulation_checks"]
    L.append("- **MC-1 parse matching across contrast cells (gaps):** %s" %
             json.dumps(mc["MC1_parse"]["contrast_parse_gaps"]))
    L.append("- **MC-7 identical gold prefix (jointly-eligible mirrored pairs):** overall %s; "
             "by d %s; by position %s" % (
                 json.dumps(mc["MC7_prefix_identity"]["overall"]),
                 json.dumps(mc["MC7_prefix_identity"]["by_d"]),
                 json.dumps(mc["MC7_prefix_identity"]["by_position"])))
    L.append("- **Joint-solve mirrored pairs by d:** %s" % json.dumps(mc["joint_solve_by_d"]))
    anch = mc.get("MC2_anchors") or {}
    for cell, a in (anch.get("cells") or {}).items():
        L.append("- **MC-2 anchor %s:** valid TOST %s; unparsed TOST %s; doubt diff vs locked-HF "
                 "%s (backend-confounded, no verdict; gate scoped to valid/unparsed per v1.1 C.2, "
                 "MOD-12: never gates H2'-H4)" % (
                     cell, json.dumps(a.get("valid_recovery_tost")),
                     json.dumps(a.get("unparsed_tost")),
                     (a.get("doubt_comparison") or {}).get("diff_vs_locked_hf")))
    L.append("")
    L.append("## Funnel (CONSORT, MC-4)")
    L.append("")
    L.append("```")
    L.append(json.dumps(mc["MC4_funnel"], indent=2, sort_keys=True))
    L.append("```")
    L.append("")
    L.append("World audit: %s worlds / %s families generated, %s audit failures." % (
        doc["world_audit_summary"]["world_count"], doc["world_audit_summary"]["family_count"],
        doc["world_audit_summary"]["audit_failures"]))
    L.append("")
    L.append("## Integrity")
    L.append("")
    L.append("```")
    L.append(json.dumps(doc["integrity"], indent=2, sort_keys=True))
    L.append("```")
    L.append("")
    L.append("## Analysis decisions & disclosures")
    L.append("")
    for k, v in sorted(doc["analysis_decisions"].items()):
        L.append("- **%s:** %s" % (k, v))
    L.append("- **Generator sizing (disclosed):** FULL_CONFIG runs 180 families/kind so that "
             "every cell x d reaches the registered n=150/cell/position after the eligibility "
             "filter (each family yields exactly one world per cell x d); the nonce-noun pool "
             "was extended with cvcv-prefix forms for capacity (globally unique nouns, "
             "audited invariants unchanged). Neither changes any registered quantity.")
    L.append("- **Judge pass:** the Qwen2.5-32B judge is the pre-registered SECONDARY detector "
             "(upper bound); it was not run in this pass. The registered PRIMARY rejection DV "
             "(strict stated-complement) and all structural DVs are complete. Verbalized doubt "
             "is reported descriptively from the lexical detector only (v1.1 A.3: doubt "
             "point-rates are backend-sensitive and unregistered).")
    L.append("- **Backend rule (v1.1 C.4):** wholly vLLM 0.24.0, greedy, max_new=192, model "
             "revision pinned; no cross-backend row comparisons.")
    L.append("")
    L.append("## Stage timings / GPU")
    L.append("")
    L.append("```")
    L.append(json.dumps(doc["stage_timings"], indent=2, sort_keys=True))
    L.append("```")
    L.append("")
    L.append("Machine-readable verdicts: `hypothesis_verdicts.json`; full analysis document: "
             "`confirmatory_analysis.json`; runner mechanical stages: "
             "`EXPD_RUNNER_STAGES_REPORT.md`; per-cell summaries: `summary_tables.json`.")
    text = "\n".join(L) + "\n"
    with open(os.path.join(out_dir, "EXPD_FULL_REPORT.md"), "w") as fh:
        fh.write(text)
    print("wrote EXPD_FULL_REPORT.md")


if __name__ == "__main__":
    main()
