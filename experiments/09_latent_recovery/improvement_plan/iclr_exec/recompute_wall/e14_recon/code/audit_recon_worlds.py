"""verify:worlds -- adversarial audit of $OUTBASE/worlds/*.jsonl (E14 recon 3x2).

READ-ONLY w.r.t. improvement_plan/ world files (it only reads them). Imports the
project's own audits and re-derives every claimed property from the world file,
never from the funnel. CPU only.

v2 additions over the 2x2 audit:
  * the `large` regime is audited on the same footing as pm1 / pm10_20
    (magnitude window, unit-digit contract, attainable-sum window, distinctness,
    within-world pairing, opacity byte-diff, gp.audit_family, discrimination,
    collision, leak, injection round trip);
  * the MEASUREMENT-SEPARATION property |out_cf - out_true| == 2*|delta| is
    re-derived per family from the interpreter, not read off the file;
  * the magnitude x last-digit cross-tab is reported per regime -- this is the
    table that shows factor A is no longer collinear with the unit-digit cue;
  * the vtrue selection effect induced by the `large` attainable-window
    constraint is measured against the v1 pool.
"""
import json, os, sys, hashlib, subprocess
from collections import Counter, defaultdict

IP = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan"
RW = os.path.join(IP, "iclr_exec", "recompute_wall")
SCRATCH = "/lfs/skampere2/0/eobbad/scratch/recon"
for _d in (SCRATCH, os.path.join(RW, "e11_generator"), os.path.join(IP, "expg")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import gen_programs as gp
import gen_depth as gd
import audit_depth as ad
import trace_format as tf
import recon_gen as rg
from interp import execute, execute_cf, parse_program, value_at

REGIMES = rg.REGIMES                      # ("pm10_20", "pm1", "large")
RECON = rg.RECON_FAMILIES                 # 6
ANCHOR = rg.ANCHOR_PAIRS
PAIRED_FIELDS = ("planted_var", "true_value", "force_after_line", "transform",
                 "opacity", "site_body_true")
MAG_OK = rg.MAG_OK
LDP = rg.LAST_DIGIT_PRESERVED
POLICY_OK = {"pm1": ("off_by_1",), "large": ("off_by_100_400",),
             "pm10_20": ("last_digit_preserving_pm10_20",)}


def targets():
    t = ad.default_targets(); t["nearest_anc_dist"] = 4; return t


class R:
    def __init__(self): self.d = defaultdict(Counter); self.ex = defaultdict(list)
    def ok(self, k, n=1): self.d[k]["pass"] += n
    def bad(self, k, detail, n=1):
        self.d[k]["fail"] += n
        if len(self.ex[k]) < 6: self.ex[k].append(detail)
    def tally(self, k, cond, detail):
        if cond: self.ok(k)
        else: self.bad(k, detail)


def audit_file(path, label):
    worlds = [json.loads(l) for l in open(path)]
    r = R()
    audit_fn = ad.make_audit_fn(targets())
    N = len(worlds)
    out = {"file": path, "label": label, "n_worlds": N,
           "md5": hashlib.md5(open(path, "rb").read()).hexdigest()}

    ids = Counter(p["program_id"] for p in worlds)
    seeds = Counter(p["e14_world_seed"] for p in worlds)
    out["dup_program_ids"] = [k for k, v in ids.items() if v > 1]
    out["dup_seeds"] = [k for k, v in seeds.items() if v > 1]

    abs_delta = {f: Counter() for f in RECON}
    signed = {f: Counter() for f in RECON}
    lastdigit_pres = {f: Counter() for f in RECON}
    sep_meas = {f: Counter() for f in RECON}          # RE-DERIVED separation
    sep_over_delta = {f: Counter() for f in RECON}    # separation / |delta|
    window_out = Counter()
    bare_operands_in_prefix = Counter()
    fam_counts = Counter()
    per_cell_worlds = Counter()
    magnitude_vs_lastdigit = {t: Counter() for t in REGIMES}
    vtrue_hist = Counter()

    lo, hi = rg.attainable_window()                    # 22, 198

    for p in worlds:
        pid = p["program_id"]
        stmts = parse_program(p["stmt_texts"])
        tw = execute(stmts)
        fams = p["families"]
        j, v, vtrue = p["site_line"], p["site_var"], p["site_true_value"]
        kn = gd._gp_knobs(gd.DKNOBS)
        lits = gp.listing_literals(stmts)
        truevals = set(tw["all_values"])
        vtrue_hist[vtrue] += 1

        for f in sorted(fams): fam_counts[f] += 1
        # ---------- structural / (a) presence ----------
        missing = [f for f in RECON if f not in fams]
        r.tally("a1_all6_families_present", not missing, "%s missing %s" % (pid, missing))
        for f in RECON:
            if f in fams: per_cell_worlds[f] += 1
        for f in ("depth_k1_bare", "depth_k1_full", "depth_k1_partial"):
            r.tally("a2_depth_anchors_retained", f in fams, "%s missing %s" % (pid, f))
        if missing: continue

        # ---------- (a) within-world pairing across ALL regime pairs ----------
        for opac in ("bare", "full"):
            specs = {t: fams["recon_%s_%s" % (opac, t)] for t in REGIMES}
            for t in REGIMES:
                s = specs[t]
                for fld in PAIRED_FIELDS:
                    r.tally("a3_regime_pair_field_%s" % fld,
                            all(specs[u][fld] == s[fld] for u in REGIMES),
                            "%s %s %s: %r" % (pid, t, fld, s[fld]))
                r.tally("a4_planted_var_is_site_var", s["planted_var"] == v,
                        "%s %s" % (pid, t))
                r.tally("a5_force_is_site_line", s["force_after_line"] == j,
                        "%s %s" % (pid, t))
                r.tally("a6_true_is_site_true", s["true_value"] == vtrue,
                        "%s %s" % (pid, t))
            # regimes differ ONLY in the trailing planted integer
            heads = {t: specs[t]["site_body"].rsplit("= ", 1)[0] for t in REGIMES}
            tails = {t: specs[t]["site_body"].rsplit("= ", 1)[1] for t in REGIMES}
            r.tally("a7_regime_bodies_differ_only_in_value",
                    len(set(heads.values())) == 1 and len(set(tails.values())) == len(REGIMES),
                    "%s %r" % (pid, {t: specs[t]["site_body"] for t in REGIMES}))
            r.tally("a8_regime_values_pairwise_distinct",
                    len({specs[t]["planted_value"] for t in REGIMES}) == len(REGIMES),
                    "%s %s" % (pid, [specs[t]["planted_value"] for t in REGIMES]))

        # ---------- (b) opacity pairing: bare vs full byte-diff ----------
        expected_span = " = " + " + ".join(str(x) for x in p["operands"])
        for tag in REGIMES:
            bare = fams["recon_bare_%s" % tag]["site_body"]
            full = fams["recon_full_%s" % tag]["site_body"]
            head, sep, tail = bare.partition("; ")
            fhead, fsep, ftail = full.partition("; ")
            r.tally("b1_opacity_diff_is_exactly_operand_span_%s" % tag,
                    ftail == tail and fhead == head + expected_span,
                    "%s bare=%r full=%r" % (pid, bare, full))
            r.tally("b2_strip_span_recovers_bare_%s" % tag,
                    fhead[:len(head)] == head
                    and fhead[len(head):] == expected_span
                    and head + "; " + ftail == bare,
                    "%s strip mismatch" % pid)
            r.tally("b4_opacity_pair_shares_plant_%s" % tag,
                    fams["recon_bare_%s" % tag]["planted_value"]
                    == fams["recon_full_%s" % tag]["planted_value"], "%s" % pid)
        r.tally("b3_operand_span_same_across_regimes",
                len({fams["recon_full_%s" % t]["site_body"].partition("; ")[0]
                     .rsplit("= ", 1)[0] for t in REGIMES}) == 1,
                "%s full heads differ" % pid)

        # ---------- (c) delta compliance + unit-digit DECOUPLING ----------
        for f in RECON:
            spec = fams[f]
            tag = spec["delta_regime"]
            d = spec["planted_value"] - vtrue
            abs_delta[f][abs(d)] += 1; signed[f][d] += 1
            pres = spec["planted_value"] % 10 == vtrue % 10
            lastdigit_pres[f][pres] += 1
            r.tally("c1_magnitude_window_%s" % tag, MAG_OK[tag](abs(d)),
                    "%s %s delta=%d" % (pid, f, d))
            r.tally("c2_delta_policy_label_%s" % tag,
                    spec["delta_policy"] in POLICY_OK[tag],
                    "%s %s pol=%s" % (pid, f, spec["delta_policy"]))
            r.tally("c3_last_digit_contract_%s" % tag, pres == LDP[tag],
                    "%s %s pres=%s want=%s" % (pid, f, pres, LDP[tag]))
            r.tally("c5_plant_positive", spec["planted_value"] > 0, "%s %s" % (pid, f))
            r.tally("c6_plant_in_knob_bounds",
                    kn["value_min"] <= spec["planted_value"] <= kn["value_max"],
                    "%s %s" % (pid, f))
            r.tally("c7_plant_in_attainable_window_%s" % tag,
                    lo <= spec["planted_value"] <= hi,
                    "%s %s pv=%d" % (pid, f, spec["planted_value"]))
            if not (lo <= spec["planted_value"] <= hi):
                window_out[f] += 1
            magnitude_vs_lastdigit[tag][(abs(d), pres)] += 1

        # ---------- (d) legality: project's own audits ----------
        fails = audit_fn(p)
        r.tally("d1_audit_depth_full_stack", not fails, "%s %s" % (pid, fails[:3]))
        pc = rg.post_checks(p)
        r.tally("d3_recon_gen_post_checks", not pc, "%s %s" % (pid, pc[:3]))

        for f in RECON:
            spec = fams[f]
            tag = spec["delta_regime"]
            ff, cfw = gp.audit_family(p, stmts, tw, spec["planted_var"],
                                      spec["planted_value"], spec["force_after_line"], kn)
            r.tally("d2_gp_audit_family_%s" % tag, not ff, "%s %s %s" % (pid, f, ff))
            # ---------- (f) discrimination ----------
            r.tally("f1_out_cf_differs_from_out_true_%s" % tag,
                    cfw["out_value"] != tw["out_value"],
                    "%s %s out=%s" % (pid, f, cfw["out_value"]))
            r.tally("f2_recorded_out_cf_matches_recompute_%s" % tag,
                    spec["out_cf"] == cfw["out_value"],
                    "%s %s rec=%s calc=%s" % (pid, f, spec["out_cf"], cfw["out_value"]))
            r.tally("f3_reads_discriminate_%s" % tag,
                    value_at(cfw, p["r1"], p["r1_var"]) != value_at(tw, p["r1"], p["r1_var"])
                    and value_at(cfw, p["r2"], p["r2_var"]) != value_at(tw, p["r2"], p["r2_var"]),
                    "%s %s" % (pid, f))
            r.tally("f4_out_true_matches_prog", tw["out_value"] == p["out_true"], pid)

            # ---------- SEPARATION, re-derived from the interpreter ----------
            s_meas = abs(cfw["out_value"] - tw["out_value"])
            sep_meas[f][s_meas] += 1
            sep_over_delta[f][s_meas / abs(spec["planted_value"] - vtrue)] += 1
            r.tally("s1_recorded_separation_matches_recompute_%s" % tag,
                    spec.get("separation") == s_meas,
                    "%s %s rec=%s calc=%s" % (pid, f, spec.get("separation"), s_meas))
            r.tally("s2_separation_is_2x_abs_delta_%s" % tag,
                    s_meas == 2 * abs(spec["planted_value"] - vtrue),
                    "%s %s sep=%d delta=%d" % (pid, f, s_meas,
                                               spec["planted_value"] - vtrue))
            r.tally("s3_recorded_out_true_matches_%s" % tag,
                    spec.get("out_true") == p["out_true"], "%s %s" % (pid, f))

            # ---------- (e) collision (actual, must be 0) ----------
            carriers = gp._plant_carriers(stmts, spec["planted_var"], spec["force_after_line"])
            cf_acc = {st["value"] for st in cfw["steps"]
                      if st["kind"] == "assign" and st["var"] not in carriers}
            pv = spec["planted_value"]
            r.tally("e1_plant_not_in_sigma_true_%s" % tag, pv not in truevals, "%s %s %d" % (pid, f, pv))
            r.tally("e2_plant_not_listing_literal_%s" % tag, pv not in lits, "%s %s %d" % (pid, f, pv))
            r.tally("e3_plant_not_noncarrier_cf_value_%s" % tag, pv not in cf_acc, "%s %s %d" % (pid, f, pv))
            r.tally("e4_plant_ne_true_%s" % tag, pv != vtrue, "%s %s" % (pid, f))

            # ---------- (g) leak ----------
            body = spec["site_body"]
            opvals = p["operands"]
            if spec["opacity"] == "full":
                r.tally("g1_full_prints_operand_values",
                        all(str(x) in body.partition(";")[0] for x in opvals),
                        "%s %r" % (pid, body))
                r.tally("g2_full_operands_sum_to_true", sum(opvals) == vtrue, "%s" % pid)
                r.tally("g3_full_line_selfcontradicts", sum(opvals) != pv, "%s %s" % (pid, f))
            else:
                lhs = body.partition(";")[0]
                r.tally("g4_bare_hides_operand_values",
                        not any(str(x) in lhs for x in opvals), "%s %r" % (pid, lhs))

            # ---------- injection round trip on the runner-emitted line ------
            line = "line %d: %s" % (j, spec["site_body"])
            pr = tf.parse_trace_line(line)
            r.tally("i1_injected_line_parses_%s" % tag,
                    pr is not None and pr.get("kind") == "assign", "%s %r" % (pid, line))
            if pr:
                r.tally("i2_roundtrip_value_%s" % tag,
                        pr["line"] == j and pr["var"] == v and pr["value"] == pv,
                        "%s %r -> %r" % (pid, line, pr))

        # (g) bare-cell leak-through-prefix: are operand values stated on
        # EARLIER trace lines (which the model sees in the prefix)?
        vbl = {int(k): vv for k, vv in p["true_values_by_line"].items()}
        earlier = {vbl[ln] for ln in vbl if ln < j}
        r.tally("g5_bare_operands_visible_earlier_in_prefix",
                all(x in earlier for x in p["operands"]), pid)
        bare_operands_in_prefix[sum(1 for x in p["operands"] if x in earlier)] += 1

        # ---------- anchor equivalence to E11 depth_k1_* ----------
        for rf, af in ANCHOR.items():
            A, B = fams[rf], fams[af]
            for fld in ("planted_var", "planted_value", "true_value", "force_after_line",
                        "delta_policy", "transform", "opacity", "out_cf", "site_body",
                        "site_body_true"):
                r.tally("h1_anchor_field_%s" % fld, A.get(fld) == B.get(fld),
                        "%s %s.%s %r vs %r" % (pid, rf, fld, A.get(fld), B.get(fld)))

    def _bucket(counter, width=20):
        b = Counter()
        for kk, vv in counter.items():
            b[(kk // width) * width] += vv
        return {str(kk): b[kk] for kk in sorted(b)}

    out["family_presence"] = dict(fam_counts)
    out["per_cell_worlds"] = dict(per_cell_worlds)
    out["abs_delta_distribution_summary"] = {
        f: {"min": min(abs_delta[f]), "max": max(abs_delta[f]),
            "distinct": len(abs_delta[f]),
            "top": dict(sorted(abs_delta[f].items())[:4])} for f in RECON}
    out["signed_delta_sign_counts"] = {
        f: {"neg": sum(v for k, v in signed[f].items() if k < 0),
            "pos": sum(v for k, v in signed[f].items() if k > 0)} for f in RECON}
    out["last_digit_preserved"] = {f: {str(k): v for k, v in sorted(lastdigit_pres[f].items())}
                                   for f in RECON}
    out["separation_recomputed"] = {
        f: {"min": min(sep_meas[f]), "max": max(sep_meas[f]),
            "distinct": len(sep_meas[f]),
            "hist_bucket20": _bucket(sep_meas[f])} for f in RECON}
    out["separation_over_abs_delta_ratios"] = {
        f: sorted(sep_over_delta[f]) for f in RECON}
    out["magnitude_x_lastdigit_by_regime"] = {
        t: {"n": sum(magnitude_vs_lastdigit[t].values()),
            "last_digit_preserved_true": sum(
                v for (m, pres), v in magnitude_vs_lastdigit[t].items() if pres),
            "last_digit_preserved_false": sum(
                v for (m, pres), v in magnitude_vs_lastdigit[t].items() if not pres),
            "abs_delta_min": min(m for (m, _p) in magnitude_vs_lastdigit[t]),
            "abs_delta_max": max(m for (m, _p) in magnitude_vs_lastdigit[t])}
        for t in REGIMES}
    out["plants_outside_attainable_window"] = dict(window_out)
    out["bare_operand_values_present_earlier_in_prefix_hist"] = dict(sorted(bare_operands_in_prefix.items()))
    out["vtrue_hist_bucket20"] = _bucket(vtrue_hist)
    out["vtrue_min"], out["vtrue_max"] = min(vtrue_hist), max(vtrue_hist)
    out["checks"] = {k: dict(v) for k, v in sorted(r.d.items())}
    out["failure_examples"] = {k: v for k, v in sorted(r.ex.items()) if v}
    out["n_failed_checks"] = sum(1 for k, v in r.d.items() if v.get("fail"))
    out["total_check_instances"] = sum(sum(v.values()) for v in r.d.values())
    return out


def pool_shift(v1_path, v2_path):
    """Measure the vtrue selection effect the `large` attainable-window
    constraint induces on the pool (it applies to the ANCHOR cells too)."""
    def load(pth):
        return [json.loads(l) for l in open(pth)]
    a, b = load(v1_path), load(v2_path)
    va = Counter(p["site_true_value"] for p in a)
    vb = Counter(p["site_true_value"] for p in b)
    ida = {p["program_id"] for p in a}
    idb = {p["program_id"] for p in b}

    def stats(c):
        n = sum(c.values())
        mean = sum(k * v for k, v in c.items()) / n
        xs = sorted(k for k in c.elements())
        return {"n": n, "mean": round(mean, 2), "median": xs[n // 2],
                "min": min(c), "max": max(c),
                "frac_in_98_125": round(sum(v for k, v in c.items()
                                            if 98 <= k <= 125) / n, 4)}
    return {"v1": stats(va), "v2": stats(vb),
            "program_ids_in_both": len(ida & idb),
            "program_ids_only_v1": len(ida - idb),
            "program_ids_only_v2": len(idb - ida),
            "note": ("v2 rejects every world whose vtrue admits no |delta|>=100 "
                     "plant inside the attainable-sum window [22,198]; that band "
                     "is vtrue in [98,125]. The restriction applies to ALL six "
                     "cells INCLUDING the pm10_20 anchors, so the E11 anchor "
                     "comparison is now on a vtrue-restricted subpopulation.")}


def determinism(outdir, n, name):
    """Regenerate into a temp dir and compare md5 -- the generator must be a
    pure function of BASE_SEED."""
    tmp = os.path.join(SCRATCH, "regen_check_v2")
    os.makedirs(tmp, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(SCRATCH, "recon_gen.py"),
                    "--out-dir", tmp, "--n", str(n), "--worlds-name", name,
                    "--overwrite"], check=True, capture_output=True)
    a = hashlib.md5(open(os.path.join(outdir, name), "rb").read()).hexdigest()
    b = hashlib.md5(open(os.path.join(tmp, name), "rb").read()).hexdigest()
    return {"file": name, "md5_pool": a, "md5_regen": b, "byte_identical": a == b}


if __name__ == "__main__":
    W = os.path.join(RW, "e14_recon", "worlds")
    W1 = os.path.join(RW, "e14_recon", "worlds_v1")
    res = []
    for fn, lab in (("recon_worlds.jsonl", "pool240"),
                    ("recon_worlds_pool1800.jsonl", "pool1800")):
        res.append(audit_file(os.path.join(W, fn), lab))
    a = open(os.path.join(W, "recon_worlds.jsonl"), "rb").readlines()
    b = open(os.path.join(W, "recon_worlds_pool1800.jsonl"), "rb").readlines()
    print(json.dumps({
        "results": res,
        "pool240_is_byte_prefix_of_pool1800": b[:len(a)] == a,
        "pool1800_len": len(b), "pool240_len": len(a),
        "determinism": [determinism(W, 240, "recon_worlds.jsonl")],
        "vtrue_selection_effect_v1_vs_v2": pool_shift(
            os.path.join(W1, "recon_worlds_pool1800.jsonl"),
            os.path.join(W, "recon_worlds_pool1800.jsonl")),
    }, indent=1, sort_keys=True))
