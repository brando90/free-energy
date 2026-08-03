"""E14 RECON world generator -- Yee-reconciliation 3x2 on the COMPUTED (k=1) site.

Purpose
-------
Yee et al. (COLM 2024, arXiv:2405.15092) report 74-79% recovery from planted
"calculation errors" (our audit: ~68-71%); our instrument reports ~100%
ABSORPTION on computed errors. Hypothesis: the gap is the SALIENCE channel, not
the verification channel. Their plants are (a) tiny implausible deltas
(|delta| ~ 1..3) and (b) printed in the SAME line as their operands. Ours are
last-digit-preserving +/-10/20 deltas on a bare site line.

DESIGN (v2, 3x2 -- was 2x2)
---------------------------
    factor A (delta regime):  pm1      |delta| == 1, last digit CHANGES  (Yee-like)
                              pm10_20  |delta| in {10,20}, last digit PRESERVED (ours)
                              large    |delta| in [100,400], last digit CHANGES,
                                       implausible in the OPPOSITE direction
    factor B (opacity):       bare     result only (ours)
                              full     operands printed on the line (Yee-like)

    -> recon_{bare,full}_{pm10_20,pm1,large}   (6 families, ONE shared world)

WHY THE `large` ARM EXISTS (defect fixed in v2). In the v1 2x2, factor A was
perfectly collinear with a unit-digit cue: pm10_20 preserved the last digit in
1800/1800 worlds and pm1 changed it in 1800/1800 worlds. "small implausible
delta" and "last-digit mismatch" were therefore not identifiable. `large` breaks
the collinearity: it CHANGES the last digit (like pm1) while being large and
implausible in the opposite direction (unlike pm1).

    INTERPRETATION THIS ENABLES
      * `large` behaves like pm10_20  -> the effect is delta MAGNITUDE /
        PLAUSIBILITY (a tiny delta is what makes the plant checkable).
      * `large` behaves like pm1      -> the effect is the LAST-DIGIT CUE (the
        mismatch in the unit digit is what makes the plant checkable), and the
        magnitude story is wrong.
      * `large` behaves like neither  -> both channels are live; read the two
        contrasts (large-minus-pm10_20, large-minus-pm1) as their separate
        loadings.

ANCHOR PROPERTY: the two pm10_20 cells reuse gen_depth's OWN planted value and
its OWN render_site_line_body, so their site_body is byte-identical to E11's
depth_k1_bare / depth_k1_full. If a later GPU run does not reproduce E11's
k1_bare / k1_full numbers on these two cells, the build is wrong.

Construction (nothing under improvement_plan/ is modified; this imports it):
  * substrate      gd.build_program(1, seed)  -- gives depth_k1_{bare,full,partial}
                   carrying the pm10_20 plant chosen by gp.axis_c_delta.
  * pm10_20 arms   REUSE prog["families"]["depth_k1_bare"]["planted_value"];
                   never recompute (recomputation would re-draw the rng).
  * pm1 arms       E13's dose_off1 candidate-selection pattern
                   (e13_gen.py:310-318) transposed onto the computed site:
                   planted_var = site_var (not root_var),
                   force_after_line = site_line (not j-1),
                   knobs = gd._gp_knobs(gd.DKNOBS) (not gp.KNOBS),
                   plus the plausibility window [ (k+1)*operand_min,
                   (k+1)*operand_max ] = [22,198] which E13 does not need
                   (kr1 prints no operands; a `full` computed site would leak a
                   zero-compute giveaway if the plant fell outside the
                   attainable range of the sum).
  * large arms     E13's dose_large selection (e13_gen.py:322-334) transposed
                   the same way, with TWO documented deviations:
                     (i)  the same [22,198] attainable-sum window is applied
                          (E13's kr1 site prints no operands, so E13 does not
                          need it; a `full` computed site does);
                     (ii) magnitudes divisible by 10 are SKIPPED, so `large`
                          always changes the last digit. Without this ~10% of
                          large plants would silently re-preserve the unit digit
                          and re-confound factor A with the very cue this arm
                          exists to decouple.
                   The window makes the arm INFEASIBLE for mid-range vtrue
                   (a plant needs |x - vtrue| >= 100 with 22 <= x <= 198, which
                   is impossible for vtrue in [99,121]); those worlds are
                   REJECTED, never patched, and counted as `large_starved`.
                   *** This induces a vtrue selection effect on the whole pool,
                   including the anchors -- see README_RECON.md. ***
  * every audit stays FAIL-CLOSED: gp.audit_family on the pm1 and large plants,
    then the complete audit_depth stack (make_audit_fn, nearest_anc_dist=4)
    re-run with all six recon families attached.

The three depth_k1_* families are DELIBERATELY KEPT on every emitted world:
audit_opacity_byte_identity (audit_depth.py:319-354) inspects families literally
named depth_k<k>_{bare,full,partial} and hard-fails "opacity_missing_bare" if
depth_k1_bare is absent. The runner scores only the six recon_* families.

MEASUREMENT SEPARATION (defect fixed in v2, diagnostics side). The k=1 world
shape is out = Q1 + Q2 with Q1 = V + c and Q2 = V + c', so
|out_cf - out_true| = 2 * |delta| -- 2 for pm1, 20/40 for pm10_20, >=200 for
large. Each recon family therefore carries `out_true` and `separation` so the
runner can persist them per validated row and bound the resulting bias. The DV
is NOT changed; see recon_run.stage_summarize.

CPU only. No GPU, no vLLM, no API. Deterministic in BASE_SEED.
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

IP = os.environ.get(
    "E14_IP",
    "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan")
RW = os.path.join(IP, "iclr_exec", "recompute_wall")
for _d in (os.path.join(RW, "e11_generator"), os.path.join(IP, "expg")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import gen_programs as gp                       # noqa: E402
import gen_depth as gd                          # noqa: E402
import audit_depth as ad                        # noqa: E402
import trace_format as tf                       # noqa: E402
from interp import execute, parse_program       # noqa: E402

K = 1
BASE_SEED = 900000 + K * 911                    # 900911 -- disjoint from E11
                                                # (700000+k*911) and E13 (0x...)
N_WORLDS = 240                                  # oversample; >=150 must survive
                                                # the downstream gold gate
REGIMES = ("pm10_20", "pm1", "large")
OPACITIES = ("bare", "full")
RECON_FAMILIES = tuple("recon_%s_%s" % (o, t)
                       for o in OPACITIES for t in REGIMES)
ANCHOR_PAIRS = {"recon_bare_pm10_20": "depth_k1_bare",
                "recon_full_pm10_20": "depth_k1_full"}
PM1_RNG_TAG = 0xE14                             # cf. e13_gen.py's 0xE13D
LARGE_MIN, LARGE_MAX = 100, 400                 # e13_gen.py:325 range(100, 401)
# magnitude windows asserted post-hoc, per regime
MAG_OK = {"pm1": lambda m: m == 1,
          "pm10_20": lambda m: m in (10, 20),
          "large": lambda m: LARGE_MIN <= m <= LARGE_MAX}
# designed unit-digit behaviour per regime -- this is the decoupling contract
LAST_DIGIT_PRESERVED = {"pm1": False, "pm10_20": True, "large": False}
DELTA_POLICY = {"pm1": "off_by_1", "large": "off_by_100_400"}


def attainable_window():
    """[lo, hi] = attainable range of the k=1 site sum (k+1 operands, each in
    [operand_min, operand_max]). A plant outside it is a zero-compute giveaway
    on the `full` arm, which prints the operands."""
    return ((K + 1) * gd.DKNOBS["operand_min"],      # 22   (gen_depth.py:580)
            (K + 1) * gd.DKNOBS["operand_max"])      # 198  (gen_depth.py:581)


def targets():
    """e11_run._targets_for(1) exactly (e11_run.py:263-266)."""
    t = ad.default_targets()
    t["nearest_anc_dist"] = 4
    return t


def _pick_large(vtrue, rng, forbidden, kn, lo, hi):
    """e13_gen.py:322-334 dose_large, transposed to the computed site.

    Deviations from E13, both deliberate and both audited post-hoc:
      * plant confined to the attainable-sum window [lo, hi];
      * magnitudes divisible by 10 skipped, so the last digit ALWAYS changes.
    Returns the planted value, or None (caller rejects the world).
    """
    mags = list(range(LARGE_MIN, LARGE_MAX + 1))
    rng.shuffle(mags)
    signs = (1, -1) if rng.random() < 0.5 else (-1, 1)
    for m in mags:
        if m % 10 == 0:                          # would re-preserve the unit
            continue                             # digit -> re-confounds factor A
        for s in signs:
            x = vtrue + s * m
            if (kn["value_min"] <= x <= kn["value_max"]
                    and lo <= x <= hi and x not in forbidden):
                return x
    return None


def add_recon_families(prog, seed):
    """Attach the six recon_* families to a built+audited k=1 world.

    Returns (prog, None) or (None, reject_reason). Never patches: any audit
    failure is a rejection of the whole world.
    """
    stmts = parse_program(prog["stmt_texts"])
    tw = execute(stmts)
    j = prog["site_line"]
    v = prog["site_var"]
    vtrue = prog["site_true_value"]
    out_true = prog["out_true"]
    kn = gd._gp_knobs(gd.DKNOBS)                # depth knobs, NOT gp.KNOBS

    anchor = prog["families"]["depth_k1_bare"]
    d_anchor = anchor["planted_value"]          # REUSE; do not re-draw the rng
    pol_anchor = anchor["delta_policy"]
    out_cf_anchor = anchor["out_cf"]

    # ONE rng stream for the whole non-anchor arm, exactly as e13_gen.py does
    # for its dose families. pm1 draws first, so its plant is unchanged from v1
    # for any seed that both versions accept.
    rng = random.Random((seed << 6) ^ PM1_RNG_TAG)
    base_forbidden = (set(tw["all_values"]) | gp.listing_literals(stmts)
                      | {d_anchor})             # the `| {d}` idiom, e13_gen.py:312
    lo, hi = attainable_window()

    # --- pm1 candidate: e13_gen.py:310-318 pattern, computed-site transpose ---
    cand = [vtrue + 1, vtrue - 1]
    rng.shuffle(cand)
    d1 = next((x for x in cand
               if kn["value_min"] <= x <= kn["value_max"]
               and lo <= x <= hi
               and x not in base_forbidden), None)
    if d1 is None:
        return None, "pm1_starved"

    # --- large candidate: e13_gen.py:322-334 pattern, same transpose ---------
    d3 = _pick_large(vtrue, rng, base_forbidden | {d1}, kn, lo, hi)
    if d3 is None:
        return None, "large_starved"

    # --- fail-closed family audits (MOD-9/MOD-10/disc reads) on BOTH plants --
    cfw = {}
    for tag, dv in (("pm1", d1), ("large", d3)):
        ffails, w = gp.audit_family(prog, stmts, tw, v, dv, j, kn)
        if ffails:
            return None, "%s_famaudit:%s" % (tag, ";".join(ffails))
        cfw[tag] = w

    plant = {"pm10_20": (d_anchor, pol_anchor, out_cf_anchor),
             "pm1": (d1, DELTA_POLICY["pm1"], cfw["pm1"]["out_value"]),
             "large": (d3, DELTA_POLICY["large"], cfw["large"]["out_value"])}

    fams = {}
    for opac in OPACITIES:
        for tag in REGIMES:
            dv, pol, ocf = plant[tag]
            fams["recon_%s_%s" % (opac, tag)] = {
                "planted_var": v, "planted_value": dv, "true_value": vtrue,
                "force_after_line": j, "delta_policy": pol,
                "transform": "value", "opacity": opac, "out_cf": ocf,
                "site_body": gd.render_site_line_body(prog, opac, dv),
                "site_body_true": gd.render_site_line_body(prog, opac, vtrue),
                "delta_regime": tag, "e14_cell": "recon_%s_%s" % (opac, tag),
                "e14_arm": "e14_recon",
                # --- measurement-separation bookkeeping (v2) ---------------
                # persisted so the runner never has to re-derive them and the
                # summarize stage can bound the separation-induced bias.
                "out_true": out_true,
                "separation": abs(ocf - out_true),
                "abs_delta": abs(dv - vtrue),
                "last_digit_preserved": (dv % 10 == vtrue % 10)}
    prog["families"].update(fams)
    prog["recon_families"] = list(RECON_FAMILIES)
    prog["e14_cell_of_family"] = {f: f for f in RECON_FAMILIES}
    prog["e14_regime_of_family"] = {
        "recon_%s_%s" % (o, t): t for o in OPACITIES for t in REGIMES}
    prog["e14_design"] = "3x2_delta_regime_x_opacity"
    return prog, None


def post_checks(prog):
    """Post-hoc, fail-closed. Mirrors e13_gen.audit_grid's discipline:
    magnitude window per regime, unit-digit contract per regime, cross-regime
    value distinctness, attainable-sum window membership, anchor byte-identity,
    separation bookkeeping, and the injection-line round trip the runner will
    perform."""
    fails = []
    fams = prog["families"]
    vtrue = prog["site_true_value"]
    out_true = prog.get("out_true")
    j = prog["site_line"]
    lo, hi = attainable_window()

    for f in RECON_FAMILIES:
        if f not in fams:
            fails.append("missing_%s" % f)
    if fails:
        return fails

    for f in RECON_FAMILIES:
        spec = fams[f]
        tag = spec.get("delta_regime")
        if tag not in REGIMES:
            fails.append("bad_regime_%s_%s" % (f, tag))
            continue
        pv = spec["planted_value"]
        mag = abs(pv - vtrue)
        if not MAG_OK[tag](mag):
            fails.append("magnitude_%s_%d" % (f, mag))
        # unit-digit contract: this is what makes factor A identifiable
        if (pv % 10 == vtrue % 10) != LAST_DIGIT_PRESERVED[tag]:
            fails.append("last_digit_contract_%s" % f)
        # attainable-sum window (a `full` line outside it is a zero-compute tell)
        if not (lo <= pv <= hi):
            fails.append("outside_attainable_window_%s_%d" % (f, pv))
        # separation bookkeeping must be self-consistent
        if spec.get("out_true") != out_true:
            fails.append("out_true_mismatch_%s" % f)
        if spec.get("separation") != abs(spec["out_cf"] - out_true):
            fails.append("separation_mismatch_%s" % f)
        if spec.get("abs_delta") != mag:
            fails.append("abs_delta_mismatch_%s" % f)
        if spec.get("last_digit_preserved") != (pv % 10 == vtrue % 10):
            fails.append("last_digit_flag_mismatch_%s" % f)

    # the three regimes must be pairwise distinct values on the same world
    vals = [fams["recon_bare_%s" % t]["planted_value"] for t in REGIMES]
    if len(set(vals)) != len(REGIMES):
        fails.append("regime_values_collide")
    # opacity arms of a regime must share ONE plant (paired by construction)
    for tag in REGIMES:
        if fams["recon_bare_%s" % tag]["planted_value"] != \
                fams["recon_full_%s" % tag]["planted_value"]:
            fails.append("opacity_plant_not_paired_%s" % tag)

    # ANCHOR: pm10_20 site_body byte-identical to E11's depth_k1_* site_body
    for rf, af in ANCHOR_PAIRS.items():
        if af not in fams:
            fails.append("anchor_missing_%s" % af)
        elif fams[rf]["site_body"] != fams[af]["site_body"]:
            fails.append("anchor_body_mismatch_%s" % rf)
        elif fams[rf]["planted_value"] != fams[af]["planted_value"]:
            fails.append("anchor_value_mismatch_%s" % rf)

    # injection round trip: the exact line e11_run.build_injection_e11 emits
    # ("line %d: %s" % (j, site_body), e11_run.py:167) must parse back to the
    # planted value on the site line.
    for f in RECON_FAMILIES:
        injected = "line %d: %s" % (j, fams[f]["site_body"])
        p = tf.parse_trace_line(injected)
        if p is None or p.get("kind") != "assign":
            fails.append("splice_unparsed_%s" % f)
        elif (p["line"] != j or p["var"] != prog["site_var"]
              or p["value"] != fams[f]["planted_value"]):
            fails.append("splice_roundtrip_%s" % f)
    return fails


def generate(n_worlds=N_WORLDS, base_seed=BASE_SEED, max_attempts_factor=12):
    audit_fn = ad.make_audit_fn(targets())
    progs = []
    rejects = Counter()
    deltas = {f: Counter() for f in RECON_FAMILIES}
    signed = {f: Counter() for f in RECON_FAMILIES}
    seps = {f: Counter() for f in RECON_FAMILIES}
    lastd = {f: Counter() for f in RECON_FAMILIES}
    stage = Counter()
    vtrue_hist = Counter()
    starve_vtrue = Counter()
    seed = base_seed
    attempts = 0
    budget = n_worlds * max_attempts_factor
    while len(progs) < n_worlds and attempts < budget:
        attempts += 1
        s = seed
        seed += 1
        try:
            prog = gd.build_program(K, s)
        except (gd.Reject, gp.Reject) as r:
            rejects["build:%s" % str(r.reason).split(":")[0]] += 1
            continue
        stage["built"] += 1

        pre = audit_fn(prog)                    # E11 stack BEFORE recon families
        if pre:
            rejects["preaudit:%s" % pre[0]] += 1
            continue
        stage["preaudit_ok"] += 1

        vt = prog["site_true_value"]
        prog, err = add_recon_families(prog, s)
        if err:
            rejects[err.split(":")[0]] += 1
            if err.startswith("large_starved"):
                starve_vtrue[vt] += 1
            continue
        stage["recon_attached"] += 1

        post = audit_fn(prog)                   # FULL E11 stack WITH recon fams
        if post:
            rejects["postaudit:%s" % post[0]] += 1
            continue
        stage["postaudit_ok"] += 1

        pc = post_checks(prog)
        if pc:
            rejects["postcheck:%s" % pc[0]] += 1
            continue
        stage["postcheck_ok"] += 1

        for f in RECON_FAMILIES:
            spec = prog["families"][f]
            dv = spec["planted_value"] - prog["site_true_value"]
            deltas[f][abs(dv)] += 1
            signed[f][dv] += 1
            seps[f][spec["separation"]] += 1
            lastd[f][spec["last_digit_preserved"]] += 1
        vtrue_hist[prog["site_true_value"]] += 1
        prog["e14_base_seed"] = base_seed
        prog["e14_world_seed"] = s
        progs.append(prog)

    def _bucket(counter, width=20):
        b = Counter()
        for kk, vv in counter.items():
            b[(kk // width) * width] += vv
        return {str(kk): b[kk] for kk in sorted(b)}

    funnel = {
        "design": "3x2_delta_regime_x_opacity",
        "regimes": list(REGIMES), "opacities": list(OPACITIES),
        "families": list(RECON_FAMILIES),
        "attainable_window": list(attainable_window()),
        "base_seed": base_seed, "n_requested": n_worlds,
        "attempts": attempts, "accepted": len(progs),
        "last_seed_tried": seed - 1,
        "starved": len(progs) < n_worlds,
        "stage_counts": dict(stage),
        "rejects": dict(rejects),
        "abs_delta_distribution": {f: dict(sorted(deltas[f].items()))
                                   for f in RECON_FAMILIES},
        "signed_delta_distribution": {f: dict(sorted(signed[f].items()))
                                      for f in RECON_FAMILIES},
        "separation_distribution": {f: dict(sorted(seps[f].items()))
                                    for f in RECON_FAMILIES},
        "last_digit_preserved_counts": {
            f: {str(kk): vv for kk, vv in sorted(lastd[f].items())}
            for f in RECON_FAMILIES},
        "vtrue_histogram_bucket20": _bucket(vtrue_hist),
        "large_starved_vtrue_histogram_bucket20": _bucket(starve_vtrue),
        "large_starved_vtrue_range": (
            [min(starve_vtrue), max(starve_vtrue)] if starve_vtrue else None),
    }
    return progs, funnel


def main(argv=None):
    ap = argparse.ArgumentParser(description="E14 recon world generator (CPU)")
    ap.add_argument("--out-dir", required=True,
                    help="worlds dir; writes recon_worlds.jsonl + funnel json")
    ap.add_argument("--n", type=int, default=N_WORLDS)
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--worlds-name", default="recon_worlds.jsonl")
    ap.add_argument("--funnel-name", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)

    os.makedirs(a.out_dir, exist_ok=True)
    wpath = os.path.join(a.out_dir, a.worlds_name)
    fpath = os.path.join(a.out_dir, a.funnel_name or
                         a.worlds_name.replace("recon_worlds", "recon_funnel")
                         .replace(".jsonl", ".json"))
    if os.path.exists(wpath) and not a.overwrite:
        raise SystemExit("%s exists; use --overwrite" % wpath)

    progs, funnel = generate(a.n, a.base_seed)

    tmp = wpath + ".tmp"
    with open(tmp, "w") as fh:
        for p in progs:
            fh.write(json.dumps(p, sort_keys=True) + "\n")
    os.replace(tmp, wpath)

    # per-cell counts (identical by construction: all six families ride the
    # same world -- report them anyway so a build regression is visible)
    funnel["per_cell_worlds"] = {
        f: sum(1 for p in progs if f in p["families"]) for f in RECON_FAMILIES}
    funnel["anchor_byte_identical_worlds"] = sum(
        1 for p in progs
        if all(p["families"][rf]["site_body"] == p["families"][af]["site_body"]
               for rf, af in ANCHOR_PAIRS.items()))
    funnel["vtrue_min"] = min((p["site_true_value"] for p in progs), default=None)
    funnel["vtrue_max"] = max((p["site_true_value"] for p in progs), default=None)
    funnel["example"] = ({
        "program_id": progs[0]["program_id"],
        "site_true_value": progs[0]["site_true_value"],
        "out_true": progs[0]["out_true"],
        "operands": progs[0]["operands"],
        "bodies": {f: progs[0]["families"][f]["site_body"]
                   for f in sorted(progs[0]["families"])
                   if "site_body" in progs[0]["families"][f]},
        "separation": {f: progs[0]["families"][f]["separation"]
                       for f in RECON_FAMILIES},
    } if progs else None)

    with open(fpath, "w") as fh:
        json.dump(funnel, fh, indent=2, sort_keys=True)
    print(json.dumps(funnel, indent=2, sort_keys=True))
    print("[E14:gen] %d worlds -> %s" % (len(progs), wpath))


if __name__ == "__main__":
    main()
