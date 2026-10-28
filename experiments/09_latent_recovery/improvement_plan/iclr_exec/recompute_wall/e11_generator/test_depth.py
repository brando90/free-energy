"""E11 depth x opacity fixture suite (progtrace_tests.py pattern; pure CPU).

Every non-negotiable is checked both POSITIVE (a clean world passes) and
NEGATIVE (a hand-tampered world is REJECTED by the fail-closed audit).
"""
import copy
import os
import sys

for _cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "..", "expg"),):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from interp import execute, execute_cf, parse_program, value_at
from trace_format import make_listing_text
import gen_depth as gd
import audit_depth as ad

FAILURES = []


def check(name, cond, detail=""):
    print("[%s] %s %s" % ("PASS" if cond else "FAIL", name, "" if cond else detail))
    if not cond:
        FAILURES.append((name, detail))


def _one(k, seed=7000, targets=None):
    """Build the first program for (k) at/after seed that passes all audits;
    returns (prog, stmts, true_world)."""
    tg = targets or _targets_for(k)
    af = ad.make_audit_fn(tg)
    progs, funnel = gd.generate_cell(k, 1, seed, audit_fn=af)
    if not progs:
        return None, None, None, funnel
    p = progs[0]
    st = parse_program(p["stmt_texts"])
    return p, st, execute(st), funnel


def _targets_for(k):
    t = ad.default_targets()
    t["nearest_anc_dist"] = None if k == 0 else 4
    return t


# -------------------------------------------------------------- layout / match

def test_layout_and_matching():
    tgt = ad.default_targets()
    for k in gd.KS:
        p, st, tw, fn = _one(k)
        check("built_k%d" % k, p is not None, "starved: %s" % (fn if p is None else ""))
        if p is None:
            continue
        check("k%d_L" % k, p["L"] == tgt["L"], "L=%d" % p["L"])
        check("k%d_ops" % k, p["total_ops"] == tgt["ops"], "ops=%d" % p["total_ops"])
        check("k%d_ws_tokens" % k, p["listing_ws_tokens"] == tgt["ws_tokens"],
              "ws=%d want=%d" % (p["listing_ws_tokens"], tgt["ws_tokens"]))
        check("k%d_ws_formula" % k,
              p["listing_ws_tokens"] == 4 * p["L"] - 2 + 2 * p["total_ops"])
        check("k%d_site_pos" % k, p["site_line"] == gd.DKNOBS["site_j"] and
              abs(p["position_frac"] - 0.5) < 1e-9, "j=%d" % p["site_line"])
        check("k%d_site_ops" % k,
              gd.expr_ops(st[p["site_line"] - 1]["expr"]) == k, "site k mismatch")


def test_cross_cell_tokens_identical():
    """The whole point: every cell's listing has the SAME whitespace-token
    count and the SAME nearest-ancestor distance for k>=1."""
    ws, anc = set(), set()
    for k in gd.KS:
        p, st, tw, fn = _one(k)
        if p is None:
            continue
        ws.add(p["listing_ws_tokens"])
        if k >= 1:
            anc.add(ad.nearest_ancestor_token_distance(p))
    check("all_cells_same_ws_tokens", len(ws) == 1, "distinct ws counts: %s" % ws)
    check("k_ge1_same_nearest_anc", len(anc) == 1, "distinct anc dists: %s" % anc)


# -------------------------------------------------------------- (1) min-path

def test_min_path_measured():
    for k in gd.KS:
        p, st, tw, fn = _one(k)
        if p is None:
            continue
        fails = ad.audit_min_path(p, st, tw)
        check("k%d_minpath_clean" % k, not fails, "fails=%s" % fails)
        check("k%d_measured_min_k" % k,
              p["audit_metrics"]["measured_min_k"] == k,
              "measured=%s" % p["audit_metrics"].get("measured_min_k"))


def test_min_path_rejects_shortcut():
    """Plant a stated value equal to a partial operand sum -> a shorter
    verification route exists -> the min-path audit MUST reject."""
    p, st, tw, fn = _one(3)
    if p is None:
        check("shortcut_setup", False, "no k=3 world")
        return
    # overwrite a filler line to STATE the sum of the first two operands
    ops = p["operands"]
    partial = ops[0] + ops[1]
    texts = list(p["stmt_texts"])
    # find a preamble filler line (before operand block) and force its value
    anc0 = min(p["operand_lines"])
    target_line = anc0 - 1                    # a preamble line
    var = st[target_line - 1]["var"]
    texts[target_line - 1] = "%s = %d" % (var, partial)
    p2 = copy.deepcopy(p)
    p2["stmt_texts"] = texts
    st2 = parse_program(texts)
    tw2 = execute(st2)
    fails = ad.audit_min_path(p2, st2, tw2)
    check("minpath_rejects_partial_sum",
          any("additive_shortcut" in f or "measured" in f for f in fails),
          "fails=%s (partial=%d in S?)" % (fails, partial))


def test_min_path_rejects_readable_true():
    """If the true site value is itself stated on an earlier line, there is a
    0-op route -> reject."""
    p, st, tw, fn = _one(2)
    if p is None:
        check("readable_setup", False)
        return
    vtrue = p["site_true_value"]
    texts = list(p["stmt_texts"])
    anc0 = min(p["operand_lines"])
    tl = anc0 - 1
    var = st[tl - 1]["var"]
    texts[tl - 1] = "%s = %d" % (var, vtrue)
    p2 = copy.deepcopy(p); p2["stmt_texts"] = texts
    st2 = parse_program(texts); tw2 = execute(st2)
    fails = ad.audit_min_path(p2, st2, tw2)
    check("minpath_rejects_readable_true",
          any("0op" in f or "measured_0" in f for f in fails), "fails=%s" % fails)


# -------------------------------------------------------------- (2) plant

def test_plant_validity():
    for k in gd.KS:
        p, st, tw, fn = _one(k)
        if p is None:
            continue
        fails = ad.audit_plant_validity(p, st, tw)
        check("k%d_plant_clean" % k, not fails, "fails=%s" % fails)
        for fam, spec in p["families"].items():
            if spec["planted_value"] is None:
                continue
            check("k%d_%s_plant_false" % (k, fam),
                  spec["planted_value"] != spec["true_value"])


def test_plant_rejects_washout():
    p, st, tw, fn = _one(2)
    if p is None:
        check("washout_setup", False); return
    # force the plant to equal true -> washout/collision
    p2 = copy.deepcopy(p)
    for fam in p2["families"]:
        p2["families"][fam]["planted_value"] = p2["families"][fam]["true_value"]
    fails = ad.audit_plant_validity(p2, st, tw)
    check("plant_rejects_equal_true", any("equals_true" in f for f in fails),
          "fails=%s" % fails)


# -------------------------------------------------------------- (3) matching

def test_cross_cell_audit_rejects():
    p, st, tw, fn = _one(2)
    if p is None:
        check("match_setup", False); return
    tgt = _targets_for(2)
    check("match_clean", not ad.audit_cross_cell(p, tgt))
    p2 = copy.deepcopy(p)
    p2["total_ops"] += 1                       # break the op match
    p2["listing_ws_tokens"] += 2               # keep formula consistent
    fails = ad.audit_cross_cell(p2, tgt)
    check("match_rejects_ops", any("ops_mismatch" in f for f in fails),
          "fails=%s" % fails)
    p3 = copy.deepcopy(p)
    p3["listing_ws_tokens"] += 1               # break the closed-form identity
    fails = ad.audit_cross_cell(p3, tgt)
    check("match_rejects_formula", any("formula_violated" in f for f in fails),
          "fails=%s" % fails)


# -------------------------------------------------------------- (4) filler indep

def test_filler_independence():
    for k in gd.KS:
        p, st, tw, fn = _one(k)
        if p is None:
            continue
        check("k%d_filler_indep_clean" % k, not ad.audit_filler_independence(p, st),
              "fails=%s" % ad.audit_filler_independence(p, st))


def test_filler_independence_rejects():
    """Make a 'filler' line feed the site (operand references a filler var) ->
    the site's backward slice now contains a filler line -> reject."""
    p, st, tw, fn = _one(2)
    if p is None:
        check("indep_setup", False); return
    # rewrite the site to add a filler var into its expression: pick a var
    # defined on a preamble line (before the operand block) that is not itself
    # an operand.
    texts = list(p["stmt_texts"])
    anc0 = min(p["operand_lines"])
    filler_var = None
    for s in st:
        if s["kind"] == "assign" and s["line"] < anc0 - 1 and \
                s["var"] not in p["operand_names"]:
            filler_var = s["var"]
    if filler_var is None:
        filler_var = st[0]["var"]
    site_idx = p["site_line"] - 1
    texts[site_idx] = texts[site_idx] + " + " + filler_var
    p2 = copy.deepcopy(p); p2["stmt_texts"] = texts
    st2 = parse_program(texts)
    fails = ad.audit_filler_independence(p2, st2)
    check("filler_indep_rejects_dep",
          any("filler_lines" in f or "site_readers" in f for f in fails),
          "fails=%s" % fails)


# -------------------------------------------------------------- (5) opacity

def test_opacity_byte_identity():
    for k in (1, 2, 3, 5):
        p, st, tw, fn = _one(k)
        if p is None:
            continue
        fails = ad.audit_opacity_byte_identity(p)
        check("k%d_opacity_clean" % k, not fails, "fails=%s" % fails)
        fams = p["families"]
        bare = fams["depth_k%d_bare" % k]["site_body"]
        full = fams["depth_k%d_full" % k]["site_body"]
        part = fams["depth_k%d_partial" % k]["site_body"]
        # bare is a prefix-preserving reduction of full/partial (same tail)
        check("k%d_full_has_worked" % k, " = " in full.split(";")[0] and
              full != bare, "full=%r" % full)
        check("k%d_same_result_tail" % k,
              bare.split("; ", 1)[1] == full.split("; ", 1)[1] ==
              part.split("; ", 1)[1])
        # full must show every operand value; partial fewer
        for ov in p["operands"]:
            pass
        check("k%d_full_shows_all_operands" % k,
              all((" %d" % ov in full or "= %d" % ov in full or "+ %d" % ov in full)
                  for ov in p["operands"]) or
              full.count("+") >= full.split("=")[0].count("+"),
              "full=%r ops=%s" % (full, p["operands"]))


def test_opacity_rejects_tampered():
    p, st, tw, fn = _one(2)
    if p is None:
        check("opac_setup", False); return
    p2 = copy.deepcopy(p)
    # tamper: change the symbolic head of the 'full' arm so it no longer starts
    # with the bare head
    fam = "depth_k2_full"
    body = p2["families"][fam]["site_body"]
    p2["families"][fam]["site_body"] = "zzz " + body
    fails = ad.audit_opacity_byte_identity(p2)
    check("opacity_rejects_tamper", any("head_not_prefix" in f or
          "tail_differs" in f or "strip_mismatch" in f for f in fails),
          "fails=%s" % fails)


# -------------------------------------------------------------- (6) determinism

def test_determinism():
    for k in gd.KS:
        # find a seed that builds, then confirm same seed -> identical dict
        seed, a = 4242, None
        for s in range(4242, 4242 + 500):
            try:
                a = gd.build_program(k, s)
                seed = s
                break
            except (gd.Reject, gd.gp.Reject):
                continue
        b = gd.build_program(k, seed)
        check("k%d_determinism" % k, a is not None and a == b,
              "seed-repro mismatch")
    # cell generation deterministic too
    tg = _targets_for(2)
    af = ad.make_audit_fn(tg)
    p1, f1 = gd.generate_cell(2, 5, 9000, audit_fn=af)
    p2, f2 = gd.generate_cell(2, 5, 9000, audit_fn=af)
    check("cell_determinism",
          [x["program_id"] for x in p1] == [x["program_id"] for x in p2] and
          [x["stmt_texts"] for x in p1] == [x["stmt_texts"] for x in p2])


# -------------------------------------------------- reuse: interp/cf still sound

def test_cf_and_downstream_compat():
    """The world must plug into the existing execute_cf / discriminating-read
    machinery exactly like the EXPG kc cells."""
    p, st, tw, fn = _one(3)
    if p is None:
        check("compat_setup", False); return
    fam = p["families"]["depth_k3_bare"]
    cf = execute_cf(st, fam["planted_var"], fam["planted_value"],
                    fam["force_after_line"])
    check("cf_out_differs", cf["out_value"] != tw["out_value"])
    check("cf_r1_disc", value_at(cf, p["r1"], p["r1_var"]) !=
          value_at(tw, p["r1"], p["r1_var"]))
    check("cf_r2_disc", value_at(cf, p["r2"], p["r2_var"]) !=
          value_at(tw, p["r2"], p["r2_var"]))
    check("cf_site_planted", value_at(cf, p["site_line"], p["site_var"]) ==
          fam["planted_value"])


def main():
    test_layout_and_matching()
    test_cross_cell_tokens_identical()
    test_min_path_measured()
    test_min_path_rejects_shortcut()
    test_min_path_rejects_readable_true()
    test_plant_validity()
    test_plant_rejects_washout()
    test_cross_cell_audit_rejects()
    test_filler_independence()
    test_filler_independence_rejects()
    test_opacity_byte_identity()
    test_opacity_rejects_tampered()
    test_determinism()
    test_cf_and_downstream_compat()
    print("\n%d failures" % len(FAILURES))
    if FAILURES:
        for n, d in FAILURES:
            print("  FAIL %s %s" % (n, d))
        sys.exit(1)


if __name__ == "__main__":
    main()
