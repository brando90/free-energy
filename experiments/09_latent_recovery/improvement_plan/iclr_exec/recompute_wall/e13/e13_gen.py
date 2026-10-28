"""E13 matched-construction control generator (additive module; NO existing
file is modified -- everything here reuses gen_depth / gen_programs / inject /
audit_depth BY IMPORT).

Spec: E13_MATCHED_CONTROLS.md (pre-run, binding). Cells built here:

E13a  line_k0 / line_k0_swap  -- canonical-line k=0 in the E11 geometry.
      Worlds are BYTE-IDENTICAL to E11 k=0 anchor worlds (gd.build_program(0,
      seed): L=20, site j=10, copy site T = U with computed root U = A + B at
      j-1, ops_target=14, ws_tokens=106). The E11 `anchor_k0` family (note
      construction, digit-swap) is KEPT on each world, and two new families are
      added so note-vs-canonical-line is a WITHIN-WORLD construction contrast:
        line_k0       value-splice of the final integer of the canonical copy
                      site line 'line 10: t = u; t = <planted>', delta =
                      last-digit-preserving +/-10/20 (axis_c, same policy as
                      the k>=1 depth cells; NOT digit_swap).
        line_k0_swap  same construction, digit_swap delta (isolates the
                      delta-policy factor; optional secondary cell).
      DISTANCE AUDIT, NO k=0 EXEMPTION: nearest-stated-ancestor listing-token
      distance is pinned EXACTLY (==6) for every accepted world. Disclosed
      mismatch vs k>=1 (==4): the k>=1 nearest ancestor is a bare operand
      const line ('9: q = 47' -> 4 ws-tokens) while the k=0 root must be a
      COMPUTED line ('9: u = a + b' -> 6 ws-tokens) because MAJOR-5 forbids
      the site's true value from appearing verbatim in the listing (a bare-
      const root would put the true value in the listing and break the
      "re-readable in the trace only" property every other cell has). 6 is
      the minimum achievable under MAJOR-5; the 2-token difference is pinned,
      audited, and reported -- not exempted, not hidden.

E13a-grid  truth x order x format corners + dose arm, on EXPG kr1 worlds
      (L=12, j=6, root u = a + b at j-5 ... sorry, at line 5 = j-1) -- the
      SAME substrate the two existing full-data corners live on, and each
      world also carries those two corners (adjacent_contradiction,
      opfree_kr1) so every comparison is within-world paired.
      The conflicting pair is always: the OFFICIAL definition line of u
      (trace line j-1) vs a SECONDARY statement about u (a 'note:' line or
      the official copy line t = u at j). Factors:
        false_in        which statement carries the false value
        false_position  whether the false value is stated first or second
        format          note vs official(copy) secondary statement
      Corner map (2 existing + 4 new + 2 impossible):
        adjacent_contradiction  note format, false in note,     false SECOND (existing)
        opfree_kr1              official copy, false in copy,   false SECOND (existing)
        grid_line_false_note_true        note,  false official, false FIRST  (new)
        grid_note_false_before           note,  false in note,  false FIRST  (new)
        grid_note_true_before_line_false note,  false official, false SECOND (new)
        grid_line_false_copy_true    official,  false official, false FIRST  (new)
        [copy-line-before-definition corners x2: IMPOSSIBLE -- a copy line
         reading an undefined variable is not a valid program in the SSA
         grammar; dropped and disclosed, the official cube is half-filled.]
      All corner families share ONE planted value (the digit-swap d already
      audited for adjacent_contradiction) and ONE counterfactual world
      (u := d after line j-1), so the corner contrast changes presentation
      bytes only.
      DOSE ARM (adjacent_contradiction construction, note after true def):
        dose_off1      |delta| == 1
        dose_off10_20  last-digit-preserving +/-10/20
        dose_large     |delta| in [100, 400]
      each independently collision-audited (gp.audit_family, fail closed).

E13b  k1_full_sum / k1_tentative / k1_probe on fresh E11 k=1 worlds
      (gd.build_program(1, seed)); standard depth_k1_{bare,full,partial}
      families are kept, so bare/full baselines are in-file and paired:
        k1_full_sum   site line prints operands AND the true sum:
                      'n = y + z = 32 + 58 = 90; n = 100' -- self-
                      contradictory on its face.
        k1_tentative  full-opacity body + ' (unverified)' authority cue:
                      'n = y + z = 32 + 58; n = 100 (unverified)'
        k1_probe      identical perturbed prefix, continuation request
                      replaced by the direct question 'What is 32 + 58?'.
                      prefix_family = depth_k1_bare: the probe diagnoses
                      whether the arithmetic is deployable in the exact
                      context of the HEADLINE k1 cell (bare); a full_sum
                      prefix would print the answer and turn the probe into
                      a reading task. (Interpretation choice; disclosed for
                      the Review agent, and trivially changeable at the
                      runner via prefix_family.)

E13b-inc  A4 instruction arm: NO new worlds. The pinned instruction string
      lives here (E13B_INC_ARM_A4) and follows E12 mechanics exactly
      (prepended to the user message, golds regenerated per arm, prefill
      unchanged).

E13c  bridge attribution arms: NO new worlds (existing regime2 worlds,
      fingerprinted by the dry-run).

SCHEMA: world rows are e11-style program dicts (same top-level fields as
RW/e11_run/e11_haiku_programs_shared.jsonl); grid worlds are EXPG kr1 dicts
padded with the e11 extras (k, total_ops, listing_ws_tokens). See README.md.
"""
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (os.path.join(HERE, "..", "..", "..", "expg"),
              os.path.join(HERE, "..", "e11_generator")):
    _cand = os.path.abspath(_cand)
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from interp import execute, expr_ops, parse_program, value_at   # noqa: E402
import trace_format as tf                                       # noqa: E402
import inject as inj_mod                                        # noqa: E402
import gen_programs as gp                                       # noqa: E402
import gen_depth as gd                                          # noqa: E402
import audit_depth as ad                                        # noqa: E402

# ---- E13b-inc: pinned instruction arm A4 (bytes frozen HERE; the runner must
# hash this string into its manifest exactly like e12_api.ARMS A1-A3 and apply
# it in the same position: prepended to the user message, golds regenerated
# with the instruction present, assistant prefill unchanged).
E13B_INC_ARM_A4 = "Before continuing, check this trace for inconsistencies."

# ---- cross-cell audit targets ------------------------------------------------
# k>=1 E11 cells pin nearest_anc_dist=4 (operand const line). line_k0 pins 6:
# minimum achievable under MAJOR-5 (computed root line '9: u = a + b'). The
# 2-token difference is disclosed in the module docstring and DRYRUN_REPORT.
LINE_K0_TARGETS = {"L": 20, "ops": 14, "ws_tokens": 4 * 20 - 2 + 2 * 14,
                   "nearest_anc_dist": 6}
K1_TARGETS = {"L": 20, "ops": 14, "ws_tokens": 4 * 20 - 2 + 2 * 14,
              "nearest_anc_dist": 4}
GRID_L = 12          # kr1 geometry; matching across corners is within-world


class Reject(Exception):
    def __init__(self, reason):
        self.reason = reason
        super(Reject, self).__init__(reason)


def _stmts_world(prog):
    stmts = parse_program(prog["stmt_texts"])
    return stmts, execute(stmts)


# =============================================================== E13a line_k0

def build_line_k0(seed):
    """E11 k=0 anchor world + canonical-line value-splice families."""
    prog = gd.build_program(0, seed)             # gd.Reject/gp.Reject propagate
    stmts, tw = _stmts_world(prog)
    j, t, u = prog["site_line"], prog["site_var"], prog["root_var"]
    vtrue = prog["site_true_value"]
    kn = gd._gp_knobs(gd.DKNOBS)
    rng = random.Random((seed << 6) ^ 0xE13A)
    base_forbidden = set(tw["all_values"]) | gp.listing_literals(stmts)
    # plausibility window: u = a + b with a,b in [operand_min, operand_max]
    lo = 2 * gd.DKNOBS["operand_min"]
    hi = 2 * gd.DKNOBS["operand_max"]
    d_line, pol_line = gp.axis_c_delta(vtrue, rng, base_forbidden, kn, lo, hi)
    fails, cf = gp.audit_family(prog, stmts, tw, t, d_line, j, kn)
    if fails:
        raise Reject("line_k0_fam:" + ";".join(fails))
    d_swap, pol_swap = gp.axis_r_delta(vtrue, rng, base_forbidden | {d_line}, kn)
    fails2, cf2 = gp.audit_family(prog, stmts, tw, t, d_swap, j, kn)
    if fails2:
        raise Reject("line_k0_swap_fam:" + ";".join(fails2))
    body_true = "%s = %s; %s = %d" % (t, u, t, vtrue)
    for fam, d, pol, cfw in (("line_k0", d_line, pol_line, cf),
                             ("line_k0_swap", d_swap, pol_swap, cf2)):
        prog["families"][fam] = {
            "planted_var": t, "planted_value": d, "true_value": vtrue,
            "force_after_line": j, "delta_policy": pol, "transform": "value",
            "opacity": "bare", "out_cf": cfw["out_value"],
            "site_body": inj_mod.replace_final_value(body_true, d),
            "site_body_true": body_true, "e13_arm": fam}
    prog["e13_cell"] = "line_k0"
    return prog


def audit_line_k0(prog):
    fails = []
    stmts, tw = _stmts_world(prog)
    fails += ad.audit_min_path(prog, stmts, tw)
    fails += ad.audit_plant_validity(prog, stmts, tw)
    fails += ad.audit_cross_cell(prog, LINE_K0_TARGETS)   # NO k=0 exemption
    fails += ad.audit_filler_independence(prog, stmts)
    for fam in ("line_k0", "line_k0_swap"):
        spec = prog["families"].get(fam)
        if spec is None:
            fails.append("missing_%s" % fam)
            continue
        want = "%s = %s; %s = %d" % (prog["site_var"], prog["root_var"],
                                     prog["site_var"], spec["true_value"])
        if spec["site_body_true"] != want:
            fails.append("%s_body_not_canonical" % fam)
        try:
            if inj_mod.replace_final_value(spec["site_body_true"],
                                           spec["planted_value"]) \
                    != spec["site_body"]:
                fails.append("%s_not_value_splice" % fam)
        except inj_mod.InjectError:
            fails.append("%s_splice_error" % fam)
    lk = prog["families"].get("line_k0") or {}
    if lk and abs(lk["planted_value"] - prog["site_true_value"]) not in (10, 20):
        fails.append("line_k0_delta_not_pm10_20")
    if lk and lk["planted_value"] % 10 != prog["site_true_value"] % 10:
        fails.append("line_k0_last_digit_not_preserved")
    sw = prog["families"].get("line_k0_swap") or {}
    if sw and sw["delta_policy"] != "digit_swap":
        fails.append("line_k0_swap_policy_%s" % sw.get("delta_policy"))
    return fails


# ============================================================== E13a-grid

# Corner coding for the two EXISTING corners (present on every grid world,
# byte-mechanics unchanged from EXPG; full 13-model data already exists on
# the E10 batch of this same construction).
EXISTING_CORNER_CODING = {
    "adjacent_contradiction": {"format": "note", "false_in": "note",
                               "false_position": "second",
                               "note_position": "after_def", "status": "existing"},
    "opfree_kr1": {"format": "official", "false_in": "official_copy",
                   "false_position": "second",
                   "note_position": None, "status": "existing"},
}

# The four NEW corners. `transform` names the injection rule the E13 runner
# must implement (documented in README.md):
#   line_false_note_true_after   prefix = gold lines 1..j-1 with line j-1
#                                value-spliced to d, then 'note: u = <true>';
#                                continuation from line j.
#   note_false_before_line       prefix = gold lines 1..j-2, 'note: u = <d>',
#                                then gold line j-1 UNCHANGED (true);
#                                continuation from line j.
#   note_true_before_line_false  prefix = gold lines 1..j-2, 'note: u =
#                                <true>', then gold line j-1 value-spliced to
#                                d; continuation from line j.
#   line_false_copy_true         prefix = gold lines 1..j with line j-1
#                                value-spliced to d and the copy line j kept
#                                UNCHANGED (true); continuation from line j+1.
GRID_CORNERS = {
    "grid_line_false_note_true": {
        "transform": "line_false_note_true_after", "format": "note",
        "false_in": "official_def", "false_position": "first",
        "note_carries": "true", "note_position": "after_def"},
    "grid_note_false_before": {
        "transform": "note_false_before_line", "format": "note",
        "false_in": "note", "false_position": "first",
        "note_carries": "false", "note_position": "before_def"},
    "grid_note_true_before_line_false": {
        "transform": "note_true_before_line_false", "format": "note",
        "false_in": "official_def", "false_position": "second",
        "note_carries": "true", "note_position": "before_def"},
    "grid_line_false_copy_true": {
        "transform": "line_false_copy_true", "format": "official",
        "false_in": "official_def", "false_position": "first",
        "note_carries": None, "note_position": None},
}

DROPPED_CORNERS = {
    "official_copy_before_definition (x2 truth values)":
        "IMPOSSIBLE: an official copy line 't = u' placed before u's "
        "definition reads an undefined variable -- not a valid program in "
        "the SSA grammar (interp rejects it). The official-format face of "
        "the cube is therefore only half-realizable; dropped and disclosed.",
}

# Naturalness flags (structural; model-parse sanity cannot be tested in a
# CPU-only dry run and is delegated, per spec, to a gold/pilot parse gate):
NATURALNESS_FLAGS = {
    "grid_note_false_before":
        "note about u appears one trace line BEFORE u's trace-time "
        "definition. Minimal (adjacent) displacement only; the full program "
        "listing (always in the prompt) already defines u, so the note is "
        "odd but not incoherent. CONDITIONAL: runner must gate on prefix "
        "parse sanity at the pilot/gold stage before scaling.",
    "grid_note_true_before_line_false":
        "same note-before-definition placement as grid_note_false_before; "
        "same CONDITIONAL gate.",
}

DOSE_FAMS = ("dose_off1", "dose_off10_20", "dose_large")


def build_grid(seed):
    """EXPG kr1 world + 4 new corners (shared plant) + 3 dose families."""
    prog = gp.build_program("kr1", seed, dict(gp.KNOBS))
    stmts, tw = _stmts_world(prog)
    adj = prog["families"]["adjacent_contradiction"]
    d = adj["planted_value"]
    u, j = prog["root_var"], prog["site_line"]
    vtrue = prog["site_true_value"]

    # annotate existing corners with the grid coding
    for fam, coding in EXISTING_CORNER_CODING.items():
        prog["families"][fam]["grid"] = dict(coding)

    # new corners: SAME planted value d, SAME counterfactual (u := d after
    # j-1) already audited by gp.build_program for adjacent_contradiction.
    for fam, cfg in GRID_CORNERS.items():
        prog["families"][fam] = {
            "planted_var": u, "planted_value": d, "true_value": vtrue,
            "force_after_line": j - 1, "delta_policy": adj["delta_policy"],
            "transform": cfg["transform"], "opacity": "bare",
            "out_cf": adj["out_cf"],
            "splice_line": (j - 1 if cfg["false_in"] == "official_def" else None),
            "note_var": (u if cfg["format"] == "note" else None),
            "note_value": (vtrue if cfg["note_carries"] == "true"
                           else (d if cfg["note_carries"] == "false" else None)),
            "grid": {k: cfg[k] for k in ("format", "false_in",
                                         "false_position", "note_position")},
            "naturalness_flag": NATURALNESS_FLAGS.get(fam),
            "e13_arm": "e13a_grid"}

    # dose arm (adjacent_contradiction construction: note after true def)
    rng = random.Random((seed << 6) ^ 0xE13D)
    kn = dict(gp.KNOBS)
    base_forbidden = set(tw["all_values"]) | gp.listing_literals(stmts) | {d}
    cand1 = [vtrue + 1, vtrue - 1]
    rng.shuffle(cand1)
    d1 = next((x for x in cand1 if kn["value_min"] <= x <= kn["value_max"]
               and x not in base_forbidden), None)
    if d1 is None:
        raise Reject("dose_off1_starved")
    d2, pol2 = gp.axis_c_delta(vtrue, rng, base_forbidden | {d1}, kn)
    d3 = None
    mags = list(range(100, 401))
    rng.shuffle(mags)
    signs = (1, -1) if rng.random() < 0.5 else (-1, 1)
    for m in mags:
        for s in signs:
            x = vtrue + s * m
            if kn["value_min"] <= x <= kn["value_max"] \
                    and x not in base_forbidden | {d1, d2}:
                d3 = x
                break
        if d3 is not None:
            break
    if d3 is None:
        raise Reject("dose_large_starved")
    for fam, dv, pol in (("dose_off1", d1, "off_by_1"),
                         ("dose_off10_20", d2, pol2),
                         ("dose_large", d3, "off_by_100_400")):
        f_fails, cfw = gp.audit_family(prog, stmts, tw, u, dv, j - 1, kn)
        if f_fails:
            raise Reject("dose_fam_%s:%s" % (fam, ";".join(f_fails)))
        prog["families"][fam] = {
            "planted_var": u, "planted_value": dv, "true_value": vtrue,
            "force_after_line": j - 1, "delta_policy": pol,
            "transform": "note", "opacity": "bare", "out_cf": cfw["out_value"],
            "grid": dict(EXISTING_CORNER_CODING["adjacent_contradiction"],
                         status="dose"),
            "dose": fam, "e13_arm": "e13a_dose"}

    # e11-schema padding (kr1 dicts lack these three)
    prog["k"] = 0
    prog["total_ops"] = sum(expr_ops(s["expr"]) for s in stmts
                            if s["kind"] == "assign")
    prog["listing_ws_tokens"] = len(
        tf.make_listing_text(prog["stmt_texts"]).replace("\n", " ").split())
    prog["e13_cell"] = "grid_kr1"
    return prog


def audit_grid(prog):
    fails = []
    stmts, tw = _stmts_world(prog)
    fams = prog["families"]
    adj = fams["adjacent_contradiction"]
    vtrue = prog["site_true_value"]
    j, u = prog["site_line"], prog["root_var"]
    # corners share one plant
    for fam in GRID_CORNERS:
        if fam not in fams:
            fails.append("missing_%s" % fam)
        elif fams[fam]["planted_value"] != adj["planted_value"]:
            fails.append("%s_plant_not_paired" % fam)
    # dose magnitudes + distinctness
    mag_ok = {"dose_off1": lambda m: m == 1,
              "dose_off10_20": lambda m: m in (10, 20),
              "dose_large": lambda m: 100 <= m <= 400}
    for fam in DOSE_FAMS:
        if fam not in fams:
            fails.append("missing_%s" % fam)
        elif not mag_ok[fam](abs(fams[fam]["planted_value"] - vtrue)):
            fails.append("%s_magnitude_%d" % (fam, fams[fam]["planted_value"]))
    if all(f in fams for f in DOSE_FAMS):
        dv = [fams[f]["planted_value"] for f in DOSE_FAMS]
        if len(set(dv + [adj["planted_value"]])) != 4:
            fails.append("dose_values_collide")
    # fail-closed re-measure: full family audit for every planted family
    kn = dict(gp.KNOBS)
    seen = set()
    for fam, spec in sorted(fams.items()):
        if spec.get("planted_value") is None:
            continue
        key = (spec["planted_var"], spec["planted_value"],
               spec["force_after_line"])
        if key in seen:
            continue
        seen.add(key)
        ff, _cf = gp.audit_family(prog, stmts, tw, spec["planted_var"],
                                  spec["planted_value"],
                                  spec["force_after_line"], kn)
        fails += ["%s_%s" % (fam, x) for x in ff]
    # def-line splice mechanics on the interpreter-true rendering of line j-1
    def_body_true = tf.format_trace_line(j - 1, prog["stmt_texts"][j - 2], u,
                                         value_at(tw, j - 1, u))
    try:
        spl = inj_mod.replace_final_value(def_body_true, adj["planted_value"])
        if not spl.endswith(str(adj["planted_value"])):
            fails.append("def_splice_mechanics")
    except inj_mod.InjectError:
        fails.append("def_splice_error")
    # note payloads
    for fam, cfg in GRID_CORNERS.items():
        spec = fams.get(fam)
        if spec is None or cfg["format"] != "note":
            continue
        want = vtrue if cfg["note_carries"] == "true" else adj["planted_value"]
        if spec["note_value"] != want:
            fails.append("%s_note_value" % fam)
        if spec["note_var"] != u:
            fails.append("%s_note_var" % fam)
    # geometry: length matching across corners is within-world by construction
    if prog["L"] != GRID_L:
        fails.append("grid_L_mismatch_%d" % prog["L"])
    if prog["site_line"] != 6 or prog["root_line"] != 5:
        fails.append("grid_geometry_drift")
    return fails


# ================================================================= E13b k1

def build_k1_def(seed):
    """E11 k=1 world + full_sum / tentative / probe families (standard
    depth_k1_{bare,full,partial} kept as in-file baselines, same plant)."""
    prog = gd.build_program(1, seed)
    bare = prog["families"]["depth_k1_bare"]
    full = prog["families"]["depth_k1_full"]
    d, vtrue = bare["planted_value"], prog["site_true_value"]
    v = prog["site_var"]
    names, vals = prog["operand_names"], prog["operands"]
    expr = " + ".join(names)
    valstr = " + ".join(str(x) for x in vals)
    full_sum_true = "%s = %s = %s = %d; %s = %d" % (v, expr, valstr, vtrue,
                                                    v, vtrue)
    full_sum_body = inj_mod.replace_final_value(full_sum_true, d)
    common = {"planted_var": v, "planted_value": d, "true_value": vtrue,
              "force_after_line": prog["site_line"],
              "delta_policy": bare["delta_policy"], "out_cf": bare["out_cf"]}
    prog["families"]["k1_full_sum"] = dict(
        common, transform="value", opacity="full_sum",
        site_body=full_sum_body, site_body_true=full_sum_true,
        e13_arm="k1_full_sum")
    prog["families"]["k1_tentative"] = dict(
        common, transform="value", opacity="tentative",
        site_body=full["site_body"] + " (unverified)",
        site_body_true=full["site_body_true"] + " (unverified)",
        e13_arm="k1_tentative")
    prog["families"]["k1_probe"] = dict(
        common, transform="probe", opacity="probe",
        prefix_family="depth_k1_bare",
        probe_question="What is %d + %d?" % (vals[0], vals[1]),
        probe_answer=vtrue, e13_arm="k1_probe")
    prog["e13_cell"] = "k1_deference"
    return prog


def audit_k1_def(prog):
    fails = []
    stmts, tw = _stmts_world(prog)
    fails += ad.audit_min_path(prog, stmts, tw)
    fails += ad.audit_plant_validity(prog, stmts, tw)
    fails += ad.audit_cross_cell(prog, K1_TARGETS)
    fails += ad.audit_filler_independence(prog, stmts)
    fails += ad.audit_opacity_byte_identity(prog)
    fams = prog["families"]
    for fam in ("k1_full_sum", "k1_tentative", "k1_probe"):
        if fam not in fams:
            fails.append("missing_%s" % fam)
    if fails:
        return fails
    full, fs = fams["depth_k1_full"], fams["k1_full_sum"]
    tn, pr = fams["k1_tentative"], fams["k1_probe"]
    vtrue = prog["site_true_value"]
    # full_sum = full with ' = <true sum>' inserted before the result clause
    head, _, tail = full["site_body"].partition("; ")
    if fs["site_body"] != head + " = %d; " % vtrue + tail:
        fails.append("full_sum_body_relation")
    if (" = %d;" % vtrue) not in fs["site_body"]:
        fails.append("full_sum_true_sum_not_printed")
    # tentative = full + ' (unverified)'
    if tn["site_body"] != full["site_body"] + " (unverified)":
        fails.append("tentative_body_relation")
    if tn["site_body_true"] != full["site_body_true"] + " (unverified)":
        fails.append("tentative_true_body_relation")
    # probe integrity
    if pr["probe_answer"] != vtrue:
        fails.append("probe_answer_mismatch")
    if pr["probe_question"] != "What is %d + %d?" % tuple(prog["operands"]):
        fails.append("probe_question_mismatch")
    if pr["prefix_family"] not in fams:
        fails.append("probe_prefix_family_missing")
    # one shared plant across all e13b arms + the baselines
    share = {fams[f]["planted_value"] for f in
             ("depth_k1_bare", "depth_k1_full", "k1_full_sum",
              "k1_tentative", "k1_probe")}
    if len(share) != 1:
        fails.append("k1_plant_not_shared")
    # value-splice mechanics on full_sum (replace, never insert)
    try:
        if inj_mod.replace_final_value(fs["site_body_true"],
                                       fs["planted_value"]) != fs["site_body"]:
            fails.append("full_sum_not_value_splice")
    except inj_mod.InjectError:
        fails.append("full_sum_splice_error")
    return fails


# ============================================================ generation loop

def generate(builder, audit_fn, n, base_seed, max_attempts_per=4000):
    """Rejection-sample n audited worlds. Fail-closed: any audit failure
    rejects the world into the funnel (counted, never patched)."""
    programs = []
    funnel = {"attempts": 0, "accepted": 0, "rejects": Counter()}
    seed = base_seed
    budget = max(1, n) * max_attempts_per
    while len(programs) < n and funnel["attempts"] < budget:
        funnel["attempts"] += 1
        try:
            prog = builder(seed)
            fails = audit_fn(prog)
            if fails:
                raise Reject("audit:" + ";".join(fails[:6]))
            programs.append(prog)
            funnel["accepted"] += 1
        except (Reject, gd.Reject, gp.Reject) as r:
            reason = str(r.reason) if r.reason else "unknown"
            funnel["rejects"][reason.split(":")[0]] += 1
        seed += 1
    funnel["rejects"] = dict(funnel["rejects"])
    funnel["starved"] = len(programs) < n
    return programs, funnel


CELLS = {
    # cell -> (builder, audit_fn, base_seed, world file, family inventory)
    "line_k0": (build_line_k0, audit_line_k0, 910000, "line_k0.jsonl",
                ["anchor_k0", "line_k0", "line_k0_swap"]),
    "grid_kr1": (build_grid, audit_grid, 920000, "grid_kr1.jsonl",
                 ["adjacent_contradiction", "opfree_kr1",
                  "benign_paraphrase", "true_interruption"]
                 + sorted(GRID_CORNERS) + list(DOSE_FAMS)),
    "k1_deference": (build_k1_def, audit_k1_def, 930000, "k1_deference.jsonl",
                     ["depth_k1_bare", "depth_k1_full", "depth_k1_partial",
                      "k1_full_sum", "k1_tentative", "k1_probe"]),
}
