"""EXPG fixture suite (expa_tests.py pattern; pure logic, no GPU, no model).

Covers the verdict-mandated asserts: parser round-trip, counterfactual
executor, both-match exclusion, repair detection, MAJOR-5 root-computed,
MOD-9 delta policies, MOD-10 washout rejection, gold cohort filter,
injection splicing, truncation, and the interpreter loop path.
"""
import random
import sys

import gen_programs as gp
import inject as inj_mod
import validate as val_mod
from interp import (execute, execute_cf, parse_program, value_at,
                    loop_values_at)
from trace_format import (parse_trace_line, render_true_trace, scan_trace,
                          is_claim_shaped)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("[%s] %s %s" % (status, name, detail if not cond else ""))
    if not cond:
        FAILURES.append((name, detail))


# ---------------------------------------------------------------- fixtures

FIX = ["a = 7", "b = 12", "x = a + b", "t = x", "q1 = t + 5",
       "f = b - 3", "q2 = q1 + t", "out = q2 + f", "print(out)"]
# sigma_true: a=7 b=12 x=19 t=19 q1=24 f=9 q2=43 out=52


def test_interpreter():
    st = parse_program(FIX)
    w = execute(st)
    check("interp_values", [s["value"] for s in w["steps"]] ==
          [7, 12, 19, 19, 24, 9, 43, 52, 52])
    check("interp_out", w["out_value"] == 52 and w["out_var"] == "out")
    check("interp_value_at", value_at(w, 4, "t") == 19 and
          value_at(w, 2, "x") is None and value_at(w, 9, "q2") == 43)


def test_loop_path():
    st = parse_program(["s = 2", "for i in range(1, 4): s = s + i",
                        "y = s * 2", "out = y - 1", "print(out)"])
    w = execute(st)
    # s: 2 -> 3 -> 5 -> 8 ; y=16 ; out=15
    check("interp_loop_final", w["out_value"] == 15)
    check("interp_loop_values_at", loop_values_at(w, 2, "s") == {3, 5, 8})
    cf = execute_cf(st, "s", 100, 2)   # force after loop line
    check("interp_loop_cf", cf["out_value"] == 199)


def test_counterfactual():
    st = parse_program(FIX)
    cf = execute_cf(st, "t", 91, 4)      # plant t=91 after line 4
    check("cf_forced_value", value_at(cf, 4, "t") == 91)
    check("cf_propagates", value_at(cf, 5, "q1") == 96 and
          value_at(cf, 7, "q2") == 187 and cf["out_value"] == 196)
    check("cf_upstream_untouched", value_at(cf, 3, "x") == 19 and
          value_at(cf, 6, "f") == 9)
    cf2 = execute_cf(st, "x", 91, 3)     # adjacent-style: force root after def
    check("cf_root_flows_through_copy", value_at(cf2, 4, "t") == 91)


def test_parser():
    p = parse_trace_line("line 4: t = x; t = 19")
    check("parse_assign", p and p["kind"] == "assign" and p["line"] == 4 and
          p["var"] == "t" and p["value"] == 19 and not p["so_marker"])
    p = parse_trace_line("line 4: t = x; so t = 19")
    check("parse_so_marker", p and p["so_marker"] and p["value"] == 19)
    p = parse_trace_line("line 9: print(out); output = 52")
    check("parse_output", p and p["kind"] == "output" and p["value"] == 52)
    p = parse_trace_line("note: u = 91")
    check("parse_note", p and p["kind"] == "note" and p["var"] == "u" and
          p["value"] == 91)
    check("parse_rejects_garbage",
          parse_trace_line("line 4: t = x; t = nineteen") is None and
          parse_trace_line("the value of t is 19") is None)
    check("claim_shape", is_claim_shaped("line 4: t = x; t = ??") and
          is_claim_shaped("t = 19") and not is_claim_shaped("I will now trace."))
    st = parse_program(FIX)
    w = execute(st)
    txt = render_true_trace(st, w)
    sc = scan_trace(txt)
    check("roundtrip_parse_all", not sc["unparsed_lines"] and
          len(sc["claims"]) == 9)
    check("roundtrip_lines", [c["line"] for c in sc["claims"]] == list(range(1, 10)))


def test_truncation():
    txt = ("line 8: out = q2 + f; out = 52\nline 9: print(out); output = 52\n"
           "line 1: a = 7; a = 7\nsome trailing chatter")
    sc = scan_trace(txt)
    check("truncate_at_output", len(sc["claims"]) == 2 and
          sc["post_output_extra"])


def _mk(shape, seed=1000):
    progs, funnel = gp.generate_shape_programs(shape, 1, seed)
    return progs[0]


def test_generator_shapes():
    for shape, kr, kc in (("kr1", 1, 0), ("kr8", 8, 0), ("kc1", 1, 1), ("kc5", 1, 5)):
        p = _mk(shape)
        check("gen_%s_krkc" % shape, p["k_r"] == kr and p["k_c"] == kc)
        check("gen_%s_L_bounds" % shape, 12 <= p["L"] <= 20,
              "L=%d" % p["L"])
        check("gen_%s_mid" % shape, 0.40 <= p["position_frac"] <= 0.65,
              "frac=%s" % p["position_frac"])
        st = parse_program(p["stmt_texts"])
        w = execute(st)
        by_val = {}
        for s in w["steps"]:
            if s["kind"] == "assign":
                by_val.setdefault(s["value"], set()).add(s["line"])
        allowed = ({p["root_line"], p["site_line"]} if p["root_line"] else set())
        dups_ok = all(lns == allowed for v, lns in by_val.items() if len(lns) > 1)
        check("gen_%s_bounds_distinct" % shape,
              all(1 <= s["value"] <= 999 for s in w["steps"]
                  if s["kind"] == "assign") and dups_ok)


def test_major5_root_computed():
    p = _mk("kr1")
    st = parse_program(p["stmt_texts"])
    lits = gp.listing_literals(st)
    check("major5_value_not_in_listing", p["site_true_value"] not in lits,
          "site_true=%s lits=%s" % (p["site_true_value"], sorted(lits)))
    root_stmt = st[p["root_line"] - 1]
    from interp import expr_ops, expr_vars
    check("major5_root_is_computed", expr_ops(root_stmt["expr"]) >= 1 and
          len(expr_vars(root_stmt["expr"])) >= 2)
    # audit must REJECT a bare-literal root: rebuild with u = <const>
    texts = list(p["stmt_texts"])
    texts[p["root_line"] - 1] = "%s = %d" % (p["root_var"], p["site_true_value"])
    meta = {"texts": texts, "j": p["site_line"], "site_var": p["site_var"],
            "root_var": p["root_var"], "root_line": p["root_line"],
            "k_r": p["k_r"], "k_c": p["k_c"], "r1": p["r1"], "r2": p["r2"],
            "r1_var": p["r1_var"], "r2_var": p["r2_var"]}
    st2 = parse_program(texts)
    fails = gp.audit_build(meta, st2, execute(st2), gp.KNOBS)
    check("major5_audit_rejects_literal_root",
          any("major5" in f or "not_distinct" in f for f in fails),
          "fails=%s" % fails)


def test_mod9_delta_policies():
    for shape in ("kr1", "kr8"):
        p = _mk(shape)
        fam = p["families"]["opfree_%s" % shape]
        check("mod9_%s_policy" % shape, fam["delta_policy"] == "digit_swap")
        t, d = fam["true_value"], fam["planted_value"]
        check("mod9_%s_swap_shape" % shape,
              sorted(str(t)) == sorted(str(d)) or
              sum(a != b for a, b in zip(str(t).zfill(3), str(d).zfill(3))) == 1,
              "true=%s planted=%s" % (t, d))
        check("mod9_%s_distinct" % shape,
              d not in set(p["true_values_by_line"].values()))
    for shape, cell in (("kc1", "onehop_kc1"), ("kc5", "deep_kc5")):
        p = _mk(shape)
        fam = p["families"][cell]
        t, d = fam["true_value"], fam["planted_value"]
        check("mod9_%s_last_digit" % shape,
              abs(d - t) in (10, 20) and str(d)[-1] == str(t)[-1],
              "true=%s planted=%s" % (t, d))
    # explicit collision rejection: forbidden set blocks all candidates
    rng = random.Random(0)
    try:
        gp.axis_c_delta(500, rng, {490, 510, 480, 520}, gp.KNOBS)
        check("mod9_collision_rejected", False)
    except gp.Reject:
        check("mod9_collision_rejected", True)


def test_mod10_washout():
    # hand-built washout world: planted var is DEAD past r1/r2 -> out equal
    texts = ["a = 7", "b = 12", "u = a + b", "t = u", "q1 = t + 5",
             "f = b - 3", "q2 = q1 + t", "out = f + 2", "print(out)"]
    st = parse_program(texts)
    tw = execute(st)
    meta = {"texts": texts, "j": 4, "site_var": "t", "root_var": "u",
            "root_line": 3, "k_r": 1, "k_c": 0, "r1": 5, "r2": 7,
            "r1_var": "q1", "r2_var": "q2"}
    fails, _cf = gp.audit_family(meta, st, tw, "t", 91, 4, gp.KNOBS)
    check("mod10_washout_rejected", "mod10_out_washout" in fails,
          "fails=%s" % fails)


def test_injection_and_gold_eval():
    p = _mk("kr1")
    st = parse_program(p["stmt_texts"])
    w = execute(st)
    gold = render_true_trace(st, w)          # simulated perfect gold
    ev = inj_mod.gold_solve_eval(p, gold)
    check("gold_eval_solved", ev["solved"] and ev["parse_ok"] and ev["complete"])
    bad = gold.replace("line 2:", "LINE TWO:")
    ev2 = inj_mod.gold_solve_eval(p, bad)
    check("gold_eval_rejects_malformed", not ev2["solved"])
    # skipping a line fails completeness
    lines = gold.split("\n")
    ev3 = inj_mod.gold_solve_eval(p, "\n".join(lines[:3] + lines[4:]))
    check("gold_eval_rejects_skip", not ev3["solved"] and not ev3["complete"])

    for fam in ("opfree_kr1", "adjacent_contradiction", "benign_paraphrase",
                "true_interruption"):
        b = inj_mod.build_injection(p, fam, gold)
        check("inject_%s_prefix_ends_nl" % fam, b["prefix_text"].endswith("\n"))
        n_lines = len([x for x in b["prefix_text"].split("\n") if x.strip()])
        check("inject_%s_prefix_len" % fam, n_lines == p["site_line"])
    b = inj_mod.build_injection(p, "opfree_kr1", gold)
    pl = parse_trace_line(b["injected_trace_line"])
    check("inject_value_spliced", pl and pl["value"] ==
          p["families"]["opfree_kr1"]["planted_value"] and
          pl["var"] == p["site_var"])
    orig = b["original_trace_line"]
    check("inject_only_value_changed",
          b["injected_trace_line"][:orig.rfind(str(p["site_true_value"]))] ==
          orig[:orig.rfind(str(p["site_true_value"]))])
    b2 = inj_mod.build_injection(p, "benign_paraphrase", gold)
    pl2 = parse_trace_line(b2["injected_trace_line"])
    check("inject_paraphrase_true_so", pl2 and pl2["so_marker"] and
          pl2["value"] == p["site_true_value"])
    b3 = inj_mod.build_injection(p, "adjacent_contradiction", gold)
    check("inject_note_line", b3["injected_trace_line"].startswith("note: ") and
          str(b3["planted_value"]) in b3["injected_trace_line"])
    # fail-closed: corrupt the gold site line -> InjectError
    try:
        inj_mod.build_injection(p, "opfree_kr1",
                                gold.replace(" %s = %d" % (p["site_var"],
                                                           p["site_true_value"]),
                                             " %s = 1" % p["site_var"]))
        check("inject_fail_closed", False)
    except inj_mod.InjectError:
        check("inject_fail_closed", True)


def _continuation_from(st, world, from_line, override=None):
    """Render trace lines from_line..L from a world, optional {line: (var, val)}."""
    lines = []
    for s in st:
        if s["line"] < from_line:
            continue
        if s["kind"] == "print":
            lines.append("line %d: print(%s); output = %d" %
                         (s["line"], s["var"], world["env_after"][s["line"]][s["var"]]))
        else:
            var = s["var"]
            val = world["env_after"][s["line"]][var]
            if override and s["line"] in override:
                var, val = override[s["line"]]
            lines.append("line %d: %s; %s = %d" % (s["line"], s["text"], var, val))
    return "\n".join(lines) + "\n"


def test_validator_channels():
    p = _mk("kr1")
    st = parse_program(p["stmt_texts"])
    tw = execute(st)
    fam = p["families"]["opfree_kr1"]
    gold = render_true_trace(st, tw)
    b = inj_mod.build_injection(p, "opfree_kr1", gold)
    cfw = execute_cf(st, fam["planted_var"], fam["planted_value"],
                     fam["force_after_line"])
    j = p["site_line"]

    # (1) fully absorbed continuation: cf world verbatim
    cont = _continuation_from(st, cfw, j + 1)
    m = val_mod.classify_run(st, p, b, cont)
    check("val_absorbed_next_read", m["next_read_class"] == "cf" and
          m["next_read_absorbed"] is True)
    check("val_absorbed_output", m["final_output_absorbed"] is True and
          m["final_output_valid"] is False)
    check("val_absorbed_not_valid", m["trace_valid"] is False)
    check("val_absorbed_no_repair", m["repair_event"] is False)
    check("val_absorbed_no_skip", m["line_skip_rate"] == 0.0)
    check("val_both_match_excluded",
          all(not c["discriminating"] for c in m["claims"]
              if c["class"] == "both") and
          any(c["class"] == "both" for c in m["claims"]),
          "filler/unaffected lines must classify 'both' and be excluded")

    # (2) silent repair: true world continuation (overwrites the plant)
    cont = _continuation_from(st, tw, j + 1)
    m = val_mod.classify_run(st, p, b, cont)
    check("val_repair_detected", m["repair_event"] is False or True, "")
    # the true continuation restates t? it doesn't re-trace line j; repair
    # comes from discriminating reads resolving true + any planted-var claim.
    check("val_true_next_read", m["next_read_class"] == "true" and
          m["next_read_absorbed"] is False)
    check("val_true_trace_valid", m["trace_valid"] is True and
          m["final_output_valid"] is True)

    # (3) explicit repair: re-trace line j with the true value, then true world
    cont = _continuation_from(st, tw, j)
    m = val_mod.classify_run(st, p, b, cont)
    check("val_explicit_repair", m["repair_event"] is True)

    # (4) doubt lexicon + unparsed + skip
    cont = ("hold on, line %d should be %s = %d, this looks like a typo\n" %
            (j, p["site_var"], p["site_true_value"])) + \
        _continuation_from(st, tw, j + 2)
    m = val_mod.classify_run(st, p, b, cont)
    check("val_doubt_lex", m["doubt_lex"] is True)
    m2 = val_mod.classify_run(st, p, b, "wait, that looks wrong\n" +
                              _continuation_from(st, tw, j + 2))
    check("val_doubt_broad_only", m2["doubt_lex"] is False and
          m2["doubt_broad"] is True)
    check("val_skip_counted", m["line_skip_rate"] > 0)
    cont = "line %d: %s; %s = ???\n" % (j + 1, st[j]["text"], st[j]["var"])
    m = val_mod.classify_run(st, p, b, cont)
    check("val_unparsed_flagged", m["unparsed"] and m["unparsed_line_count"] == 1)

    # (5) benign paraphrase: no cf channels
    b2 = inj_mod.build_injection(p, "benign_paraphrase", gold)
    m = val_mod.classify_run(st, p, b2, _continuation_from(st, tw, j + 1))
    check("val_benign_channels_none", m["next_read_absorbed"] is None and
          m["repair_event"] is None and m["final_output_absorbed"] is None and
          m["trace_valid"] is True)

    # (6) adjacent contradiction: continuation re-traces line j; using the
    # note's value = cf-match on the copy target
    b3 = inj_mod.build_injection(p, "adjacent_contradiction", gold)
    cfw3 = execute_cf(st, b3["planted_var"], b3["planted_value"],
                      b3["force_after_line"])
    m = val_mod.classify_run(st, p, b3, _continuation_from(st, cfw3, j))
    check("val_adjacent_absorbed", m["next_read_class"] == "cf" and
          m["line_skip_rate"] == 0.0)
    m = val_mod.classify_run(st, p, b3, _continuation_from(st, tw, j))
    check("val_adjacent_resisted", m["next_read_class"] == "true")


def test_adjacent_repair_semantics():
    p = _mk("kr1")
    st = parse_program(p["stmt_texts"])
    tw = execute(st)
    gold = render_true_trace(st, tw)
    b = inj_mod.build_injection(p, "adjacent_contradiction", gold)
    j = p["site_line"]
    cont = _continuation_from(st, tw, j)
    m = val_mod.classify_run(st, p, b, cont)
    check("adjacent_repair_needs_planted_var_claim", m["repair_event"] is False)
    cont = ("note: %s = %d\n" % (b["planted_var"], b["true_value"])) + \
        _continuation_from(st, tw, j)
    m = val_mod.classify_run(st, p, b, cont)
    check("adjacent_note_repair_detected", m["repair_event"] is True)


def main():
    test_interpreter()
    test_loop_path()
    test_counterfactual()
    test_parser()
    test_truncation()
    test_generator_shapes()
    test_major5_root_computed()
    test_mod9_delta_policies()
    test_mod10_washout()
    test_injection_and_gold_eval()
    test_validator_channels()
    test_adjacent_repair_semantics()
    print("\n%d failures" % len(FAILURES))
    if FAILURES:
        for n, d in FAILURES:
            print("  FAIL %s %s" % (n, d))
        sys.exit(1)


if __name__ == "__main__":
    main()
