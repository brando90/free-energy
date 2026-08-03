"""E11 CPU self-test (NO model, NO GPU). Validates the runner's non-model
seams on real generator worlds:

  1. worlds generate + pass the full fail-closed audit stack (per k).
  2. own-trace gold (interpreter-true) solves gold_solve_eval.
  3. build_injection_e11 produces the byte-correct opacity site line, and the
     prefix parses back to the planted value.
  4. classify_run's cf world out_value == the generator's stored out_cf
     (independent re-derivation agreement).
  5. the 3-way DV labels behave on synthetic absorbed / corrected / flagged
     continuations.

Exit non-zero on any failure. Usage: python e11_cpu_selftest.py [n_per_k]
"""
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e11_run as R                              # noqa: E402
import trace_format as tf                        # noqa: E402
from interp import parse_program, execute, execute_cf   # noqa: E402
import validate as val_mod                        # noqa: E402
import inject as inj_mod                          # noqa: E402


def gold_true_text(prog):
    stmts = parse_program(prog["stmt_texts"])
    return tf.render_true_trace(stmts, execute(stmts))


def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        raise SystemExit(1)


def main(n=6):
    fails = 0
    for k in R.KS:
        progs, funnel = R.gd.generate_cell(k, n, 500000 + k * 131,
                                           audit_fn=R._audit_fn_for(k))
        check(len(progs) == n, "k=%d starved (%d/%d) funnel=%s" %
              (k, len(progs), n, funnel["rejects"]))
        for p in progs:
            gtext = gold_true_text(p)
            ev = inj_mod.gold_solve_eval(p, gtext)
            check(ev["solved"], "k=%d %s: interpreter-true gold did not solve: %s" %
                  (k, p["program_id"], ev["first_error"]))
            stmts = parse_program(p["stmt_texts"])
            for fam in R.families_of(p):
                spec = p["families"][fam]
                b = R.build_injection_e11(p, fam, gtext)
                # planted value must be recoverable from the injected line
                parsed = tf.parse_trace_line(b["injected_trace_line"])
                if spec["transform"] == "value":
                    check(parsed is not None and parsed["value"] == spec["planted_value"],
                          "k=%d %s: injected line does not parse to planted value: %r"
                          % (k, fam, b["injected_trace_line"]))
                    # opacity body byte-check: strip worked span -> bare
                    check(spec["site_body"] in b["injected_trace_line"],
                          "k=%d %s: site_body not spliced verbatim" % (k, fam))
                # cf world agreement
                if b["cf"]:
                    cf = execute_cf(stmts, b["planted_var"], b["planted_value"],
                                    b["force_after_line"])
                    check(cf["out_value"] == spec["out_cf"],
                          "k=%d %s: cf out %s != generator out_cf %s" %
                          (k, fam, cf["out_value"], spec["out_cf"]))
                # ---- 3-way DV behaviour on synthetic continuations
                inj = {kk: b[kk] for kk in ("cf", "planted_var", "planted_value",
                                            "force_after_line", "expected_lines_from")}
                # (a) ABSORBED: continue the cf world to its (false) output
                cf_full = tf.render_true_trace(stmts, cf) if b["cf"] else gtext
                # tail after the injected prefix
                npref = len([x for x in b["prefix_text"].split("\n") if x.strip()])
                absorbed_cont = "\n".join(cf_full.split("\n")[npref:])
                mA = val_mod.classify_run(stmts, p, inj, absorbed_cont)
                labA = R.three_way_label({**mA, "judge_reject": None})
                check(labA == "absorbed",
                      "k=%d %s: cf-world tail labelled %s, expected absorbed" %
                      (k, fam, labA))
                # (b) SILENTLY_CORRECTED: continue the TRUE world
                true_cont = "\n".join(gtext.split("\n")[npref:])
                mC = val_mod.classify_run(stmts, p, inj, true_cont)
                labC = R.three_way_label({**mC, "judge_reject": None})
                check(labC in ("silently_corrected", "absorbed", "flagged", "unresolved"),
                      "bad label")
                check(labC == "silently_corrected",
                      "k=%d %s: true-world tail labelled %s, expected silently_corrected"
                      % (k, fam, labC))
                # (c) FLAGGED: prepend a doubt-lexicon marker
                flagged_cont = "line %d: this looks like a bug, should be different\n%s" % (
                    p["site_line"] + 1, true_cont)
                mF = val_mod.classify_run(stmts, p, inj, flagged_cont)
                labF = R.three_way_label({**mF, "judge_reject": None})
                check(labF == "flagged",
                      "k=%d %s: doubt-marked tail labelled %s, expected flagged" %
                      (k, fam, labF))
        print("k=%d: %d worlds OK (gold-solve, injection, cf-agreement, 3-way DV)"
              % (k, len(progs)))
    print("ALL CPU SELF-TESTS PASSED")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
