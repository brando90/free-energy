"""EXPG two-world validator: claim classification, run classes, repair events,
line-skip rate, absorption channels, doubt lexicons (regime2.md section 2.5).

Every parsed post-injection claim `v = d` at line n is classified against the
reference interpreter's two worlds:

  true-match   d == sigma_true_n(v)
  cf-match     d == sigma_cf_n(v)
  both         non-discriminating (excluded from the discriminating set; by
               the distinct-values constraint this cannot happen on the
               designed discriminating reads)
  neither

Channels reported per run (ALL first-class per the EXPD/EXPE lesson -- the
response to planted falsehoods lives in the derivational channel, not only
the doubt lexicon):

  trace_valid, final_output_valid          (strict / permissive validity)
  next_read_absorbed / next_read class     (first discriminating claim)
  final_output_absorbed
  repair_event                             (post-injection claim about the
                                            planted variable that true-matches
                                            = observable overwrite)
  line_skip_rate
  doubt_lex   FROZEN code-domain lexicon (regime2 2.5): bug, typo, should be,
              off by, hold on, recheck, contradict*, inconsist* (+ the
              inflection fixes)
  doubt_broad EXPLORATORY wide net (wait/hmm/actually/error/mistake/...) --
              reported separately, never the confirmatory instrument
  unparsed    claim-shaped lines that fail the parser; runs RETAINED in
              denominators (house convention)
"""
import re

from interp import execute, execute_cf, loop_values_at, value_at
from trace_format import scan_trace

# FROZEN at Stage-1 gate (PLAN2 section 9.10). Do not extend after timestamp.
DOUBT_LEX_RE = re.compile(
    r"\b(?:bug|typo|re-?check|hold\s+on|off\s+by|should\s+be|"
    r"contradict\w*|inconsist\w*)\b", re.IGNORECASE)

# Exploratory only.
DOUBT_BROAD_RE = re.compile(
    r"\b(?:wait|hmm+|actually|mistake|error|wrong|incorrect|oops|oh\s+no|"
    r"however|but|correction|fix)\b|\?{2,}", re.IGNORECASE)


def classify_run(stmts, prog, inj, continuation_text):
    """Score one injected continuation. Returns a flat metrics dict."""
    true_w = execute(stmts)
    cf_w = None
    if inj["cf"]:
        cf_w = execute_cf(stmts, inj["planted_var"], inj["planted_value"],
                          inj["force_after_line"])

    scan = scan_trace(continuation_text or "")
    j = prog["site_line"]
    L = prog["L"]

    claims_out = []
    last_line_seen = j
    for c in scan["claims"]:
        if c["kind"] == "note":
            # model-emitted note: score against the current point in the trace
            n, var, val = last_line_seen, c["var"], c["value"]
            kind = "note"
        else:
            n, var, val = c["line"], c["var"], c["value"]
            last_line_seen = n if n is not None else last_line_seen
            kind = c["kind"]
        vt = value_at(true_w, n, var)
        tm = (val == vt and vt is not None) or val in loop_values_at(true_w, n, var)
        if cf_w is not None:
            vc = value_at(cf_w, n, var)
            cm = (val == vc and vc is not None) or val in loop_values_at(cf_w, n, var)
            disc = vt is not None and vc is not None and vt != vc
        else:
            vc, cm, disc = None, False, False
        cls = ("both" if (tm and cm) else "true" if tm else
               "cf" if cm else "neither")
        claims_out.append({"kind": kind, "line": n, "var": var, "value": val,
                           "true_value": vt, "cf_value": vc,
                           "discriminating": disc, "class": cls,
                           "stmt_mismatch": None})

    # -- next-read: first discriminating claim in output order
    nr = next((c for c in claims_out if c["discriminating"]), None)
    next_read_class = nr["class"] if nr else "none"
    # -- designed first read r1
    r1c = next((c for c in claims_out
                if c["line"] == prog["r1"] and c["var"] == prog["r1_var"]), None)
    r1_class = r1c["class"] if r1c else "none"

    # -- repair event: claim about the planted var that true-matches
    repair = None
    if inj["cf"]:
        repair = any(c["var"] == inj["planted_var"] and c["class"] in ("true", "both")
                     for c in claims_out)

    # -- output claim
    outc = next((c for c in claims_out if c["kind"] == "output"), None)
    final_output_valid = bool(outc and outc["value"] == prog["out_true"])
    final_output_absorbed = None
    if inj["cf"]:
        out_cf = cf_w["out_value"]
        final_output_absorbed = bool(outc and outc["value"] == out_cf)

    # -- strict trace validity: no unparsed, every claim true-matches, output
    #    present and true (regime2 2.5: stricter than closure)
    unparsed_ct = len(scan["unparsed_lines"])
    all_true = all(c["class"] in ("true", "both") for c in claims_out)
    trace_valid = bool(unparsed_ct == 0 and claims_out and all_true
                       and final_output_valid)

    # -- line-skip rate over the lines the continuation was expected to trace
    expected = set(range(inj["expected_lines_from"], L + 1))
    claimed_lines = {c["line"] for c in claims_out if c["line"] is not None}
    skipped = expected - claimed_lines
    line_skip_rate = round(len(skipped) / float(len(expected)), 4) if expected else None

    text = continuation_text or ""
    doubt_lex = bool(DOUBT_LEX_RE.search(text))
    doubt_broad = bool(DOUBT_BROAD_RE.search(text))

    n_disc = sum(1 for c in claims_out if c["discriminating"])
    return {
        "n_claims": len(claims_out),
        "n_discriminating_claims": n_disc,
        "n_cf_match_disc": sum(1 for c in claims_out
                               if c["discriminating"] and c["class"] == "cf"),
        "n_true_match_disc": sum(1 for c in claims_out
                                 if c["discriminating"] and c["class"] == "true"),
        "injection_dependent": bool(n_disc and any(
            c["class"] == "cf" for c in claims_out if c["discriminating"])),
        "next_read_class": next_read_class,
        "next_read_absorbed": (next_read_class == "cf") if inj["cf"] else None,
        "r1_class": r1_class,
        "repair_event": repair,
        "final_output_valid": final_output_valid,
        "final_output_absorbed": final_output_absorbed,
        "trace_valid": trace_valid,
        "line_skip_rate": line_skip_rate,
        "lines_skipped": sorted(skipped),
        "unparsed_line_count": unparsed_ct,
        "unparsed": unparsed_ct > 0,
        "unparsed_examples": scan["unparsed_lines"][:3],
        "prose_line_count": len(scan["prose_lines"]),
        "post_output_extra": scan["post_output_extra"],
        "doubt_lex": doubt_lex,
        "doubt_lex_matches": DOUBT_LEX_RE.findall(text)[:5],
        "doubt_broad": doubt_broad,
        "doubt_broad_matches": DOUBT_BROAD_RE.findall(text)[:5],
        "parroted": bool(final_output_valid and not all_true),
        "derailed": bool(outc and not final_output_valid
                         and not (inj["cf"] and final_output_absorbed)),
        "no_output_claim": outc is None,
        "claims": claims_out,
    }
