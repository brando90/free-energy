"""EXPG injection constructors: build the perturbed continuation prefix from
the model's OWN gold trace (string-prefix mechanics exactly as
arith_pilot.py:104-110; under vLLM the prefix is passed as token ids).

Families (regime2.md section 2.4 transfer map, pilot subset):

  benign_paraphrase       same line, same true value, style variant
                          ("...; so t = 19")                    [R1/N11 anchor]
  true_interruption       step replaced by a TRUE note about another live
                          variable ("note: f = 31")
  adjacent_contradiction  step replaced by a note wrongly restating the value
                          stated on the IMMEDIATELY PRECEDING trace line
                          ("note: u = 91" right after "line 5: u = a + b;
                          u = 19") -- d~0, zero staleness, zero compute
  opfree_kr1 / opfree_kr8 stated result value on the model's own copy-site
                          line replaced by the digit-swap value (k_c=0; true
                          value re-readable k_r lines back in the trace ONLY,
                          per MAJOR-5)
  onehop_kc1 / deep_kc5   stated result value on the binop site line replaced
                          by the last-digit-preserving +/-10/20 value

Fail-closed: the gold trace line at the site must parse and carry exactly the
designed (var, true value); otherwise the run is rejected and counted.
"""
import re

from trace_format import format_note_line, parse_trace_line, scan_trace

_FINAL_INT = re.compile(r"(-?\d+)(\s*[.;]?\s*)$")
_VALUE_CLAUSE = re.compile(r"([;,]\s*)([A-Za-z_][A-Za-z0-9_]*\s*=\s*-?\d+\s*[.;]?\s*)$")


class InjectError(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super(InjectError, self).__init__(reason)


def replace_final_value(raw_line, new_value):
    """Splice the planted value over the final integer token, preserving every
    other byte of the model's own line."""
    m = _FINAL_INT.search(raw_line)
    if not m:
        raise InjectError("no_final_int")
    return raw_line[:m.start(1)] + str(int(new_value)) + raw_line[m.end(1):]


def paraphrase_line(raw_line):
    """Insert the 'so ' marker before the result clause (value unchanged)."""
    m = _VALUE_CLAUSE.search(raw_line)
    if not m:
        raise InjectError("no_value_clause")
    return raw_line[:m.end(1)] + "so " + raw_line[m.end(1):]


def gold_trace_raw_lines(gold_full_text):
    """Non-empty raw lines of the gold trace, in order."""
    return [ln for ln in gold_full_text.split("\n") if ln.strip()]


def build_injection(prog, family, gold_full_text):
    """Returns dict with the injected prefix and the counterfactual spec.

    prefix_text ends with a newline after the injected line; the continuation
    prompt is chat(user) + prefix_text at the token level.
    """
    spec = prog["families"][family]
    j = prog["site_line"]
    raws = gold_trace_raw_lines(gold_full_text)
    # cohort golds are complete and ordered: the trace line for program line j
    # is the j-th claim line; locate it fail-closed.
    idx = None
    for k, ln in enumerate(raws):
        p = parse_trace_line(ln)
        if p is not None and p.get("line") == j and p["kind"] == "assign":
            idx = k
            break
    if idx is None:
        raise InjectError("site_line_not_found_in_gold")
    p = parse_trace_line(raws[idx])
    if p["var"] != prog["site_var"]:
        raise InjectError("site_var_mismatch")
    if p["value"] != prog["site_true_value"]:
        raise InjectError("site_value_mismatch")

    tf = spec["transform"]
    if tf == "value":
        injected = replace_final_value(raws[idx], spec["planted_value"])
    elif tf == "paraphrase":
        injected = paraphrase_line(raws[idx])
    elif tf == "note":
        injected = format_note_line(spec["planted_var"], spec["planted_value"])
    elif tf == "note_true":
        injected = format_note_line(spec["note_var"], spec["note_value"])
    else:
        raise InjectError("unknown_transform")

    prefix_lines = raws[:idx] + [injected]
    step_replaced = tf in ("note", "note_true")
    return {
        "family": family,
        "site_line": j,
        "injected_trace_line": injected,
        "original_trace_line": raws[idx],
        "prefix_text": "\n".join(prefix_lines) + "\n",
        "step_replaced": step_replaced,
        # lines the continuation is expected to trace (for line-skip rate)
        "expected_lines_from": j if step_replaced else j + 1,
        "cf": spec["planted_value"] is not None,
        "planted_var": spec["planted_var"],
        "planted_value": spec["planted_value"],
        "true_value": spec["true_value"],
        "force_after_line": spec["force_after_line"],
        "delta_policy": spec["delta_policy"],
    }


def gold_solve_eval(prog, gold_full_text):
    """Cohort double-filter (regime2 section 2.2): every claim-shaped line
    parses; trace covers exactly lines 1..L in order; every claim matches the
    interpreter; final output correct. Returns dict of flags + first error."""
    L = prog["L"]
    scan = scan_trace(gold_full_text)
    parse_ok = not scan["unparsed_lines"]
    claims = scan["claims"]
    first_error = None
    values = {int(k): v for k, v in prog["true_values_by_line"].items()}

    lines_seen = [c.get("line") for c in claims]
    complete = lines_seen == list(range(1, L + 1))
    correct = True
    for c in claims:
        n = c.get("line")
        if c["kind"] == "note":
            correct = False
            first_error = first_error or {"line": None, "type": "stray_note"}
            continue
        if c["kind"] == "output":
            ok = (n == L and c["value"] == prog["out_true"])
            if not ok:
                correct = False
                first_error = first_error or {"line": n, "type": "wrong_output",
                                              "claimed": c["value"],
                                              "true": prog["out_true"]}
            continue
        truev = values.get(n)
        if truev is None or c["value"] != truev:
            correct = False
            first_error = first_error or {"line": n, "type": "wrong_value",
                                          "claimed": c["value"], "true": truev}
    if not parse_ok:
        first_error = first_error or {"line": None, "type": "parse",
                                      "raw": scan["unparsed_lines"][0][:120]}
    elif not complete:
        first_error = first_error or {"line": None, "type": "incomplete_or_out_of_order",
                                      "lines_seen": lines_seen[:24]}
    solved = parse_ok and complete and correct
    return {"parse_ok": parse_ok, "complete": complete, "correct": correct,
            "solved": solved, "first_error": first_error,
            "n_claim_lines": len(claims) + len(scan["unparsed_lines"]),
            "n_unparsed_lines": len(scan["unparsed_lines"]),
            "post_output_extra": scan["post_output_extra"]}
