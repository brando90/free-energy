"""EXPG prompt builder + strict trace-line parser (regime2.md section 2.1).

Trace format (dataflow-controlled, result-only): one step per line,

    line 4: t = x + y; t = 15

A step states the SYMBOLIC expression and the RESULT value only -- operand
values are never restated inline (this is what makes re-read distance k_r a
real variable). Final line:

    line 12: print(out); output = 132

Parser: anchored regex per line, newline-delimited (no dependence on the
'". "' sentence split -- skeptic S6/N1). Unparsed accounting is the caller's
job; parse_trace_line returns None for a non-matching line.
"""
import hashlib
import re

INSTRUCTION = (
    "You will be given a short Python program with numbered lines. Trace its "
    "execution line by line. For each line, write exactly one trace line in the "
    "format \"line N: <code>; <var> = <value>\", where <value> is the value the "
    "variable has after that line runs. For the final print line, write "
    "\"line N: print(<var>); output = <value>\". Write one trace line per program "
    "line, in order, and nothing else.\n\n"
)

# 3-shot exemplars: cover const lines, a computed root, a copy/alias line, a
# multi-op expression traced result-only, subtraction, one multiplication, and
# the print/output convention. Values obey the pilot value bounds.
EXEMPLARS = """Program:
1: a = 7
2: b = 12
3: x = a + b
4: t = x
5: q = t - 4
6: out = q + a
7: print(out)

Trace:
line 1: a = 7; a = 7
line 2: b = 12; b = 12
line 3: x = a + b; x = 19
line 4: t = x; t = 19
line 5: q = t - 4; q = 15
line 6: out = q + a; out = 22
line 7: print(out); output = 22

Program:
1: c = 4
2: d = 9
3: e = 3
4: f = 7
5: n = 2
6: g = 8
7: v = c + d + e + f + n + g
8: w = v - 6
9: out = w + e
10: print(out)

Trace:
line 1: c = 4; c = 4
line 2: d = 9; d = 9
line 3: e = 3; e = 3
line 4: f = 7; f = 7
line 5: n = 2; n = 2
line 6: g = 8; g = 8
line 7: v = c + d + e + f + n + g; v = 33
line 8: w = v - 6; w = 27
line 9: out = w + e; out = 30
line 10: print(out); output = 30

Program:
1: g = 8
2: h = 3
3: k = g * h
4: m = k + 17
5: n = m - g
6: out = n
7: print(out)

Trace:
line 1: g = 8; g = 8
line 2: h = 3; h = 3
line 3: k = g * h; k = 24
line 4: m = k + 17; m = 41
line 5: n = m - g; n = 33
line 6: out = n; out = 33
line 7: print(out); output = 33

"""

GOLD_PREFILL = "line 1:"


def prompt_sha256():
    return hashlib.sha256((INSTRUCTION + EXEMPLARS).encode()).hexdigest()


def make_listing_text(stmt_texts):
    return "\n".join("%d: %s" % (i + 1, t) for i, t in enumerate(stmt_texts))


def make_user_message(listing_text):
    return INSTRUCTION + EXEMPLARS + "Program:\n" + listing_text + "\n\nTrace:"


def format_trace_line(line_no, stmt_text, var, value):
    return "line %d: %s; %s = %d" % (line_no, stmt_text, var, value)


def format_output_line(line_no, var, value):
    return "line %d: print(%s); output = %d" % (line_no, var, value)


def format_note_line(var, value):
    return "note: %s = %d" % (var, value)


def render_true_trace(stmts, world):
    """Interpreter-true trace text (fixtures/debug; gold comes from the model)."""
    by_line = {}
    lines = []
    for st in world["steps"]:
        stmt = next(s for s in stmts if s["line"] == st["line"])
        if st["kind"] == "print":
            lines.append(format_output_line(st["line"], st["var"], st["value"]))
        elif stmt["kind"] == "for":
            lines.append("line %d (i=%d): %s; %s = %d" % (
                st["line"], st["iter"], stmt["text"], st["var"], st["value"]))
        else:
            lines.append(format_trace_line(st["line"], stmt["text"], st["var"], st["value"]))
        by_line.setdefault(st["line"], []).append(lines[-1])
    return "\n".join(lines)


VAR = r"[A-Za-z_][A-Za-z0-9_]*"
OUTPUT_RE = re.compile(
    r"^\s*line\s+(\d+)\s*(?:\(i=(-?\d+)\))?\s*:\s*print\(\s*(%s)\s*\)\s*[;,]\s*"
    r"(?:output|prints|out)\s*[=:]?\s*(-?\d+)\s*[.;]?\s*$" % VAR, re.IGNORECASE)
ASSIGN_RE = re.compile(
    r"^\s*line\s+(\d+)\s*(?:\(i=(-?\d+)\))?\s*:\s*(.*?)\s*[;,]\s*(so\s+)?"
    r"(%s)\s*=\s*(-?\d+)\s*[.;]?\s*$" % VAR, re.IGNORECASE)
NOTE_RE = re.compile(r"^\s*note:\s*(%s)\s*=\s*(-?\d+)\s*[.;]?\s*$" % VAR)


def parse_trace_line(s):
    """Parse one trace line. Returns dict or None (unparseable).

    kinds: 'output'  {line, var, value}
           'assign'  {line, iter, stmt_text, var, value, so_marker}
           'note'    {var, value}
    """
    m = OUTPUT_RE.match(s)
    if m:
        return {"kind": "output", "line": int(m.group(1)), "var": m.group(3),
                "value": int(m.group(4))}
    m = ASSIGN_RE.match(s)
    if m:
        return {"kind": "assign", "line": int(m.group(1)),
                "iter": int(m.group(2)) if m.group(2) is not None else None,
                "stmt_text": m.group(3), "so_marker": bool(m.group(4)),
                "var": m.group(5), "value": int(m.group(6))}
    m = NOTE_RE.match(s)
    if m:
        return {"kind": "note", "var": m.group(1), "value": int(m.group(2))}
    return None


_CLAIM_SHAPE = re.compile(r"(^\s*line\s+\d+)|(^\s*note:)|(=\s*-?\d+\s*[.;]?\s*$)",
                          re.IGNORECASE)


def is_claim_shaped(s):
    """A line that looks like it asserts a traced value (and therefore must
    parse, else it is 'unparsed' -- house convention keeps such runs in
    denominators)."""
    if not s.strip():
        return False
    return bool(_CLAIM_SHAPE.search(s))


def split_lines(text):
    return [ln for ln in (text or "").split("\n")]


def scan_trace(text):
    """Scan a trace/continuation text.

    Returns dict:
      claims          parsed claim dicts in output order (assign/output/note),
                      each with '_raw' -- scanning STOPS after the first
                      output claim (post-output content recorded separately)
      unparsed_lines  claim-shaped lines that failed to parse (pre-output)
      prose_lines     non-claim-shaped, non-empty lines (doubt scan happens on
                      the full text anyway)
      post_output_extra  True if any non-empty content followed the output claim
    """
    claims, unparsed, prose = [], [], []
    post_output_extra = False
    seen_output = False
    for ln in split_lines(text):
        if not ln.strip():
            continue
        if seen_output:
            post_output_extra = True
            break
        p = parse_trace_line(ln)
        if p is not None:
            p = dict(p)
            p["_raw"] = ln
            claims.append(p)
            if p["kind"] == "output":
                seen_output = True
        elif is_claim_shaped(ln):
            unparsed.append(ln)
        else:
            prose.append(ln)
    return {"claims": claims, "unparsed_lines": unparsed, "prose_lines": prose,
            "post_output_extra": post_output_extra}
