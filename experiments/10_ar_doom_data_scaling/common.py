"""Compact generator, trace format and grader for the planted-computed-error test.

Mirrors Elyas's fine-tuning protocol (paper Appendix B): 20-line programs, planted
line 10, +/-10 or +/-20 plant, result-only trace lines, check = one "re-check" line.
Depth k = number of arithmetic ops needed to verify line 10 (k+1 operands set on the
k+1 lines right before line 10). Self-contained: no dependency on expg/.
"""
import random
import re

L = 20
SITE = 10
NAMES = list("abcdefghijklmnopqrstuvwxyz")
INSTRUCTION = (
    "You will be given a short Python program with numbered lines. Trace its execution "
    "line by line. For each line, write exactly one trace line in the format "
    "\"line N: <code>; <var> = <value>\", where <value> is the value the variable has after "
    "that line runs. For the final print line, write \"line N: print(<var>); output = <value>\". "
    "Write one trace line per program line, in order, and nothing else.\n\n"
)


def gen_program(depth, seed):
    """Returns dict: stmts (list of (var, expr_text)), values by line, y info."""
    rng = random.Random(seed)
    names = rng.sample(NAMES, len(NAMES))
    it = iter(names)
    env = {}
    stmts = []  # (line_no, var, rhs)

    def add_const(lo=2, hi=40):
        v = next(it)
        c = rng.randint(lo, hi)
        env[v] = c
        stmts.append((len(stmts) + 1, v, str(c)))
        return v

    # lines 1 .. 9-depth-1: fillers (consts)
    n_fill = SITE - 1 - (depth + 1)
    clean = []
    for _ in range(n_fill):
        clean.append(add_const())
    ops = [add_const() for _ in range(depth + 1)]  # lines SITE-depth-1 .. SITE-1
    y = next(it)
    rhs = " + ".join(ops)
    yv = sum(env[o] for o in ops)
    env[y] = yv
    stmts.append((SITE, y, rhs))
    assert len(stmts) == SITE
    # line 11: filler const
    clean.append(add_const())
    # line 12: z = y + c ; line 13: w = y - c' ; rest fillers/clean ops
    z = next(it); c1 = rng.randint(2, 30); env[z] = yv + c1; stmts.append((12, z, "%s + %d" % (y, c1)))
    stmts.append  # noqa
    # reorder: line 11 const is already stmt 11
    w = next(it); c2 = rng.randint(2, 30)
    if yv - c2 < 1:
        c2 = rng.randint(1, max(1, yv - 1))
    env[w] = yv - c2; stmts.append((13, w, "%s - %d" % (y, c2)))
    # lines 14..18 fillers (consts or clean-var ops)
    for ln in range(14, L - 1):
        v = next(it)
        if rng.random() < 0.4 and clean:
            a = rng.choice(clean); c = rng.randint(2, 30)
            env[v] = env[a] + c
            stmts.append((ln, v, "%s + %d" % (a, c)))
        else:
            c = rng.randint(2, 40); env[v] = c
            stmts.append((ln, v, str(c)))
        clean.append(v)
    out = next(it)
    env[out] = env[z] + env[w]
    stmts.append((L - 1, out, "%s + %s" % (z, w)))
    assert len(stmts) == L - 1, len(stmts)
    prog = {"depth": depth, "seed": seed, "stmts": stmts, "out_var": out,
            "y": y, "ops": ops, "z": z, "w": w, "c1": c1, "c2": c2}
    return prog


def listing(prog):
    lines = ["%d: %s = %s" % (n, v, r) for n, v, r in prog["stmts"]]
    lines.append("%d: print(%s)" % (L, prog["out_var"]))
    return "\n".join(lines)


def true_values(prog):
    env = {}
    vals = {}
    for n, v, r in prog["stmts"]:
        env[v] = eval(r, {}, dict(env))
        vals[n] = env[v]
    return vals


def trace_line(n, v, r, val):
    return "line %d: %s = %s; %s = %d" % (n, v, r, v, val)


def true_trace_lines(prog, override=None):
    """override: {line: value} to force a value (used for counterfactual)."""
    env, out = {}, []
    for n, v, r in prog["stmts"]:
        val = eval(r, {}, dict(env))
        if override and n in override:
            val = override[n]
        env[v] = val
        out.append(trace_line(n, v, r, val))
    out.append("line %d: print(%s); output = %d" % (L, prog["out_var"], env[prog["out_var"]]))
    return out


def plant(prog, seed):
    rng = random.Random(seed * 7919 + 13)
    yv = true_values(prog)[SITE]
    cands = [yv + d for d in (-20, -10, 10, 20) if yv + d >= 1]
    return rng.choice(cands)


def check_line(prog, planted):
    y, ops = prog["y"], prog["ops"]
    vals = true_values(prog)
    # operand values in listing order
    opv = [vals[n] for n, v, r in prog["stmts"] if v in ops]
    return "re-check line %d: %s = %s = %s = %d, not %d" % (
        SITE, y, " + ".join(ops), " + ".join(str(x) for x in opv), vals[SITE], planted)


def instruction_with_exemplar():
    ex = gen_program(1, 999_001)
    return (INSTRUCTION + "Example\nProgram:\n" + listing(ex) + "\n\nTrace:\n" +
            "\n".join(true_trace_lines(ex)) + "\n\nNow trace this program.\n\n")


def user_message(prog):
    return instruction_with_exemplar() + "Program:\n" + listing(prog) + "\n\nTrace:\n"


def build_example(prog, kind, seed):
    """kind: 'perturbed' or 'control'. Returns (prefix_text, target_text)."""
    lines = true_trace_lines(prog)
    pre = lines[:SITE - 1]
    if kind == "control":
        pre = pre + [lines[SITE - 1]]
        return "\n".join(pre) + "\n", "\n".join(lines[SITE:])
    pv = plant(prog, seed)
    y = prog["y"]
    planted_line = trace_line(SITE, y, dict((n, r) for n, v, r in prog["stmts"])[SITE], pv)
    tgt = [check_line(prog, pv), lines[SITE - 1]] + lines[SITE:]
    return "\n".join(pre + [planted_line]) + "\n", "\n".join(tgt)


CLAIM = re.compile(r"^line (\d+): (.+?); (\w+) = (-?\d+)\s*$")
OUT = re.compile(r"^line (\d+): print\((\w+)\); output = (-?\d+)\s*$")


def grade(prog, planted, continuation, kind):
    """Classify a continuation after the (planted or true) line 10.
    kind='perturbed': propagated / corrected / other. kind='control': flagged or not,
    and whether the rest of the trace is exactly right."""
    vals = true_values(prog)
    env_true = true_trace_lines(prog)
    flagged = bool(re.search(r"re-?check|not \d+|correct(ed|ion)|mistake|error|wrong", continuation, re.I))
    recheck = bool(re.search(r"re-?check", continuation, re.I))
    claims = {}
    for ln in continuation.splitlines():
        m = CLAIM.match(ln.strip())
        if m:
            claims.setdefault(int(m.group(1)), []).append((m.group(3), int(m.group(4))))
    tv = vals[SITE]
    cf_z = (planted if planted is not None else tv) + prog["c1"]
    true_z = tv + prog["c1"]
    z12 = claims.get(12, [(None, None)])[-1][1]
    l10 = claims.get(SITE, [(None, None)])[-1][1]
    rest_ok = continuation.strip().splitlines()[-1:] == env_true[-1:] and \
        all(v == vals.get(n) for n, lst in claims.items() if n in vals for _, v in lst)
    if kind == "perturbed":
        if recheck or l10 == tv or z12 == true_z:
            label = "corrected"
        elif z12 == cf_z:
            label = "propagated"
        else:
            label = "other"
        return {"label": label, "recheck": recheck, "z12": z12, "l10": l10, "rest_ok": rest_ok}
    return {"label": "flagged" if recheck else "clean", "recheck": recheck, "z12": z12,
            "rest_ok": rest_ok}
