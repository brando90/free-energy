"""EXPG reference + counterfactual interpreter for the restricted grammar.

Grammar (reports/regime2.md section 2.1):

    v = c                          (integer constant)
    v = u                          (copy / alias)
    v = <expr>                     expr = term (('+'|'-') term)* ;
                                   term = atom | atom '*' atom ; atom = var | const
    for i in range(a, b): s = s <op> <expr>    (single accumulator statement body)
    print(v)                       (final line only)

HARD RULE (verdict / regime2 section 2.5): no exec()/eval() on generated
programs anywhere -- this is a hand-rolled parser + evaluator. The
counterfactual executor forces a variable to the planted value after a given
line and runs forward (sigma_cf).

World representation returned by execute()/execute_cf():

    {
      "steps":  [ {"line": n, "iter": k_or_None, "kind": "assign"|"print",
                   "var": v, "value": int, } ... ]   (one per executed step)
      "env_after": { line_no: {var: value, ...} }     (env after the line's
                                                       last execution)
      "values_at": { (line_no, var): set_of_values }  (post-step values of var
                                                       at that line; >1 entry
                                                       only for loop lines)
      "out_var": str, "out_value": int,
      "all_values": set of every assigned value in the run,
    }
"""
import re

VAR_RE = r"[A-Za-z_][A-Za-z0-9_]*"
_ASSIGN_LINE = re.compile(r"^\s*(%s)\s*=\s*(.+?)\s*$" % VAR_RE)
_PRINT_LINE = re.compile(r"^\s*print\((%s)\)\s*$" % VAR_RE)
_FOR_LINE = re.compile(
    r"^\s*for\s+(%s)\s+in\s+range\((-?\d+)\s*,\s*(-?\d+)\)\s*:\s*(.+?)\s*$" % VAR_RE)
_ATOM_RE = re.compile(r"^(-?\d+|%s)$" % VAR_RE)


class ProgramError(ValueError):
    pass


def parse_expr(rhs):
    """rhs -> list of (sign, [atom, ...]) ; atom = ('const', int) | ('var', name).

    Tokens must be whitespace-separated (the generator always emits canonical
    spacing); '*' binds tighter than '+'/'-'.
    """
    toks = rhs.split()
    if not toks or len(toks) % 2 == 0:
        raise ProgramError("bad expression: %r" % rhs)
    def atom(tok):
        if not _ATOM_RE.match(tok):
            raise ProgramError("bad atom: %r" % tok)
        if re.match(r"^-?\d+$", tok):
            return ("const", int(tok))
        return ("var", tok)
    terms = []
    cur_sign, cur_term = 1, [atom(toks[0])]
    for i in range(1, len(toks), 2):
        op, nxt = toks[i], toks[i + 1]
        if op == "*":
            cur_term.append(atom(nxt))
            if len(cur_term) > 2:
                raise ProgramError("term with >2 factors: %r" % rhs)
        elif op in ("+", "-"):
            terms.append((cur_sign, cur_term))
            cur_sign, cur_term = (1 if op == "+" else -1), [atom(nxt)]
        else:
            raise ProgramError("bad operator %r in %r" % (op, rhs))
    terms.append((cur_sign, cur_term))
    return terms


def expr_ops(terms):
    """Number of arithmetic operations in a parsed expression."""
    n = len(terms) - 1                      # +/- between terms
    n += sum(len(t) - 1 for _, t in terms)  # '*' inside terms
    return n


def expr_vars(terms):
    return [a[1] for _, t in terms for a in t if a[0] == "var"]


def expr_consts(terms):
    return [a[1] for _, t in terms for a in t if a[0] == "const"]


def parse_program(lines):
    """lines: list of statement strings (no numbering) -> list of stmt dicts."""
    stmts = []
    for idx, raw in enumerate(lines):
        n = idx + 1
        m = _PRINT_LINE.match(raw)
        if m:
            stmts.append({"line": n, "kind": "print", "var": m.group(1), "text": raw})
            continue
        m = _FOR_LINE.match(raw)
        if m:
            ivar, a, b, body = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
            bm = _ASSIGN_LINE.match(body)
            if not bm:
                raise ProgramError("bad loop body: %r" % raw)
            stmts.append({"line": n, "kind": "for", "ivar": ivar, "a": a, "b": b,
                          "var": bm.group(1), "expr": parse_expr(bm.group(2)),
                          "text": raw})
            continue
        m = _ASSIGN_LINE.match(raw)
        if m:
            stmts.append({"line": n, "kind": "assign", "var": m.group(1),
                          "expr": parse_expr(m.group(2)), "text": raw})
            continue
        raise ProgramError("unparseable program line %d: %r" % (n, raw))
    if not stmts or stmts[-1]["kind"] != "print":
        raise ProgramError("last line must be print(v)")
    if any(s["kind"] == "print" for s in stmts[:-1]):
        raise ProgramError("print before last line")
    return stmts


def _eval_expr(terms, env, line_no):
    total = 0
    for sign, term in terms:
        prod = 1
        for kind, val in term:
            if kind == "const":
                prod *= val
            else:
                if val not in env:
                    raise ProgramError("line %d reads undefined var %r" % (line_no, val))
                prod *= env[val]
        total += sign * prod
    return total


def _run(stmts, force=None):
    """force: None or (force_var, force_value, force_after_line).

    The override is applied to the environment immediately after the given
    line finishes executing (force_after_line=0 -> before the first line).
    """
    env = {}
    steps = []
    env_after = {}
    values_at = {}
    out_var = out_value = None
    fvar, fval, fline = force if force else (None, None, None)

    def note(line, var, value, it=None, kind="assign"):
        steps.append({"line": line, "iter": it, "kind": kind, "var": var, "value": value})
        values_at.setdefault((line, var), set()).add(value)

    if force and fline == 0:
        env[fvar] = fval
    for s in stmts:
        n = s["line"]
        if s["kind"] == "assign":
            env[s["var"]] = _eval_expr(s["expr"], env, n)
            note(n, s["var"], env[s["var"]])
        elif s["kind"] == "for":
            if s["a"] >= s["b"]:
                raise ProgramError("empty loop at line %d" % n)
            for it in range(s["a"], s["b"]):
                env[s["ivar"]] = it
                env[s["var"]] = _eval_expr(s["expr"], env, n)
                note(n, s["var"], env[s["var"]], it=it)
        else:  # print
            if s["var"] not in env:
                raise ProgramError("print of undefined var %r" % s["var"])
            out_var, out_value = s["var"], env[s["var"]]
            note(n, s["var"], out_value, kind="print")
        if force and n == fline:
            env[fvar] = fval
        env_after[n] = dict(env)
    return {"steps": steps, "env_after": env_after, "values_at": values_at,
            "out_var": out_var, "out_value": out_value,
            "all_values": {st["value"] for st in steps if st["kind"] == "assign"}}


def execute(stmts):
    """Reference world sigma_true."""
    return _run(stmts, force=None)


def execute_cf(stmts, force_var, force_value, force_after_line):
    """Counterfactual world sigma_cf: force_var := force_value immediately
    after line force_after_line executes; run forward."""
    return _run(stmts, force=(force_var, force_value, force_after_line))


def value_at(world, line_no, var):
    """Value of var in the environment after line_no (None if undefined or
    line_no out of range). For claims citing a line, this is the reference
    the claim is scored against."""
    if line_no is None:
        return None
    known = [n for n in world["env_after"] if n <= line_no]
    if not known:
        return None
    env = world["env_after"][max(known)]
    return env.get(var)


def loop_values_at(world, line_no, var):
    """Set of per-iteration values var takes AT line_no (empty if none)."""
    return world["values_at"].get((line_no, var), set())
