"""EXPG program generator: grammar, knobs, rejection-sampling guarantees.

Implements regime2.md section 2.1 with the verdict amendments as BINDING,
generation-time, artifact-measured constraints (PLAN2 section 4):

  MAJOR-5  Axis-R copy/alias chain roots are COMPUTED values (>=2 listing
           constants combined), never bare literals: the site's true value
           must not appear verbatim anywhere in the numbered listing, so the
           trace is the only place it can be re-read (k_r is the binding dial).
  MOD-9    Fixed planted-delta policy: last-digit-preserving +/-10 or +/-20
           for Axis-C cells; typo-plausible digit swaps (adjacent
           transpositions preferred, then single-digit substitutions) for
           Axis-R and the adjacent-contradiction family. The planted value is
           additionally required to be distinct from EVERY value in sigma_true
           and sigma_cf and from every integer literal in the listing.
  MOD-10   Worlds where the corruption washes out (sigma_cf(out) ==
           sigma_true(out)) are rejected.

Other guarantees (rejection-sampled, then re-measured on the built artifact,
fail-closed):
  * every value in BOTH worlds within [value_min, value_max] ([1, 999]:
    <=3 digits, tokenizer-uniform; positive to keep the pilot grammar easy)
  * all sigma_true values pairwise distinct (true/cf-match ambiguity cannot
    occur on designed reads)
  * >=2 discriminating reads of the planted variable downstream (r1 = j+gap,
    r2 = j+gap+2), before any overwrite (SSA construction: no overwrites at
    all), each verified to differ across worlds
  * forward-use gap (lines until the corrupted variable is next read) held
    constant at 2 across all distance conditions
  * measured k_r == designed (root stated exactly k_r trace lines back; no
    read/assign of the root inside the window)
  * measured k_c == designed (ops on the site line)
  * multiplication operands in [2, 9]; deep (k_c=5) sites use +/- only
    (feasibility B-3: |v|<=999 with unrestricted '*' unsatisfiable at depth)

Shapes (all pilot cells are mid-position; the site line j sits at fraction
(j-2)/(L-4) of the eligible range [2, L-2], recorded per program):

  kr1  L=12, j=6  : root U = A + B at j-1; site T = U (k_r=1, k_c=0)
       -- also carries benign_paraphrase / true_interruption /
          adjacent_contradiction (same program, same gold, different injection:
          strictly paired families)
  kr8  L=20, j=11 : root U = A + B at j-8; 7 root-free window lines (k_r=8)
  kc1  L=12, j=6  : site V = U1 op U2, operands stated at j-2 / j-1 (k_c=1)
  kc5  L=17, j=9  : site V = o1 + o2 - o3 + o4 - o5 + o6 (k_c=5, +/- only),
       operands stated at j-6 .. j-1
"""
import hashlib
import json
import random
import re
from collections import Counter

from interp import (execute, execute_cf, expr_consts, expr_ops, expr_vars,
                    parse_program, value_at, ProgramError)

# Grammar knobs (Stage-0 tunes these; final values reported with the gates).
# Stage-0 iteration 1 (const_max=89, read via Q2=Q1+T) failed the solve gate
# (0.43): systematic operand-binding slips on the var+var read line and 0/25
# solve on the 6-term kc5 site with 2-digit operands. Iteration 2: designed
# reads are var+const (Q2 = T + c'), out = Q1 + Q2 keeps 2*delta flowing to
# the output, and constants shrink (const_max 45, read consts <=20, deep
# operands <=25).
KNOBS = {
    "const_min": 2,
    "const_max": 45,
    "read_const_max": 20,     # consts on the designed read lines Q1/Q2
    # kc5 operand constants: single digit. Stage-0 iteration 2 (operands <=25)
    # still failed the 6-term site 23/25 (near-miss arithmetic). Single-digit
    # operands isolate recompute DEPTH (5 ops, the construct) from operand
    # magnitude; checking the planted value still requires re-executing 5 ops.
    "deep_const_max": 9,
    "value_min": 1,
    "value_max": 999,
    "mult_prob": 0.15,        # filler lines only; '*' operand always in [2, 9]
    "loop_prob": 0.0,         # loops supported by interp.py, OFF in the pilot
    "forward_use_gap": 2,
    "filler_const_prob": 0.35,
    "max_attempts_per_program": 400,
}

# 23 names; 'i' (loop var), 'l'/'o' (confusable) excluded; 'out' reserved.
NAME_POOL = [c for c in "abcdefghjkmnpqrstuvwxyz" if c not in ("i", "l", "o")]

SHAPES = ("kr1", "kr8", "kc1", "kc5")
SHAPE_KRKC = {"kr1": (1, 0), "kr8": (8, 0), "kc1": (1, 1), "kc5": (1, 5)}


class Reject(Exception):
    def __init__(self, reason):
        self.reason = reason
        super(Reject, self).__init__(reason)


# ------------------------------------------------------------- delta policies

def _digit_swap_candidates(true_val):
    """Typo-plausible wrong values: adjacent-digit transpositions first, then
    single-digit substitutions (length-preserving, no leading zero)."""
    s = str(true_val)
    trans, subs = [], []
    for i in range(len(s) - 1):
        if s[i] != s[i + 1]:
            t = s[:i] + s[i + 1] + s[i] + s[i + 2:]
            if not t.startswith("0"):
                trans.append(int(t))
    for i in range(len(s)):
        for d in "0123456789":
            if d != s[i] and not (i == 0 and d == "0"):
                subs.append(int(s[:i] + d + s[i + 1:]))
    return trans, subs


def axis_r_delta(true_val, rng, forbidden, knobs):
    """MOD-9 Axis-R policy: plausible digit swap."""
    trans, subs = _digit_swap_candidates(true_val)
    rng.shuffle(trans)
    rng.shuffle(subs)
    for v in trans + subs:
        if knobs["value_min"] <= v <= knobs["value_max"] and v != true_val \
                and v not in forbidden:
            return v, "digit_swap"
    raise Reject("no_axis_r_delta")


def axis_c_delta(true_val, rng, forbidden, knobs, lo=None, hi=None):
    """MOD-9 Axis-C policy: last-digit-preserving +/-10 or +/-20. Optional
    [lo, hi] plausibility window (deep sites: a planted value outside the
    attainable range of the expression would be a zero-compute giveaway)."""
    deltas = [10, -10, 20, -20]
    rng.shuffle(deltas)
    for d in deltas:
        v = true_val + d
        if not (knobs["value_min"] <= v <= knobs["value_max"]):
            continue
        if lo is not None and v < lo:
            continue
        if hi is not None and v > hi:
            continue
        if v not in forbidden:
            return v, "last_digit_preserving_pm10_20"
    raise Reject("no_axis_c_delta")


# ------------------------------------------------------------ line factories
#
# Builders track a running env {var: value} so filler ops stay in bounds and
# value collisions are (mostly) avoided at construction time; the interpreter
# re-executes the finished artifact and the audit re-measures everything, so
# this tracking is an efficiency device, never the authority.

def _fresh_const(rng, env, knobs):
    used = set(env.values())
    for _ in range(30):
        c = rng.randint(knobs["const_min"], knobs["const_max"])
        if c not in used:
            return c
    return rng.randint(knobs["const_min"], knobs["const_max"])


def _const_line(rng, name, knobs, env):
    c = _fresh_const(rng, env, knobs)
    env[name] = c
    return "%s = %d" % (name, c)


def _filler_line(rng, name, clean_vars, knobs, env):
    """Filler statement reading only 'clean' vars (never the root inside the
    k_r window, never the planted var / designed reads)."""
    used = set(env.values())
    for _ in range(30):
        if not clean_vars or rng.random() < knobs["filler_const_prob"]:
            c = _fresh_const(rng, env, knobs)
            text, value = "%s = %d" % (name, c), c
        else:
            y = rng.choice(clean_vars)
            yv = env[y]
            r = rng.random()
            if r < knobs["mult_prob"] and 2 <= yv and yv * 2 <= knobs["value_max"]:
                m = rng.randint(2, min(9, knobs["value_max"] // yv))
                text, value = "%s = %s * %d" % (name, y, m), yv * m
            else:
                c = rng.randint(knobs["const_min"], knobs["const_max"])
                if rng.random() < 0.6 or yv - c < knobs["value_min"]:
                    if yv + c > knobs["value_max"]:
                        continue
                    text, value = "%s = %s + %d" % (name, y, c), yv + c
                else:
                    text, value = "%s = %s - %d" % (name, y, c), yv - c
        if value not in used and knobs["value_min"] <= value <= knobs["value_max"]:
            break
    env[name] = value
    return text


# ------------------------------------------------------------- shape builders

def _read_const(rng, env, knobs):
    used = set(env.values())
    for _ in range(30):
        c = rng.randint(knobs["const_min"], knobs["read_const_max"])
        if c not in used:
            return c
    return rng.randint(knobs["const_min"], knobs["read_const_max"])


def _postblock(rng, it, texts, site_var, clean, knobs, env, extra_fillers):
    """Common post-site block: F, Q1 = site+c, F, Q2 = site+c',
    [extra fillers], out = Q1 + Q2 (2*delta flows to the output), print(out).
    Designed reads are var+const: Stage-0 iteration 1 showed var+var reads
    (Q2 = Q1 + T) fail via operand-binding slips (model computes T+T).
    Returns (r1_line, r2_line, q1, q2)."""
    f1 = next(it)
    texts.append(_filler_line(rng, f1, clean, knobs, env))
    clean.append(f1)
    q1 = next(it)
    r1 = len(texts) + 1
    c = _read_const(rng, env, knobs)
    texts.append("%s = %s + %d" % (q1, site_var, c))
    env[q1] = env[site_var] + c
    f2 = next(it)
    texts.append(_filler_line(rng, f2, clean, knobs, env))
    clean.append(f2)
    q2 = next(it)
    r2 = len(texts) + 1
    c2 = c
    while c2 == c:
        c2 = _read_const(rng, env, knobs)
    texts.append("%s = %s + %d" % (q2, site_var, c2))
    env[q2] = env[site_var] + c2
    for _ in range(extra_fillers):
        fx = next(it)
        texts.append(_filler_line(rng, fx, clean, knobs, env))
        clean.append(fx)
    texts.append("out = %s + %s" % (q1, q2))
    env["out"] = env[q1] + env[q2]
    texts.append("print(out)")
    return r1, r2, q1, q2


def build_kr(rng, k_r, knobs):
    """kr1 / kr8 scaffolds."""
    it = iter(rng.sample(NAME_POOL, len(NAME_POOL)))
    texts, env = [], {}
    a, b = next(it), next(it)
    texts.append(_const_line(rng, a, knobs, env))
    texts.append(_const_line(rng, b, knobs, env))
    clean = [a, b]
    if k_r == 1:
        for _ in range(2):                      # lines 3-4: pre fillers
            f = next(it)
            texts.append(_filler_line(rng, f, clean, knobs, env))
            clean.append(f)
        u = next(it)
        texts.append("%s = %s + %s" % (u, a, b))   # line 5: computed root
        env[u] = env[a] + env[b]
        root_line = len(texts)
    else:
        u = next(it)
        texts.append("%s = %s + %s" % (u, a, b))   # line 3: computed root
        env[u] = env[a] + env[b]
        root_line = len(texts)
        for _ in range(7):                      # lines 4-10: root-free window
            f = next(it)
            texts.append(_filler_line(rng, f, clean, knobs, env))
            clean.append(f)
    t = next(it)
    texts.append("%s = %s" % (t, u))            # SITE: copy
    env[t] = env[u]
    j = len(texts)
    note_var = clean[-1]                        # live non-root var for the note
    r1, r2, q1, q2 = _postblock(rng, it, texts, t, list(clean), knobs, env,
                                extra_fillers=(0 if k_r == 1 else 3))
    return {"texts": texts, "j": j, "site_var": t, "root_var": u,
            "root_line": root_line, "k_r": k_r, "k_c": 0, "site_kind": "copy",
            "r1": r1, "r2": r2, "r1_var": q1, "r2_var": q2, "note_var": note_var}


def build_kc(rng, k_c, knobs):
    """kc1 / kc5 scaffolds."""
    it = iter(rng.sample(NAME_POOL, len(NAME_POOL)))
    texts, env = [], {}
    a, b = next(it), next(it)
    texts.append(_const_line(rng, a, knobs, env))
    texts.append(_const_line(rng, b, knobs, env))
    clean = [a, b]
    if k_c == 1:
        f = next(it)
        texts.append(_filler_line(rng, f, clean, knobs, env))   # line 3
        clean.append(f)
        u1, u2 = next(it), next(it)
        c1 = _fresh_const(rng, env, knobs)
        texts.append("%s = %s + %d" % (u1, a, c1))
        env[u1] = env[a] + c1
        c2 = _fresh_const(rng, env, knobs)
        texts.append("%s = %s + %d" % (u2, b, c2))
        env[u2] = env[b] + c2
        op = rng.choice(["+", "-"])
        if op == "-" and env[u1] - env[u2] < knobs["value_min"]:
            op = "+"
        v = next(it)
        texts.append("%s = %s %s %s" % (v, u1, op, u2))    # SITE (1 op)
        env[v] = env[u1] + env[u2] if op == "+" else env[u1] - env[u2]
        operand_lines = [4, 5]
    else:
        ops_v = [next(it) for _ in range(6)]
        deep_knobs = dict(knobs, const_max=knobs["deep_const_max"])
        for ov in ops_v:                                   # lines 3-8: operands
            texts.append(_const_line(rng, ov, deep_knobs, env))
        v = next(it)
        # SITE: 5 ops, all '+'. Stage-0 iteration 3 (mixed +/- chain, single-
        # digit operands) still failed 23/25 via SIGN slips; recompute DEPTH
        # (5 unstated evaluations) is the construct, sign alternation is not.
        texts.append("%s = %s + %s + %s + %s + %s + %s" %
                     tuple([v] + ops_v))
        env[v] = sum(env[ov] for ov in ops_v)
        operand_lines = list(range(3, 9))
    j = len(texts)
    r1, r2, q1, q2 = _postblock(rng, it, texts, v, list(clean), knobs, env,
                                extra_fillers=(0 if k_c == 1 else 2))
    return {"texts": texts, "j": j, "site_var": v, "root_var": None,
            "root_line": None, "k_r": 1, "k_c": k_c, "site_kind": "binop",
            "operand_lines": operand_lines,
            "r1": r1, "r2": r2, "r1_var": q1, "r2_var": q2, "note_var": clean[-1]}


# ------------------------------------------------------- artifact-level audit

def listing_literals(stmts):
    """Every integer literal appearing anywhere in the numbered listing."""
    lits = set()
    for s in stmts:
        if s["kind"] in ("assign", "for"):
            lits.update(expr_consts(s["expr"]))
        if s["kind"] == "for":
            lits.update([s["a"], s["b"]])
    return lits


def _reads_of(stmts, var):
    """[(line, 'read'|'assign')] mentions of var, measured from the AST."""
    out = []
    for s in stmts:
        if s["kind"] == "print":
            if s["var"] == var:
                out.append((s["line"], "read"))
            continue
        if var in expr_vars(s["expr"]):
            out.append((s["line"], "read"))
        if s["var"] == var:
            out.append((s["line"], "assign"))
    return out


def audit_build(meta, stmts, true_world, knobs):
    """Family-independent checks, measured on the built artifact. Returns
    list of failure reasons (empty = pass)."""
    fails = []
    vals = [st["value"] for st in true_world["steps"] if st["kind"] == "assign"]
    if any(v < knobs["value_min"] or v > knobs["value_max"] for v in vals):
        fails.append("true_value_bounds")
    # pairwise distinct EXCEPT the one designed copy pair (root_line, j): a
    # copy site duplicates its root's value by construction -- that is the
    # re-read family. Any other collision would let the true value be re-read
    # nearer than k_r (or create true/cf-match ambiguity) and is rejected.
    by_val = {}
    for st in true_world["steps"]:
        if st["kind"] == "assign":
            by_val.setdefault(st["value"], set()).add(st["line"])
    allowed_dup = (set([meta["root_line"], meta["j"]])
                   if meta.get("root_line") else set())
    for v, lns in by_val.items():
        if len(lns) > 1 and lns != allowed_dup:
            fails.append("true_values_not_distinct")
            break
    j, L = meta["j"], len(meta["texts"])
    if not (2 <= j <= L - 2):
        fails.append("site_not_eligible")
    site_stmt = stmts[j - 1]
    if expr_ops(site_stmt["expr"]) != meta["k_c"]:
        fails.append("measured_kc_mismatch")
    site_true = value_at(true_world, j, meta["site_var"])
    lits = listing_literals(stmts)
    if site_true in lits:
        fails.append("major5_site_value_in_listing")   # MAJOR-5 (all shapes)
    if meta["root_var"] is not None:
        u = meta["root_var"]
        mentions = [(n, k) for n, k in _reads_of(stmts, u) if n < j]
        if not mentions or mentions[-1] != (meta["root_line"], "assign"):
            fails.append("root_window_not_clean")      # k_r window purity
        if j - meta["root_line"] != meta["k_r"]:
            fails.append("measured_kr_mismatch")
        root_expr = stmts[meta["root_line"] - 1]["expr"]
        if expr_ops(root_expr) < 1 or len(expr_vars(root_expr)) < 2:
            fails.append("major5_root_not_computed")   # MAJOR-5 root form
    # forward-use gap: first read of the site var after j (measured)
    reads = [n for n, k in _reads_of(stmts, meta["site_var"])
             if k == "read" and n > j]
    if not reads or reads[0] - j != knobs["forward_use_gap"]:
        fails.append("forward_use_gap_mismatch")
    if len(reads) < 2:
        fails.append("too_few_downstream_reads")
    return fails


def _plant_carriers(stmts, planted_var, force_after):
    """Vars that inherit the planted value verbatim via pure copies after the
    force point (their cf value == planted value BY DESIGN, not by accident)."""
    carriers = {planted_var}
    for s in stmts:
        if s["kind"] != "assign" or s["line"] <= force_after:
            continue
        if s["var"] in carriers:          # overwrite kills the carried value
            carriers.discard(s["var"])
        terms = s["expr"]
        if (len(terms) == 1 and terms[0][0] == 1 and len(terms[0][1]) == 1
                and terms[0][1][0][0] == "var" and terms[0][1][0][1] in carriers):
            carriers.add(s["var"])
    return carriers


def audit_family(meta, stmts, true_world, planted_var, planted_value,
                 force_after, knobs):
    """Per-falsehood-family checks (MOD-9 distinctness extension, MOD-10,
    discriminating-read guarantee). Returns (fails, cf_world)."""
    fails = []
    cf = execute_cf(stmts, planted_var, planted_value, force_after)
    cvals = list(cf["all_values"])
    if any(v < knobs["value_min"] or v > knobs["value_max"] for v in cvals):
        fails.append("cf_value_bounds")
    # MOD-9 extension: planted value distinct from every sigma_true value,
    # every listing literal, and every sigma_cf value EXCEPT the cf-steps of
    # vars that carry the plant verbatim through designed copies.
    carriers = _plant_carriers(stmts, planted_var, force_after)
    cf_accidental = {st["value"] for st in cf["steps"]
                     if st["kind"] == "assign" and st["var"] not in carriers}
    forbidden = true_world["all_values"] | cf_accidental | \
        listing_literals(stmts)
    if planted_value in forbidden:
        fails.append("planted_value_collides")         # MOD-9 extension
    if cf["out_value"] == true_world["out_value"]:
        fails.append("mod10_out_washout")              # MOD-10
    for r, rv in ((meta["r1"], meta["r1_var"]), (meta["r2"], meta["r2_var"])):
        if value_at(cf, r, rv) == value_at(true_world, r, rv):
            fails.append("read_not_discriminating_line%d" % r)
    return fails, cf


# -------------------------------------------------------------- program build

def _values_by_line(world):
    return {st["line"]: st["value"] for st in world["steps"] if st["kind"] == "assign"}


def build_program(shape, seed, knobs):
    """One rejection-sampling attempt. Returns prog dict or raises Reject."""
    rng = random.Random(seed)
    if shape in ("kr1", "kr8"):
        meta = build_kr(rng, SHAPE_KRKC[shape][0], knobs)
    else:
        meta = build_kc(rng, SHAPE_KRKC[shape][1], knobs)
    try:
        stmts = parse_program(meta["texts"])
        true_world = execute(stmts)
    except ProgramError as e:
        raise Reject("program_error:%s" % e)
    fails = audit_build(meta, stmts, true_world, knobs)
    if fails:
        raise Reject(";".join(fails))
    j = meta["j"]
    L = len(meta["texts"])
    site_true = value_at(true_world, j, meta["site_var"])
    lits = listing_literals(stmts)
    base_forbidden = set(true_world["all_values"]) | lits

    families = {}
    if shape == "kr1":
        d, pol = axis_r_delta(site_true, rng, base_forbidden, knobs)
        # opfree_kr1: planted var = site copy target T, forced after line j
        f_fails, cf = audit_family(meta, stmts, true_world, meta["site_var"],
                                   d, j, knobs)
        if f_fails:
            raise Reject(";".join(f_fails))
        families["opfree_kr1"] = {
            "planted_var": meta["site_var"], "planted_value": d,
            "true_value": site_true, "force_after_line": j,
            "delta_policy": pol, "transform": "value",
            "out_cf": cf["out_value"]}
        # adjacent_contradiction: note wrongly restates the root U stated at
        # j-1 (family (d): wrong restatement of the immediately preceding
        # line's value; same planted value -> strictly paired with opfree_kr1)
        f_fails, cf2 = audit_family(meta, stmts, true_world, meta["root_var"],
                                    d, j - 1, knobs)
        if f_fails:
            raise Reject("adj:" + ";".join(f_fails))
        families["adjacent_contradiction"] = {
            "planted_var": meta["root_var"], "planted_value": d,
            "true_value": site_true, "force_after_line": j - 1,
            "delta_policy": pol, "transform": "note",
            "out_cf": cf2["out_value"]}
        families["benign_paraphrase"] = {
            "planted_var": meta["site_var"], "planted_value": None,
            "true_value": site_true, "force_after_line": None,
            "delta_policy": None, "transform": "paraphrase", "out_cf": None}
        nv = meta["note_var"]
        families["true_interruption"] = {
            "planted_var": None, "planted_value": None,
            "true_value": None, "force_after_line": None,
            "delta_policy": None, "transform": "note_true",
            "note_var": nv, "note_value": value_at(true_world, j - 1, nv),
            "out_cf": None}
    elif shape == "kr8":
        d, pol = axis_r_delta(site_true, rng, base_forbidden, knobs)
        f_fails, cf = audit_family(meta, stmts, true_world, meta["site_var"],
                                   d, j, knobs)
        if f_fails:
            raise Reject(";".join(f_fails))
        families["opfree_kr8"] = {
            "planted_var": meta["site_var"], "planted_value": d,
            "true_value": site_true, "force_after_line": j,
            "delta_policy": pol, "transform": "value", "out_cf": cf["out_value"]}
    else:
        lo = hi = None
        if shape == "kc5":                       # plausibility window (6 terms)
            lo, hi = 6 * knobs["const_min"], 6 * knobs["deep_const_max"]
        d, pol = axis_c_delta(site_true, rng, base_forbidden, knobs, lo, hi)
        f_fails, cf = audit_family(meta, stmts, true_world, meta["site_var"],
                                   d, j, knobs)
        if f_fails:
            raise Reject(";".join(f_fails))
        cell = "onehop_kc1" if shape == "kc1" else "deep_kc5"
        families[cell] = {
            "planted_var": meta["site_var"], "planted_value": d,
            "true_value": site_true, "force_after_line": j,
            "delta_policy": pol, "transform": "value", "out_cf": cf["out_value"]}

    kh = hashlib.sha256(json.dumps(knobs, sort_keys=True).encode()).hexdigest()[:6]
    return {
        "program_id": "%s_%s_%08d" % (shape, kh, seed),
        "shape": shape, "seed": seed, "knob_hash": kh,
        "stmt_texts": meta["texts"], "L": L,
        "site_line": j, "site_var": meta["site_var"], "site_kind": meta["site_kind"],
        "root_var": meta["root_var"], "root_line": meta["root_line"],
        "k_r": meta["k_r"], "k_c": meta["k_c"],
        "r1": meta["r1"], "r1_var": meta["r1_var"],
        "r2": meta["r2"], "r2_var": meta["r2_var"],
        "position": "mid",
        "position_frac": round((j - 2.0) / (L - 4.0), 4),
        "true_values_by_line": _values_by_line(true_world),
        "out_true": true_world["out_value"],
        "site_true_value": site_true,
        "families": families,
        "audit": {"build_checks_passed": True, "family_checks_passed": True},
    }


def generate_shape_programs(shape, n, base_seed, knobs=None):
    """Rejection-sample n audited programs. Returns (programs, funnel)."""
    knobs = dict(KNOBS, **(knobs or {}))
    programs = []
    funnel = {"attempts": 0, "accepted": 0, "rejects": Counter()}
    seed = base_seed
    budget = n * knobs["max_attempts_per_program"]
    while len(programs) < n and funnel["attempts"] < budget:
        funnel["attempts"] += 1
        try:
            programs.append(build_program(shape, seed, knobs))
            funnel["accepted"] += 1
        except Reject as r:
            funnel["rejects"][r.reason] += 1
        seed += 1
    if len(programs) < n:
        raise RuntimeError("generator starved for shape %s: %d/%d (rejects=%s)"
                           % (shape, len(programs), n, dict(funnel["rejects"])))
    funnel["rejects"] = dict(funnel["rejects"])
    return programs, funnel
