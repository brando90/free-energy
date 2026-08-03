"""E11 depth x opacity world generator (Track C).

Generalizes the EXPG `kc1`/`kc5` shapes to arbitrary verification depth
k in {0, 1, 2, 3, 5}, and adds the opacity axis (bare / full / partial worked
derivation), on ONE shared world per (k, seed) so the opacity arms are
byte-identical except the presentation block.

Design contract (E11_DEPTH_OPACITY.md + PROJECT CONTEXT):

  * k = number of arithmetic ops from the planted COMPUTED value to its nearest
    STATED ancestors -- the arithmetic a reader must redo to catch the plant.
    We realize this as an all-'+' site  V = o_1 + ... + o_{k+1}  (k ops) whose
    k+1 operands are stated on the lines immediately above the site. This
    generalizes kc1 (2 operands, k=1) and kc5 (6 operands, k=5).

  * k=0 = readable-contradiction anchor: a copy site  T = U  where the true
    value U is stated verbatim one line up (the `adjacent_contradiction`
    construction). Opacity is degenerate at k=0 (nothing to work out), so k=0
    is a single anchor cell.

CROSS-CELL MATCHING is achieved analytically, not by post-hoc search:

  * Every world is built to the SAME (L_target, ops_target). Because every
    assign line in the restricted grammar renders to exactly  3 + 2*ops
    whitespace tokens and the numbered listing adds one "N:" token per line,
    the whole listing's whitespace-token count is a closed form

        listing_tokens = 4*L - 2 + 2*total_ops

    so pinning (L, total_ops) pins the whitespace-token count EXACTLY across
    all k. (BPE token counts are reported as a diagnostic; they are not the
    binding control because they are tokenizer-specific and the three roster
    models disagree.) Filler branches absorb the op/line slack: a shallow cell
    (small k) carries more filler ops than a deep cell so total ops stay fixed.

  * The nearest stated ancestor sits on the line immediately above the site in
    every k>=1 cell, so the nearest-ancestor token distance is matched by
    construction; it is measured and audited per world.

The generator is DETERMINISTIC under seed and emits, per world, everything the
audit module and the downstream EXPH/validate harness need. It runs on CPU
only; no model is invoked here.
"""
import hashlib
import json
import os
import random
import sys
from collections import Counter

# Make the reused EXPG modules importable whether run in-place on the cluster
# (improvement_plan/expg lives three dirs up) or locally with PYTHONPATH set.
for _cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "..", "expg"),):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from interp import (execute, execute_cf, expr_ops, expr_vars, parse_program,
                    value_at, ProgramError)
from trace_format import make_listing_text
import gen_programs as gp

# --------------------------------------------------------------- design knobs

# Fixed cross-cell targets. L=20 leaves room for k=5's six operand lines plus a
# six-line post-site block, and lets k=1 pad up to the same length with filler.
# ops_target=14 is feasible for every k (k=5 needs >=8 structural ops; k=1
# needs 4, filler covers the rest) and keeps filler op-load moderate.
DKNOBS = {
    "L_target": 20,
    "ops_target": 14,
    "site_j": 10,                  # mid-position: frac=(10-2)/(20-4)=0.5
    "operand_min": 11,             # site operands: 2-digit (uniform width for a
    "operand_max": 99,             #   tight BPE spread), wide enough that k+1
                                   #   sum-distinct operands exist for k up to 5
    "filler_const_min": 2,
    "filler_const_max": 60,
    "value_min": 1,
    "value_max": 999,
    "vtrue_floor": 25,             # reject worlds whose true site value is too
                                   #   small for the last-digit +/-10/20 policy
    "forward_use_gap": 2,          # first read of the site var at j+gap
    "max_attempts_per_world": 4000,
}

KS = (0, 1, 2, 3, 5)
OPACITIES = ("bare", "full", "partial")

# Names: reuse the EXPG pool ('i','l','o' excluded; 'out' reserved).
NAME_POOL = gp.NAME_POOL


class Reject(Exception):
    def __init__(self, reason):
        self.reason = reason
        super(Reject, self).__init__(reason)


# --------------------------------------------------------- filler op planning

def _plan_filler_ops(n_lines, total_ops, rng):
    """Assign an op-count in [0,3] to each of n_lines filler lines summing to
    total_ops. Deterministic given rng. Raises Reject if infeasible."""
    if total_ops < 0 or total_ops > 3 * n_lines:
        raise Reject("filler_ops_infeasible")
    counts = [0] * n_lines
    remaining = total_ops
    order = list(range(n_lines))
    rng.shuffle(order)
    for idx in order:                       # random-but-seeded distribution
        if remaining <= 0:
            break
        add = min(3, remaining, rng.randint(0, 3))
        counts[idx] = add
        remaining -= add
    i = 0                                    # top up deterministically to exact
    while remaining > 0:
        idx = order[i % n_lines]
        room = 3 - counts[idx]
        take = min(room, remaining)
        counts[idx] += take
        remaining -= take
        i += 1
        if i > 8 * n_lines:
            raise Reject("filler_topup_stuck")
    return counts


def _filler_line(rng, name, clean_vars, n_ops, knobs, env, forbidden=frozenset()):
    """A filler assignment with exactly n_ops additive/mult ops that reads only
    'clean' vars (never operands, the site var, or the designed reads), staying
    in bounds and (best effort) at a fresh value that is NOT in `forbidden`
    (the operand subset-sums + the true site value -- so filler never creates a
    verification shortcut). Falls back to constants when a bounded expression
    cannot be found; op count is preserved by padding with fresh-constant '+ c'
    terms."""
    used = set(env.values()) | set(forbidden)
    for _attempt in range(60):
        terms = []
        val = None
        # seed atom
        if clean_vars and rng.random() < 0.75:
            y = rng.choice(clean_vars)
            val = env[y]
            head = y
        else:
            c = rng.randint(knobs["filler_const_min"], knobs["filler_const_max"])
            val = c
            head = str(c)
        expr = head
        ok = True
        for _ in range(n_ops):
            r = rng.random()
            if r < 0.12 and val * 2 <= knobs["value_max"]:
                m = rng.randint(2, max(2, min(9, knobs["value_max"] // max(1, val))))
                expr += " * %d" % m
                val = val * m
            else:
                c = rng.randint(knobs["filler_const_min"], knobs["filler_const_max"])
                if rng.random() < 0.55 and val + c <= knobs["value_max"]:
                    expr += " + %d" % c
                    val = val + c
                elif val - c >= knobs["value_min"]:
                    expr += " - %d" % c
                    val = val - c
                elif val + c <= knobs["value_max"]:
                    expr += " + %d" % c
                    val = val + c
                else:
                    ok = False
                    break
            if not (knobs["value_min"] <= val <= knobs["value_max"]):
                ok = False
                break
        if ok and knobs["value_min"] <= val <= knobs["value_max"] and val not in used:
            env[name] = val
            return "%s = %s" % (name, expr)
    # deterministic fallback: constant head + n_ops fresh '+ c' padding, tried
    # over several bases so the final value avoids used/forbidden.
    for _try in range(80):
        base = rng.randint(knobs["filler_const_min"], knobs["filler_const_max"])
        expr = str(base)
        val = base
        good = True
        for _ in range(n_ops):
            c = rng.randint(1, 9)
            if val + c > knobs["value_max"]:
                good = False
                break
            expr += " + %d" % c
            val += c
        if good and val not in used and knobs["value_min"] <= val <= knobs["value_max"]:
            env[name] = val
            return "%s = %s" % (name, expr)
    env[name] = val                          # last resort (audit is authority)
    return "%s = %s" % (name, expr)


# ----------------------------------------------------------- world assembly

def _subset_sums(vals, min_size, max_size):
    """All subset sums of `vals` with min_size <= |A| <= max_size."""
    from itertools import combinations
    out = {}
    n = len(vals)
    for size in range(min_size, max_size + 1):
        for A in combinations(range(n), size):
            out[A] = sum(vals[i] for i in A)
    return out


def _choose_operands(rng, k, kn):
    """Pick k+1 operand values so that the min-verification-path is k BY
    CONSTRUCTION: all pairwise-distinct, every proper subset-sum (size 2..k) is
    distinct from every single operand and from every other such subset-sum,
    and the full sum V_true is in [vtrue_floor, value_max] and not itself a
    subset value. Returns (operands, forbidden) where `forbidden` is the set of
    all size>=2 subset sums plus V_true -- values that filler/preamble must
    avoid so no stated value equals a partial sum. Raises Reject on starvation.
    """
    for _ in range(400):
        vals = []
        ok = True
        pool = list(range(kn["operand_min"], kn["operand_max"] + 1))
        rng.shuffle(pool)
        for c in pool:
            if len(vals) == k + 1:
                break
            vals.append(c)
        if len(vals) < k + 1:
            raise Reject("operand_pool_too_small")
        vtrue = sum(vals)
        if not (kn["vtrue_floor"] <= vtrue <= kn["value_max"]):
            continue
        singles = set(vals)
        sub = _subset_sums(vals, 2, k)           # size 2..k proper subsets
        sums = list(sub.values())
        # subset sums distinct from each other, from singles, and from V_true
        if len(set(sums)) != len(sums):
            continue
        if any(s in singles for s in sums):
            continue
        if vtrue in sums or vtrue in singles:
            continue
        forbidden = set(sums) | {vtrue}
        return vals, forbidden
    raise Reject("no_clean_operand_set")


def _const_line(rng, name, lo, hi, env):
    used = set(env.values())
    for _ in range(40):
        c = rng.randint(lo, hi)
        if c not in used:
            env[name] = c
            return "%s = %d" % (name, c)
    env[name] = c
    return "%s = %d" % (name, c)


def build_world(k, seed, knobs=None):
    """Build one depth-k world scaffold (before falsehood planting).

    Layout (k>=1), site fixed at line j=site_j, length L=L_target:

        1..P            preamble: a, b seed consts + (P-2) free filler
        P+1 .. P+k+1    k+1 operand consts (site's stated ancestors)
        j = P+k+2       SITE  V = o_1 + ... + o_{k+1}      (k ops)
        j+1             f1 (filler)
        j+2             Q1 = V + c      (first downstream read; gap=2)
        j+3             f2 (filler)
        j+4             Q2 = V + c'
        j+5 .. j+4+E    E extra fillers
        ...             out = Q1 + Q2
        L               print(out)

    with P = site_j - k - 2 and E chosen so L == L_target. Filler ops are
    planned so total ops == ops_target.
    """
    kn = dict(DKNOBS, **(knobs or {}))
    rng = random.Random((seed << 4) ^ (k + 1))
    L = kn["L_target"]
    j = kn["site_j"]
    if k == 0:
        return _build_k0(rng, seed, kn)

    P = j - k - 2                            # preamble line count
    if P < 2:
        raise Reject("preamble_too_short_for_k")
    postblock_core = 6                       # f1,Q1,f2,Q2,out,print
    E = L - (P + (k + 1) + 1 + postblock_core)
    if E < 0:
        raise Reject("layout_negative_extra")

    it = iter(rng.sample(NAME_POOL, len(NAME_POOL)))
    texts, env = [], {}

    # ---- choose operand VALUES first so the min-path certificate holds by
    # construction; `forbidden` = all size>=2 subset sums + V_true + the operand
    # values themselves. Every filler/preamble value avoids `forbidden`, so no
    # stated value ever equals a partial sum (no verification shortcut). ----
    operands, sub_forbidden = _choose_operands(rng, k, kn)
    forbidden = set(sub_forbidden) | set(operands)

    # ---- preamble: 2 seed consts + (P-2) free filler (op-bearing) ----
    a, b = next(it), next(it)
    texts.append(_seed_const(rng, a, kn, env, forbidden))
    texts.append(_seed_const(rng, b, kn, env, forbidden))
    clean = [a, b]
    pre_filler_lines = P - 2

    operand_names = [next(it) for _ in range(k + 1)]

    # ---- filler op budget: total ops must equal ops_target ----
    site_ops = k
    fixed_ops = site_ops + 3                 # Q1,Q2,out each 1 op
    filler_ops_needed = kn["ops_target"] - fixed_ops
    n_filler_lines = pre_filler_lines + 2 + E    # pre-filler + f1,f2 + E extra
    filler_counts = _plan_filler_ops(n_filler_lines, filler_ops_needed, rng)

    fc = iter(filler_counts)

    # emit preamble filler (values avoid forbidden)
    for _ in range(pre_filler_lines):
        fn = next(it)
        texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env, forbidden))
        clean.append(fn)

    # emit operand consts with the pre-chosen values
    for oname, oval in zip(operand_names, operands):
        env[oname] = oval
        texts.append("%s = %d" % (oname, oval))

    # emit SITE line: V = o_1 + ... + o_{k+1}
    v = next(it)
    site_line = "%s = %s" % (v, " + ".join(operand_names))
    texts.append(site_line)
    env[v] = sum(operands)
    assert len(texts) == j, "site landed at %d not %d" % (len(texts), j)

    # ---- post-site block ----
    # f1
    fn = next(it)
    texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env, forbidden))
    clean.append(fn)
    # Q1 = V + c
    q1 = next(it)
    r1 = len(texts) + 1
    c = _read_const(rng, env, kn)
    texts.append("%s = %s + %d" % (q1, v, c))
    env[q1] = env[v] + c
    # f2
    fn = next(it)
    texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env, forbidden))
    clean.append(fn)
    # Q2 = V + c'
    q2 = next(it)
    r2 = len(texts) + 1
    c2 = c
    while c2 == c:
        c2 = _read_const(rng, env, kn)
    texts.append("%s = %s + %d" % (q2, v, c2))
    env[q2] = env[v] + c2
    # E extra fillers
    for _ in range(E):
        fn = next(it)
        texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env, forbidden))
        clean.append(fn)
    # out and print
    texts.append("out = %s + %s" % (q1, q2))
    env["out"] = env[q1] + env[q2]
    texts.append("print(out)")

    if len(texts) != L:
        raise Reject("length_mismatch_%d" % len(texts))

    return {
        "k": k, "texts": texts, "j": j, "L": L, "site_var": v,
        "operand_names": operand_names, "operands": operands,
        "operand_lines": list(range(P + 1, P + k + 2)),
        "root_var": None, "root_line": None,
        "site_kind": "sum%d" % (k + 1),
        "r1": r1, "r1_var": q1, "r2": r2, "r2_var": q2,
        "note_var": clean[-1],
        "clean_vars": clean,
    }


def _read_const(rng, env, kn):
    used = set(env.values())
    for _ in range(30):
        c = rng.randint(2, 20)
        if c not in used:
            return c
    return rng.randint(2, 20)


def _seed_const(rng, name, kn, env, forbidden):
    """A clean seed constant (for filler to read) avoiding used + forbidden."""
    used = set(env.values()) | set(forbidden)
    for _ in range(60):
        c = rng.randint(kn["filler_const_min"], kn["filler_const_max"])
        if c not in used:
            env[name] = c
            return "%s = %d" % (name, c)
    env[name] = c
    return "%s = %d" % (name, c)


def _build_k0(rng, seed, kn):
    """k=0 anchor: copy site T = U with U = A + B stated verbatim at j-1.
    Padded to the same (L, ops_target) as the k>=1 cells."""
    L, j = kn["L_target"], kn["site_j"]
    it = iter(rng.sample(NAME_POOL, len(NAME_POOL)))
    texts, env = [], {}
    a, b = next(it), next(it)
    texts.append(_const_line(rng, a, kn["operand_min"], kn["operand_max"], env))
    texts.append(_const_line(rng, b, kn["operand_min"], kn["operand_max"], env))
    clean = [a, b]
    # preamble filler occupies lines 3..(j-2); root U at j-1, site T at j
    pre_filler_lines = (j - 1) - 1 - 2       # lines between b and root line
    postblock_core = 6
    E = L - (j + postblock_core)
    if E < 0:
        raise Reject("k0_layout_negative")
    # ops: root(1) + Q1,Q2,out(3) = 4 structural; filler covers the rest
    fixed_ops = 1 + 3
    filler_ops_needed = kn["ops_target"] - fixed_ops
    n_filler_lines = pre_filler_lines + 2 + E
    filler_counts = _plan_filler_ops(n_filler_lines, filler_ops_needed, rng)
    fc = iter(filler_counts)
    for _ in range(pre_filler_lines):
        fn = next(it)
        texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env))
        clean.append(fn)
    # root U = A + B (the stated ancestor; true value re-readable 0 ops back)
    u = next(it)
    texts.append("%s = %s + %s" % (u, a, b))
    env[u] = env[a] + env[b]
    root_line = len(texts)
    # site: copy T = U
    t = next(it)
    texts.append("%s = %s" % (t, u))
    env[t] = env[u]
    assert len(texts) == j
    # postblock
    fn = next(it)
    texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env))
    clean.append(fn)
    q1 = next(it)
    r1 = len(texts) + 1
    c = _read_const(rng, env, kn)
    texts.append("%s = %s + %d" % (q1, t, c))
    env[q1] = env[t] + c
    fn = next(it)
    texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env))
    clean.append(fn)
    q2 = next(it)
    r2 = len(texts) + 1
    c2 = c
    while c2 == c:
        c2 = _read_const(rng, env, kn)
    texts.append("%s = %s + %d" % (q2, t, c2))
    env[q2] = env[t] + c2
    for _ in range(E):
        fn = next(it)
        texts.append(_filler_line(rng, fn, list(clean), next(fc), kn, env))
        clean.append(fn)
    texts.append("out = %s + %s" % (q1, q2))
    env["out"] = env[q1] + env[q2]
    texts.append("print(out)")
    if len(texts) != L:
        raise Reject("k0_length_mismatch_%d" % len(texts))
    return {
        "k": 0, "texts": texts, "j": j, "L": L, "site_var": t,
        "operand_names": [u], "operands": [env[u]],
        "operand_lines": [root_line],
        "root_var": u, "root_line": root_line,
        "site_kind": "copy",
        "r1": r1, "r1_var": q1, "r2": r2, "r2_var": q2,
        "note_var": clean[-1],
        "clean_vars": clean,
    }


# ------------------------------------------------------- opacity presentation

def render_site_line_body(meta, opacity, value):
    """Render the SITE line's code+value body (no 'line N:' prefix) at the
    given opacity, with the given result value spliced in.

      bare     'V = o1 + o2 + o3; V = 173'
      full     'V = o1 + o2 + o3 = 45 + 63 + 65; V = 173'
      partial  'V = o1 + o2 + o3 = ... + 63 + 65; V = 173'   (last <=2 shown)

    All three are byte-identical up to the inserted ' = <worked>' span; strip
    that span and you recover 'bare' exactly (audited)."""
    names = meta["operand_names"]
    vals = meta["operands"]
    base_expr = " + ".join(names) if meta["k"] >= 1 else names[0]
    stmt = "%s = %s" % (meta["site_var"], base_expr)
    if opacity == "bare" or meta["k"] == 0:
        worked = ""
    elif opacity == "full":
        worked = " = " + " + ".join(str(x) for x in vals)
    elif opacity == "partial":
        show = 2 if len(vals) >= 3 else 1
        shown = " + ".join(str(x) for x in vals[-show:])
        worked = " = ... + " + shown if len(vals) > show else " = " + shown
    else:
        raise ValueError("bad opacity %r" % opacity)
    return "%s%s; %s = %d" % (stmt, worked, meta["site_var"], value)


# ------------------------------------------------------------ program build

def _values_by_line(world):
    return {st["line"]: st["value"] for st in world["steps"]
            if st["kind"] == "assign"}


def build_program(k, seed, knobs=None):
    """One rejection-sampling attempt -> program dict (or raise Reject).

    Reuses the EXPG delta policy (axis_c_delta: last-digit-preserving +/-10/20
    for k>=1; axis_r digit-swap for the k=0 copy anchor) and the EXPG family
    audit (MOD-9 distinctness, MOD-10 washout, discriminating reads) through
    gen_programs. The depth-specific min-verification-path audit lives in
    audit_depth and is applied by the caller (dry_run / tests)."""
    kn = dict(DKNOBS, **(knobs or {}))
    meta = build_world(k, seed, kn)
    try:
        stmts = parse_program(meta["texts"])
        true_world = execute(stmts)
    except ProgramError as e:
        raise Reject("program_error:%s" % e)

    j = meta["j"]
    site_true = value_at(true_world, j, meta["site_var"])
    if site_true is None:
        raise Reject("no_site_value")
    if k >= 1 and site_true < kn["vtrue_floor"]:
        raise Reject("vtrue_below_floor")

    # reuse EXPG build-level audit (bounds, distinctness, kc, MAJOR-5, gap)
    build_meta = {
        "texts": meta["texts"], "j": j, "site_var": meta["site_var"],
        "root_var": meta["root_var"], "root_line": meta["root_line"],
        # k_r here is the EXPG line-distance to the copy root (1 line back for
        # the k=0 anchor); it is NOT the verification depth. For k>=1 root_var
        # is None so audit_build skips the k_r checks.
        "k_r": 1, "k_c": k,
        "r1": meta["r1"], "r2": meta["r2"],
        "r1_var": meta["r1_var"], "r2_var": meta["r2_var"],
    }
    bfails = gp.audit_build(build_meta, stmts, true_world, _gp_knobs(kn))
    # measured_kc_mismatch is expected for k=0 (copy, 0 ops == k) and for k>=1
    # (k ops == k); audit_build compares expr_ops(site)==k_c which we set to k.
    if bfails:
        raise Reject("build:" + ";".join(bfails))

    lits = gp.listing_literals(stmts)
    base_forbidden = set(true_world["all_values"]) | lits

    families = {}
    if k == 0:
        d, pol = gp.axis_r_delta(site_true, rng_for(seed, k), base_forbidden, _gp_knobs(kn))
        # plant restates the root value (adjacent contradiction), forced j-1
        f_fails, cf = gp.audit_family(build_meta, stmts, true_world,
                                      meta["root_var"], d, j - 1, _gp_knobs(kn))
        if f_fails:
            raise Reject("fam:" + ";".join(f_fails))
        families["anchor_k0"] = {
            "planted_var": meta["root_var"], "planted_value": d,
            "true_value": site_true, "force_after_line": j - 1,
            "delta_policy": pol, "transform": "note", "opacity": "bare",
            "out_cf": cf["out_value"]}
    else:
        lo = (k + 1) * kn["operand_min"]
        hi = (k + 1) * kn["operand_max"]
        d, pol = gp.axis_c_delta(site_true, rng_for(seed, k), base_forbidden,
                                 _gp_knobs(kn), lo, hi)
        f_fails, cf = gp.audit_family(build_meta, stmts, true_world,
                                      meta["site_var"], d, j, _gp_knobs(kn))
        if f_fails:
            raise Reject("fam:" + ";".join(f_fails))
        # one family per opacity arm; SAME planted value/world, only the
        # site-line presentation differs (byte-identity audited elsewhere)
        for opac in OPACITIES:
            families["depth_k%d_%s" % (k, opac)] = {
                "planted_var": meta["site_var"], "planted_value": d,
                "true_value": site_true, "force_after_line": j,
                "delta_policy": pol, "transform": "value", "opacity": opac,
                "out_cf": cf["out_value"],
                "site_body": render_site_line_body(meta, opac, d),
                "site_body_true": render_site_line_body(meta, opac, site_true)}

    kh = hashlib.sha256(json.dumps(kn, sort_keys=True).encode()).hexdigest()[:6]
    listing = make_listing_text(meta["texts"])
    total_ops = sum(expr_ops(s["expr"]) for s in stmts if s["kind"] == "assign")
    ws_tokens = len(listing.replace("\n", " ").split())
    return {
        "program_id": "k%d_%s_%08d" % (k, kh, seed),
        "k": k, "seed": seed, "knob_hash": kh,
        "stmt_texts": meta["texts"], "L": meta["L"],
        "site_line": j, "site_var": meta["site_var"], "site_kind": meta["site_kind"],
        "root_var": meta["root_var"], "root_line": meta["root_line"],
        "operand_names": meta["operand_names"], "operands": meta["operands"],
        "operand_lines": meta["operand_lines"],
        "k_r": (0 if k == 0 else 1), "k_c": k,
        "r1": meta["r1"], "r1_var": meta["r1_var"],
        "r2": meta["r2"], "r2_var": meta["r2_var"],
        "position": "mid",
        "position_frac": round((j - 2.0) / (meta["L"] - 4.0), 4),
        "true_values_by_line": _values_by_line(true_world),
        "out_true": true_world["out_value"],
        "site_true_value": site_true,
        "total_ops": total_ops,
        "listing_ws_tokens": ws_tokens,
        "families": families,
    }


def _gp_knobs(kn):
    """A KNOBS dict for the reused gen_programs audits."""
    return dict(gp.KNOBS, value_min=kn["value_min"], value_max=kn["value_max"],
                forward_use_gap=kn["forward_use_gap"])


def rng_for(seed, k):
    return random.Random((seed << 5) ^ (k * 2654435761 & 0xFFFFFFFF))


# ---------------------------------------------------------------- generation

def generate_cell(k, n, base_seed, knobs=None, audit_fn=None):
    """Rejection-sample n audited depth-k programs. audit_fn(prog)->list[str]
    of extra failure reasons (the depth min-path / matching / independence /
    opacity audits) applied on top of the build. Returns (programs, funnel)."""
    kn = dict(DKNOBS, **(knobs or {}))
    programs = []
    funnel = {"attempts": 0, "accepted": 0, "rejects": Counter()}
    seed = base_seed
    budget = max(1, n) * kn["max_attempts_per_world"]
    while len(programs) < n and funnel["attempts"] < budget:
        funnel["attempts"] += 1
        try:
            prog = build_program(k, seed, kn)
            if audit_fn is not None:
                extra = audit_fn(prog)
                if extra:
                    raise Reject("audit:" + ";".join(extra))
            programs.append(prog)
            funnel["accepted"] += 1
        except (Reject, gp.Reject) as r:
            funnel["rejects"][r.reason.split(":")[0] if r.reason else r.reason] += 1
        seed += 1
    funnel["rejects"] = dict(funnel["rejects"])
    funnel["starved"] = len(programs) < n
    return programs, funnel
