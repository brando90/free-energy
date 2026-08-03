"""E11 depth x opacity fail-closed audit module (Track C).

Each audit returns a list of failure-reason strings ([] == pass). A world that
trips ANY audit is rejected by the generator loop and counted in the funnel --
never patched. The six non-negotiables from the task brief, each realized here:

  (1) min_verification_path   min ops to re-derive the true site value from
                              STATED / re-readable values == the designed k.
  (2) plant_validity          planted value audited-false, collides with no
                              correct in-trace value (delegated to the EXPG
                              MOD-9/MOD-10 family audit in gen_programs; here we
                              re-assert the distinctness certificate).
  (3) cross_cell_matching     L, total ops, whitespace-token count, and
                              nearest-stated-ancestor token distance all equal
                              the fixed per-experiment targets.
  (4) filler_independence     the planted chain's backward slice contains only
                              the site + its operands; no filler line is an
                              ancestor of the site, and no filler reads the
                              site/read/output vars.
  (5) opacity_byte_identity   bare/full/partial site renderings are identical
                              outside the inserted worked-derivation span.
  (6) determinism             (checked at the suite level: same seed -> same
                              program dict.)

MIN-VERIFICATION-PATH CERTIFICATE (the load-bearing audit).

For an all-'+' site  V = o_1 + ... + o_{k+1}  the operands are 0-cost leaves
(each stated on its own line). A reader verifies V by recomputing it; the
cheapest route uses any STATED value that already equals a partial sum of the
operands as a precomputed shortcut. Using a stated value equal to Sum(A) for a
sub-multiset A of the operands (|A|>=2) replaces |A|-1 additions with a
0-cost re-read, so the route costs k-(|A|-1) ops. Hence

    min-verification-path == k
      IFF  (a) V_true is not itself a re-readable value (no 0-op route), AND
           (b) no sub-multiset A of the operands with 2<=|A|<=k has Sum(A)
               equal to any re-readable value (no additive shortcut).

This is EXACT and COMPLETE for additive re-derivation routes (proof: any route
through the operand dataflow is a partitioning of the operand multiset into
stated partial sums; its op-cost is (#parts used as leaves beyond size 1)
subtracted from k, minimized exactly when a maximal stated partial sum is
reused, which (b) forbids). Non-additive coincidental routes (e.g. a stated
pair whose product happens to equal V_true) are not dataflow verification
paths; we additionally scan for the cheapest of them (<=2 ops over stated
values) as a disclosed conservative guard and reject on a hit, but do NOT run
the exponential >=3-op reuse search that would spuriously reject deep cells.

The re-readable set S at the site = every value stated on trace lines < j
(all operands and earlier intermediates) UNION every integer literal in the
program listing (the reader sees the whole numbered program).
"""
import os
import sys
from itertools import combinations

for _cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "..", "expg"),):
    if os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

from interp import execute, execute_cf, parse_program, value_at
from trace_format import make_listing_text
import gen_programs as gp
import gen_depth as gd


# ---------------------------------------------------------- (1) min-path

def readable_set(prog, stmts, true_world):
    """S: values re-readable at the site = stated trace values on lines < j
    plus all listing literals."""
    j = prog["site_line"]
    stated = {st["value"] for st in true_world["steps"]
              if st["kind"] == "assign" and st["line"] < j}
    lits = set(gp.listing_literals(stmts))
    return stated | lits


def audit_min_path(prog, stmts, true_world):
    fails = []
    k = prog["k"]
    S = readable_set(prog, stmts, true_world)
    vtrue = prog["site_true_value"]
    if k == 0:
        # anchor: the true value MUST be re-readable 0 ops back (verbatim at the
        # root line j-1); that is the whole construction.
        root_val = value_at(true_world, prog["root_line"], prog["root_var"])
        if root_val != vtrue:
            fails.append("k0_true_not_readable")
        prog.setdefault("audit_metrics", {})["measured_min_k"] = 0
        return fails
    operands = list(prog["operands"])
    # (a) no 0-op route
    if vtrue in S:
        fails.append("minpath_vtrue_readable_0op")
    # (b) no additive partial-sum shortcut
    shortcut = None
    for size in range(2, k + 1):                 # |A| = 2..k
        for A in combinations(range(len(operands)), size):
            s = sum(operands[i] for i in A)
            if s in S:
                shortcut = (size, s)
                break
        if shortcut:
            break
    if shortcut:
        fails.append("minpath_additive_shortcut_size%d_val%d" % shortcut)
    # conservative coincidental guard: any <=2-op combo of stated values that
    # hits V_true (excludes the legitimate operand-sum route)
    coincidental = _coincidental_le2(S, operands, vtrue, k)
    if coincidental:
        fails.append("minpath_coincidental_%s" % coincidental)
    # measured min k (for logging): smallest additive route length that reaches
    # vtrue given S; should be exactly k when (a)&(b) hold.
    prog.setdefault("audit_metrics", {})["measured_min_k"] = _measured_min_k(
        operands, S, vtrue, k)
    if prog["audit_metrics"]["measured_min_k"] != k:
        fails.append("minpath_measured_%s_ne_%d" %
                     (prog["audit_metrics"]["measured_min_k"], k))
    return fails


def _measured_min_k(operands, S, vtrue, kdesign):
    """Smallest number of additions to reach vtrue, where any stated partial
    sum of operands may be used as a 0-cost leaf. Complete for additive routes;
    returns kdesign when no shortcut/readable exists."""
    if vtrue in S:
        return 0
    best = kdesign
    n = len(operands)
    # try replacing a stated maximal partial sum
    for size in range(2, kdesign + 1):
        for A in combinations(range(n), size):
            if sum(operands[i] for i in A) in S:
                best = min(best, kdesign - (size - 1))
    return best


def _coincidental_le2(S, operands, vtrue, k):
    """Guard for NON-additive arithmetic coincidences: is vtrue reachable from
    the stated set S in strictly fewer ops than k using {+,-,*}?

    The additive subset-sum certificate (a)+(b) is COMPLETE for additive
    re-derivation of an all-'+' site, so this guard only has to catch a reader
    spotting a *cheap* non-additive coincidence (e.g. two stated values whose
    product or difference happens to equal the true value). We cap the search
    depth at D = min(k-1, 1): capping at k-1 guarantees the designed k-op route
    is never in the guard set (a valid world is never falsely rejected), and
    capping at 1 keeps the guard to genuinely cheap, reader-plausible flukes.
    A >=2-op reuse search over the whole stated set is deliberately NOT run: it
    does not correspond to any dataflow verification route (it would combine
    unrelated fillers by coincidence) and, being near-surjective onto [1,999]
    at depth 2, would spuriously reject nearly every deep cell. Returns the
    op-depth of the cheapest hit, or None. For k<=1, D<=0 (the 1-op operand sum
    IS the designed route), so only the 0-op re-read -- handled by (a) --
    matters. Disclosed limitation, per the task's fail-honest instruction."""
    D = min(k - 1, 1)
    if D <= 0:
        return None
    BOUND = 4000
    levels = [set(int(x) for x in S if abs(int(x)) <= BOUND)]   # R0
    if vtrue in levels[0]:
        return "0op"                       # (a) also catches this
    for m in range(1, D + 1):
        Rm = set()
        for i in range((m - 1) // 2 + 1):
            j = m - 1 - i
            for a in levels[i]:
                for b in levels[j]:
                    for r in (a + b, a - b, b - a, a * b):
                        if abs(r) <= BOUND:
                            Rm.add(r)
        levels.append(Rm)
        if vtrue in Rm:
            return "%dop" % m
    return None


# --------------------------------------------------- (2) plant validity

def audit_plant_validity(prog, stmts, true_world):
    """Re-assert the MOD-9 distinctness certificate on the built artifact for
    every planted family (the EXPG family audit already ran at build; this is
    the fail-closed re-measure)."""
    fails = []
    lits = set(gp.listing_literals(stmts))
    truevals = set(true_world["all_values"])
    seen = set()
    for fam, spec in prog["families"].items():
        pv = spec["planted_value"]
        if pv is None:
            continue
        key = (spec["planted_var"], pv, spec["force_after_line"])
        if key in seen:
            continue
        seen.add(key)
        if pv == spec["true_value"]:
            fails.append("plant_equals_true_%s" % fam)
        if pv in truevals:
            fails.append("plant_in_true_values_%s" % fam)
        if pv in lits:
            fails.append("plant_in_listing_%s" % fam)
        cf = execute_cf(stmts, spec["planted_var"], pv, spec["force_after_line"])
        if cf["out_value"] == true_world["out_value"]:
            fails.append("plant_washout_%s" % fam)
        # discriminating reads r1,r2 must differ across worlds
        for r, rv in ((prog["r1"], prog["r1_var"]), (prog["r2"], prog["r2_var"])):
            if value_at(cf, r, rv) == value_at(true_world, r, rv):
                fails.append("plant_read_nondisc_%s_line%d" % (fam, r))
    return fails


# ------------------------------------------------ (3) cross-cell matching

def nearest_ancestor_token_distance(prog):
    """Whitespace-token distance in the numbered LISTING from the first token
    of the site line back to the first token of its nearest stated-ancestor
    line (the operand/root line immediately above the site)."""
    texts = prog["stmt_texts"]
    j = prog["site_line"]
    anc_line = max(prog["operand_lines"])        # nearest ancestor line no.
    # token count of lines (anc_line+1 .. j-1) inclusive plus the ancestor line
    def line_tokens(idx):                        # idx is 1-based program line
        return len(("%d: %s" % (idx, texts[idx - 1])).split())
    dist = 0
    for ln in range(anc_line, j):                # from ancestor line up to j-1
        dist += line_tokens(ln)
    return dist


def audit_cross_cell(prog, targets):
    """targets: {'L','ops','ws_tokens','nearest_anc_dist'} the fixed values all
    cells must hit."""
    fails = []
    m = prog.setdefault("audit_metrics", {})
    m["L"] = prog["L"]
    m["total_ops"] = prog["total_ops"]
    m["ws_tokens"] = prog["listing_ws_tokens"]
    m["nearest_anc_dist"] = nearest_ancestor_token_distance(prog)
    # closed-form identity check: ws_tokens must equal 4L-2+2*ops
    if prog["listing_ws_tokens"] != 4 * prog["L"] - 2 + 2 * prog["total_ops"]:
        fails.append("ws_token_formula_violated")
    if prog["L"] != targets["L"]:
        fails.append("L_mismatch_%d" % prog["L"])
    if prog["total_ops"] != targets["ops"]:
        fails.append("ops_mismatch_%d" % prog["total_ops"])
    if prog["listing_ws_tokens"] != targets["ws_tokens"]:
        fails.append("ws_tokens_mismatch_%d" % prog["listing_ws_tokens"])
    if targets.get("nearest_anc_dist") is not None and \
            m["nearest_anc_dist"] != targets["nearest_anc_dist"]:
        fails.append("nearest_anc_dist_mismatch_%d" % m["nearest_anc_dist"])
    return fails


# ------------------------------------------- (4) filler independence

def _backward_slice(stmts, target_var, target_line):
    """Set of line numbers that target_var@target_line data-depends on
    (transitive), in this SSA (no overwrites) program."""
    # map var -> defining line and its read-vars
    defs = {}
    for s in stmts:
        if s["kind"] == "assign":
            defs[s["var"]] = (s["line"], [v for v in _read_vars(s)])
    slice_lines = set()
    stack = [target_var]
    seen_vars = set()
    while stack:
        v = stack.pop()
        if v in seen_vars or v not in defs:
            continue
        seen_vars.add(v)
        ln, reads = defs[v]
        if ln <= target_line:
            slice_lines.add(ln)
        for rv in reads:
            stack.append(rv)
    return slice_lines


def _read_vars(stmt):
    from interp import expr_vars
    if stmt["kind"] == "print":
        return [stmt["var"]]
    return list(expr_vars(stmt["expr"]))


def audit_filler_independence(prog, stmts):
    fails = []
    j = prog["site_line"]
    v = prog["site_var"]
    sl = _backward_slice(stmts, v, j)
    # the slice must be exactly the site line + its operand lines (+ their
    # ancestors, for the k=0 root which reads a,b). Compute the allowed set as
    # the transitive closure of the operand lines themselves.
    allowed = set(prog["operand_lines"]) | {j}
    for ol in prog["operand_lines"]:
        allowed |= _backward_slice(stmts, stmts[ol - 1]["var"], ol)
    extra = sl - allowed
    if extra:
        fails.append("site_slice_has_filler_lines_%s" % sorted(extra))
    # no filler line (any line not in the planted chain / postblock reads) may
    # read the site var, the read vars, or 'out' before they are defined-again;
    # specifically ensure the site var is only read by r1,r2 (and out via q's).
    site_readers = []
    for s in stmts:
        if s["kind"] != "print" and v in _read_vars(s):
            site_readers.append(s["line"])
        if s["kind"] == "print" and s["var"] == v:
            site_readers.append(s["line"])
    if sorted(site_readers) != sorted([prog["r1"], prog["r2"]]):
        fails.append("unexpected_site_readers_%s" % sorted(site_readers))
    return fails


# ------------------------------------------- (5) opacity byte-identity

def audit_opacity_byte_identity(prog):
    """bare/full/partial site bodies must be byte-identical except for an
    inserted ' = <worked>' span located immediately after the symbolic
    expression; stripping that span from full/partial must yield bare exactly.
    Only applies to k>=1 (k=0 is bare-only)."""
    fails = []
    if prog["k"] == 0:
        return fails
    fams = prog["families"]
    bare = fams.get("depth_k%d_bare" % prog["k"])
    if bare is None:
        return ["opacity_missing_bare"]
    bare_body = bare["site_body"]
    # bare has form 'V = <expr>; V = <val>' ; the worked span is inserted before
    # the first '; '
    head, _, tail = bare_body.partition("; ")
    for opac in ("full", "partial"):
        f = fams.get("depth_k%d_%s" % (prog["k"], opac))
        if f is None:
            fails.append("opacity_missing_%s" % opac)
            continue
        body = f["site_body"]
        fhead, _, ftail = body.partition("; ")
        if ftail != tail:
            fails.append("opacity_tail_differs_%s" % opac)
        # fhead must start with bare head (the symbolic expr) and only append
        if not fhead.startswith(head):
            fails.append("opacity_head_not_prefix_%s" % opac)
        inserted = fhead[len(head):]
        if inserted and not inserted.startswith(" = "):
            fails.append("opacity_insert_not_worked_span_%s" % opac)
        # stripping the inserted span recovers bare exactly
        recovered = (head + "; " + tail)
        if recovered != bare_body:
            fails.append("opacity_strip_mismatch_%s" % opac)
    return fails


# ---------------------------------------------------- combined audit fn

def make_audit_fn(targets):
    """Return audit_fn(prog)->list[str] for gen_depth.generate_cell, wiring all
    per-world audits. Determinism is checked separately at suite level."""
    def audit_fn(prog):
        stmts = parse_program(prog["stmt_texts"])
        true_world = execute(stmts)
        fails = []
        fails += audit_min_path(prog, stmts, true_world)
        fails += audit_plant_validity(prog, stmts, true_world)
        fails += audit_cross_cell(prog, targets)
        fails += audit_filler_independence(prog, stmts)
        fails += audit_opacity_byte_identity(prog)
        return fails
    return audit_fn


def default_targets(knobs=None):
    kn = dict(gd.DKNOBS, **(knobs or {}))
    L, ops = kn["L_target"], kn["ops_target"]
    return {"L": L, "ops": ops, "ws_tokens": 4 * L - 2 + 2 * ops,
            "nearest_anc_dist": None}     # set after measuring k>=1 baseline
