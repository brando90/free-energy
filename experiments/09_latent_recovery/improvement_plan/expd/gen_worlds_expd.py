"""EXPD world generator: deterministic matched refutation-distance worlds.

Implements the EXPD world construction of reports/causal2.md section 1.1 with the
PLAN2.md section 4 binding amendments (verdict MAJOR-1, MAJOR-2, MOD-13, minor-c):

  * Goal chain: premise "E is a C0." + rules C0->C1->...->C_{L-1} -> goal_adj,
    L in {6,7,8} (verdict minor-c). Target = chain-end predicate.
  * Refutation spur (the d dial): premise "E is a S0." + rules S0->...->S_{d-1}
    + terminal refuting rule whose head is the complement of the planted claim.
    The spur is off the goal chain, so d is static through compliant continuations.
  * D_MAX padding: every world carries exactly D_MAX spur+filler block rules
    ((d-1) spur cat rules + 1 refuting rule + (D_MAX - d) filler-chain rules),
    so rule count and question word count are constant across d within a family.
    The refuting rule is pinned at the last block slot (constant token distance
    from the question end across d).
  * Mirrored quadruples: world B is byte-identical to world A except one polarity
    token ("not") in the refuting rule head (or the d=0 premise fact).
  * Planted-token question frequency == 1 for the attribute family and the inert
    categorical family. STRUCTURAL DEVIATION (disclosed, audited): the usable
    categorical family is >= 2 by construction (refuting head + usable-branch
    body); it is held CONSTANT ACROSS d instead (2 at every d incl. inf, via a
    frequency-mention filler rule at d=inf).
  * MAJOR-1: attribute cells exist ONLY at finite d in {0,1,2,3,5}; requesting an
    attribute world at d=inf raises. All inf anchors live in the categorical
    family (genuinely CWA-false). Theorems T1/T2/T3 are printed in the
    world-audit manifest.
  * MAJOR-2: the usable categorical branch routes through >= 1 fresh intermediate
    category (bridge) underivable without the planted atom:
        "Every <z> is a <bridge>. Every <bridge> is a <C_{L-1}>."
    audited fail-closed: bridge not in D(true state) and bridge in
    D(true state + planted atom). The inert twin replaces the branch body with a
    disconnected decoy noun (one-token difference), so usable-vs-inert is itself
    a matched pair.

All audits are fail-closed: any assertion failure rejects the world, is counted
in the audit funnel, and aborts generation (a deterministic constructor that
fails its own audit is a bug, not a sample).

Distance function: exactly `shortest_rule_distance` from src/expc_polarity_control.py
(the canonical owner per PLAN2 section 3). Nothing is reimplemented.

Usage:
  python gen_worlds_expd.py --out-dir <dir> --seed 0 --config pilot
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_src():
    cands = [
        os.environ.get("LR_SRC"),
        os.path.abspath(os.path.join(HERE, "..", "..", "src")),        # cluster: improvement_plan/expd/../../src
        os.path.abspath(os.path.join(HERE, "..", "exp", "src")),       # local mirror: latent_recovery/expd_dev/../exp/src
        os.path.abspath(os.path.join(HERE, "..", "..", "exp", "src")),
    ]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "validator.py")):
            return c
    raise SystemExit("cannot locate src/ with validator.py; set LR_SRC")


SRC = _find_src()
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from validator import (  # noqa: E402
    closure,
    derivable,
    parse_world,
    validate_continuation,
)
from expc_polarity_control import (  # noqa: E402
    audit_truth_status,
    shortest_rule_distance,
    statement_predicate,
    world_state,
)

D_MAX = 6                      # spur+filler block size; > max finite d (5)
GOAL_L_CYCLE = (6, 7, 8)       # goal-chain rule count, verdict minor-c
ATTR_D_ALLOWED = (0, 1, 2, 3, 5)   # MAJOR-1: no attribute cell at d=inf
CAT_D_ALLOWED = (1, 2, 3, 5, "inf")

FEWSHOT_NOUNS = {"yumpus", "dumpus", "tumpus", "gorpus", "sterpus", "borpus"}
FEWSHOT_ADJS = {"bright", "red"}
ENTITIES = (
    "Stella", "Marcus", "Bianca", "Hector", "Portia", "Gideon", "Amelia",
    "Rosina", "Tobias", "Cedric", "Ingrid", "Lorenz", "Miriam", "Nestor",
    "Quincy", "Sylvia", "Ursula", "Victor", "Walter", "Zelina",
)
# no overlap with fewshot adjectives or the validator DOUBT lexicon
ADJECTIVES = (
    "shiny", "fuzzy", "salty", "vivid", "moist", "tepid", "dusky", "plush",
    "stark", "timid", "vapid", "murky", "balmy", "crisp", "gaunt", "husky",
    "jolly", "livid", "perky", "rigid", "slick", "tangy", "waxen", "zesty",
    "downy", "runny", "milky", "gauzy", "lanky", "nutty",
)
CONSONANTS = "bcdfghjklmnprstvwz"
VOWELS = "aeiou"

THEOREMS = {
    "T1": "ON-PATH x AFFIRMATIVE x FALSE is logically unconstructible: on-path "
          "positive entity facts are entailed-true by construction.",
    "T2": "Rule bodies are always positive categories (validator.py:33-54): "
          "attribute claims are derivationally inert by grammar. Polarity purity "
          "is tested on attributes, reuse on categoricals - by necessity, not choice.",
    "T3": "Attribute truth-decidability <=> finite d (attribute audit is "
          "open-world): attribute cells cannot exist at d=inf. Every inf anchor "
          "lives in the categorical family (cat_false_inf, genuinely CWA-false).",
}

PILOT_CONFIG = {
    "name": "pilot_200",
    "attr": [
        {"families": 20, "d_levels": [1, 3, 5]},   # crossed with d (within-family dose)
        {"families": 15, "d_levels": [1]},          # d1-only top-up for the matched polarity contrast
    ],
    "cat": [
        {"families": 25, "d_levels": [1]},
    ],
}

# Full confirmatory grid (PLAN2 v1.1 section C.5). NOT run by default; the
# runner refuses it without an explicit pre-registration-timestamp confirmation
# flag. Family count sizing: each family yields exactly ONE world per
# (cell, d), so n=150/cell/position requires >=150 ELIGIBLE worlds per cell x d.
# Pilot eligibility was 0.90 (joint-solve at d5 implies ~0.85-0.87/world), so
# 180 families/kind targets ~155-165 eligible per cell x d; any cell landing
# under 150 keeps all its eligible worlds and the shortfall is reported.
FULL_CONFIG = {
    "name": "full_confirmatory",
    "attr": [
        {"families": 180, "d_levels": [0, 1, 2, 3, 5]},
    ],
    "cat": [
        {"families": 180, "d_levels": [1, 3, "inf"]},
    ],
}

CONFIGS = {"pilot": PILOT_CONFIG, "full": FULL_CONFIG}


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def d_key(d):
    return "inf" if d == "inf" else "%d" % d


def noun_stream(rng):
    """Nonce '-pus' nouns drawn without replacement (globally unique).

    FULL-GRID CAPACITY NOTE (disclosed): the original cvc pool (18*5*18 = 1,620)
    supports ~78 families at ~20 nouns/family; the PLAN2 v1.1 section-C.5 grid
    (n=150/cell/position) needs ~360 families (~7,400 nouns). The pool is
    extended with cvcv prefixes (18*5*18*5 = 8,100; e.g. 'bavipus'), shuffled
    together with the cvc forms so prefix length is randomized across families
    and constant WITHIN a family across d/version (the audited invariants -
    per-family word/rule-count constancy and one-token mirrors - are unchanged).
    """
    combos = [c1 + v + c2 + "pus" for c1 in CONSONANTS for v in VOWELS for c2 in CONSONANTS]
    combos += [c1 + v1 + c2 + v2 + "pus" for c1 in CONSONANTS for v1 in VOWELS
               for c2 in CONSONANTS for v2 in VOWELS]
    rng.shuffle(combos)
    for n in combos:
        if n in FEWSHOT_NOUNS:
            continue
        yield n


# ------------------------------------------------------------- family builders

def build_attr_family(fam_idx, d_levels, nouns, seed):
    """One mirrored attribute family: shared vocabulary, instantiated at each d
    in d_levels x versions {A, B}. MAJOR-1: refuses d=inf / d not in ATTR_D_ALLOWED."""
    for d in d_levels:
        if d == "inf" or d not in ATTR_D_ALLOWED:
            raise ValueError(
                "MAJOR-1 violation: attribute family requested at d=%r "
                "(allowed: %r; inf anchors live in the categorical family only)" % (d, ATTR_D_ALLOWED))
    L = GOAL_L_CYCLE[fam_idx % len(GOAL_L_CYCLE)]
    fam = {
        "family_id": "attr_f%03d" % fam_idx,
        "family_idx": fam_idx,
        "kind": "attr",
        "L": L,
        "d_levels": list(d_levels),
        "entity": ENTITIES[fam_idx % len(ENTITIES)],
        "planted_adj": ADJECTIVES[(2 * fam_idx) % len(ADJECTIVES)],
        "goal_adj": ADJECTIVES[(2 * fam_idx + 1) % len(ADJECTIVES)],
        "chain": [next(nouns) for _ in range(L)],
        "t0": next(nouns),
        "spur": [next(nouns) for _ in range(max(ATTR_D_ALLOWED))],
        "filler": [next(nouns) for _ in range(D_MAX + 1)],
    }
    assert fam["planted_adj"] != fam["goal_adj"]
    worlds = []
    for d in d_levels:
        for ver in ("A", "B"):
            worlds.append(build_attr_world(fam, d, ver, seed))
    for w in worlds:
        w["partner_world_id"] = w["world_id"][:-1] + ("B" if w["version"] == "A" else "A")
    return fam, worlds


def _block_rules(fam, d, refuting_rule, seed, extra_tag=""):
    """(d-1) spur cat rules + (D_MAX - d) filler-chain rules, seeded shuffle,
    then the refuting rule pinned at the last slot. d=0 or d='inf': no spur."""
    nd = 0 if d in (0, "inf") else d
    spur = fam["spur"][:nd]
    srules = ["Every %s is a %s." % (spur[i], spur[i + 1]) for i in range(max(0, nd - 1))]
    if d == "inf":
        n_fill = D_MAX - 1     # the frequency-mention rule occupies the refuting slot
    elif d == 0:
        n_fill = D_MAX         # complement carried by a premise fact instead
    else:
        n_fill = D_MAX - nd
    fill = fam["filler"][: n_fill + 1] if n_fill > 0 else []
    frules = ["Every %s is a %s." % (fill[i], fill[i + 1]) for i in range(n_fill)]
    body = srules + frules
    rng = random.Random(json.dumps([seed, fam["family_idx"], d_key(d), extra_tag]))
    rng.shuffle(body)
    if refuting_rule is not None:
        body.append(refuting_rule)
    return body


def _base_sentences(fam):
    chain, L = fam["chain"], fam["L"]
    goal_rules = ["Every %s is a %s." % (chain[i], chain[i + 1]) for i in range(L - 1)]
    goal_rules.append("Every %s is %s." % (chain[-1], fam["goal_adj"]))
    true_branch = "Every %s is a %s." % (chain[0], fam["t0"])
    return goal_rules, true_branch


def _canonical_proof(fam):
    E, chain = fam["entity"], fam["chain"]
    steps = ["%s is a %s." % (E, chain[0])]
    for i in range(len(chain) - 1):
        steps.append("Every %s is a %s." % (chain[i], chain[i + 1]))
        steps.append("%s is a %s." % (E, chain[i + 1]))
    steps.append("Every %s is %s." % (chain[-1], fam["goal_adj"]))
    steps.append("%s is %s." % (E, fam["goal_adj"]))
    return steps


def build_attr_world(fam, d, ver, seed):
    E, adj = fam["entity"], fam["planted_adj"]
    neg = ver == "A"   # world A carries the NEGATIVE refuting head (claim "E is adj." false)
    goal_rules, true_branch = _base_sentences(fam)
    if d == 0:
        block = _block_rules(fam, d, None, seed)
        comp_fact = "%s is %s%s." % (E, "not " if neg else "", adj)
        facts = ["%s is a %s." % (E, fam["chain"][0]), comp_fact]
    else:
        head = ("not " if neg else "") + adj
        refuting = "Every %s is %s." % (fam["spur"][d - 1], head)
        block = _block_rules(fam, d, refuting, seed)
        facts = ["%s is a %s." % (E, fam["chain"][0]), "%s is a %s." % (E, fam["spur"][0])]
    question = " ".join(goal_rules + [true_branch] + block + facts)
    target = "%s is %s." % (E, fam["goal_adj"])
    aff = "%s is %s." % (E, adj)
    negs = "%s is not %s." % (E, adj)
    if ver == "A":
        cells = [
            {"cell": "aff_false_attr_d%s" % d_key(d), "injected_statement": aff,
             "designed_truth": "false", "designed_d": d, "polarity": "affirmative",
             "claim_family": "attribute", "planted_token": adj, "designed_token_freq": 1},
            {"cell": "neg_true_attr_d%s" % d_key(d), "injected_statement": negs,
             "designed_truth": "true", "designed_d": d, "polarity": "negative",
             "claim_family": "attribute", "planted_token": adj, "designed_token_freq": 1},
        ]
    else:
        cells = [
            {"cell": "neg_false_attr_d%s" % d_key(d), "injected_statement": negs,
             "designed_truth": "false", "designed_d": d, "polarity": "negative",
             "claim_family": "attribute", "planted_token": adj, "designed_token_freq": 1},
            {"cell": "aff_true_attr_d%s" % d_key(d), "injected_statement": aff,
             "designed_truth": "true", "designed_d": d, "polarity": "affirmative",
             "claim_family": "attribute", "planted_token": adj, "designed_token_freq": 1},
        ]
    return {
        "world_id": "%s_d%s_%s" % (fam["family_id"], d_key(d), ver),
        "family_id": fam["family_id"],
        "family_idx": fam["family_idx"],
        "kind": "attr",
        "d": d,
        "version": ver,
        "entity": E,
        "L": fam["L"],
        "chain": fam["chain"],
        "t0": fam["t0"],
        "spur_used": fam["spur"][:d] if d not in (0, "inf") else [],
        "planted_token": adj,
        "goal_adjective": fam["goal_adj"],
        "question": question,
        "target": target,
        "query": "Prove: %s" % target,
        "canonical_proof_steps": _canonical_proof(fam),
        "cells": cells,
        "anchors_supported": ver == "A",
    }


def build_cat_family(fam_idx, d_levels, nouns, seed):
    """One categorical-reuse family: usable/inert twin worlds at each d.
    Usable branch per MAJOR-2: z -> bridge -> C_{L-1}, bridge fresh."""
    for d in d_levels:
        if d not in CAT_D_ALLOWED:
            raise ValueError("categorical family d=%r not in %r" % (d, CAT_D_ALLOWED))
    L = GOAL_L_CYCLE[fam_idx % len(GOAL_L_CYCLE)]
    max_fd = max([d for d in d_levels if d != "inf"] or [1])
    fam = {
        "family_id": "cat_f%03d" % fam_idx,
        "family_idx": fam_idx,
        "kind": "cat",
        "L": L,
        "d_levels": list(d_levels),
        "entity": ENTITIES[fam_idx % len(ENTITIES)],
        "goal_adj": ADJECTIVES[(2 * fam_idx + 1) % len(ADJECTIVES)],
        "planted_adj": None,
        "chain": [next(nouns) for _ in range(L)],
        "t0": next(nouns),
        "spur": [next(nouns) for _ in range(max(max_fd, 1))],
        "filler": [next(nouns) for _ in range(D_MAX + 1)],
        "z": next(nouns),
        "bridge": next(nouns),
        "decoy": next(nouns),
    }
    worlds = []
    for d in d_levels:
        for usability in ("usable", "inert"):
            worlds.append(build_cat_world(fam, d, usability, seed))
    for w in worlds:
        w["partner_world_id"] = w["world_id"].replace(
            "_usable", "_XX").replace("_inert", "_usable").replace("_XX", "_inert")
    return fam, worlds


def build_cat_world(fam, d, usability, seed):
    E, z, bridge = fam["entity"], fam["z"], fam["bridge"]
    goal_rules, true_branch = _base_sentences(fam)
    if d == "inf":
        # CWA-only falsity: the complement is underivable from the entity's
        # state (canonical PLAN2 section-3 definition of d=inf). To keep the
        # lexical stimulus matched across d (same planted-token count, same
        # "not"-token count, same word count), the frequency-mention rule
        # carries the same refuting-shaped head on an ENTITY-DISCONNECTED body
        # (the filler-chain tail, unreachable from the entity's premises).
        # The world-audit asserts measured d is None and the falsity basis is
        # unentailed_category_closed_world. Disclosed in the audit manifest.
        freq_mention = "Every %s is not a %s." % (fam["filler"][D_MAX - 1], z)
        block = _block_rules(fam, d, freq_mention, seed)
    else:
        refuting = "Every %s is not a %s." % (fam["spur"][d - 1], z)
        block = _block_rules(fam, d, refuting, seed)
    branch_body = z if usability == "usable" else fam["decoy"]
    branch = [
        "Every %s is a %s." % (branch_body, bridge),
        "Every %s is a %s." % (bridge, fam["chain"][-1]),
    ]
    facts = ["%s is a %s." % (E, fam["chain"][0]), "%s is a %s." % (E, fam["spur"][0])]
    question = " ".join(goal_rules + [true_branch] + block + branch + facts)
    target = "%s is %s." % (E, fam["goal_adj"])
    planted = "%s is a %s." % (E, z)
    cells = [{
        "cell": "cat_false_%s_d%s" % (usability, d_key(d)),
        "injected_statement": planted,
        "designed_truth": "false",
        "designed_d": d,
        "polarity": "affirmative",
        "claim_family": "category",
        "planted_token": z,
        "designed_token_freq": 2 if usability == "usable" else 1,
    }]
    return {
        "world_id": "%s_d%s_%s" % (fam["family_id"], d_key(d), usability),
        "family_id": fam["family_id"],
        "family_idx": fam["family_idx"],
        "kind": "cat",
        "d": d,
        "version": usability,
        "entity": E,
        "L": fam["L"],
        "chain": fam["chain"],
        "t0": fam["t0"],
        "spur_used": fam["spur"][:d] if d != "inf" else [],
        "z": z,
        "bridge": bridge,
        "decoy": fam["decoy"],
        "planted_token": z,
        "goal_adjective": fam["goal_adj"],
        "question": question,
        "target": target,
        "query": "Prove: %s" % target,
        "canonical_proof_steps": _canonical_proof(fam),
        "cells": cells,
        "anchors_supported": False,
    }


# ------------------------------------------------------------------ audits

def one_token_diff(qa, qb):
    """Return (ok, detail): ok iff the two strings differ by exactly one
    whitespace token (one substitution, or one insertion/deletion)."""
    ta, tb = qa.split(), qb.split()
    if ta == tb:
        return False, "identical"
    if len(ta) == len(tb):
        diffs = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
        if len(diffs) == 1:
            return True, "substitution@%d:%s->%s" % (diffs[0], ta[diffs[0]], tb[diffs[0]])
        return False, "%d token substitutions" % len(diffs)
    if abs(len(ta) - len(tb)) == 1:
        lo, hi = (ta, tb) if len(ta) < len(tb) else (tb, ta)
        i = 0
        while i < len(lo) and lo[i] == hi[i]:
            i += 1
        if hi[:i] + hi[i + 1:] == lo:
            return True, "insertion@%d:%s" % (i, hi[i])
        return False, "length differs by 1 but not a single insertion"
    return False, "token counts differ by %d" % abs(len(ta) - len(tb))


def token_freq(question, token):
    return len(re.findall(r"\b%s\b" % re.escape(token), question))


class WorldAuditError(AssertionError):
    pass


def _req(cond, world_id, msg):
    if not cond:
        raise WorldAuditError("%s: %s" % (world_id, msg))


def audit_world(world):
    """Fail-closed single-world audit. Returns the audit record on success."""
    wid, q, E = world["world_id"], world["question"], world["entity"]
    rules, facts, unparsed = parse_world(q)
    _req(not unparsed, wid, "question has unparsed sentences: %r" % unparsed)
    _, direct, reach, state, _ = world_state(q, E, ())

    # canonical gold proof validates and solves
    proof = " ".join(world["canonical_proof_steps"])
    v = validate_continuation(q, [], None, proof, world["target"], E)
    _req(v["class"] == "valid_rederivation", wid, "canonical proof class=%s" % v["class"])

    # off-path guarantees: spur nouns and t0 never appear in the canonical proof
    for noun in list(world["spur_used"]) + [world["t0"]]:
        _req(noun not in proof, wid, "off-path noun %s appears in canonical proof" % noun)

    cell_audits = []
    for cell in world["cells"]:
        stmt = cell["injected_statement"]
        pred = statement_predicate(stmt, E)
        _req(pred is not None, wid, "planted statement unparsed: %r" % stmt)
        truth = audit_truth_status(q, stmt, E, ())
        _req(truth["truth_status"] == cell["designed_truth"], wid,
             "%s: truth %s != designed %s (basis=%s)" % (
                 cell["cell"], truth["truth_status"], cell["designed_truth"], truth["audit_basis"]))
        dd = cell["designed_d"]
        if cell["designed_truth"] == "false":
            md = truth["local_rule_distance"]   # distance to the COMPLEMENT (expc canonical)
            if dd == "inf":
                _req(md is None, wid, "%s: measured d=%r, designed inf" % (cell["cell"], md))
                _req(truth["audit_basis"] == "unentailed_category_closed_world", wid,
                     "%s: inf falsity basis=%s (must be CWA)" % (cell["cell"], truth["audit_basis"]))
            else:
                _req(md == dd, wid, "%s: measured d=%r != designed %r" % (cell["cell"], md, dd))
        else:
            md = 0 if pred in state else shortest_rule_distance(pred, state, direct)
            _req(md == dd, wid, "%s: true claim entailed at %r != designed %r" % (cell["cell"], md, dd))
        tf = token_freq(q, cell["planted_token"])
        _req(tf == cell["designed_token_freq"], wid,
             "%s: planted-token freq %d != designed %d" % (cell["cell"], tf, cell["designed_token_freq"]))
        _req("attr" not in cell["cell"] or d_key(dd) != "inf", wid,
             "MAJOR-1: attribute cell at d=inf may not exist")
        cell_audits.append({
            "cell": cell["cell"], "measured_d": md, "audit_basis": truth["audit_basis"],
            "truth_status": truth["truth_status"], "planted_token_freq": tf,
            "grammar_template": truth.get("grammar_template"),
        })

    major2 = None
    if world["kind"] == "cat":
        zp, bp = ("cat", world["z"]), ("cat", world["bridge"])
        chain_end = ("cat", world["chain"][-1])
        bridge_feeds_goal = (world["bridge"], chain_end) in rules
        _req(bridge_feeds_goal, wid, "bridge does not feed the goal chain")
        bridge_underivable = not derivable(bp, state, reach)
        _req(bridge_underivable, wid, "MAJOR-2: bridge derivable from true state")
        st2 = set(state)
        st2.add(zp)
        reach2 = closure(rules)
        bridge_with_plant = derivable(bp, st2, reach2)
        z_out_rules = [r for r in rules if r[0] == world["z"]]
        if world["version"] == "usable":
            _req(bridge_with_plant, wid, "MAJOR-2: bridge not derivable even with planted atom")
            _req(len(z_out_rules) == 1, wid, "usable z must have exactly the branch rule")
        else:
            _req(not z_out_rules, wid, "inert z has outgoing rules")
            _req(not bridge_with_plant, wid, "inert bridge reachable via planted atom")
        major2 = {
            "bridge": world["bridge"],
            "bridge_underivable_without_plant": bridge_underivable,
            "bridge_derivable_with_plant": bridge_with_plant,
            "bridge_feeds_goal_chain": bridge_feeds_goal,
        }

    return {
        "world_id": wid,
        "rule_count": len(rules),
        "fact_count": len(facts),
        "word_count": len(q.split()),
        "char_count": len(q),
        "distinct_category_count": len({r[0] for r in rules}
                                       | {p[1] for _, p in rules if p[0] == "cat"}
                                       | {p[1] for _, p in facts if p[0] == "cat"}),
        "canonical_proof_class": v["class"],
        "cells": cell_audits,
        "major2": major2,
    }


def audit_family(fam, worlds, audits_by_id):
    """Cross-world audits: mirror one-token diff + constancy across d."""
    recs = []
    by = {(w["d"], w["version"]): w for w in worlds}
    vers = sorted({w["version"] for w in worlds})
    fid = fam["family_id"]
    pair = ("A", "B") if fam["kind"] == "attr" else ("usable", "inert")
    for d in fam["d_levels"]:
        wa, wb = by.get((d, pair[0])), by.get((d, pair[1]))
        if wa and wb:
            ok, detail = one_token_diff(wa["question"], wb["question"])
            _req(ok, fid, "d=%s mirrored pair not one-token: %s" % (d_key(d), detail))
            recs.append({"family_id": fid, "d": d_key(d), "pair": pair, "one_token_diff": detail})
    for ver in vers:
        ws = [by[(d, ver)] for d in fam["d_levels"] if (d, ver) in by]
        rcs = {audits_by_id[w["world_id"]]["rule_count"] for w in ws}
        wcs = {audits_by_id[w["world_id"]]["word_count"] for w in ws}
        dcs = {audits_by_id[w["world_id"]]["distinct_category_count"] for w in ws}
        _req(len(rcs) == 1, fid, "rule count varies across d for %s: %r" % (ver, rcs))
        _req(len(wcs) == 1, fid, "word count varies across d for %s: %r" % (ver, wcs))
        recs.append({"family_id": fid, "version": ver, "rule_count": rcs.pop(),
                     "word_count": wcs.pop(), "distinct_category_counts": sorted(dcs)})
    return recs


# --------------------------------------------------------------- generation

def generate_worlds(seed, config):
    rng = random.Random(seed)
    nouns = noun_stream(rng)
    families, worlds = [], []
    fam_idx = 0
    for spec in config["attr"]:
        for _ in range(spec["families"]):
            fam, ws = build_attr_family(fam_idx, spec["d_levels"], nouns, seed)
            families.append(fam)
            worlds.extend(ws)
            fam_idx += 1
    for spec in config["cat"]:
        for _ in range(spec["families"]):
            fam, ws = build_cat_family(fam_idx, spec["d_levels"], nouns, seed)
            families.append(fam)
            worlds.extend(ws)
            fam_idx += 1
    return families, worlds


def run_audits(families, worlds):
    audits, failures = {}, []
    for w in worlds:
        try:
            audits[w["world_id"]] = audit_world(w)
        except WorldAuditError as e:
            failures.append(str(e))
    fam_recs = []
    by_fam = {}
    for w in worlds:
        by_fam.setdefault(w["family_id"], []).append(w)
    for fam in families:
        try:
            fam_recs.extend(audit_family(fam, by_fam[fam["family_id"]], audits))
        except WorldAuditError as e:
            failures.append(str(e))
        except KeyError as e:
            failures.append("%s: missing per-world audit %s" % (fam["family_id"], e))
    for w in worlds:  # MAJOR-1 manifest-name check
        for cell in w["cells"]:
            if re.search(r"attr.*_dinf", cell["cell"]):
                failures.append("%s: forbidden attribute-inf cell name %s" % (w["world_id"], cell["cell"]))
    return audits, fam_recs, failures


def write_outputs(out_dir, seed, config, families, worlds, audits, fam_recs, failures):
    os.makedirs(out_dir, exist_ok=True)
    ds = {}
    for w in worlds:
        ds[w["world_id"]] = {"test_example": {
            "question": w["question"], "query": w["query"],
            "chain_of_thought": w["canonical_proof_steps"],
        }}
    with open(os.path.join(out_dir, "worlds.json"), "w") as fh:
        json.dump(ds, fh, indent=1, sort_keys=True)
    with open(os.path.join(out_dir, "world_manifest.jsonl"), "w") as fh:
        for w in worlds:
            row = dict(w)
            row["audit"] = audits.get(w["world_id"])
            row["created_at"] = now_iso()
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    sha = hashlib.sha256(open(os.path.join(out_dir, "worlds.json"), "rb").read()).hexdigest()
    audit_doc = {
        "experiment": "EXPD world generation",
        "created_at": now_iso(),
        "seed": seed,
        "config": config,
        "grammar_theorems": THEOREMS,
        "design_notes": {
            "planted_token_frequency": (
                "attribute cells and inert categorical cells: exactly 1. "
                "STRUCTURAL DEVIATION (disclosed): usable categorical cells are 2 by "
                "construction (refuting head + usable-branch body per MAJOR-2); held "
                "constant across d (incl. d=inf via a frequency-mention filler rule)."),
            "negation_tokens": (
                "attribute world A / cat worlds contain exactly one 'not' (the "
                "refuting-shaped head), constant across d INCLUDING d=inf; the planted "
                "aff_false sentence itself is negation-free and byte-identical across d."),
            "d_inf_semantics": (
                "cat d=inf worlds carry the refuting-shaped head on an "
                "entity-disconnected body (filler-chain tail), so the complement is "
                "underivable from the entity's state (canonical d=inf per PLAN2 "
                "section 3, audited: measured d is None, falsity basis is CWA) while "
                "the lexical stimulus stays matched across d. This strengthens "
                "causal2's 'no refuting rule anywhere' wording to 'no REACHABLE "
                "refutation' - disclosed."),
            "refuting_rule_position": "pinned to the last block slot before the facts at every d",
        },
        "world_count": len(worlds),
        "family_count": len(families),
        "per_world_audits_passed": len(audits),
        "family_audit_records": fam_recs,
        "audit_failures": failures,
        "worlds_json_sha256": sha,
        "distance_function": "expc_polarity_control.shortest_rule_distance (canonical, PLAN2 section 3)",
        "src_path": SRC,
    }
    with open(os.path.join(out_dir, "world_audit.json"), "w") as fh:
        json.dump(audit_doc, fh, indent=2, sort_keys=True)
    return audit_doc


def generate_and_write(out_dir, seed, config_name):
    config = CONFIGS[config_name]
    families, worlds = generate_worlds(seed, config)
    audits, fam_recs, failures = run_audits(families, worlds)
    doc = write_outputs(out_dir, seed, config, families, worlds, audits, fam_recs, failures)
    if failures:
        raise SystemExit("FAIL-CLOSED: %d world audit failures (see world_audit.json):\n%s"
                         % (len(failures), "\n".join(failures[:20])))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", choices=sorted(CONFIGS), default="pilot")
    args = ap.parse_args(argv)
    doc = generate_and_write(args.out_dir, args.seed, args.config)
    print(json.dumps({k: doc[k] for k in
                      ("world_count", "family_count", "per_world_audits_passed",
                       "audit_failures", "worlds_json_sha256")}, indent=2))


if __name__ == "__main__":
    main()
