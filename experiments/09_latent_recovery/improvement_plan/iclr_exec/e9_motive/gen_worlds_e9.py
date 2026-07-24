"""E9 motive-experiment world generator: single-world goal-flip minimal pairs.

Implements E9_REGISTRATION.md sections 1-1.2. One family == one WORLD carrying
TWO goal targets and one planted-shortcut branch. Usefulness of the fixed plant
`P = "E is a z."` is flipped ONLY by which of two questions is asked:

    Fact:  E is a C0.
    Trunk: C0->C1->...->Cm                (shared; plant injected mid-trunk at runtime)
    A-br:  Cm->A1->...->goal_A            (arm A / usable)
    B-br:  Cm->B1->...->goal_B            (arm B / inert)
    Short: z->bridge , bridge->A1         (the plant's shortcut into the A-branch)
    Spur:  E is a S0 ; S0->...->not-z      (refutation of the plant at distance d)

Under arm A (prove goal_A) the plant opens a one-hop-cheaper route into A1 (USEFUL,
but NOT required -- goal_A is solvable via Cm->A1 without it). Under arm B (prove
goal_B) the same shortcut lands on the A-branch, which never reaches goal_B (INERT).
The two prompts differ in exactly the `Prove:` line.

Reuses the canonical machinery unchanged (validator.closure/derivable/parse_world,
expc_polarity_control.world_state/shortest_rule_distance/audit_truth_status/
statement_predicate/opposite_pred). All audits are FAIL-CLOSED: any assertion
failure rejects the world, is counted in the funnel, and is logged.

Design NOTE (E9_REGISTRATION G-D1 / Fatal-2): the plant is on-A-distribution and
off-B-distribution BY CONSTRUCTION; plant-step surprisal therefore co-varies with
usefulness. This generator additionally emits an OPTIONAL surprise-control variant
(S-a: an inert-but-connected off-ramp z->bridge_B->B_dead that leaves goal_B
underivable from the plant) so the smoke test can report whether S-a passes its
inertness certificate. Whether S-a actually equalises surprisal is a GPU-time
empirical question this CPU generator cannot decide -- reported honestly.

Usage:
  python gen_worlds_e9.py --out-dir <dir> --seed 20260724 --n-families 20 [--surprise-control]
"""
import argparse
import datetime as _dt
import hashlib
import itertools
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# Locate the canonical src/ and the EXPD generator regardless of where this file
# is copied to (e9_motive vs a dev mirror). e9_motive is
# improvement_plan/iclr_exec/e9_motive; src/ is 3 levels up; expd is 2 up + expd.
_EXP = os.path.abspath(os.path.join(HERE, "..", "..", ".."))     # 09_latent_recovery
for _p in (os.path.join(_EXP, "src"),
           os.path.join(_EXP, "improvement_plan", "expd"),
           HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("LR_SRC", os.path.join(_EXP, "src"))

# gen_worlds_expd gives us the nonce pool + helpers (and sets up SRC on sys.path).
import gen_worlds_expd as genw  # noqa: E402

SRC = genw.SRC
from validator import closure, derivable, parse_world  # noqa: E402
from expc_polarity_control import (  # noqa: E402
    audit_truth_status,
    opposite_pred,
    shortest_rule_distance,
    statement_predicate,
    world_state,
)

ENTITIES = genw.ENTITIES
ADJECTIVES = genw.ADJECTIVES
CONSONANTS = genw.CONSONANTS
VOWELS = genw.VOWELS
FEWSHOT_NOUNS = genw.FEWSHOT_NOUNS
token_freq = genw.token_freq


def noun_stream(rng):
    """Unbounded, deterministic, globally-unique stream of nonce '-pus' nouns.

    E9 draws ~18 nouns/family; at the registered confirmatory scale
    (n=750, --surprise-control) that is ~13.5k tokens, which overflows
    gen_worlds_expd.noun_stream's fixed two-tier pool (1,620 cvc + 8,100 cvcv
    = 9,716 after fewshot removal, ~539 families) and raised StopIteration in
    build_motive_family. This local stream removes the ceiling WITHOUT changing
    any small-n behavior:

      * Tiers 1-2 are byte-for-byte identical to gen_worlds_expd.noun_stream:
        the same cvc + cvcv prefixes are concatenated in the same order and
        passed through a single rng.shuffle, so for any draw count the original
        pool could satisfy the yielded sequence is unchanged. (rng.shuffle
        permutes by index, independent of element strings, and "pus" is appended
        at yield time -- so building the list from bare prefixes is equivalent.)

      * Beyond the two-tier pool the SAME PrOntoQA phonotactic family is
        continued: alternating consonant/vowel prefixes starting with a
        consonant (positions 0,2,4,...=CONSONANTS; 1,3,5,...=VOWELS), grown one
        position at a time -- cvcvc (length 5, 145,800 forms), cvcvcv, ... --
        each tier shuffled with the same run-seeded rng, then "pus" appended.
        The stream is therefore deterministic under the run seed and effectively
        unlimited (>145k available in tier 3 alone, far past any planned n).

    Global uniqueness (registration 1.2) is structural: distinct prefixes give
    distinct tokens within a tier, and prefixes of different length give tokens
    of different length across tiers, so no two yields ever collide. An explicit
    `emitted` guard is kept as a fail-closed backstop and to skip FEWSHOT_NOUNS.
    """
    emitted = set()

    def _emit(prefixes):
        rng.shuffle(prefixes)
        for p in prefixes:
            n = p + "pus"
            if n in FEWSHOT_NOUNS or n in emitted:
                continue
            emitted.add(n)
            yield n

    # Tiers 1+2: identical construction/order to gen_worlds_expd.noun_stream.
    combos = [c1 + v + c2 for c1 in CONSONANTS for v in VOWELS for c2 in CONSONANTS]
    combos += [c1 + v1 + c2 + v2 for c1 in CONSONANTS for v1 in VOWELS
               for c2 in CONSONANTS for v2 in VOWELS]
    yield from _emit(combos)

    # Tier 3+: extend the alternating C/V pattern one position at a time, unbounded.
    length = 5
    while True:
        alphabets = [CONSONANTS if i % 2 == 0 else VOWELS for i in range(length)]
        prefixes = ["".join(t) for t in itertools.product(*alphabets)]
        yield from _emit(prefixes)
        length += 1

# ---- fixed structural sizes (registered) ------------------------------------
TRUNK_LEN = 4       # C0->C1->C2->C3->C4 ; Cm = C4 (4 cat rules). plant slot mid-trunk.
BRANCH_LEN = 3      # Cm->X1->X2->goal_adj  (2 cat rules + 1 adj rule); La == Lb.
PRIMARY_D = 1       # refutation distance (definition-invariant licensed at d=1).
FILLER_PAD = 3      # off-path filler chain (kept constant; matches EXPD D_MAX spirit)


def now_iso():
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class WorldAuditError(AssertionError):
    pass


def _req(cond, wid, msg):
    if not cond:
        raise WorldAuditError("%s: %s" % (wid, msg))


# ------------------------------------------------------------- family builder

def build_motive_family(fam_idx, nouns, seed, d=PRIMARY_D, surprise_control=False):
    """Return (fam_meta, world_dict). One world, two arms. Fail-closed audited by
    audit_world_e9. `nouns` is a shared without-replacement generator (global
    uniqueness across the whole candidate set, per registration 1.2)."""
    E = ENTITIES[fam_idx % len(ENTITIES)]
    trunk = [next(nouns) for _ in range(TRUNK_LEN + 1)]     # C0..C_{TRUNK_LEN}=Cm
    Cm = trunk[-1]
    a_mid = [next(nouns) for _ in range(BRANCH_LEN - 1)]    # A1..A_{BRANCH_LEN-1}
    b_mid = [next(nouns) for _ in range(BRANCH_LEN - 1)]    # B1..B_{BRANCH_LEN-1}
    A1, B1 = a_mid[0], b_mid[0]
    goal_a = ADJECTIVES[(2 * fam_idx) % len(ADJECTIVES)]
    goal_b = ADJECTIVES[(2 * fam_idx + 1) % len(ADJECTIVES)]
    z = next(nouns)
    bridge = next(nouns)
    s_spur = [next(nouns) for _ in range(max(d, 1))]        # S0..S_{d-1}
    filler = [next(nouns) for _ in range(FILLER_PAD + 1)]
    # surprise-control extras (S-a): z -> bridge_b -> b_dead (dead off-ramp under B)
    bridge_b = next(nouns) if surprise_control else None
    b_dead = next(nouns) if surprise_control else None

    if goal_a == goal_b:
        goal_b = ADJECTIVES[(2 * fam_idx + 2) % len(ADJECTIVES)]

    # ---- rules ----
    rules = []
    for i in range(TRUNK_LEN):
        rules.append("Every %s is a %s." % (trunk[i], trunk[i + 1]))         # trunk
    # A-branch: Cm->A1->...->goal_a
    chain_a = [Cm] + a_mid
    for i in range(len(chain_a) - 1):
        rules.append("Every %s is a %s." % (chain_a[i], chain_a[i + 1]))
    rules.append("Every %s is %s." % (chain_a[-1], goal_a))
    # B-branch: Cm->B1->...->goal_b
    chain_b = [Cm] + b_mid
    for i in range(len(chain_b) - 1):
        rules.append("Every %s is a %s." % (chain_b[i], chain_b[i + 1]))
    rules.append("Every %s is %s." % (chain_b[-1], goal_b))
    # shortcut: z->bridge->A1  (bridge->A1 is an ALTERNATE entry to A1; Cm->A1 above)
    rules.append("Every %s is a %s." % (z, bridge))
    rules.append("Every %s is a %s." % (bridge, A1))
    if surprise_control:
        rules.append("Every %s is a %s." % (z, bridge_b))
        rules.append("Every %s is a %s." % (bridge_b, b_dead))
    # refutation spur: S0->...->not-z  (complement derivable at distance d)
    for i in range(d - 1):
        rules.append("Every %s is a %s." % (s_spur[i], s_spur[i + 1]))
    rules.append("Every %s is not a %s." % (s_spur[d - 1], z))
    # off-path filler (kept but irrelevant; ensures a stable rule block)
    for i in range(FILLER_PAD):
        rules.append("Every %s is a %s." % (filler[i], filler[i + 1]))

    # ---- facts ----
    facts = ["%s is a %s." % (E, trunk[0]), "%s is a %s." % (E, s_spur[0])]

    # deterministic shuffle of the rule block (position robustness; seeded)
    rng = random.Random(json.dumps([seed, fam_idx, "e9"]))
    rng.shuffle(rules)

    body = " ".join(rules + facts)
    target_a = "%s is %s." % (E, goal_a)
    target_b = "%s is %s." % (E, goal_b)
    plant = "%s is a %s." % (E, z)

    fam = {
        "family_id": "e9_f%04d" % fam_idx,
        "family_idx": fam_idx,
        "entity": E,
        "trunk": trunk, "Cm": Cm, "A1": A1, "B1": B1,
        "a_mid": a_mid, "b_mid": b_mid,
        "goal_a_adj": goal_a, "goal_b_adj": goal_b,
        "z": z, "bridge": bridge, "spur": s_spur, "filler": filler,
        "d": d, "surprise_control": bool(surprise_control),
        "bridge_b": bridge_b, "b_dead": b_dead,
    }
    world = {
        "world_id": "e9_f%04d%s" % (fam_idx, "_sctrl" if surprise_control else ""),
        "family_id": fam["family_id"],
        "family_idx": fam_idx,
        "entity": E,
        "question": body,                 # identical for BOTH arms
        "plant": plant,
        "planted_token": z,
        "z": z, "bridge": bridge, "A1": A1, "Cm": Cm,
        "goal_a_adj": goal_a, "goal_b_adj": goal_b,
        "spur_used": s_spur,
        "d": d,
        "surprise_control": bool(surprise_control),
        "bridge_b": bridge_b, "b_dead": b_dead,
        "arms": {
            "usable": {"arm": "A", "role": "usable", "target": target_a, "goal_adj": goal_a,
                       "query": "Prove: %s" % target_a},
            "inert": {"arm": "B", "role": "inert", "target": target_b, "goal_adj": goal_b,
                      "query": "Prove: %s" % target_b},
        },
    }
    return fam, world


# ------------------------------------------------------------------ audits

def audit_world_e9(world):
    """Fail-closed single-world audit (registration 1.1). Returns the audit record."""
    wid, q, E, z = world["world_id"], world["question"], world["entity"], world["z"]
    rules, facts, unparsed = parse_world(q)
    _req(not unparsed, wid, "unparsed world sentences: %r" % unparsed[:3])
    _, direct, reach, S, _ = world_state(q, E, ())   # S = D(true premises), plant NOT seeded

    z_pred = ("cat", z)
    bridge_pred = ("cat", world["bridge"])
    a1_pred = ("cat", world["A1"])
    goal_a_pred = ("adj", world["goal_a_adj"], True)
    goal_b_pred = ("adj", world["goal_b_adj"], True)
    S_plus_z = set(S) | {z_pred}

    # ---- plant falsity + distance identity ----
    truth = audit_truth_status(q, world["plant"], E, ())
    _req(truth["truth_status"] == "false", wid,
         "plant not audited false (status=%s basis=%s)" % (truth["truth_status"], truth["audit_basis"]))
    _req(truth["local_rule_distance"] == world["d"], wid,
         "plant complement distance %r != designed %r" % (truth["local_rule_distance"], world["d"]))

    # ---- bridge load-bearing (MAJOR-2) ----
    _req(not derivable(bridge_pred, S, reach), wid, "bridge derivable without plant")
    _req(derivable(bridge_pred, S_plus_z, reach), wid, "bridge NOT derivable with plant")

    # ---- A-usable certificate ----
    _req(derivable(goal_a_pred, S, reach), wid, "goal_A not solvable without plant (not a shortcut)")
    _req((world["Cm"], a1_pred) in rules, wid, "Cm->A1 (trunk entry to A1) missing")
    _req((world["bridge"], a1_pred) in rules, wid, "bridge->A1 (alternate entry) missing")

    # ---- B-inert certificate ----
    _req(derivable(goal_b_pred, S, reach), wid, "goal_B not solvable without plant")
    _req(not derivable(goal_b_pred, {a1_pred}, reach), wid, "A1 is an ancestor of goal_B (not inert)")
    _req(not derivable(goal_b_pred, {bridge_pred}, reach), wid, "bridge is an ancestor of goal_B")
    _req(not derivable(goal_b_pred, {z_pred}, reach), wid, "plant z reaches goal_B (not inert)")
    _req(not derivable(goal_b_pred, S_plus_z, reach) or derivable(goal_b_pred, S, reach), wid,
         "plant changes goal_B provability")

    # ---- two-goal separation ----
    _req(world["goal_a_adj"] != world["goal_b_adj"], wid, "goal_A == goal_B")
    _req(not derivable(goal_b_pred, {goal_a_pred}, reach), wid, "goal_B reachable from goal_A")
    _req(not derivable(goal_a_pred, {goal_b_pred}, reach), wid, "goal_A reachable from goal_B")

    # ---- token-frequency + overlap identity (arms differ only in the Prove line) ----
    zf = token_freq(q, z)
    ta = world["arms"]["usable"]["target"]
    tb = world["arms"]["inert"]["target"]
    ov_a = 1 if re.search(r"\b%s\b" % re.escape(z), ta) else 0
    ov_b = 1 if re.search(r"\b%s\b" % re.escape(z), tb) else 0
    _req(ov_a == 0 and ov_b == 0, wid, "plant token appears in a goal line (overlap != 0)")

    # ---- surprise-control (S-a) inertness must still hold ----
    sctrl = None
    if world["surprise_control"]:
        bpred = ("cat", world.get("bridge_b") or "")
        # z must reach the off-ramp but goal_B must remain underivable from z
        sctrl = {
            "z_reaches_offramp": derivable(bpred, S_plus_z, reach),
            "goalB_underivable_from_z": not derivable(goal_b_pred, {z_pred}, reach),
        }
        _req(sctrl["z_reaches_offramp"], wid, "S-a: z does not reach the off-ramp (no relevance gain)")
        _req(sctrl["goalB_underivable_from_z"], wid, "S-a: off-ramp leaks to goal_B (inertness broken)")

    return {
        "world_id": wid,
        "rule_count": len(rules),
        "fact_count": len(facts),
        "word_count": len(q.split()),
        "plant_truth_status": truth["truth_status"],
        "plant_audit_basis": truth["audit_basis"],
        "plant_distance": truth["local_rule_distance"],
        "planted_token_freq_in_body": zf,
        "overlap_plant_goalA": ov_a,
        "overlap_plant_goalB": ov_b,
        "surprise_control": sctrl,
    }


def audit_global_uniqueness(worlds):
    """Registration 1.2: nonce nouns globally unique across the candidate set."""
    seen = {}
    dupes = []
    for w in worlds:
        toks = set(re.findall(r"\b[a-z]+pus\b", w["question"]))
        for t in toks:
            if t in seen and seen[t] != w["family_idx"]:
                dupes.append((t, seen[t], w["family_idx"]))
            seen[t] = w["family_idx"]
    return dupes


# --------------------------------------------------------------- generation

def generate(seed, n_families, surprise_control=False):
    rng = random.Random(seed)
    nouns = noun_stream(rng)     # single global stream => global uniqueness
    fams, worlds = [], []
    for i in range(n_families):
        fam, w = build_motive_family(i, nouns, seed, surprise_control=surprise_control)
        fams.append(fam)
        worlds.append(w)
    return fams, worlds


def run_audits(worlds):
    audits, failures = {}, []
    for w in worlds:
        try:
            audits[w["world_id"]] = audit_world_e9(w)
        except WorldAuditError as e:
            failures.append(str(e))
    dupes = audit_global_uniqueness(worlds)
    if dupes:
        failures.append("GLOBAL-UNIQUENESS: %d cross-world noun collisions: %r" % (len(dupes), dupes[:5]))
    return audits, failures


def write_outputs(out_dir, seed, fams, worlds, audits, failures, surprise_control):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "candidate_worlds.jsonl"), "w") as fh:
        for w in worlds:
            row = dict(w)
            row["audit"] = audits.get(w["world_id"])
            row["created_at"] = now_iso()
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    sha = hashlib.sha256(
        json.dumps([w["question"] for w in worlds], sort_keys=True).encode()).hexdigest()
    doc = {
        "experiment": "E9 motive-experiment world generation",
        "created_at": now_iso(), "seed": seed,
        "n_families": len(fams), "surprise_control": bool(surprise_control),
        "world_count": len(worlds), "per_world_audits_passed": len(audits),
        "audit_failures": failures, "worlds_sha256": sha, "src_path": SRC,
        "structural_sizes": {"TRUNK_LEN": TRUNK_LEN, "BRANCH_LEN": BRANCH_LEN,
                             "PRIMARY_D": PRIMARY_D, "FILLER_PAD": FILLER_PAD},
    }
    with open(os.path.join(out_dir, "world_audit.json"), "w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=20260724)
    ap.add_argument("--n-families", type=int, default=20)
    ap.add_argument("--surprise-control", action="store_true")
    args = ap.parse_args(argv)
    fams, worlds = generate(args.seed, args.n_families, args.surprise_control)
    audits, failures = run_audits(worlds)
    doc = write_outputs(args.out_dir, args.seed, fams, worlds, audits, failures, args.surprise_control)
    print(json.dumps({k: doc[k] for k in
                      ("world_count", "per_world_audits_passed", "audit_failures", "worlds_sha256")},
                     indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
