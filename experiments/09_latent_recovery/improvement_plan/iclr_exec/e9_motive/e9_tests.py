"""CPU fixture tests for the E9 generator + audits (pattern: expd_tests.py).

Every fail-closed audit is exercised in both directions: PASS on good worlds,
FAIL on deliberately tampered ones. Run: python e9_tests.py
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_worlds_e9 as g9  # noqa: E402
from expc_polarity_control import statement_predicate, opposite_pred  # noqa: E402
from validator import derivable  # noqa: E402
from expc_polarity_control import world_state  # noqa: E402


def check(name, cond):
    if not cond:
        raise AssertionError("TEST FAILED: %s" % name)
    print("  ok: %s" % name)


def build(n=6, sc=False):
    fams, worlds = g9.generate(20260724, n, surprise_control=sc)
    audits, failures = g9.run_audits(worlds)
    return fams, worlds, audits, failures


def test_determinism():
    _, w1, _, _ = build()
    _, w2, _, _ = build()
    import json
    check("generator determinism",
          json.dumps([w["question"] for w in w1]) == json.dumps([w["question"] for w in w2]))


def test_good_worlds_pass():
    _, worlds, audits, failures = build()
    check("all good worlds pass audits (failures=%r)" % failures[:2], not failures)
    check("every world audited", len(audits) == len(worlds))


def test_two_arms_identical_body():
    _, worlds, _, _ = build()
    for w in worlds:
        check("arms share the identical body (only Prove line differs) [%s]" % w["world_id"],
              w["arms"]["usable"]["target"] != w["arms"]["inert"]["target"])
        check("plant token absent from both goal lines [%s]" % w["world_id"],
              w["z"] not in w["arms"]["usable"]["target"] and w["z"] not in w["arms"]["inert"]["target"])


def test_plant_false_at_d1():
    _, worlds, audits, _ = build()
    for w in worlds:
        a = audits[w["world_id"]]
        check("plant audited false [%s]" % w["world_id"], a["plant_truth_status"] == "false")
        check("plant complement distance == 1 [%s]" % w["world_id"], a["plant_distance"] == 1)


def test_usable_inert_certificates():
    _, worlds, _, _ = build()
    for w in worlds:
        _, direct, reach, S, _ = world_state(w["question"], w["entity"], ())
        z_pred = ("cat", w["z"]); bridge_pred = ("cat", w["bridge"])
        gA = ("adj", w["goal_a_adj"], True); gB = ("adj", w["goal_b_adj"], True)
        check("goal_A solvable without plant [%s]" % w["world_id"], derivable(gA, S, reach))
        check("goal_B solvable without plant [%s]" % w["world_id"], derivable(gB, S, reach))
        check("bridge load-bearing on plant [%s]" % w["world_id"],
              (not derivable(bridge_pred, S, reach)) and derivable(bridge_pred, S | {z_pred}, reach))
        check("plant z does NOT reach goal_B (inert) [%s]" % w["world_id"],
              not derivable(gB, {z_pred}, reach))
        check("plant z DOES reach goal_A via shortcut [%s]" % w["world_id"],
              derivable(gA, {z_pred}, reach))


def test_tamper_plant_not_false():
    """If we delete the refuting spur rule, the plant is no longer audited false."""
    _, worlds, _, _ = build()
    w = copy.deepcopy(worlds[0])
    import re
    # remove the 'Every S0 is not a z.' refuting rule
    w["question"] = re.sub(r"Every \w+ is not a %s\.\s*" % re.escape(w["z"]), "", w["question"])
    try:
        g9.audit_world_e9(w); failed = False
    except g9.WorldAuditError as e:
        failed = "false" in str(e) or "distance" in str(e)
    check("removing the refuting spur fails the plant-falsity audit", failed)


def test_tamper_inert_leak():
    """Add a rule bridge->B1 so the plant reaches goal_B: inertness must fail."""
    fams, worlds = g9.generate(20260724, 6)
    fam = fams[0]
    w = copy.deepcopy(worlds[0])
    w["question"] += " Every %s is a %s." % (w["bridge"], fam["B1"])
    try:
        g9.audit_world_e9(w); failed = False
    except g9.WorldAuditError as e:
        failed = "inert" in str(e) or "ancestor" in str(e)
    check("bridge->B1 leak breaks the inertness certificate", failed)


def test_tamper_bridge_independent():
    """If bridge is independently derivable (C0->bridge), load-bearing must fail."""
    fams, worlds = g9.generate(20260724, 6)
    w = copy.deepcopy(worlds[0])
    w["question"] += " Every %s is a %s." % (fams[0]["trunk"][0], w["bridge"])
    try:
        g9.audit_world_e9(w); failed = False
    except g9.WorldAuditError as e:
        failed = "bridge derivable without plant" in str(e)
    check("independently-derivable bridge fails load-bearing", failed)


def test_tamper_goal_a_requires_plant():
    """Remove Cm->A1 so goal_A needs the plant: the 'shortcut-not-requirement' cert fails."""
    fams, worlds = g9.generate(20260724, 6)
    w = copy.deepcopy(worlds[0]); fam = fams[0]
    import re
    w["question"] = re.sub(r"Every %s is a %s\.\s*" % (re.escape(fam["Cm"]), re.escape(w["A1"])),
                           "", w["question"], count=1)
    try:
        g9.audit_world_e9(w); failed = False
    except g9.WorldAuditError as e:
        failed = "without plant" in str(e) or "alternate entry" in str(e) or "Cm->A1" in str(e)
    check("removing Cm->A1 (plant becomes required) fails the usable certificate", failed)


def test_global_uniqueness():
    _, worlds, _, failures = build(n=12)
    dupes = g9.audit_global_uniqueness(worlds)
    check("no cross-world nonce collisions across 12 families", not dupes)


def test_surprise_control_constructs():
    _, worlds, audits, failures = build(n=6, sc=True)
    check("surprise-control worlds pass their audits (failures=%r)" % failures[:2], not failures)
    for w in worlds:
        sc = audits[w["world_id"]]["surprise_control"]
        check("S-a: z reaches the off-ramp [%s]" % w["world_id"], sc["z_reaches_offramp"])
        check("S-a: goal_B stays underivable from z [%s]" % w["world_id"], sc["goalB_underivable_from_z"])


def main():
    tests = [
        ("determinism", test_determinism),
        ("good worlds pass audits", test_good_worlds_pass),
        ("two arms share body", test_two_arms_identical_body),
        ("plant false at d=1", test_plant_false_at_d1),
        ("usable/inert certificates", test_usable_inert_certificates),
        ("tamper: plant not false", test_tamper_plant_not_false),
        ("tamper: inert leak", test_tamper_inert_leak),
        ("tamper: bridge independent", test_tamper_bridge_independent),
        ("tamper: goal_A requires plant", test_tamper_goal_a_requires_plant),
        ("global nonce uniqueness", test_global_uniqueness),
        ("surprise-control constructs", test_surprise_control_constructs),
    ]
    for name, fn in tests:
        print("[e9_tests] %s" % name)
        fn()
    print("E9 tests passed (%d groups)" % len(tests))


if __name__ == "__main__":
    main()
