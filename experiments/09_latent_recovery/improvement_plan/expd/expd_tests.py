"""Fixture tests for the EXPD world generator + runner audits (CPU-only).

Pattern: expa_tests.py / expc_tests.py. Every fail-closed audit is exercised in
both directions: it must PASS on constructed-good worlds and FAIL on
deliberately tampered ones.

Run:  python expd_tests.py
"""
import argparse
import copy
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_worlds_expd as genw
import expd_matched_gradient as runner

from validator import validate_continuation  # noqa: E402  (src on path via genw)
from expc_polarity_control import audit_truth_status, injection_points  # noqa: E402

TEST_CONFIG = {
    "name": "test_grid",
    "attr": [{"families": 2, "d_levels": [0, 1, 3, 5]}],
    "cat": [{"families": 2, "d_levels": [1, 3, "inf"]}],
}


def check(name, cond):
    if not cond:
        raise AssertionError("TEST FAILED: %s" % name)
    print("  ok: %s" % name)


def build():
    fams, worlds = genw.generate_worlds(0, TEST_CONFIG)
    audits, fam_recs, failures = genw.run_audits(fams, worlds)
    return fams, worlds, audits, fam_recs, failures


def wmap(worlds):
    return {w["world_id"]: w for w in worlds}


def test_determinism():
    _, w1, _, _, _ = build()
    _, w2, _, _, _ = build()
    check("generator determinism", json.dumps(w1, sort_keys=True) == json.dumps(w2, sort_keys=True))


def test_good_worlds_pass():
    _, worlds, audits, _, failures = build()
    check("all constructed worlds pass their audits (failures=%r)" % failures[:3], not failures)
    check("expected world count", len(worlds) == 2 * 4 * 2 + 2 * 3 * 2)
    check("every world audited", len(audits) == len(worlds))


def test_distances_measured_equals_designed():
    _, worlds, audits, _, _ = build()
    for w in worlds:
        for cell, ca in zip(w["cells"], audits[w["world_id"]]["cells"]):
            dd = cell["designed_d"]
            want = None if dd == "inf" else dd
            check("distance %s (measured=%r designed=%r)" % (ca["cell"], ca["measured_d"], dd),
                  ca["measured_d"] == want if cell["designed_truth"] == "false"
                  else ca["measured_d"] == dd)


def test_attr_inf_refused():
    rng_nouns = genw.noun_stream(__import__("random").Random(0))
    try:
        genw.build_attr_family(0, [1, "inf"], rng_nouns, 0)
        raised = False
    except ValueError:
        raised = True
    check("MAJOR-1: attribute family at d=inf raises", raised)
    try:
        genw.build_attr_family(0, [4], rng_nouns, 0)
        raised = False
    except ValueError:
        raised = True
    check("attribute family at off-grid d=4 raises", raised)


def test_no_attr_inf_cell_names():
    _, worlds, _, _, _ = build()
    names = [c["cell"] for w in worlds for c in w["cells"]]
    check("no *_attr_*inf cell name exists in the manifest",
          not any("attr" in n and "inf" in n for n in names))
    check("cat inf cells exist and carry the inf anchors",
          any(n == "cat_false_usable_dinf" for n in names))


def test_mirror_one_token_and_tamper():
    _, worlds, _, _, _ = build()
    by = wmap(worlds)
    wa, wb = by["attr_f000_d3_A"], by["attr_f000_d3_B"]
    ok, detail = genw.one_token_diff(wa["question"], wb["question"])
    check("mirrored pair differs by exactly one token (%s)" % detail, ok)
    check("the differing token is the polarity token", "not" in detail)
    # tamper: change an additional noun in B -> audit must fail
    bad = copy.deepcopy(wb)
    bad["question"] = bad["question"].replace(bad["chain"][2], "hackpus")
    ok2, _ = genw.one_token_diff(wa["question"], bad["question"])
    check("two-token tamper detected", not ok2)
    # d=0 mirrored pair: polarity token lives in the premise fact
    ok3, det3 = genw.one_token_diff(by["attr_f000_d0_A"]["question"], by["attr_f000_d0_B"]["question"])
    check("d=0 mirrored pair one-token diff (%s)" % det3, ok3 and "not" in det3)


def test_planted_freq_and_tamper():
    _, worlds, _, _, _ = build()
    by = wmap(worlds)
    w = by["attr_f000_d1_A"]
    check("attr planted-token freq == 1",
          genw.token_freq(w["question"], w["planted_token"]) == 1)
    bad = copy.deepcopy(w)
    bad["question"] += " Every %s is %s." % (bad["chain"][1], bad["planted_token"])
    try:
        genw.audit_world(bad)
        failed = False
    except genw.WorldAuditError as e:
        failed = "freq" in str(e) or "truth" in str(e)
    check("extra planted-token mention fails the world audit", failed)
    u = by["cat_f002_d1_usable"]
    i = by["cat_f002_d1_inert"]
    check("cat usable planted-token freq == 2 (disclosed deviation)",
          genw.token_freq(u["question"], u["planted_token"]) == 2)
    check("cat inert planted-token freq == 1",
          genw.token_freq(i["question"], i["planted_token"]) == 1)
    uinf = by["cat_f002_dinf_usable"]
    check("cat usable freq constant at d=inf",
          genw.token_freq(uinf["question"], uinf["planted_token"]) == 2)


def test_major2_usable_branch():
    _, worlds, audits, _, _ = build()
    by = wmap(worlds)
    u = by["cat_f002_d3_usable"]
    a = audits[u["world_id"]]["major2"]
    check("MAJOR-2: bridge underivable without planted atom", a["bridge_underivable_without_plant"])
    check("MAJOR-2: bridge derivable with planted atom", a["bridge_derivable_with_plant"])
    check("MAJOR-2: bridge feeds the goal chain", a["bridge_feeds_goal_chain"])
    # tamper: make the usable branch mechanically-zero (z -> C_{L-1} direct, no
    # fresh intermediate) - exactly the broken pre-MAJOR-2 design; must fail.
    bad = copy.deepcopy(u)
    bad["question"] = bad["question"].replace(
        "Every %s is a %s." % (bad["z"], bad["bridge"]),
        "Every %s is a %s." % (bad["z"], bad["chain"][-1]))
    try:
        genw.audit_world(bad)
        failed = False
    except genw.WorldAuditError as e:
        failed = "bridge" in str(e)
    check("mechanically-zero usable branch (no fresh intermediate) fails the audit", failed)
    # tamper: make the bridge independently derivable (defeats injection-dependence)
    bad2 = copy.deepcopy(u)
    bad2["question"] += " Every %s is a %s." % (bad2["chain"][0], bad2["bridge"])
    try:
        genw.audit_world(bad2)
        failed2 = False
    except genw.WorldAuditError as e:
        failed2 = "MAJOR-2" in str(e)
    check("independently-derivable bridge fails the audit", failed2)


def test_constancy_across_d():
    _, worlds, audits, _, _ = build()
    for fid, ver in (("attr_f000", "A"), ("attr_f000", "B"), ("cat_f002", "usable"), ("cat_f002", "inert")):
        ws = [w for w in worlds if w["family_id"] == fid and w["version"] == ver]
        rcs = {audits[w["world_id"]]["rule_count"] for w in ws}
        wcs = {audits[w["world_id"]]["word_count"] for w in ws}
        ccs = {audits[w["world_id"]]["char_count"] for w in ws}
        dcs = {audits[w["world_id"]]["distinct_category_count"] for w in ws}
        check("rule count constant across d (%s/%s)" % (fid, ver), len(rcs) == 1)
        check("word count constant across d (%s/%s)" % (fid, ver), len(wcs) == 1)
        check("distinct category count constant across d (%s/%s): %r" % (fid, ver, dcs), len(dcs) == 1)
        print("    (%s/%s char counts across d: %r)" % (fid, ver, sorted(ccs)))


def test_truth_statuses():
    _, worlds, _, _, _ = build()
    by = wmap(worlds)
    wa = by["attr_f000_d3_A"]
    for cell in wa["cells"]:
        t = audit_truth_status(wa["question"], cell["injected_statement"], wa["entity"], ())
        check("truth %s == %s" % (cell["cell"], cell["designed_truth"]),
              t["truth_status"] == cell["designed_truth"])
    uinf = by["cat_f002_dinf_usable"]
    t = audit_truth_status(uinf["question"], uinf["cells"][0]["injected_statement"], uinf["entity"], ())
    check("cat inf falsity is CWA (basis=%s)" % t["audit_basis"],
          t["truth_status"] == "false" and t["audit_basis"] == "unentailed_category_closed_world")
    check("cat inf measured distance is None", t["local_rule_distance"] is None)


def test_validator_wiring():
    _, worlds, _, _, _ = build()
    by = wmap(worlds)
    u = by["cat_f002_d1_usable"]
    E, z, bridge, chain = u["entity"], u["z"], u["bridge"], u["chain"]
    proof = " ".join(u["canonical_proof_steps"])
    v = validate_continuation(u["question"], [], None, proof, u["target"], E)
    check("canonical proof validates", v["class"] == "valid_rederivation")
    # poisoned: continuation routes through the planted atom via the usable branch
    steps = u["canonical_proof_steps"]
    prefix = steps[:3]
    planted = "%s is a %s." % (E, z)
    cont = ("%s is a %s. %s is a %s. Every %s is %s. %s" % (
        E, bridge, E, chain[-1], chain[-1], u["goal_adjective"], u["target"]))
    v = validate_continuation(u["question"], prefix, planted, cont, u["target"], E)
    check("usable-branch reuse classes poisoned (M4 derivational)", v["class"] == "poisoned")
    # the same continuation in the INERT world cannot be poisoned (no z rules)
    i = by["cat_f002_d1_inert"]
    v2 = validate_continuation(i["question"], i["canonical_proof_steps"][:3], planted, cont,
                               i["target"], E)
    check("inert world: same reuse is NOT derivational (class=%s)" % v2["class"],
          v2["class"] != "poisoned")
    # doubt flag wiring
    v3 = validate_continuation(u["question"], prefix, planted,
                               "Wait, that contradicts the rules. " + " ".join(steps[3:]),
                               u["target"], E)
    check("doubt lexicon fires", bool(v3["acknowledged"]))


def test_anchor_audits():
    _, worlds, _, _, _ = build()
    by = wmap(worlds)
    w = by["attr_f001_d1_A"]
    steps = w["canonical_proof_steps"]
    pts = injection_points(steps, w["entity"])
    check("canonical proof has 3 injection sites", pts is not None)
    si = pts["mid"]
    para = runner.make_paraphrase(steps, si, idx=0)
    t = audit_truth_status(w["question"], para, w["entity"], steps[:si])
    check("benign paraphrase audits true", t["truth_status"] == "true")
    ti = "%s is a %s." % (w["entity"], w["t0"])
    t2 = audit_truth_status(w["question"], ti, w["entity"], steps[:si])
    check("true interruption audits true", t2["truth_status"] == "true")
    check("t0 is off the canonical gold path", w["t0"] not in " ".join(steps))


class _FakeBackend:
    def token_count(self, text):
        return len((text or "").split())


def _mk_args(tmp, cells, positions=("mid",)):
    return argparse.Namespace(
        out_dir=tmp, model="fake", model_revision="rev0", seed=0,
        cells=tuple(cells), positions=tuple(positions), cap=35, backend="vllm",
        max_new_tokens=192, batch_size=8, overwrite=False,
        world_config="pilot", confirm_timestamped_plan2=False,
    )


def test_runner_injection_audit_fail_closed():
    _, worlds, _, _, _ = build()
    by = wmap(worlds)
    w = by["attr_f000_d3_A"]
    steps = list(w["canonical_proof_steps"])
    pts = injection_points(steps, w["entity"])
    tmp = tempfile.mkdtemp(prefix="expd_test_")
    try:
        paths = runner.out_paths(tmp)
        args = _mk_args(tmp, ["aff_false_attr_d3"])
        good = {"world_id": w["world_id"], "family_id": w["family_id"], "family_idx": 0,
                "kind": "attr", "d": 3, "version": "A", "entity": w["entity"],
                "target": w["target"], "eligible": True, "gold_steps": steps,
                "injection_points": pts, "partner_world_id": w["partner_world_id"]}
        planned = runner.plan_injections([w], [good], args, paths, _FakeBackend(), "rev0")
        check("clean gold trace plans the injection", len(planned) == 1)
        check("manifest records measured d == designed",
              planned[0]["measured_d_from_prefix"] == 3)
        # gold trace that WALKED THE SPUR: measured d from prefix shrinks -> reject
        spur1 = w["question"].split("Every %s is a " % w["spur_used"][0])[1].split(".")[0]
        bad_steps = steps[:2] + ["%s is a %s." % (w["entity"], spur1)] + steps[2:]
        bad_pts = injection_points(bad_steps, w["entity"])
        bad = dict(good)
        bad["gold_steps"] = bad_steps
        bad["injection_points"] = bad_pts
        planned2 = runner.plan_injections([w], [bad], args, paths, _FakeBackend(), "rev0")
        check("spur-walking gold trace is rejected fail-closed", len(planned2) == 0)
        audits = runner.read_jsonl(paths["audit"])
        check("rejection logged with measured_d_mismatch",
              any(a.get("reason") == "measured_d_mismatch" for a in audits))
        # ineligible world is skipped and logged
        inel = dict(good)
        inel["eligible"] = False
        inel["reason"] = "gold_not_solved"
        planned3 = runner.plan_injections([w], [inel], args, paths, _FakeBackend(), "rev0")
        check("ineligible gold is skipped", len(planned3) == 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pilot_cells_exist_in_pilot_config():
    fams, worlds = genw.generate_worlds(0, genw.PILOT_CONFIG)
    names = {c["cell"] for w in worlds for c in w["cells"]}
    for cell in runner.PILOT_CELLS:
        if cell in runner.ANCHOR_CELLS:
            continue
        check("pilot cell %s constructible from pilot worlds" % cell, cell in names)
    check("pilot world count == 200", len(worlds) == 200)
    n_anchor_hosts = sum(1 for w in worlds if w.get("anchors_supported"))
    check("anchor host worlds available (>=35)", n_anchor_hosts >= 35)


def main():
    tests = [
        ("determinism", test_determinism),
        ("good worlds pass audits", test_good_worlds_pass),
        ("measured distance == designed d", test_distances_measured_equals_designed),
        ("MAJOR-1 attr-inf refusal", test_attr_inf_refused),
        ("MAJOR-1 manifest cell names", test_no_attr_inf_cell_names),
        ("mirrored one-token diff + tamper", test_mirror_one_token_and_tamper),
        ("planted-token frequency + tamper", test_planted_freq_and_tamper),
        ("MAJOR-2 usable branch + tampers", test_major2_usable_branch),
        ("constancy across d", test_constancy_across_d),
        ("designed truth statuses", test_truth_statuses),
        ("validator wiring (poisoned/inert/doubt)", test_validator_wiring),
        ("anchor audits", test_anchor_audits),
        ("runner fail-closed injection audit", test_runner_injection_audit_fail_closed),
        ("pilot cells constructible", test_pilot_cells_exist_in_pilot_config),
    ]
    for name, fn in tests:
        print("[expd_tests] %s" % name)
        fn()
    print("EXPD tests passed (%d groups)" % len(tests))


if __name__ == "__main__":
    main()
