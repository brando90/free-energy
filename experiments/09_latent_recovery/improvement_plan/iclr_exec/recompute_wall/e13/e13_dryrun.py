"""E13 CPU dry-run: generate + audit N worlds/cell, write world files and the
fail-closed audit report. NO model, NO GPU, NO API.

Usage: python e13_dryrun.py [N]        (default N=200)

Outputs (all under RW/e13/):
  worlds/line_k0.jsonl, worlds/grid_kr1.jsonl, worlds/k1_deference.jsonl
  dryrun/DRYRUN_REPORT.md
  dryrun/dryrun_summary.json

FAIL CLOSED: a cell that cannot reach N audited worlds within the attempt
budget, or that fails the determinism re-build, is marked FAILED in the
report; nothing is patched around.
"""
import hashlib
import json
import os
import statistics
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import e13_gen as eg                              # noqa: E402
import trace_format as tf                         # noqa: E402
import inject as inj_mod                          # noqa: E402
from interp import parse_program, execute, value_at   # noqa: E402

WORLDS_DIR = os.path.join(HERE, "worlds")
DRY_DIR = os.path.join(HERE, "dryrun")

# E13c: existing regime2 worlds (e10bridge PROG_SRC); fingerprint only.
EXP_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
REGIME2_PROGRAMS = os.path.join(EXP_ROOT, "results",
                                "EXPH2_FRONTIER_FOLLOWUPS", "regime2",
                                "programs.jsonl")


def _splice(true_line, planted):
    return inj_mod.replace_final_value(true_line, planted)


def _true_line(prog, tw, line_no, var):
    return tf.format_trace_line(line_no, prog["stmt_texts"][line_no - 1], var,
                                value_at(tw, line_no, var))


def example_lines(cell, prog):
    """One verbatim planted line (with minimal conflict context) per family."""
    stmts = parse_program(prog["stmt_texts"])
    tw = execute(stmts)
    j = prog["site_line"]
    out = {}
    fams = prog["families"]
    if cell == "line_k0":
        for fam in ("line_k0", "line_k0_swap"):
            out[fam] = "line %d: %s" % (j, fams[fam]["site_body"])
        a = fams["anchor_k0"]
        out["anchor_k0"] = "note: %s = %d" % (a["planted_var"],
                                              a["planted_value"])
    elif cell == "grid_kr1":
        u = prog["root_var"]
        d = fams["adjacent_contradiction"]["planted_value"]
        vtrue = prog["site_true_value"]
        def_true = _true_line(prog, tw, j - 1, u)
        def_false = _splice(def_true, d)
        site_true = _true_line(prog, tw, j, prog["site_var"])
        out["adjacent_contradiction"] = "%s  ||  note: %s = %d" % (def_true, u, d)
        out["opfree_kr1"] = "%s  ||  %s" % (def_true, _splice(site_true, d))
        out["grid_line_false_note_true"] = "%s  ||  note: %s = %d" % (
            def_false, u, vtrue)
        out["grid_note_false_before"] = "note: %s = %d  ||  %s" % (u, d, def_true)
        out["grid_note_true_before_line_false"] = "note: %s = %d  ||  %s" % (
            u, vtrue, def_false)
        out["grid_line_false_copy_true"] = "%s  ||  %s" % (def_false, site_true)
        for fam in eg.DOSE_FAMS:
            out[fam] = "%s  ||  note: %s = %d" % (def_true, u,
                                                  fams[fam]["planted_value"])
    else:                                          # k1_deference
        for fam in ("depth_k1_bare", "k1_full_sum", "k1_tentative"):
            out[fam] = "line %d: %s" % (j, fams[fam]["site_body"])
        pr = fams["k1_probe"]
        out["k1_probe"] = "[prefix=%s plant] %s  (true answer %d)" % (
            pr["prefix_family"], pr["probe_question"], pr["probe_answer"])
    return out


def determinism_check(builder, seeds):
    for s in seeds:
        try:
            a = json.dumps(builder(s), sort_keys=True)
            b = json.dumps(builder(s), sort_keys=True)
        except Exception as e:                     # seed rejected: rebuild must
            try:                                   # reject identically
                builder(s)
                return False, "seed %d: reject-then-accept nondeterminism" % s
            except Exception as e2:
                if type(e2) != type(e) or str(e2) != str(e):
                    return False, "seed %d: divergent rejects" % s
                continue
        if a != b:
            return False, "seed %d: dict mismatch" % s
    return True, "ok"


def fingerprint_regime2():
    if not os.path.exists(REGIME2_PROGRAMS):
        return {"path": REGIME2_PROGRAMS, "status": "MISSING"}
    h = hashlib.sha256()
    n = 0
    shapes = Counter()
    with open(REGIME2_PROGRAMS, "rb") as fh:
        data = fh.read()
    h.update(data)
    for ln in data.decode().splitlines():
        if ln.strip():
            n += 1
            try:
                shapes[json.loads(ln).get("shape", "?")] += 1
            except ValueError:
                shapes["unparseable"] += 1
    return {"path": REGIME2_PROGRAMS, "status": "ok", "sha256": h.hexdigest(),
            "n_programs": n, "shapes": dict(shapes)}


def run(N=200):
    os.makedirs(WORLDS_DIR, exist_ok=True)
    os.makedirs(DRY_DIR, exist_ok=True)
    summary = {"N_target": N, "cells": {}, "e13c_regime2_fingerprint":
               fingerprint_regime2(),
               "e13b_inc_arm_A4": eg.E13B_INC_ARM_A4,
               "dropped_corners": eg.DROPPED_CORNERS,
               "naturalness_flags": eg.NATURALNESS_FLAGS}
    R = []
    ap = R.append
    ap("# E13 dry-run audit report (CPU only; fail-closed)")
    ap("")
    ap("N target per cell: %d. Worlds: `worlds/*.jsonl` (e11-style program "
       "dicts; see README.md for schema)." % N)
    ap("")

    overall_ok = True
    for cell in ("line_k0", "grid_kr1", "k1_deference"):
        builder, audit_fn, base_seed, fname, fam_inventory = eg.CELLS[cell]
        progs, funnel = eg.generate(builder, audit_fn, N, base_seed)
        det_ok, det_msg = determinism_check(builder,
                                            [base_seed, base_seed + 1,
                                             base_seed + 7])
        wpath = os.path.join(WORLDS_DIR, fname)
        with open(wpath, "w") as fh:
            for p in progs:
                fh.write(json.dumps(p, sort_keys=True) + "\n")
        ws = sorted({p["listing_ws_tokens"] for p in progs})
        ops = sorted({p["total_ops"] for p in progs})
        Ls = sorted({p["L"] for p in progs})
        anc = sorted({p.get("audit_metrics", {}).get("nearest_anc_dist")
                      for p in progs if p.get("audit_metrics")}) \
            if cell != "grid_kr1" else ["within-world (paired families)"]
        mink = sorted({p.get("audit_metrics", {}).get("measured_min_k")
                       for p in progs if p.get("audit_metrics")}) \
            if cell != "grid_kr1" else ["n/a (kr1 substrate, k_c=0)"]
        fam_missing = [f for f in fam_inventory
                       if progs and f not in progs[0]["families"]]
        passed = (len(progs) == N and not funnel["starved"] and det_ok
                  and not fam_missing)
        overall_ok &= passed
        cellrec = {
            "accepted": len(progs), "attempts": funnel["attempts"],
            "accept_rate": round(len(progs) / max(1, funnel["attempts"]), 4),
            "rejects": funnel["rejects"], "starved": funnel["starved"],
            "determinism": det_msg, "world_file": wpath,
            "families": fam_inventory, "family_missing": fam_missing,
            "L_set": Ls, "ops_set": ops, "ws_tokens_set": ws,
            "nearest_anc_dist_set": anc, "measured_min_k_set": mink,
            "status": "PASS" if passed else "FAILED",
        }
        if cell == "grid_kr1" and progs:
            wtok = [p["listing_ws_tokens"] for p in progs]
            cellrec["ws_tokens_stats"] = {
                "mean": round(statistics.mean(wtok), 2),
                "sd": round(statistics.pstdev(wtok), 2),
                "min": min(wtok), "max": max(wtok)}
        summary["cells"][cell] = cellrec

        ap("## Cell set `%s` -- %s" % (cell, cellrec["status"]))
        ap("")
        ap("* accepted %d/%d worlds (attempts %d, accept rate %.3f, "
           "starved=%s); determinism: %s" %
           (len(progs), N, funnel["attempts"], cellrec["accept_rate"],
            funnel["starved"], det_msg))
        ap("* world file: `%s`" % wpath)
        ap("* L=%s ops=%s ws_tokens=%s nearest_anc_dist=%s measured_min_k=%s" %
           (Ls, ops, ws, anc, mink))
        if "ws_tokens_stats" in cellrec:
            ap("* grid ws-token spread (diagnostic; corners are within-world "
               "paired so cross-corner matching is exact): %s"
               % cellrec["ws_tokens_stats"])
        ap("* reject funnel: %s" % (funnel["rejects"] or "{none}"))
        if fam_missing:
            ap("* MISSING FAMILIES: %s" % fam_missing)
        if progs:
            ap("* per-family verbatim planted lines (world 1, `%s`):"
               % progs[0]["program_id"])
            ap("")
            for fam, line in example_lines(cell, progs[0]).items():
                ap("  * `%s`: `%s`" % (fam, line))
        ap("")

    ap("## Distance-audit disclosure (line_k0, no k=0 exemption)")
    ap("")
    ap("nearest-stated-ancestor listing-token distance is PINNED at exactly 6 "
       "for every line_k0 world and audited fail-closed (target in "
       "LINE_K0_TARGETS). k>=1 E11 cells pin 4. The 2-token difference is a "
       "hard floor: the k>=1 ancestor is a bare operand const line "
       "('9: q = 47' = 4 ws-tokens) while the k=0 copy root must be a "
       "COMPUTED line ('9: u = a + b' = 6 ws-tokens) because MAJOR-5 forbids "
       "the site's true value appearing verbatim in the listing. Pinned + "
       "disclosed, not exempted.")
    ap("")
    ap("## E13a-grid corner accounting")
    ap("")
    ap("Existing corners already covered (full 13-model data, same kr1 "
       "construction, carried on every new world for paired reruns):")
    for fam, c in eg.EXISTING_CORNER_CODING.items():
        ap("  * `%s`: format=%s, false_in=%s, false_position=%s" %
           (fam, c["format"], c["false_in"], c["false_position"]))
    ap("")
    ap("New corners generated: %s" % ", ".join("`%s`" % f
                                               for f in sorted(eg.GRID_CORNERS)))
    ap("")
    ap("Dropped corners (fail-closed disclosure):")
    for k, v in eg.DROPPED_CORNERS.items():
        ap("  * %s: %s" % (k, v))
    ap("")
    ap("Naturalness check (structural; CPU dry-run cannot test model parse "
       "sanity -- CONDITIONAL corners must pass a pilot/gold parse gate "
       "before scaling, per spec):")
    for k, v in eg.NATURALNESS_FLAGS.items():
        ap("  * `%s`: %s" % (k, v))
    ap("")
    ap("## E13b-inc (no worlds)")
    ap("")
    ap("A4 instruction string pinned in e13_gen.E13B_INC_ARM_A4: `%s` -- "
       "apply with e12_api mechanics (prepended to user message, golds "
       "regenerated per arm, prefill unchanged)." % eg.E13B_INC_ARM_A4)
    ap("")
    ap("## E13c (no new worlds) -- regime2 world fingerprint")
    ap("")
    ap("`%s`" % json.dumps(summary["e13c_regime2_fingerprint"]))
    ap("")
    ap("## Overall: %s" % ("ALL CELLS PASS" if overall_ok
                           else "AT LEAST ONE CELL FAILED (fail closed)"))

    with open(os.path.join(DRY_DIR, "dryrun_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    with open(os.path.join(DRY_DIR, "DRYRUN_REPORT.md"), "w") as fh:
        fh.write("\n".join(R) + "\n")
    print("\n".join(R))
    print("\n[wrote %s and %s]" % (os.path.join(DRY_DIR, "DRYRUN_REPORT.md"),
                                   os.path.join(DRY_DIR, "dryrun_summary.json")))
    return summary


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
