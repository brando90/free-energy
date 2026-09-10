"""E13b-inc: A4 inconsistency-framed instruction arm (haiku), copy-derived
from e12_api.py WITHOUT modifying it.

Arm A4 (bytes pinned in e13_gen.E13B_INC_ARM_A4):
    "Before continuing, check this trace for inconsistencies."

Mechanics are EXACTLY the E12 ladder's (A1-A3): the instruction is prepended
to the USER message (E9 VERIF_INSTR position), assistant prefill unchanged,
and it is applied at BOTH gold generation and continuation (golds regenerated
per arm with the instruction present -- e12_api's pinned gold_policy). This
file registers "A4" in e12_api.ARMS at RUNTIME (module state only; the e12
file on disk is untouched) and delegates the live run to e12_api.main(), so
run mechanics, run-id namespace, manifest fields, DV, and the recompute
signature stay byte-identical to arms A0-A3. Cells: adjacent_contradiction
(k0), onehop_kc1 (k1), deep_kc5 (k5) on the SHARED E12 programs file.

A run manifest pinning the instruction bytes + sha256 + gold policy is written
by e12_api BEFORE any generation (out_base/A4/run_manifest.json); this wrapper
additionally writes out_base/A4_pin.json with the same pins plus provenance.

--dry assembles the A4 gold + continuation prompts (interpreter-true stand-in
golds, labelled), writes example prompts + a per-arm spend projection, and
makes ZERO API calls.

Usage:
  python e13_inc.py --out-base results/E13_inc \
      --programs ../e12_fixes/programs.jsonl --dry
  source <keys env> && python e13_inc.py --out-base results/E13_inc \
      --programs ../e12_fixes/programs.jsonl --n-target 60 --R 3 --usd-cap 15
"""
import argparse
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e13_lib as L                        # noqa: E402
import e13_gen                             # noqa: E402
import e11_run as e11                      # noqa: E402
import e12_api                             # noqa: E402
import trace_format as tf                  # noqa: E402
import inject as inj_mod                   # noqa: E402

ARM = "A4"
INSTR = e13_gen.E13B_INC_ARM_A4
PRICE_IN, PRICE_OUT = 1.00, 5.00           # claude-haiku-4-5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-base", required=True)
    ap.add_argument("--programs", required=True,
                    help="SHARED e12 programs.jsonl (identical worlds across "
                         "arms A0-A4; intersection analysis reuse)")
    ap.add_argument("--n-target", type=int, default=60)
    ap.add_argument("--world-cap", type=int, default=110)
    ap.add_argument("--R", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--usd-cap", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    assert INSTR == "Before continuing, check this trace for inconsistencies.", \
        "A4 bytes drifted from the e13_gen pin"
    os.makedirs(a.out_base, exist_ok=True)
    pin = {
        "experiment": "E13B_INC", "arm": ARM, "model": e12_api.MODEL_ID,
        "created_at": e11.now_iso(),
        "instruction_verbatim": INSTR,
        "instruction_sha256_full": L.sha256_text(INSTR),
        "instruction_sha16_e12style": e11.sha([INSTR]),
        "instruction_source": "e13_gen.E13B_INC_ARM_A4 (pinned Build 1/2)",
        "mechanics": ("e12_api A1-A3 mechanics byte-identical: instruction "
                      "prepended to user message (E9 VERIF_INSTR position); "
                      "prefill unchanged; applied at gold AND continuation"),
        "gold_policy": ("golds REGENERATED for arm A4 with the A4 instruction "
                        "present in the same position (e12_api pinned "
                        "gold_policy; per-arm gate disclosed)"),
        "programs_path": os.path.abspath(a.programs),
        "cells": {"k0_bare": "adjacent_contradiction",
                  "k1_bare": "onehop_kc1", "k5_bare": "deep_kc5"},
        "n_target": a.n_target, "world_cap": a.world_cap, "R": a.R,
        "temp": a.temp, "seed": a.seed, "dry": bool(a.dry),
    }
    e11.write_json(os.path.join(a.out_base, "A4_pin.json"), pin)

    # register the arm in module state only (the e12_api.py FILE is untouched)
    e12_api.ARMS[ARM] = INSTR

    if a.dry:
        return dry_run(a, pin)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY not set. Source the keys env file "
                         "first; this script never prompts for a credential.")
    argv = ["--arm", ARM, "--out-base", a.out_base, "--programs", a.programs,
            "--n-target", str(a.n_target), "--world-cap", str(a.world_cap),
            "--R", str(a.R), "--temp", str(a.temp),
            "--usd-cap", str(a.usd_cap), "--seed", str(a.seed)]
    old = sys.argv
    sys.argv = [os.path.join(os.path.dirname(os.path.abspath(
        e12_api.__file__)), "e12_api.py")] + argv
    try:
        e12_api.main()
    finally:
        sys.argv = old


def dry_run(a, pin):
    out_dir = os.path.join(a.out_base, ARM)
    os.makedirs(out_dir, exist_ok=True)
    programs = [p for p in e11.read_jsonl(a.programs)
                if p["k"] in e12_api.TARGET_KS]
    capped, per_k = [], defaultdict(int)
    for p in sorted(programs, key=lambda x: (x["k"], x["program_id"])):
        if per_k[p["k"]] < a.world_cap:
            capped.append(p)
            per_k[p["k"]] += 1
    programs = capped
    examples = defaultdict(list)
    jobs = defaultdict(list)
    n_cont_worlds = defaultdict(int)
    for p in programs:
        um = e12_api._apply_instruction(
            tf.make_user_message(tf.make_listing_text(p["stmt_texts"])), INSTR)
        jobs["A4_gold"].append((L.est_tokens(um), 1))
        if len(examples["A4_gold"]) < 2:
            examples["A4_gold"].append({
                "program_id": p["program_id"], "k": p["k"],
                "api_messages": [
                    {"role": "user", "content": um},
                    {"role": "assistant (prefill)", "content": tf.GOLD_PREFILL}]})
        gfull = L.mock_gold_full(p)          # DRY stand-in; live uses A4 golds
        for fam in e11.families_of(p):
            kk, opac = e11.cell_of(p, fam)
            if (kk, opac) not in e12_api.TARGET_CELLS:
                continue
            if n_cont_worlds[fam] >= a.n_target:
                continue
            try:
                b = e11.build_injection_e11(p, fam, gfull)
            except inj_mod.InjectError:
                continue
            n_cont_worlds[fam] += 1
            arm_key = "A4_cont_%s" % e12_api.CELL_ALIAS[(kk, opac)]
            jobs[arm_key].append((L.est_tokens(um) +
                                  L.est_tokens(b["prefix_text"]), 1 + a.R))
            if len(examples[arm_key]) < 2:
                examples[arm_key].append({
                    "program_id": p["program_id"], "cell":
                        e12_api.CELL_ALIAS[(kk, opac)], "family": fam,
                    "api_messages": [
                        {"role": "user", "content": um},
                        {"role": "assistant (prefill, trailing ws stripped "
                                 "for the request)",
                         "content": b["prefix_text"]}]})
    L.dump_examples(os.path.join(out_dir, "example_prompts.json"),
                    dict(examples))
    proj = L.project_spend("e13_inc A4 (haiku)", jobs, PRICE_IN, PRICE_OUT,
                           320)
    e11.write_json(os.path.join(out_dir, "spend_projection.json"), proj)
    e11.write_json(os.path.join(out_dir, "queue_status.json"),
                   {"dry": True, "n_programs": len(programs),
                    "cont_worlds_per_cell": dict(n_cont_worlds),
                    "updated_at": e11.now_iso()})
    print(json.dumps({"arm": ARM, "n_programs": len(programs),
                      "cont_worlds_per_cell": dict(n_cont_worlds),
                      "instruction_sha256": pin["instruction_sha256_full"]},
                     indent=2))
    print("[e13_inc:dry] DONE (zero API calls made)")


if __name__ == "__main__":
    main()
