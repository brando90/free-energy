#!/usr/bin/env python3
"""E10 WIDEN -- CPU pre-arming validation of world prep.

Checks, for every model out-dir, that `prepare` produced a spec-conforming world
substrate BEFORE the GPU is armed:
  1. programs.jsonl exists and has exactly OVERSAMPLE * len(SHAPES) rows.
  2. per-shape accepted counts == OVERSAMPLE (all 4 shapes fully provisioned).
  3. generator_funnels present in run_metadata.json; every reject reason is a KNOWN
     fail-closed audit category (no unexpected/silent rejects), and accepted==OVERSAMPLE.
  4. run_metadata.json records the right model/revision and prompt_sha256 == pilot anchor.
  5. all five models share an identical world substrate (same program_id set) -- the
     model-independence invariant.
Exit 0 iff every check passes for every model; else prints FAIL lines and exits 1.
"""
import collections
import hashlib
import json
import os
import sys

RESULTS = sys.argv[1]
OVERSAMPLE = int(sys.argv[2])
SHAPES = ("kr1", "kr8", "kc1", "kc5")
PILOT_PROMPT_SHA = "a1a134d3c0b753c273365be903dba2cd5c5fc2d33fb443891448d4c384de578b"
KNOWN_REJECTS = {
    "major5_site_value_in_listing", "true_values_not_distinct",
    "no_axis_c_delta", "true_values_not_distinct;major5_site_value_in_listing",
    "major5_site_value_in_listing;true_values_not_distinct",
}
EXPECT = {  # slug -> (repo, revision)
    "qwen1p5b": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
    "qwen7b":   ("Qwen/Qwen2.5-7B-Instruct",   "a09a35458c702b33eeacc393d103063234e8bc28"),
    "olmo7b":   ("allenai/OLMo-2-1124-7B-Instruct", "470b1fba1ae01581f270116362ee4aa1b97f4c84"),
    "llama8b":  ("NousResearch/Meta-Llama-3.1-8B-Instruct", "d10aef7999a2b5ba950ab3974312feeedbfe0b77"),
    "qwen32b":  ("Qwen/Qwen2.5-32B-Instruct",  "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
}

fails = []
prog_sets = {}
for slug, (repo, rev) in EXPECT.items():
    d = os.path.join(RESULTS, "EXPG_PROGTRACE_" + slug)
    pj = os.path.join(d, "programs.jsonl")
    mj = os.path.join(d, "run_metadata.json")
    if not os.path.exists(pj):
        fails.append("%s: programs.jsonl MISSING" % slug); continue
    rows = [json.loads(l) for l in open(pj) if l.strip()]
    by_shape = collections.Counter(r["shape"] for r in rows)
    if len(rows) != OVERSAMPLE * len(SHAPES):
        fails.append("%s: programs.jsonl has %d rows, expected %d"
                     % (slug, len(rows), OVERSAMPLE * len(SHAPES)))
    for s in SHAPES:
        if by_shape.get(s, 0) != OVERSAMPLE:
            fails.append("%s: shape %s has %d programs, expected %d"
                         % (slug, s, by_shape.get(s, 0), OVERSAMPLE))
    prog_sets[slug] = frozenset(r["program_id"] for r in rows)
    if not os.path.exists(mj):
        fails.append("%s: run_metadata.json MISSING" % slug); continue
    meta = json.load(open(mj))
    if meta.get("model") != repo:
        fails.append("%s: metadata model=%r expected %r" % (slug, meta.get("model"), repo))
    if meta.get("model_revision") != rev:
        fails.append("%s: metadata revision=%r expected %r" % (slug, meta.get("model_revision"), rev))
    if meta.get("prompt", {}).get("instruction_plus_exemplars_sha256") != PILOT_PROMPT_SHA:
        fails.append("%s: prompt_sha256 != pilot anchor" % slug)
    fun = meta.get("generator_funnels")
    if not fun:
        fails.append("%s: generator_funnels missing from metadata" % slug); continue
    for s in SHAPES:
        f = fun.get(s, {})
        if f.get("accepted") != OVERSAMPLE:
            fails.append("%s: funnel[%s] accepted=%s expected %d"
                         % (slug, s, f.get("accepted"), OVERSAMPLE))
        bad = set(f.get("rejects", {})) - KNOWN_REJECTS
        if bad:
            fails.append("%s: funnel[%s] UNKNOWN reject reasons %s" % (slug, s, sorted(bad)))

# model-independence invariant: identical program_id set across all prepared models
if len(prog_sets) > 1:
    base_slug = next(iter(prog_sets))
    base = prog_sets[base_slug]
    for slug, ps in prog_sets.items():
        if ps != base:
            fails.append("%s: program_id set differs from %s (worlds not identical)"
                         % (slug, base_slug))
    # a stable fingerprint of the shared substrate, for the report
    fp = hashlib.sha256("|".join(sorted(base)).encode()).hexdigest()[:16]
    print("shared world substrate: %d programs, fingerprint %s" % (len(base), fp))

if fails:
    print("WORLD-PREP VALIDATION: FAIL (%d issues)" % len(fails))
    for f in fails:
        print("  FAIL " + f)
    sys.exit(1)
print("WORLD-PREP VALIDATION: PASS -- %d models x %d shapes x %d programs, audits clean"
      % (len(EXPECT), len(SHAPES), OVERSAMPLE))
