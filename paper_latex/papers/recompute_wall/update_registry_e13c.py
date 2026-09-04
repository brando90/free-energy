#!/usr/bin/env python3
"""Add the e13_bridge block (E13c attribution arms) to verified_numbers.json.

Source artifact: e13/results/E13_bridge/summary_attribution.json (run 2026-07-29).
All-or-nothing: backs up to .bak_e13c, validates the new JSON, replaces atomically.
Refuses to run if e13_bridge already exists. Reads the results dir, never writes it.
"""
import json
import os
import shutil
import sys

PAPER = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall"
SRC = ("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/"
       "improvement_plan/iclr_exec/recompute_wall/e13/results/E13_bridge/"
       "summary_attribution.json")
REG = os.path.join(PAPER, "results", "verified_numbers.json")
BAK = REG + ".bak_e13c"

ARMS = ["baseline", "self", "other_model", "human"]
CELLS = ["onehop_kc1", "adjacent_contradiction", "opfree_kr1"]

src = json.load(open(SRC))
reg = json.load(open(REG))

if "e13_bridge" in reg:
    sys.exit("e13_bridge already present in registry; refusing to overwrite")

block = {}
computed_rates = []
readable_rates = []
for model, cells in src["models"].items():
    m = {}
    for arm in ARMS:
        for cell in CELLS:
            e = cells[f"{arm}|{cell}"]
            # one greedy continuation per program: Wilson is the right CI (fig2 policy).
            # sonnet-5 onehop_kc1 is n=59 in all four arms (one program excluded
            # at gating, arm-uniform); everything else is n=60.
            assert e["n"] == e["program_n"] and e["n"] in (59, 60), \
                (model, arm, cell, e["n"], e["program_n"])
            m[f"{arm}|{cell}"] = {
                "rate": e["absorbed"]["rate"],
                "wilson95": e["absorbed"]["wilson95"],
                "n": e["n"],
            }
            if cell == "onehop_kc1":
                computed_rates.append(e["absorbed"]["rate"])
            elif cell == "adjacent_contradiction":
                readable_rates.append(e["absorbed"]["rate"])
    block[model] = m

opus = block["claude-opus-4-8"]
g41 = block["gpt-4.1"]
block["_prose"] = {
    "computed_min_across_arms_models": min(computed_rates),
    "computed_max_across_arms_models": max(computed_rates),
    "readable_max_across_arms_models": max(readable_rates),
    "opus_copy_by_arm": {a: opus[f"{a}|opfree_kr1"]["rate"] for a in ARMS},
    "gpt41_copy_min": min(g41[f"{a}|opfree_kr1"]["rate"] for a in ARMS),
    "gpt41_copy_max": max(g41[f"{a}|opfree_kr1"]["rate"] for a in ARMS),
    "n_per_cell_min": 59,
    "n_per_cell_max": 60,
    "n_note": "sonnet-5 onehop_kc1 n=59 in all four arms (arm-uniform program exclusion); all other cells n=60",
    "models": sorted(src["models"].keys()),
    "source_created_at": src["created_at"],
}

reg["e13_bridge"] = block
reg["_sources"]["e13_bridge"] = (
    "e13/results/E13_bridge/summary_attribution.json models.<m>.'<arm>|<cell>'.absorbed "
    "— E13c attribution arms (baseline/self/other_model/human framings over byte-identical "
    "perturbed traces), bridge presentation, greedy, one continuation per program "
    "(n=60=program_n) so Wilson95 CIs; cited in Sec 5, Sec 9, and app:attribution"
)

shutil.copy2(REG, BAK)
tmp = REG + ".tmp_e13c"
with open(tmp, "w") as f:
    json.dump(reg, f, indent=1, sort_keys=True)
    f.write("\n")
json.load(open(tmp))  # round-trip validation before replacing
os.replace(tmp, REG)

print("e13_bridge added for models:", block["_prose"]["models"])
print("prose anchors:", json.dumps(block["_prose"], indent=1, sort_keys=True))
print("backup at:", BAK)
