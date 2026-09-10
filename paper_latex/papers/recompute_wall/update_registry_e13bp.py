#!/usr/bin/env python3
"""Add k1_tentative + k1_probe numbers to the e13 registry blocks.

Sources: e13/results/{E13_haiku,E13_open_llama8b,E13_open_qwen7b}/
summary_tables.json cells.k1_deference.all.{k1_tentative,k1_probe}.
All-or-nothing, .bak_e13bp backup, refuses to overwrite existing keys.
"""
import json
import os
import shutil
import sys

PAPER = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall"
EXP = ("/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/"
       "improvement_plan/iclr_exec/recompute_wall")
REG = os.path.join(PAPER, "results", "verified_numbers.json")
SRC = {"haiku": "E13_haiku", "llama8b": "E13_open_llama8b", "qwen7b": "E13_open_qwen7b"}

reg = json.load(open(REG))
for mk in SRC:
    if "k1_tentative" in reg["e13"].get(mk, {}):
        sys.exit("k1_tentative already present for %s; refusing" % mk)

for mk, dname in SRC.items():
    s = json.load(open(os.path.join(EXP, "e13", "results", dname,
                                    "summary_tables.json")))
    cell = s["cells"]["k1_deference"]["all"]
    t = cell["k1_tentative"]
    p = cell["k1_probe"]
    reg["e13"].setdefault(mk, {})
    reg["e13"][mk]["k1_tentative"] = {
        "rate": t["absorbed"]["rate"], "cboot": t["absorbed"]["cluster_boot95"],
        "n": t["n"]}
    reg["e13"][mk]["k1_probe"] = {
        "answer_anywhere": p["probe_answer_anywhere_rate"],
        "planted_echo": p["probe_planted_echo_rate"], "n": p["n"]}
    print(mk, "tentative", reg["e13"][mk]["k1_tentative"]["rate"],
          "probe_anywhere", reg["e13"][mk]["k1_probe"]["answer_anywhere"],
          "echo", reg["e13"][mk]["k1_probe"]["planted_echo"])

reg["_sources"]["e13_tentative_probe"] = (
    "e13/results/{E13_haiku,E13_open_llama8b,E13_open_qwen7b}/summary_tables.json "
    "cells.k1_deference.all.{k1_tentative.absorbed, k1_probe.probe_answer_anywhere_rate/"
    "probe_planted_echo_rate} — tentative '(unverified)' tag + direct-probe cells; "
    "probe reported as answer-anywhere (first-integer metric is a restating artifact, "
    "disclosed in appendix); cited in Sec 5 and the matched-controls appendix paragraph")

shutil.copy2(REG, REG + ".bak_e13bp")
tmp = REG + ".tmp_e13bp"
with open(tmp, "w") as f:
    json.dump(reg, f, indent=1, sort_keys=True)
    f.write("\n")
json.load(open(tmp))
os.replace(tmp, REG)
print("registry updated; backup at .bak_e13bp")
