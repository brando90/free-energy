#!/usr/bin/env python3
"""Correct e16_probe_steer: the steer_stub battery counted its own baseline row as a condition
(25 -> 24; total 151 -> 150). Backup .bak_sep4_e16b, atomic replace."""
import json, os, shutil
REG = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall/results/verified_numbers.json"
reg = json.load(open(REG)); b = reg["e16_probe_steer"]["batteries"]["steer_stub_generation_position"]
assert b["n_conditions"] == 25 and b["n_control_ok"] == 25
b["n_conditions"] = 24; b["n_control_ok"] = 24; b["note"] = "baseline_a0.0_fNone row excluded from the condition count (it is the baseline)"
reg["e16_probe_steer"]["_prose"]["n_conditions_total"] = sum(v["n_conditions"] for v in reg["e16_probe_steer"]["batteries"].values())
assert reg["e16_probe_steer"]["_prose"]["n_conditions_total"] == 150
shutil.copy2(REG, REG + ".bak_sep4_e16b"); tmp = REG + ".tmp_fix"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG); print("fixed: n_conditions_total =", reg["e16_probe_steer"]["_prose"]["n_conditions_total"])
