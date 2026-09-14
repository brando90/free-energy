#!/usr/bin/env python3
"""Item 44/B10 footer: add fig4.inset_lines.note_caption (read by make_figs.py). Backup .bak_sep14_fig4."""
import json, os, shutil, sys
REG = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall/results/verified_numbers.json"
reg = json.load(open(REG)); ins = reg["fig4"]["inset_lines"]
if "note_caption" in ins: sys.exit("note_caption exists; refusing")
ins["note_caption"] = "'full' prints the operands 32 + 58 beside the planted 100; the sum 90 is not written, and 100 is still copied."
shutil.copy2(REG, REG + ".bak_sep14_fig4"); tmp = REG + ".tmp_fig4"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG); print("note_caption added")
