#!/usr/bin/env python3
"""Add e17_boundary block from iclr_exec/recompute_wall/e17_boundary/results.json. Backup .bak_sep14_e17."""
import json, os, shutil, sys
REG = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall/results/verified_numbers.json"
SRC = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e17_boundary/results.json"
reg = json.load(open(REG)); r = json.load(open(SRC))
if "e17_boundary" in reg: sys.exit("exists; refusing")
led = json.load(open(os.path.dirname(SRC) + "/usage_ledger.json"))
blk = {k: r[k] for k in ("ORIG", "RENUM", "POOLED")}; blk["n_items"] = r["n_items"]; blk["usd"] = round(led["cost_usd"], 3); blk["model"] = "gpt-4o"
o = r["ORIG"]; pl = r["POOLED"]
blk["_prose"] = {"orig_recovery_numeral_pct": round(100 * o["corr_numeral"]["recovered_rate"], 1), "orig_recovery_statement_pct": round(100 * o["corr_statement"]["recovered_rate"], 1),
    "orig_paired_diff_pts": round(100 * o["paired_corrupted"]["recovery_diff_numeral_minus_statement"], 1), "orig_mcnemar_p": o["paired_corrupted"]["mcnemar_exact_p"],
    "renum_recovery_numeral_pct": round(100 * r["RENUM"]["corr_numeral"]["recovered_rate"], 1), "renum_recovery_statement_pct": round(100 * r["RENUM"]["corr_statement"]["recovered_rate"], 1),
    "pooled_paired_diff_pts": round(100 * pl["paired_corrupted"]["recovery_diff_numeral_minus_statement"], 1), "pooled_mcnemar_p": pl["paired_corrupted"]["mcnemar_exact_p"], "pooled_n": pl["paired_corrupted"]["n_pairs"],
    "pooled_flip_table": {k: pl["paired_corrupted"][k] for k in ("both_recovered", "numeral_only", "statement_only", "neither")},
    "clean_numeral_pct": round(100 * pl["clean_numeral"]["recovered_rate"], 1), "clean_statement_pct": round(100 * pl["clean_statement"]["recovered_rate"], 1),
    "verdict": "PARTIAL per registered rule (5-15 pt band): boundary lowers recovery ~9 pts in both numberings, pooled p=0.077; statement completion alone does not produce a wall on GSM8K",
    "registration": "E17_STATEMENT_BOUNDARY.md commits 201a0dd (spec) + ec7717d (amendment 1, pre-run)"}
reg["e17_boundary"] = blk; reg["_sources"]["e17_boundary"] = "e17_boundary/results.json (scorer e17_score.py over responses.jsonl; items from e17_build.py); paired exact McNemar; Wilson over items; cited in Sec 8 and app:renumber"
shutil.copy2(REG, REG + ".bak_sep14_e17"); tmp = REG + ".tmp_e17"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG); print(json.dumps(blk["_prose"], indent=1))
