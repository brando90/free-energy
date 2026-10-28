#!/usr/bin/env python3
"""Add yee_prefix_endings block: how Yee et al.'s released GPT-4 GSM8K perturbed prefixes end
(random perturbation, annotated CSVs). Computed directly from the audit clone. Backup .bak_sep14_yee."""
import csv, re, json, os, shutil, sys
csv.field_size_limit(10**9)
R = "/lfs/skampere2/0/eobbad/scratch/yee_audit/repo/results/gsm8k/"
REG = "/lfs/skampere2/0/eobbad/free-energy/paper_latex/papers/recompute_wall/results/verified_numbers.json"
reg = json.load(open(REG))
if "yee_prefix_endings" in reg: sys.exit("block exists; refusing")
blk = {}
for name in ("calc", "copy", "propcalc"):
    rows = [r for r in csv.DictReader(open(R + f"test_adjusted_position-{name}_perturbation-random_annotated.csv")) if r.get("Model Name", "").startswith("gpt-4")]
    txt = [r["Question"] or "" for r in rows]; n = len(txt)
    blk[name] = {"n_rows_gpt4": n,
        "ends_with_number": sum(bool(re.search(r"\d\s*$", t.rstrip())) for t in txt),
        "ends_with_equals_number": sum(bool(re.search(r"=\s*\$?-?\d[\d,\.]*\s*$", t.rstrip())) for t in txt),
        "ends_with_sentence_terminator": sum(bool(re.search(r"[.!?]\s*$", t.rstrip())) for t in txt),
        "contains_calculator_annotation": sum("<<" in t for t in txt),
        "ends_inside_unclosed_annotation": sum(t.count("<<") > t.count(">>") for t in txt)}
    blk[name]["frac_equals_number"] = round(blk[name]["ends_with_equals_number"] / n, 4)
blk["_prose"] = {"calc_frac_mid_statement": 1.0, "calc_frac_after_equals": blk["calc"]["frac_equals_number"],
    "calc_frac_with_annotation": round(blk["calc"]["contains_calculator_annotation"] / blk["calc"]["n_rows_gpt4"], 4),
    "note": "rows = GPT-4 entries (all prompt styles) in the annotated random-perturbation CSVs; 'Question' column holds the perturbed prefix as presented. Their prefix always ends right after the corrupted numeral (never at a sentence terminator); for calculation errors 83% end immediately after an equals sign. Only ~28% of calc items carry <<>> annotations at all, so the July 31 'always inside an unclosed annotation' reading was wrong in its specifics; 'always mid-statement' is correct."}
reg["yee_prefix_endings"] = blk
reg["_sources"]["yee_prefix_endings"] = "scratch/yee_audit/repo/results/gsm8k/test_adjusted_position-{calc,copy,propcalc}_perturbation-random_annotated.csv (Yee et al. release), GPT-4 rows, 'Question' column; computed by update_registry_yee_prefix.py 2026-09-14; cited in Sec 8 (boundary sentence) and E17 registration"
shutil.copy2(REG, REG + ".bak_sep14_yee"); tmp = REG + ".tmp_yee"
with open(tmp, "w") as f: json.dump(reg, f, indent=1, sort_keys=True); f.write("\n")
json.load(open(tmp)); os.replace(tmp, REG); print(json.dumps(blk, indent=1))
