#!/usr/bin/env python3
"""Produce baseline + fixed copies of extract_natural_errors.py in the S7 slug dir.

Baseline: identical analysis, only OUT_DIR redirected (do NOT clobber the locked
          natural_errors/ outputs).
Fixed   : OUT_DIR redirected + the ONE-LINE discourse-marker fix:
          the failure test stops trusting the stored `solved` flag (computed by
          common.solved()/expa.solved(), which compare the RAW final sentence and
          never strip leading discourse markers) and instead uses the marker-aware
          validator field validate_continuation(...)["final_ok"], which is exactly
          `norm(strip_marker(final_sentence)) == norm(target)`.
"""
import os

SLUG = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/s7_solved_fix_rowcount"
ORIG = os.path.join(SLUG, "extract_natural_errors_ORIG.py")

with open(ORIG) as fh:
    src = fh.read()

OUT_LINE = 'OUT_DIR = os.path.join(EXP, "improvement_plan", "natural_errors")'
assert src.count(OUT_LINE) == 1, "OUT_DIR line not unique"

FIX_OLD = '        is_fail = (not r["solved"]) or v["class"] != "valid_rederivation"'
FIX_NEW = ('        # S7 FIX: stored `solved` came from common.solved()/expa.solved(),\n'
           '        # which compare the RAW final sentence and never strip discourse markers.\n'
           '        # v["final_ok"] is norm(strip_marker(final_sent))==norm(target): marker-aware.\n'
           '        is_fail = (not v["final_ok"]) or v["class"] != "valid_rederivation"')
assert src.count(FIX_OLD) == 1, "is_fail line not unique"

# baseline: only redirect OUT_DIR
baseline = src.replace(OUT_LINE, 'OUT_DIR = os.path.join(EXP, "improvement_plan", "iclr_exec", "s7_solved_fix_rowcount", "out_baseline")')
with open(os.path.join(SLUG, "extract_baseline.py"), "w") as fh:
    fh.write(baseline)

# fixed: redirect OUT_DIR + one-line fix
fixed = src.replace(OUT_LINE, 'OUT_DIR = os.path.join(EXP, "improvement_plan", "iclr_exec", "s7_solved_fix_rowcount", "out_fixed")')
fixed = fixed.replace(FIX_OLD, FIX_NEW)
with open(os.path.join(SLUG, "extract_fixed.py"), "w") as fh:
    fh.write(fixed)

print("wrote extract_baseline.py and extract_fixed.py")
print("baseline OUT_DIR ->", "out_baseline" in baseline)
print("fixed OUT_DIR ->", "out_fixed" in fixed, "| fixed is_fail ->", 'not v["final_ok"]' in fixed)
