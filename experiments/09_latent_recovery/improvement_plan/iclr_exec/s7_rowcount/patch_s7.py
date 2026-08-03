#!/usr/bin/env python3
"""S7 patcher: produce baseline + fixed variants of extract_natural_errors.py.

- Reads the LOCKED original (read-only) from natural_errors/extract_natural_errors.py.
- Writes two runnable copies into my slug dir with OUT_DIR redirected so the locked
  outputs under improvement_plan/natural_errors/ are NEVER touched.
- 'fixed' additionally applies the one-line discourse-marker fix at the is_fail test:
  the stale marker-blind stored flag r["solved"] -> the validator's marker-aware
  v["final_ok"] (validator.py:153, = norm(strip_marker(final_sentence))==norm(target)).
"""
import os

SLUG = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/s7_rowcount"
ORIG = "/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/natural_errors/extract_natural_errors.py"

OUT_LINE = 'OUT_DIR = os.path.join(EXP, "improvement_plan", "natural_errors")'
FAIL_ORIG = '        is_fail = (not r["solved"]) or v["class"] != "valid_rederivation"'
FAIL_FIXED = '        is_fail = (not v["final_ok"]) or v["class"] != "valid_rederivation"  # S7 fix: marker-aware'

src = open(ORIG).read()
assert OUT_LINE in src, "OUT_DIR anchor not found"
assert FAIL_ORIG in src, "is_fail anchor not found"

# keep a verbatim copy for provenance
open(os.path.join(SLUG, "extract_natural_errors_ORIG.py"), "w").write(src)

# baseline: only redirect OUT_DIR
base = src.replace(OUT_LINE, f'OUT_DIR = os.path.join("{SLUG}", "out_baseline")')
open(os.path.join(SLUG, "extract_baseline.py"), "w").write(base)

# fixed: redirect OUT_DIR + one-line fix
fixed = src.replace(OUT_LINE, f'OUT_DIR = os.path.join("{SLUG}", "out_fixed")')
assert fixed.count(FAIL_ORIG) == 1
fixed = fixed.replace(FAIL_ORIG, FAIL_FIXED)
open(os.path.join(SLUG, "extract_fixed.py"), "w").write(fixed)

for d in ("out_baseline", "out_fixed"):
    os.makedirs(os.path.join(SLUG, d), exist_ok=True)
print("patched OK; baseline + fixed written to", SLUG)
print("FAIL_ORIG occurrences in fixed:", fixed.count(FAIL_ORIG), "| FAIL_FIXED:", fixed.count(FAIL_FIXED))
