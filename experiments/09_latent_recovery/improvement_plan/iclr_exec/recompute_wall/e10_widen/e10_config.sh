#!/bin/bash
# E10 WIDEN (Track B, open-weights half) -- shared config.
# Registration (timestamped plan): iclr_exec/recompute_wall/E10_WIDEN.md.
# Substrate/machinery REUSED UNCHANGED: improvement_plan/expg/expg_progtrace.py
# (+ gen_programs.py, inject.py, validate.py, interp.py, trace_format.py). This
# config edits NOTHING in expg/; it only drives the locked stages
# prepare -> generate -> validate -> judge -> summarize -> report.
#
# Nothing here is written into $EXP/results/ or any pre-existing improvement_plan
# subdir: all E10 outputs live under this recompute_wall/e10_widen/ tree.

EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
EXPG=$EXP/improvement_plan/expg
E10DIR=$EXP/improvement_plan/iclr_exec/recompute_wall/e10_widen
RESULTS=$E10DIR/results
PY_VLLM=$EXP/improvement_plan/tooling/.venv-vllm/bin/python
PY_MAIN=/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python
EXPG_DRIVER=$EXPG/expg_progtrace.py

# --- EXPG cell set (E10 §1, reused verbatim; all 7 are PILOT_CELLS) ----------
# readable anchor (k=0) + recompute copy/alias (kr1,kr8) + recompute op (kc1,kc5)
# + two TRUE-plant controls.
CELLS="benign_paraphrase,true_interruption,adjacent_contradiction,opfree_kr1,opfree_kr8,onehop_kc1,deep_kc5"

# --- budget knobs ------------------------------------------------------------
# CAP: target n/cell after the gold-solve + fail-closed injection filters (E10 §5.2, n>=150).
# OVERSAMPLE: programs generated per shape before the cohort filter. 240 == 4x the
#   pilot world budget (pilot oversample_per_shape=60), per E10 §3 ("4x for every
#   non-anchor model"); applied uniformly incl. the re-anchor so every model shares
#   an identical world substrate (worlds are model-independent) and each injects its
#   OWN gold rollouts. kc5 is the expected attrition driver (pilot solve 0.45); cells
#   that land <150 are scored/CI-widened/excluded per the honest rule in §5.2/§5.3.
OVERSAMPLE=240
CAP=150

# Decoding: EXPG pilot machinery values (gold 512 / cont 384), the calibrated
# settings that produced the anchor numbers we re-anchor against. NOTE: E10_WIDEN.md
# §3 says "max_new=192, as EXPG"; that conflates the E1 setting -- the EXPG pilot
# used gold=512/cont=384. We follow the pilot machinery for re-anchor comparability
# and FLAG the §3 discrepancy for review (see REPORT / handoff).
MAXNEW_GOLD=512
MAXNEW_CONT=384
BATCH=256
SEED=0

# --confirm-timestamped-plan-progtrace: EXPG's honesty gate for anything beyond the
# pilot scope (cap>35). The timestamped plan is E10_WIDEN.md. --ignore-gate-a: the
# per-model G-A pilot gate is NOT a hard abort for the widen -- grammar is already
# locked (pilot passed G-A on Qwen-7B); every model must produce continuations so the
# honest per-cell / <25%-cohort exclusion rules (E10 §5) can be applied at analysis.
# cohort.json still records each model's measured G-A for disclosure.
CONFIRM_FLAGS="--confirm-timestamped-plan-progtrace --ignore-gate-a"

# --- roster: SLUG|HF_REPO|REVISION (E10 §3; revisions pinned to E1 verified cache) -
# Run order = smallest first (validates the pipeline fastest; E1 pattern):
#   qwen1p5b(1.5B) -> qwen7b(7B, re-anchor) -> olmo7b(7B) -> llama8b(8B) -> qwen32b(32B).
E10_MODELS=(
  "qwen1p5b|Qwen/Qwen2.5-1.5B-Instruct|989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
  "qwen7b|Qwen/Qwen2.5-7B-Instruct|a09a35458c702b33eeacc393d103063234e8bc28"
  "olmo7b|allenai/OLMo-2-1124-7B-Instruct|470b1fba1ae01581f270116362ee4aa1b97f4c84"
  "llama8b|NousResearch/Meta-Llama-3.1-8B-Instruct|d10aef7999a2b5ba950ab3974312feeedbfe0b77"
  "qwen32b|Qwen/Qwen2.5-32B-Instruct|5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
)
QUEUE_ORDER="qwen1p5b,qwen7b,olmo7b,llama8b,qwen32b"

# per-model out dir (basename MUST contain EXPG_PROGTRACE -- guard_out_dir requires it)
e10_outdir() { echo "$RESULTS/EXPG_PROGTRACE_$1"; }
