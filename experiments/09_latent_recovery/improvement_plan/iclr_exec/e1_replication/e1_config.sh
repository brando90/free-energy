#!/bin/bash
# E1 replication — shared config (sourced by run_e1_queue.sh and smoke_e1.sh).
# Registration: iclr_exec/registrations/E1_REGISTRATION.md
# (committed bdb098a1d85b33063274481e53cb9d7130dc32c6, 2026-07-24T10:50:06-07:00).
# NOTHING here modifies the locked expd/ generator or src/; the generate/validate/
# summarize/report stages call the LOCKED expd_matched_gradient.py unchanged.

EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery
EXPD=$EXP/improvement_plan/expd
E1DIR=$EXP/improvement_plan/iclr_exec/e1_replication
RESULTS=$E1DIR/results
PY_VLLM=$EXP/improvement_plan/tooling/.venv-vllm/bin/python
PY_MAIN=/lfs/skampere2/0/eobbad/free-energy/.venv/bin/python

E1_REG_COMMIT=bdb098a1d85b33063274481e53cb9d7130dc32c6

# --- reduced EXPD core grid (registration section 4) -------------------------
# Pass 1: attribute-false quartet + anchors, positions early/mid/late.
ATTR_FALSE_CELLS="aff_false_attr_d0,aff_false_attr_d1,aff_false_attr_d3,aff_false_attr_d5,neg_false_attr_d0,neg_false_attr_d1,neg_false_attr_d3,neg_false_attr_d5"
ANCHOR_CELLS="benign_paraphrase,true_interruption"
PASS1_CELLS="$ATTR_FALSE_CELLS,$ANCHOR_CELLS"
PASS1_POS="early,mid,late"
# Pass 2: MC true cells (false-positive floor), mid only.
PASS2_CELLS="aff_true_attr_d1,neg_true_attr_d1"
PASS2_POS="mid"
# Pass 3: categorical usable/inert reuse cells, early/mid.
PASS3_CELLS="cat_false_usable_d1,cat_false_usable_d3,cat_false_usable_dinf,cat_false_inert_d1,cat_false_inert_d3,cat_false_inert_dinf"
PASS3_POS="early,mid"

CAP=150
MAXNEW=192
BATCH=256

# --- subject models: SLUG|HF_REPO|REVISION|OVERSAMPLE_SEEDS -------------------
# Run order (registration decision-rule note): 1.5B first (fastest, validates the
# pipeline end-to-end), then Llama-8B, OLMo-7B, Qwen2.5-32B last (largest).
# OVERSAMPLE_SEEDS: comma-separated world-generation seeds; K seeds == K x 2,880
# worlds merged (2x for >=7B-class M1/M3, 4x for weak M2/M4). Generator reused
# UNCHANGED; multi-seed shards merged under an s{seed}_ id prefix by prepare_e1.py.
E1_MODELS=(
  "qwen1p5b|Qwen/Qwen2.5-1.5B-Instruct|989aa7980e4cf806f80c7fef2b1adb7bc71aa306|0,1,2,3"
  "llama8b|NousResearch/Meta-Llama-3.1-8B-Instruct|d10aef7999a2b5ba950ab3974312feeedbfe0b77|0,1"
  "olmo7b|allenai/OLMo-2-1124-7B-Instruct|470b1fba1ae01581f270116362ee4aa1b97f4c84|0,1,2,3"
  "qwen32b|Qwen/Qwen2.5-32B-Instruct|5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd|0,1"
)
