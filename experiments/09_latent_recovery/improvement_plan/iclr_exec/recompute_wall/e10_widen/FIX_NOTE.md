# e10_widen — driver fix note (rc=127)

**Date:** 2026-07-27
**File fixed:** `run_e10_queue.sh` (source of truth on skampere2). Backup:
`run_e10_queue.sh.prefix127.bak`.

## What happened
The first skampere2 launch fired at 2026-07-27T12:01 (waiter pid 119622 → queue pid
946305 on GPU0) and **failed instantly**: every model's GPU `generate` stage returned
**rc=127**, and the queue "completed" in ~1s having produced **zero generations**.

## Root cause
`GEN` and `JUDGE` are bash **functions**, but `run_gpu()` launched GPU stages with
`setsid "$@"` where `"$@"` began with the function name (`run_gpu … generate GEN …`).
`setsid` execs its argument as an **external program** and cannot see bash functions →
`setsid: failed to execute GEN: No such file or directory` → rc=127. This broke every
GPU stage (generate + judge) on any node. CPU stages were unaffected (they use in-shell
`"$@"`).

## Fix (minimal, semantics-preserving)
1. Export the stage functions + the vars they reference, right before the model loop:
   ```
   export -f GEN JUDGE
   export PY_VLLM EXPG_DRIVER CELLS CAP OVERSAMPLE SEED MAXNEW_GOLD MAXNEW_CONT BATCH CONFIRM_FLAGS
   ```
2. Run GPU stages through a real interpreter that can resolve the exported function,
   keeping the new-session / process-group "EngineCore reap":
   ```
   setsid bash -c 'set -u; "$@"' _ "$@" >>"$log" 2>&1 &
   ```

## Verification
- `bash -n run_e10_queue.sh` → OK.
- No-GPU dry-run of the patched exec path on a trivial exported function `FOO(){…;return 7;}`:
  function executed (log printed) **and** `RUN_GPU_RC=7` (exit code propagates). No rc=127.

## Data status
**No experiment data existed before this fix.** The failed attempt produced only the
pre-computed worlds (`programs.jsonl`) + `run_metadata.json`; no generations. The failed
attempt's live-state files (`queue_status.json`, `queue.out`, `queue.pid`, `launch.lock`,
`*.generate.log`, `waiter.log`) are archived under `failed_attempt_1/` so they cannot be
mistaken for data. World-prep logs (`*.prepare.log`, `prepare_worlds.out`) are valid and
left in `logs/`.

## Where the fixed run actually ran
The corrected run was launched on **skampere1** (skampere2 GPUs were the reason for the
whole staging effort). The skampere1 copy carries additional NODE adaptations documented
in the RUNBOOK / staging report (adapted `env.sh` cache paths → /var/tmp, `PY_MAIN`
repointed, combined HF cache so llama8b's pinned snapshot resolves offline). This
skampere2 source file itself is only path-generic; those node deltas live on the
skampere1 copy, not here.

## Addendum 2026-07-27 -- judge stage KV-cache fix (judge-only max_model_len)

On the skampere1 run (80GB A100s) the 32B judge (JUDGE_MODEL=Qwen/Qwen2.5-32B-Instruct)
failed EngineCore init: after ~62 GiB weights at gpu_memory_utilization=0.85, only ~3.74 GiB
KV cache remained -- too little for max_model_len=32768 (a 32768-token seq needs ~8.2 GiB KV
for this model). skampere2's H200s (143 GB) never exposed this.

Fix (GENERATION PARAMS UNTOUCHED): tooling/vllm_gen.py `_get_llm` now reads two OPTIONAL env
vars -- `EXPG_LLM_MAX_MODEL_LEN` and `EXPG_LLM_GPU_MEM_UTIL`. Unset => byte-identical prior
behavior (no max_model_len cap; gpu_memory_utilization=0.85). The judge stage sets
`EXPG_LLM_MAX_MODEL_LEN=8192` (judge prompt = continuation + 4 new tokens, far under 8192;
~2.05 GiB KV/seq fits the 3.74 GiB budget at 0.85). If 8192 still won't fit on some node,
also set `EXPG_LLM_GPU_MEM_UTIL=0.92`. Backup: `vllm_gen.py.prejudgefix.bak`.

This run used judge-AFTER: the queue ran generate+validate for all 5 models (judge failed
non-fatally in-queue), then a separate `judge_repass.sh` re-ran judge over the on-disk
outputs with the override. Generation params/decoding were not touched anywhere.

## Addendum 2 2026-07-27 -- qwen32b GENERATE also needed the node-fit cap

On skampere1 (80GB A100) the 32B *generate* stage hit the SAME KV wall as the judge
(3.84 GiB KV < the 8.0 GiB a 32768-token seq needs). expg_progtrace's generate retried
engine-init indefinitely (looped ~59 min) and deadlocked the sequential driver, so the
first queue pass recorded qwen32b generate as failed (rc=137 after an operator kill to let
the driver reach terminal). The 4 smaller models were unaffected (their weights leave ample
KV).

Decision (coordinator-approved): apply `EXPG_LLM_MAX_MODEL_LEN=8192` to qwen32b's GENERATE
and rerun it. Rationale for the record: max_model_len is a MEMORY-ALLOCATION cap, not a
decoding parameter. With prompt + max_new (gold 512 / cont 384) at <=~900 tokens, no
sequence approaches 8192, so nothing is truncated and the sampled distribution is
byte-identical (greedy, same seed, same decoding). This is a node-fitting adaptation of the
same class as the judge fix -- decoding params are NOT touched. Verified: the run log
asserts the maximum observed prompt+generation token count stays far below 8192 (see
qwen32b.generate_rerun.log / the measurement in the final report).
