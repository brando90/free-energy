# E1 LAUNCH — multi-model replication of the boundary conditions — EXEC REPORT

Date: 2026-07-24 (~10:40–10:56 PDT). Agent: E1 launch. Slug dir (cluster):
`$EXP/improvement_plan/iclr_exec/e1_replication/` where
`$EXP=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`.

**Bottom line:** Step 1 (registration finalized + committed) DONE. Step 2 (pipeline built +
CPU-validated) DONE. **Step 3 (smoke + launch) NOT DONE — no GPU was fully free (0 MiB) at setup
time; per the GPU rules I refused to squeeze onto an occupied GPU.** All CPU work is complete; the
queue is one command away from launch the moment a GPU frees.

---

## STEP 1 — Registration finalized + committed (DONE)

- Finalized file: `iclr_exec/registrations/E1_REGISTRATION.md` (from `E1_REGISTRATION_DRAFT.md`).
- **Commit hash: `bdb098a1d85b33063274481e53cb9d7130dc32c6`**
- **Committer timestamp: `2026-07-24T10:50:06-07:00`** (author = Elyas Obbad <eobbad@gmail.com>, matching repo convention).
- Branch `paper-framing-polish`; parent `3986290` (the sibling "E9 registration" commit). 1 file, 170 insertions. **NOT pushed** (branch ahead of origin by 4).
- `git status` was checked first; only `E1_REGISTRATION.md` was staged/committed (verified: `git show --stat` = 1 file).

**Open parameters resolved conservatively (registration §12):**
- M1 Llama = `NousResearch/Meta-Llama-3.1-8B-Instruct` @ `d10aef79…` (the `meta-llama` Instruct repo is a gated 76 KB stub — s7 Part C).
- M2 OLMo = **Instruct** variant `allenai/OLMo-2-1124-7B-Instruct` @ `470b1fba…` (cached complete; chosen + disclosed).
- M3 = `Qwen/Qwen2.5-32B-Instruct` @ `5ede1c97…`; M4 = `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa798…`.
- 5th model slot **DEFERRED/dropped** (stated: 2/4 subjects Qwen-family; judge=M3 disclosed).
- Magnitude thresholds accepted as drafted and made immutable (B1 ≥ +0.10, B2 ≥ +0.30, B3 ±0.10/±0.15); program rule ≥3/4 full replications, floor 3; B-P stays secondary (m=3/model).
- Dated addendum appended: **"Run approved by Elyas 2026-07-24 (chat); Brando review pending at registration time; any change after this commit is a disclosed deviation."**

**Provenance caveat (disclose to Elyas):** the repo has a **concurrent writer** — a sibling agent
committing the analogous "E9 registration". Its reflog shows a `commit → reset HEAD~1 → re-commit`
pattern that twice wiped my *staged* file out of the shared index (`git reset` unstages). I therefore
committed **race-safely** via an isolated `GIT_INDEX_FILE` + `commit-tree` + compare-and-swap
`update-ref` (never `git add` on the shared index, never a force). My commit landed on attempt 1 on
top of their HEAD. **Residual risk:** if that sibling agent does another `git reset` past my commit,
it could drop `bdb098a1` from the branch tip. Verify with
`git -C $REPO cat-file -e paper-framing-polish:experiments/09_latent_recovery/improvement_plan/iclr_exec/registrations/E1_REGISTRATION.md`
before relying on it; the file content is also preserved in this report's local bundle copy.

---

## STEP 2 — Pipeline built + CPU-validated (DONE)

All files under `iclr_exec/e1_replication/` (new dir). **The locked EXPD machinery
(`improvement_plan/expd/*.py`) and `src/*` are reused UNCHANGED** — the pipeline calls
`expd_matched_gradient.py` for generate/validate/summarize/report and imports `gen_worlds_expd.py`
for world prep, editing neither.

| File | Role |
|---|---|
| `e1_config.sh` | model table (repo/rev/oversample-seeds), reduced-grid cells/positions, paths, E1 commit |
| `prepare_e1.py` | CPU world prep; own-trace golds are inherent to the EXPD `generate` stage (phase-1 gold rollouts → eligibility ladder → injection manifest → perturbed continuations). Oversampling = the UNCHANGED full generator run under K seeds, merged under an `s{seed}_` id prefix + `family_idx` offset; each shard fail-closed audited before merge. Writes the exact files `generate` reads (`candidate_pool.jsonl`, `worlds/`) |
| `run_e1_queue.sh` | sequential queue over the 4 models on ONE pinned GPU; per model: prepare→generate pass1/2/3→validate→summarize→report; rewrites `queue_status.json` (stage, model, started, done, rc) after every stage transition |
| `status_writer.py` | atomic (temp+rename) `queue_status.json` updater |
| `smoke_e1.sh` + `check_smoke.py` | the ~10-problem 1.5B end-to-end smoke (pilot config so gold phase ≈200 rollouts); asserts validator parses ≥50% of 1.5B outputs and gold attrition is nonzero-but-viable |

**Design choices worth knowing:**
- **Own-trace golds:** each model solves its own worlds (phase-1 gold), and only golds that are
  `solved ∧ valid_rederivation ∧ injection-points-available` (EXPA double filter) are perturbed —
  inherent to the locked `generate`, reused as-is.
- **Oversampling mechanism.** The EXPD full config = 2,880 worlds, sized for Qwen-7B's ~0.90
  eligibility. Weaker models attrit more, so `prepare_e1.py` generates K seed-shards (K=4 for
  1.5B/OLMo, K=2 for Llama/32B) of the **unchanged** generator and merges them. The generator source
  is never edited; the per-cell cap (150 eligible) is unchanged, so surplus worlds are simply unused.
- **Run order** (registration note): `qwen1p5b` first (fastest, validates the pipeline
  end-to-end), then `llama8b`, `olmo7b`, `qwen32b` last (largest). Each model is its own
  subprocess, so vLLM tears down between models (no memory bleed).
- **Reduced grid** (registration §4, disclosed): attr-false quartet × d{0,1,3,5} × 3 pos + anchors ×
  3 pos + cat usable/inert × d{1,3,∞} × {early,mid} + MC true cells @ mid. No d=2, no full true
  quartet, no satellites.
- vLLM env (`.venv-vllm`), `HF_HUB_OFFLINE=1`, `CUDA_VISIBLE_DEVICES` pinned by the operator at
  launch (the script refuses to run without it and does NOT pick a GPU itself).

**CPU validation performed (no GPU needed, all passed):**
1. `prepare_e1 --seeds 0 --config full` → 2,880 worlds, **2,880/2,880 audits passed, 0 failures**.
2. `prepare_e1 --seeds 0,1` → **5,760 worlds, 5,760/5,760 audits passed**; 5,760 unique `world_id`,
   720 unique `family_id`, 720 unique `family_idx`, **0 dangling partners**, balanced shards
   (s0=2,880 / s1=2,880); every required field present; each replicated cell has 360 worlds at 2×.
3. **Anchor check (strong):** the unchanged generator regenerated seed-0 full worlds with
   `worlds_json_sha256 = 0bb17731…`, **byte-identical to the real
   `results/EXPD_MATCHED_GRADIENT/worlds/world_audit.json` hash**. The pipeline reproduces EXPD's
   worlds exactly (modulo the id prefix).

---

## STEP 3 — Smoke + Launch (NOT DONE — GPU blocked)

GPU snapshot at setup (2026-07-24 ~10:56 PDT), `nvidia-smi memory.used / util`:

| GPU | mem.used | util | state |
|---|---|---|---|
| 0 | 133,653 MiB | 100% | busy |
| 1 | 130,337 MiB | 100% | busy |
| 2 | 128,289 MiB | 69% | busy (was 45 GiB at 10:41, labmate expanded) |
| 3 | 111,265 MiB | 0% | **parked weights — memory occupied, not free** |
| 4 | 138,897 MiB | 91% | busy |
| 5 | 137,753 MiB | 100% | busy |
| 6 | 130,329 MiB | 0% | **parked weights — memory occupied, not free** |
| 7 | 130,329 MiB | 0% | **parked weights — memory occupied, not free** |

**No GPU shows 0 MiB used.** The task's GPU rule is explicit: use only a 0-MiB GPU; "if none is
fully free, report blocked rather than squeezing next to someone's job." GPU 2 — noted as fully free
at 13:05 EDT in the task brief — was taken by a labmate before setup began. GPUs 3/6/7 are idle
(0% util) but each holds 111–130 GiB of loaded weights, so they are NOT free. **Smoke and launch are
therefore skipped.** No queue was launched → **no PID**. `queue_status.json` was pre-seeded with a
`NOT_LAUNCHED` marker so a progress check is unambiguous.

### What remains (all GPU-gated; ~everything else is done)
1. **Smoke** (~5–10 min once a GPU frees):
   `CUDA_VISIBLE_DEVICES=<free_gpu> bash $E1DIR/smoke_e1.sh`
   Confirm `SMOKE: PASS` (validator parses 1.5B output; gold attrition nonzero-but-viable). Fix any
   1.5B chat-template parse issue before the full launch.
2. **Launch** the full queue (returns immediately; runs under nohup):
   `CUDA_VISIBLE_DEVICES=<free_gpu> nohup bash $E1DIR/run_e1_queue.sh > $E1DIR/logs/queue.out 2>&1 &`

Where `$E1DIR=$EXP/improvement_plan/iclr_exec/e1_replication` and `<free_gpu>` is a device showing
**0 MiB used** in `nvidia-smi` at launch (re-check immediately before — a labmate may take it).

### Progress-check commands (after launch)
- Status file: `cat $E1DIR/results/queue_status.json` (`.current` = stage/model/started/done/rc;
  `.history` = full transition log).
- Live log: `tail -f $E1DIR/logs/<slug>.<stage>.log` (slugs: `qwen1p5b`, `llama8b`, `olmo7b`,
  `qwen32b`; stages: `prepare`, `generate_pass1/2/3`, `validate`, `summarize`, `report`).
- Queue stdout: `tail -f $E1DIR/logs/queue.out`; GPU: `nvidia-smi`.
- Outputs land in `$E1DIR/results/<slug>/` (`validated_outputs.jsonl`, `summary_tables.json`,
  `gold_cohort.jsonl`, `EXPD_PILOT_REPORT.md`).

### Per-model ETA (single H200, from EXPD's 0.09 GPU-h full-grid gen + gold oversample)
- `qwen1p5b` (4×, 11,520 golds): ~15–25 min · `llama8b` (2×, 5,760 golds): ~20–40 min ·
  `olmo7b` (4×, 11,520 golds): ~30–50 min · `qwen32b` (2×, 5,760 golds + largest weights): ~1.5–2.5 h.
- **Total ~3–4 GPU-h realistic, well under the registration's <8 GPU-h budget.** (Analysis/scoring
  of the confirmatory family is a separate downstream step, not in this queue.)

---

## Files (cluster, all under `$EXP/improvement_plan/iclr_exec/`)
- `registrations/E1_REGISTRATION.md` (committed `bdb098a1`)
- `e1_replication/{e1_config.sh, prepare_e1.py, run_e1_queue.sh, status_writer.py, smoke_e1.sh, check_smoke.py}`
- `e1_replication/results/queue_status.json` (NOT_LAUNCHED marker)
- Local copy of this report: `exec_reports/e1_replication_REPORT.md`; registration copy:
  `E1_REGISTRATION.md` (both in the review bundle scratchpad).
