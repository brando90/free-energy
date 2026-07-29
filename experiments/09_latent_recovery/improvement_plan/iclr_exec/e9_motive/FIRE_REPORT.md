# E9 motive experiment — FIRE report

**Date:** 2026-07-24 · **Agent:** E9-fire subagent. Authorized by Elyas 2026-07-24 in chat.
`EXP = /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`.

## TL;DR
All three launch gates are **armed and verified**: E1-terminal = **yes**, human-GO = **yes**. The run
is now **purely GPU-availability-gated** (gate 2). A live waiter (PID **1180214**, 120 s poll) will
auto-fire the moment a GPU goes idle, pinning that GPU and never touching an occupied one. Generation
was **not yet running at report time** because no GPU was idle (all 8 in use). The four gated decisions
were resolved per Elyas's authorization, recorded in a committed amendment, and encoded in `E9_HUMAN_GO`.

## 1. E1 terminal state (step 1)
- Authoritative file `e1_replication/results/queue_status.json` was **already corrected** by the E1
  adjudication agent: top-level `state = queue_complete` (history[-1] = queue_complete rc=0 at
  2026-07-24T18:50:08Z; all 4 models rc=0; world_audit_failures=0). mtime 2026-07-24T13:07 PDT
  (>5 min before I touched anything — no race). **I did not modify it.**
- **Gap found + fixed:** the waiter (`run_e9_queue.sh`) reads E1 state from a *different* path —
  top-level `e1_replication/queue_status.json` — which did **not exist**, and its accepted terminal
  vocabulary `{done,complete,completed,terminal,failed,aborted,error}` does **not** include the literal
  `queue_complete`. I created that gate file with `state="completed"` (an accepted synonym), mirroring
  the authoritative `current`/commit fields and a provenance note pointing back to `results/`. Verified
  against the waiter's exact `e1_terminal()` check → **GATE1 = PASS**.

## 2. Amendment (step 2) — four resolutions, committed pre-run
File: `e9_motive/E9_REGISTRATION_AMENDMENT.md`. Records verbatim:
- **G-D2:** crossed low/high-verification factor **IS in the confirmatory family** (doubles generation;
  accepted). Holm m=3. → `verifications:["low","high"]`, `gd2_in_family:true`.
- **G-D3:** n raised to **300 eligible pairs**; candidate pool scaled to **n_families=750** (conservative
  ~0.4 yield, within §5.5 cap 1200); power note updated per §5.4 grid (n=300 → ~0.80 power at δ=0.10).
- **G-D1:** surprise measurement + conditioning + **S-a surprise-control arm INCLUDED**
  (`surprise_control:true`); surprise-vs-motive interpretation limit stays a **disclosed limitation,
  Brando review pending**.
- **G-D4:** blind-rater classifier audit **cannot run today** (needs human non-author) → all
  licensed-count results labeled **PROVISIONAL**; the **>0.05 per-arm error-difference rule stands** and
  applies when the audit runs.
- Status line included verbatim: *"authorized by Elyas 2026-07-24 in chat; Brando review pending;
  amendment committed before any confirmatory generation."*

**No code change was required** — all four map onto existing `e9_run.py`/`run_e9_queue.sh` flags driven
by `E9_HUMAN_GO`. Verified against `e9_run.py`: `prepare --n-families/--surprise-control`,
`rollout --verification low high`, `analyze --gd2-in-family` all present.
`surprise_control=true` augments every world with the S-a off-ramp while retaining the base
usable/inert arms (superset), so the S-a arm is included without dropping the primary contrast.
`final_n` is documentary (the waiter never passes it to a stage); the closed-pre-registration guarantee
rests on the fixed 750-candidate pool + outcome-independent G1–G3 eligibility (no optional stopping).

## 3. Amendment commit (step 3)
- **Hash:** `d367a972818d61c2e2ca2531b2065ce1cec2eec1`
- **Time:** 2026-07-24T13:24:35-07:00 · branch `paper-framing-polish` · **not pushed**.
- **Exactly 1 file** in the commit (the amendment). Staging area was clean beforehand.

## 4. E9_HUMAN_GO (step 4)
`e9_motive/run_confirmatory/E9_HUMAN_GO` (valid JSON — plain prose would break the waiter's `json.load`
and silently fall back to wrong defaults, e.g. `surprise_control=false`):
```json
{"authorization":"Authorized by Elyas 2026-07-24 in chat via Claude; see E9_REGISTRATION_AMENDMENT.md (commit d367a972818d61c2e2ca2531b2065ce1cec2eec1).",
 "n_families":750,"final_n":300,"gd2_in_family":true,"surprise_control":true,
 "verifications":["low","high"],"brando_review":"pending",
 "gd4_licensed_counts":"PROVISIONAL pending blind non-author HARD-RULE-5 audit; >0.05 per-arm error-diff rule applies"}
```
Verified the waiter's exact field reads: n_families=750, gd2=true, surprise_control→`--surprise-control`,
verifications=`low high`.

## 5. Waiter (step 5)
- The original waiter (PID 4021112, 600 s poll) was **demonstrably missing the cluster's transient idle
  windows** (GPUs idx3/6/7 freed at some polls but were re-occupied by the next 10-min poll). Diagnosis:
  poll interval coarser than the free-window duration.
- **Action:** killed 4021112 cleanly (no EngineCore/GPU children — only a `sleep`; cleanup trap fired via
  killing the sleep child), and **relaunched the same script, same gate logic**, with `E9_POLL_SEC=120`.
- **New waiter:** PID **1180214**, poll 120 s, detached (setsid+nohup). Current status:
  `{"state":"waiting","detail":"E1=yes GPUfree=no(idx=none) humanGO=yes"}`.
- It pins only a GPU with `<1024 MiB` used (genuinely idle) and never touches occupied ones; on all-3
  gates it runs prepare→gold→pair→rollout→score→analyze on the pinned GPU.

## 6. Fire conditions, ETA, progress check
- **Remaining gate:** GPU-free. Will auto-fire within ≤120 s of any GPU going idle.
- **ETA once running:** ~1–2 GPU-h at n=300 with the crossed factor (registration §6, EXPD anchor
  44 gen/s); the S-a arm adds ~n×8 rollouts. With n_families=750 and yield ~0.4–0.7, expect
  ~300–525 eligible pairs → upper end ~2–2.5 GPU-h wall. **Binding constraint is GPU availability, not
  hours.**
- **Status-file path:** `EXP/improvement_plan/iclr_exec/e9_motive/run_confirmatory/queue_status.json`
- **Progress check (one line):**
  ```
  ssh skampere2 'D=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e9_motive/run_confirmatory; cat $D/queue_status.json; echo; tail -4 $D/waiter.out; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits'
  ```
- **To stop the waiter:** `ssh skampere2 'kill 1180214'` (then kill its `sleep` child to trigger the trap).

## 7. Honest status
Steps 1–4 are **complete and verified**. Step 5: the waiter is **armed and correctly waiting on the
single remaining external gate (an idle GPU)**; per the run's own safety rule and the task constraint I
did **not** commandeer any occupied GPU, so generation had **not started** as of report time. It will
start automatically at the next idle-GPU window. See §8 for the observed outcome during the monitoring
window.

## 8. Monitoring window outcome
Monitored ~13:29–13:33 PDT (2026-07-24). Across every poll the waiter held at
`state=waiting, detail="E1=yes GPUfree=no(idx=none) humanGO=yes"` — i.e. **2 of 3 gates clear, GPU
gate never opened**. The cluster stayed **fully saturated** the entire window (all 8 GPUs
106–139 GB used; idx3, which had flickered idle earlier in the hour, climbed 11 GB → 100 GB →
106 GB). No `worlds/`, `gold_rollouts.jsonl`, or `raw_generations.jsonl` were created.

**Conclusion: generation has NOT started** as of 2026-07-24T20:33Z, purely because no GPU went idle.
The waiter (PID 1180214, 120 s poll) is confirmed alive and will fire automatically at the first idle
window. This is the correct, safe terminal state for this task — per the run's own rule and the task
constraint, I did not commandeer any of the occupied GPUs. Nothing further is blocking on a human: the
E1 and human-GO gates are permanently satisfied; only GPU availability remains.
