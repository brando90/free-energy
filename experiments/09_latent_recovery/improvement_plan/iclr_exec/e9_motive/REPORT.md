# E9 motive experiment — finalize / implement / arm — REPORT

**Date:** 2026-07-24 · **Author agent** (subagent). Local copy of the report also at
`$EXP/improvement_plan/iclr_exec/e9_motive/REPORT.md`.
`EXP = /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`.

## TL;DR

The design was revised per **every** required_change from both adversarial critics, the revised
registration was **committed before any generation**, the full zero-GPU pipeline (generator +
audits + scoring + analysis + runner pairing-audit) is **implemented and green**, the CPU smoke
shows **100% world-construction viability** (20/20 base, 20/20 surprise-control), and the GPU run is
**armed but human-gated**: a triple-gated waiter is live under nohup, consuming no GPU, and **cannot
start generation until (1) E1 is terminal, (2) a GPU frees, and (3) a human writes `E9_HUMAN_GO`
after resolving four decisions**. Two critic findings are structural identification threats that code
alone cannot close (plant-surprise ≡ usefulness by construction; compliance-interference ≡
usefulness in a 2-arm pair); the run should **not** be un-gated until Elyas/Brando decide G-D1..G-D4.

## 1. Registration commit (pre-run timestamp)

- **File:** `improvement_plan/iclr_exec/e9_motive/E9_REGISTRATION.md` (+ identical local copy at
  `scratchpad/latent_recovery_review/e9_design/E9_REGISTRATION.md`).
- **Commit hash:** `3986290adb0ce2638beacd6913e655e9054a983d`
- **Commit time:** `2026-07-24T10:48:09-07:00` (branch `paper-framing-polish`; **not pushed**).
- Commit contains **only** the registration file (an already-staged `E1_REGISTRATION.md` was kept out
  and left in its prior staged state). The commit **predates all generation** — no gold/continuation
  rollouts have run; the only generation performed is deterministic CPU nonce-world construction for
  the smoke test.
- Provenance line is in the doc verbatim: *"Design adversarially reviewed (2 critics) before
  registration; run approved by Elyas 2026-07-24 (chat); Brando review pending; hypothesis origin
  disclosed as exploratory (E5 licensed-split) — this registration is its first confirmatory test."*

## 2. Design revision — every required_change actioned

Accepted and registered (see `E9_REGISTRATION.md` sections in parentheses): crossed
verification-pressure factor to break instrumentality≡compliance (§3.1); plant-step **surprise
measurement + conditioning + surprise-control-arm hooks** and the corrected D4 disclosure (§2.2);
realized-effort logging + conditioning (§2.3); **plant↔active-reasoning-path** overlap replacing the
vacuous plant↔question overlap (§2.4); SPARC realized-trajectory proxy + D19 softening (§2.5);
**arm-blind HARD-RULE-5 classifier-transfer audit gate** (§4.3); G2 reframed as a **conditioning
event** with stricter/looser robustness + estimand disclosure (§1.3); **power re-derived against the
actual bootstrap decision rule** across a deconfounded-effect grid + fixed-n rule (§5.4); **symmetric
equivalence null branch** (artifact only if CI excludes 0.164) (§5.3); rollout **engagement gate** +
ITT sensitivity (§4.4); **closed pre-registration** — finite seed sequence, candidate cap 1200, fixed
final n, fixed Holm m, no data-dependent n→200 (§5.5); S2 equivalence power (§5.6); **judge_v2 as an
independent secondary DV** + global nonce uniqueness (§4/§1.2); Trust-but-Don't-Verify
synthesis/isolation rival conceded (§3). Nothing was rejected outright; the old D6 overlap-2×2 is
demoted to an exploratory satellite (its covariate is the vacuous one §2.4 replaces).

## 3. The four human decisions that gate launch (G-D1..G-D4)

These are why the run is **armed but not fired**. They are documented in `E9_REGISTRATION.md` §0.1.

- **G-D1 (surprise-matched control — Fatal 2, the crux).** The plant is on-A-path / off-B-path **by
  construction**, so its goal-conditioned token surprisal co-varies *perfectly* with usefulness. A
  "check locally-surprising statements" (prediction-error) mechanism predicts the exact inert>useful
  result with **no motive**. §2.2 registers direct surprisal measurement + conditioning and a
  surprise-control arm (options S-a/S-b); the generator builds and audits the S-a variant (passes its
  inertness certificate on 20/20 worlds), but whether S-a actually equalises surprisal is a GPU-time
  empirical question, and whether a clean control is even constructible in this grammar is a **design
  decision for Brando**. If resolved as "measurement only," the causal headline is softened
  (branch d).
- **G-D2 (crossed verification factor — Fatal 1).** Confirm the low/high verification factor is in the
  confirmatory family (its interaction discriminates motive from completion-pull). Doubles
  continuation generation; power re-derived.
- **G-D3 (sample size).** Against the actual decision rule and the deconfounded gap (which "may be
  below 0.164"), **n=150 is badly underpowered** (~0.31–0.52 power at δ=0.08–0.10; §5.4 grid). Either
  raise n (n≈300 at δ=0.10, n≈450 at δ=0.08 — rescaling candidate pool + GPU budget) or formally
  declare 0.164 the minimum effect of interest.
- **G-D4 (classifier transfer).** The E5 six-anchor validation does **not** transfer to E9 worlds; a
  **non-author** blind rater must run the HARD-RULE-5 audit (§4.3); if per-arm classifier error
  differs by >0.05 the primary is reported classifier-confounded, not causal.

The human writes these into `run_confirmatory/E9_HUMAN_GO` (a JSON:
`{"n_families":…,"final_n":…,"gd2_in_family":…,"surprise_control":…,"verifications":[…]}`), which
both records the decision and drives the run.

## 4. Smoke-audit pass rates (CPU, zero-GPU)

- **Fixture tests** (`e9_tests.py`): **11/11 groups pass**, each fail-closed audit exercised in both
  directions (good worlds pass; tampered worlds — deleted spur, bridge→B1 leak, independently-derivable
  bridge, removed Cm→A1 — all correctly rejected).
- **World construction, base:** `gen_worlds_e9.py --n-families 20` → **20/20 worlds pass all
  fail-closed certificates** (plant false at d=1; bridge load-bearing; A-usable & B-inert; two-goal
  separation; token-freq & zero-overlap identity; global nonce uniqueness). **Pass rate 100%.**
- **World construction, surprise-control (S-a):** **20/20 pass**, including the S-a inertness
  certificate (z reaches an off-ramp yet goal_B stays underivable from z). **Pass rate 100%.**
- **Runner pairing audit** (`e9_run.py --self-test`): G1–G3 pass on a shared-trunk pair (plant slot
  inside the trunk) and correctly reject an unsolved arm and a divergent-opening pair.
- **Scoring + analysis** validated on synthetic rows: 100 licensed / 200 reuse correctly counted;
  primary paired cluster-bootstrap, equivalence-vs-0.164 null, S1, S2 TOST, S3 interaction, power
  curve, and the descriptive by-arm block all compute. The descriptive block **surfaces the two
  Fatal-flaw signals directly**: synthetic usable-arm plant-surprisal 0.99 vs inert 3.01, and
  active-path overlap 0.63 vs 0.08 — exactly the confounds §2.2/§2.4 instrument.

**Bottleneck honesty:** the deterministic constructor passing 100% is expected (a constructor that
fails its own audit is a bug, not a sample — same as EXPD). The **binding** viability is the
GPU-gated **G1–G3 pair eligibility** (joint-solve of TWO different questions + byte-identical trunk
prefix for both), which **cannot be measured on CPU**. From EXPD priors (one-token joint-solve ~0.89;
identical-prefix ~0.93–0.96/direction) and the fact that arm B is a *genuinely different* question,
the realistic net pair yield is plausibly **~0.4–0.7** — the reason the registration provisions 1200
candidate families and a fail-closed under-power→inconclusive rule. This yield is the first thing the
gold+pair stages will report once armed; if it comes in low, G-D3's n decision must absorb it.

## 5. Armed run — waiter state

- **Waiter PID:** `4021112` (`bash run_e9_queue.sh`, launched under nohup, **consumes no GPU while
  waiting**, 600 s poll, heartbeat logging).
- **Status file:** `$EXP/improvement_plan/iclr_exec/e9_motive/run_confirmatory/queue_status.json`
  (currently `{"state":"waiting","detail":"E1=no GPUfree=no humanGO=no"}` — verified holding on all
  gates even though GPU 3 briefly freed, proving gates 1 & 3 are load-bearing).
- **Waiter log:** `…/e9_motive/run_confirmatory/waiter.out`.
- **E1 gate file (does not yet exist):** `…/iclr_exec/e1_replication/queue_status.json` — fail-closed:
  absent ⇒ treated as non-terminal.
- **Human-GO sentinel (to be written by Elyas):** `…/e9_motive/run_confirmatory/E9_HUMAN_GO`.
- **One-line progress check:**
  `ssh skampere2 'cat /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e9_motive/run_confirmatory/queue_status.json; tail -3 …/run_confirmatory/waiter.out'`
- **To stop the waiter:** `ssh skampere2 'kill 4021112'`.

The waiter, once all three gates clear, runs prepare → gold → pair → rollout → score → analyze on the
freed GPU (`CUDA_VISIBLE_DEVICES` pinned, EngineCore cleanup trap copied from `e7_judge/run_e7.sh`),
reading n and factor choices from `E9_HUMAN_GO`, and writes its own `queue_status.json`.

## 6. Files delivered (all under `$EXP/improvement_plan/iclr_exec/e9_motive/`)

- `E9_REGISTRATION.md` — committed pre-run (hash above).
- `gen_worlds_e9.py` — `build_motive_family` (two goals + shortcut branch + EXPD refuting spur) +
  fail-closed `audit_world_e9` + global-uniqueness audit + S-a surprise-control variant.
- `e9_tests.py` — 11 CPU fixture groups (both directions of every audit).
- `e9_run.py` — staged runner (prepare/gold/pair/rollout) reusing `tooling/vllm_gen`; CPU pairing
  audit (G1–G3) + plant-surprisal capture via prompt-logprobs; `--self-test` (CPU) green.
- `e9_score.py` — licensed split via `e5_classify.classify_row` (verbatim) + reuse (D11) + unlicensed
  (D11.1) + engagement gate + realized effort + active-path overlap; emits the arm-blind audit sample.
- `e9_analyze.py` — registered contrasts: primary paired cluster-bootstrap (percentile + BCa) vs
  MES=0.05, equivalence-vs-0.164 null, S1/S2 TOST, S3 verification interaction, surprise/effort
  descriptives, power-curve recompute.
- `run_e9_queue.sh` — the triple-gated waiter (armed, live).
- `smoke_base/`, `smoke_sctrl/`, `smoke_run/` — smoke artifacts (evidence).

## 7. Recommendation

Do **not** create `E9_HUMAN_GO` until G-D1..G-D4 are decided. G-D1 (surprise identification) is the
scientific crux and should go to Brando — it is exactly the "usefulness vs prediction-error" confound
that determines whether a confirming result can carry the causal C4 headline at all. Everything else
(generator, audits, scoring, analysis, waiter) is ready for a short validated GPU burst the moment the
gates clear.
