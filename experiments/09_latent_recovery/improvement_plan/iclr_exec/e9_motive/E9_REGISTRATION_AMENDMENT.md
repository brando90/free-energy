# E9 — Registration Amendment (pre-run): four gated decisions resolved

**Parent registration:** `E9_REGISTRATION.md` (commit `3986290adb0ce2638beacd6913e655e9054a983d`, filed pre-run).
**Amendment status line (verbatim, required):**
> Authorized by Elyas 2026-07-24 in chat; Brando review pending; amendment committed before any
> confirmatory generation.

This amendment records the resolution of the four launch-gating decisions G-D1..G-D4 defined in
`E9_REGISTRATION.md` §0.1. It is committed **before any gold or continuation generation** (the only
generation performed to date is deterministic CPU nonce-world construction for smoke tests). It fixes
the run configuration consumed by `run_e9_queue.sh` via the `E9_HUMAN_GO` sentinel.

---

## Resolutions (verbatim)

### G-D2 — crossed low/high verification-pressure factor
The crossed low/high-verification factor **IS in the confirmatory family** (doubles generation;
accepted). Holm family size therefore m = 3 (S1, S2, S3-interaction) per §5.5. Implemented via
`rollout --verification low high`; `e9_analyze.py --gd2-in-family`.
`E9_HUMAN_GO`: `"gd2_in_family": true`, `"verifications": ["low","high"]`.

### G-D3 — sample size / power
n raised to **300 eligible pairs**; the candidate pool is scaled accordingly per the registration's
yield estimate (~0.4–0.7). Using the conservative low end (~0.4), the candidate pool is set to
**n_families = 750** (single seed 20260724; within the §5.5 candidate cap of 1200) so that expected
eligible pairs ≥ 300. The power note is updated per the stats critique's grid (§5.4): at n=300, power
≈ 0.99 at δ=0.164, ≈ 0.93 at δ=0.12, ≈ 0.80 at δ=0.10, ≈ 0.58 at δ=0.08. The confirmatory n is the
count of all G1–G3-eligible pairs from this **fixed, pre-declared** candidate pool; eligibility (G1–G3)
is **outcome-independent** (gold-trace solvability + shared-trunk geometry only, never the DV), so the
realized n is fixed-in-advance in the closed-pre-registration sense (no optional stopping, no
data-dependent n). `final_n = 300` is the pre-declared target; it is not applied as a hard truncation,
so any yield overshoot above 300 is analyzed in full (strictly more power, no fishing). Per §5.5
run-level fail-closed: if eligible pairs < 300 after the pool is exhausted, the run is
**underpowered → primary inconclusive**; no gate is relaxed.
`E9_HUMAN_GO`: `"n_families": 750`, `"final_n": 300`.

### G-D1 — surprise measurement + conditioning + S-a surprise-control arm
Surprise measurement + conditioning **and** the **S-a surprise-control arm are INCLUDED**. The
interpretation limit (surprise vs motive — the plant is on-A-path/off-B-path by construction, so
plant-step surprisal co-varies with usefulness) **stays a disclosed limitation** (§0.1, §2.2, §7 branch
(d)), **Brando review pending**. Implemented via `prepare --surprise-control`, which augments every
world with the S-a off-ramp (`z → bridge_b → b_dead`, a dead off-ramp under B that raises B-side
relevance without leaking goal_B), retaining the base usable/inert arms; plant-step surprisal is
captured per arm in `rollout` via prompt-logprobs regardless.
`E9_HUMAN_GO`: `"surprise_control": true`.

### G-D4 — classifier transfer / blind-rater audit
The blind-rater HARD-RULE-5 classifier audit (§4.3) **CANNOT run today** — it requires a human
non-author rater. Therefore **all licensed-count results are labeled PROVISIONAL** pending that audit.
The registered **>0.05 per-arm classifier-error-difference rule stands** and will be applied when the
audit is performed: if per-arm classifier error differs by more than 0.05, the primary is reported
classifier-confounded, not causal. No `E9_HUMAN_GO` field; enforced as a reporting stance on the
results.

---

## Authorization & provenance
Authorized by **Elyas 2026-07-24 in chat via Claude**. **Brando review pending** (G-D1 surprise
identification is the causal-identification crux and is the standing open scientific question). This
**amendment committed before any confirmatory generation**. No change to the analysis plan, contrasts,
decision rules, MES (0.05), or the symmetric-null branch registered in `E9_REGISTRATION.md`; this
amendment only fixes the four run-configuration decisions the registration deferred to a human.

## Implementation note (no code change required)
All four resolutions are realized through existing `e9_run.py` / `run_e9_queue.sh` flags driven by
`E9_HUMAN_GO` (`n_families`, `gd2_in_family`, `surprise_control`, `verifications`); no source change
was made to the E9 pipeline for this amendment. `final_n` is documentary (the waiter does not pass it
to any stage); the closed-pre-registration guarantee rests on the fixed candidate pool + outcome-
independent eligibility, as described under G-D3.
