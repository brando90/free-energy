# E9 — Adversarial Verification of the Confirmatory Adjudication

**Verified:** 2026-07-24 · **Verifier:** independent recomputation from raw scored rows
**Target:** `run_confirmatory/ADJUDICATION.md` (BRANCH (b), NULL / world-side artifact, PROVISIONAL)
**Method:** re-derived every headline number from `scored_rows.jsonl` with a standalone script that
does **not** import `e9_analyze.py`; cross-checked the bootstrap two ways (independent RNG at 40k
resamples, and an exact-RNG reproduction under the registered seed 20260724).

## Bottom line

**The adjudication is CONFIRMED.** Every load-bearing number reproduces — most to the exact digit,
the percentile bootstrap byte-exactly. The registered decision rules were applied verbatim, the
exclusion counts match the registration exactly, and the BRANCH (b) selection follows mechanically
from the pre-committed rules. The adjudication's caveats are complete and honest; I found no
overstatement and no undisclosed weakness. Two cosmetic numeric imprecisions in the write-up
(reuse ceiling to the hundredth; which surprisal-Δ is quoted) are immaterial and reconciled below.

## 1. Point estimates — exact reproduction

| Quantity | e9_results.json / ADJUDICATION | Independent recompute | Match |
|---|---|---|---|
| Primary point (inert−usable, low, engaged) | 0.0037061994609164 | 0.0037061994609164 | exact |
| n_pairs (primary) | 371 | 371 | exact |
| n_informative (discordant) | 35 | 35 | exact |
| Primary lower_95 percentile | −0.0012321909896034 | −0.0012321909896034 | exact (seed 20260724) |
| Primary upper_95 | 0.0086798870491593 | 0.0086798870491593 | exact |
| Equivalence CI95 two-sided | [−0.0022077, 0.0095784] | [−0.0022077, 0.0095784] | exact |
| S1 reuse point (usable−inert) | 0.0120299063021435 | 0.0120299063021435 | exact |
| S3 gap_low / gap_high | 0.0037263 / 0.0077139 | 0.0037263 / 0.0077139 | exact |
| S3 interaction (low−high) | −0.0039876 | −0.0039876 | exact |
| n_pairs S3 common | 369 | 369 | exact |

The percentile bootstrap reproduced **byte-exactly** under seed 20260724, which means the resample
draws and the per-pair diff vector are identical; the BCa lower bound (−0.00109) is a deterministic
bias/acceleration transform of that same resample set and the same jackknife, so it too is confirmed.
An independent-RNG bootstrap (seed 12345, 40k resamples) gives lower_95 = −0.00113 and
CI = [−0.00199, 0.00979] — indistinguishable, confirming the CI is not seed-fragile.

## 2. Decision rules — applied as written

1. **Primary** (reject H0 iff one-sided 95% lower bound > MES 0.05): lower bound −0.00109 (BCa) /
   −0.00123 (pct) / −0.00113 (indep) — all ≪ 0.05 → **H0 NOT rejected.** Correct.
2. **Symmetric-null / equivalence** (artifact concluded only if 95% CI excludes 0.164):
   upper 0.00958 < 0.164 → excludes; `ci_admits_effect_above_MES` = false (0.00958 < 0.05) →
   **not INCONCLUSIVE**, licensed NULL. Correct — the two sub-conditions were both checked.
3. **Branch (b)** (NULL properly licensed): equivalence excludes 0.164 → fires. Correct.
4. **Branch-(b) worst case** (also kills S1): S1 lower_95 = 0.00159 (indep) / 0.00126 (BCa) > 0 →
   S1 directionally positive, **not** co-killed. Correct.
5. **Branch (c)** (dissociation-fails; needs S2 equivalence to FAIL under power): S2 CI90 [0,0],
   equivalent = true → (c) does **not** fire. Correct.
6. **Branch (d) / (a) / INCONCLUSIVE**: all correctly excluded — no gap clears MES (a out), no
   confirmed gap to mis-attribute (d moot), CI excludes anchor + admits no effect > MES (INCONCLUSIVE
   out). Correct.

The branch call is mechanically forced by the pre-registered rules; I could not construct a reading
of the registration under which a different branch fires.

## 3. Exclusions & integrity — match the registration exactly

- `eligibility_audit.jsonl`: 750 rows, **377 excluded / 373 eligible, yield 0.4973** (within registered
  0.4–0.7). Reasons: **G1_not_both_solved 302, G1_gold_not_valid 50, G3_trunk_too_short 20,
  G2_no_shared_trunk 5** — identical to the adjudication table. G1 (both-solved) dominates (352/377);
  G2 conditioning-event drops only 5 → the §1.3 null-biasing worry is minimal, as claimed.
- `world_audit.json`: **750/750 per-world audits passed, `audit_failures: []`**, surprise_control true,
  structural sizes TRUNK_LEN 4 / BRANCH_LEN 3 / PRIMARY_D 1 / FILLER_PAD 3,
  `worlds_sha256 = fdd49f81…f1b9` (matches DEVIATIONS.md). Clean.
- Rollout grid: all 1492 (world×arm×verif) cells have **exactly 8** rollouts; 373×2×2×8 = 11936 scored
  rows, arm-balanced (5968/5968), verif-balanced (5968/5968). No missing or partial cells.
- Primary n=371 vs eligible 373: the 2-pair gap is the two worlds where one arm has 0 **engaged** low-
  verif rollouts (engaged-denominator drop), reconciled — ITT (all-rollout denom) recovers all 373.
- `audit_sample.jsonl`: **67 rows, 0 rater labels**, no `arm` field exposed (blind by design) →
  confirms the G-D4 audit was not adjudicated → PROVISIONAL is correctly applied.

## 4. Supporting figures — all reproduce

- Pooled licensed (low, engaged): **inert 0.01101 vs usable 0.00737** (~1%), vs E5 exploratory inert
  0.187 → order-of-magnitude baseline collapse. Confirmed.
- Engagement: **usable 0.9502 vs inert 0.8318 pooled** (11.8 pp asymmetry) — matches; the registered
  §4.4 symmetry expectation is violated, correctly flagged ⚠.
- ITT primary (adjudicator-run, queue omitted `--itt`): **point +0.02044, lower_95 0.0124**, CI upper
  0.0308 excludes 0.164, admits no effect > MES; sign agrees with engaged (+0.0037). ITT S1 point
  **0.0942, lower 0.0737**. All match the adjudication. Headline robust across denominators confirmed.
- Descriptives (low): active_path_overlap 0.8047/0.7773, realized_n_steps 10.151/14.432,
  plant_surprisal 3.5555/3.5537 — exact. plant_surprisal overall σ 0.772. Confirmed.

## 5. Minor imprecisions (immaterial, reconciled)

- **Reuse ceiling hundredths.** ADJUDICATION §2.5 cites reuse "inert 0.904, usable 0.905 pooled";
  pooled-over-rollouts I get usable 0.9009 / inert 0.8959 (both verif) or usable 0.9028 / inert 0.8970
  (low). All ~0.90; the qualitative claim (ceiling in both arms, dissociation barely materializes) is
  fully supported. The last-digit discrepancy is a pooled-vs-per-pair-mean choice and changes nothing.
- **Which surprisal-Δ is quoted.** ADJUDICATION quotes "Δ 0.0035 nats". The low-verif descriptive arm
  difference is 0.0018; the **all-rows** arm difference is exactly **0.00349** → the 0.0035 figure is
  the all-rows value and is correct. Both are ≈0 against σ 0.77 (arms surprise-matched either way).

## 6. Weaknesses — verified as genuine, and already disclosed

Every substantive caveat in the adjudication holds up under recomputation and none is missing:

1. **PROVISIONAL is load-bearing.** Primary + equivalence rest entirely on the E5 `classify_row`
   transferring to E9 worlds, never validated (G-D4 unrun, 0/67 labels). If the per-arm classifier
   error-difference exceeds 0.05, the primary is classifier-confounded, not causal. Correctly central.
2. **Equivalence exclusion of 0.164 is floor-driven.** Both arms sit at ~1% checking; excluding 0.164
   is near-guaranteed once the baseline itself collapsed. The mechanical branch-(b) rule fires
   correctly, but "world-side artifact" vs "E9 paradigm is simply insensitive to licensed checking" is
   **not** separated by the data. The adjudication says exactly this. No overclaim.
3. **Engagement asymmetry (~12 pp)** is a real §4.4 violation; ITT sign-agreement is the right
   mitigation and it does agree. Disclosed.
4. **S1 positive control is weak** (reuse at ~0.90 ceiling in both arms; predicted 0.6–0.8-vs-0
   dissociation appears only under ITT at 0.09). Verified. This is the main threat to a strong
   artifact claim, and the adjudication names it as such.
5. **Plan-vs-code gaps:** ITT (§4.4) omitted by the queue (run post-hoc by adjudicator); surprise-
   stratified (§2.2) and effort-conditioned (§2.3) analyses not implemented (descriptive means only);
   `power_curve.json` degenerate (constant-shift bug collapses variance → 1.0 every cell, uninformative,
   plays no role). All verified and disclosed; all moot under the null.
6. **Provenance:** pipeline code untracked in git (only `gen_worlds_e9.py` + registration MDs committed).
   Mitigated by full reproducibility from raw rows, which this verification demonstrates.

## Verdict

**CONFIRMED.** The E9 confirmatory adjudication is numerically exact, rule-faithful, and honestly
caveated. BRANCH (b) — properly-licensed world-side-artifact NULL, PROVISIONAL pending the G-D4 blind
classifier-transfer audit — is the correct call under the pre-registration + amendment. Recommendation
stands: report as an honest second registered reversal for TMLR; withhold the ICLR causal C4 headline;
treat the outstanding blind-rater audit as the gate before any licensed-count conclusion is finalized.
