# E9 — Adjudication of the completed motive (goal-flip) experiment

**Adjudicated:** 2026-07-24 · **Run:** `run_confirmatory/` (queue `state=done`, complete)
**Registration:** `E9_REGISTRATION.md` (commit `3986290`) + `E9_REGISTRATION_AMENDMENT.md` (commit `d367a97`)
**Implementation deviation:** `DEVIATIONS.md` + `gen_worlds_e9.py` fix (commit `f447bc5`)

**VERDICT (one line): BRANCH (b) — NULL / properly-licensed world-side artifact.** The exploratory
E5 0.187-vs-0.023 (δ≈0.164) licensed-checking split does **not** reproduce as a within-world paired
effect on clean minimal-pair worlds; the 95% CI on the deconfounded gap positively **excludes** the
0.164 anchor. **Label: PROVISIONAL** — the G-D4 blind non-author HARD-RULE-5 classifier-transfer audit
has not been performed, so every licensed-count-driven conclusion here is provisional pending that
audit and the registered >0.05 per-arm error-difference rule.

---

## 1. Integrity checks

| Check | Registered expectation | Observed | Pass |
|---|---|---|---|
| World-audit failures | 0 (fail-closed) | `world_audit.json`: 750/750 per-world audits passed, `audit_failures: []` | ✅ |
| Candidate pool | 750 families, seed 20260724 (amendment G-D3) | 750 generated, sha256 `fdd49f81…f1b9` | ✅ |
| Pair-eligibility yield | registered ~0.4–0.7 → ≥300 | **373 / 750 = 0.497** eligible pairs | ✅ |
| Final n vs target | `final_n=300` (documentary; overshoot analyzed in full) | 371 pairs analyzed (primary), ≥300 met | ✅ |
| G1–G3 audit | outcome-independent, logged | ineligible reasons: **G1_not_both_solved 302, G1_gold_not_valid 50, G3_trunk_too_short 20, G2_no_shared_trunk 5** | ✅ |
| G2 conditioning-event bias (§1.3) | flagged risk of null-biasing | only **5** worlds excluded on G2 → minimal selection pressure from the "A-like B" conditioning | ✅ (low) |
| Per-arm row counts | 750×2 gold; 373 pairs ×2 arm ×2 verif ×8 rollout | gold 1500; rollout low 746×8 + high 746×8; **scored 11936** | ✅ |
| Engagement bucket (§4.4) | symmetric across arms | **usable 0.950 vs inert 0.832** (11.8 pp asymmetry; inert engages less, as feared for goal_B) | ⚠ asymmetric |
| Plant-surprisal logging (G-D1) | measured per pair per arm | present on **11936/11936** rows; usable μ=3.549 σ=0.772, inert μ=3.546 σ=0.772 (non-degenerate) | ✅ |
| G-D4 blind-rater audit | gates trusting licensed counts | `audit_sample.jsonl` = **67 rows emitted, 0 rater labels (unadjudicated)** | ⛔ not run → PROVISIONAL |

**Implementation deviation (adjudicated benign).** The first confirmatory `prepare` aborted with
`StopIteration` (nonce-noun pool exhausted at n=750; `failed_attempt_1/`). Commit `f447bc5` extended the
noun stream. Verified: the failure fired during **CPU world construction before any world row or GPU
rollout was written** (zero outcome data existed); the fix keeps the first 9,716 yields byte-identical
to the old pool and only appends longer same-class `-pus` nonces; 750/750 audits pass; global-nonce
uniqueness holds. This is a legitimate pre-data implementation fix, not a design change.

**Provenance gap (note, not a blocker).** The pipeline code (`e9_run.py`, `e9_score.py`,
`e9_analyze.py`, `run_e9_queue.sh`) is **untracked in git** (only `gen_worlds_e9.py` + the registration
MDs are committed). The analysis is reproducible from the scored rows but the analysis code itself is
not version-pinned. I independently re-derived the primary and S1 point estimates from `scored_rows.jsonl`
and they match `e9_results.json` exactly (primary 0.0037062, n_informative 35, S1 0.0120299).

---

## 2. Registered decision rules (verbatim) and their application

### 2.1 Primary
> **Decision rule:** reject H0 iff the one-sided 95% lower bound of a **cluster bootstrap** (resample
> pairs w/ replacement, 5000 resamples, registered seed; **BCa** variant registered … ) exceeds
> **MES = 0.05**.

Primary = mean over pairs of `licensed_strict(inert) − licensed_strict(usable)` at low-verification,
engaged denominator. **n_pairs = 371, n_informative (discordant) = 35.**

- point = **0.00371**, lower_95_pct = **−0.00123**, lower_95_bca = **−0.00109**, upper_95 = 0.00868
- **−0.00109 < 0.05 → H0 NOT rejected.** Primary does **not** clear the MES hurdle. (BCa and percentile agree.)

### 2.2 Symmetric null / equivalence branch
> **Symmetric null branch.** … "Artifact" is concluded **only** if the one-sided **upper** bound of the
> deconfounded gap falls below a pre-registered artifact-consistent threshold (registered: the 95% CI
> **excludes 0.164** …). A bootstrap lower bound ≤ 0.05 whose CI still admits an effect materially above
> MES is reported **INCONCLUSIVE (underpowered)**, never as confirming the artifact hypothesis.

- CI95 two-sided = **[−0.00221, 0.00958]**; `excludes_anchor_0164 = true` (upper 0.0096 ≪ 0.164);
  `ci_admits_effect_above_MES = false` (upper 0.0096 < 0.05).
- **→ The equivalence/upper-bound test PASSES.** This is **not** the INCONCLUSIVE case (the CI does not
  admit any effect above MES). Branch **(b)** is licensed, not "INCONCLUSIVE".

### 2.3 Outcome branch selected (verbatim)
> **(b) NULL / REVERSES (properly licensed)** — only if the equivalence/upper-bound test (§5.3)
> positively excludes the exploratory 0.164; then the exploratory split was a **world-side artifact**
> (EXPD's freq-2-vs-1 + noun swap), reported openly as a second registered reversal (à la C8). S1 is
> expected to survive (C3 is world-side-robust); a branch-(b) that *also* kills S1 is flagged as the most
> consequential failure.

**Selected: branch (b).** Corroborating mechanism (descriptive): E9's inert-arm licensed-checking rate
is **~0.011** (pooled, engaged) vs the E5 exploratory inert rate of **0.187** — an order of magnitude
lower. The gap cannot reproduce because the inert-arm checking **baseline itself did not reproduce** on
clean minimal-pair worlds. This is exactly the world-side-artifact signature branch (b) describes.

### 2.4 Crossed verification interaction (S3, in-family per amendment G-D2)
> **Motive account:** the usefulness gap **survives** under high verification. **Compliance account:**
> the gap **shrinks toward zero** under high verification.

- gap_low = 0.00373, gap_high = 0.00771, interaction (low−high) = **−0.00399** (gap does **not** shrink;
  nominally grows). n_pairs = 369.
- **Interpretation: MOOT.** Both gaps are within noise of zero; there is no gap to survive or collapse,
  so the motive-vs-compliance lever cannot discriminate. Directionally the gap does not shrink under high
  verification (weakly motive-consistent), but this carries no evidential weight given the null primary.

### 2.5 Secondaries (Holm family m = 3: S1, S2, S3)
- **S1 (reuse usable − inert > 0, directional).** Engaged: point **0.0120**, lower_95_bca **0.00126 > 0**
  → directionally positive, so S1 is **not killed** (branch (b)'s "most consequential failure" does NOT
  obtain). **But** the magnitude is negligible and far below the registered expectation (~0.6–0.8 vs ~0):
  reuse-of-plant sits at a **~0.90 ceiling in BOTH arms** (inert 0.904, usable 0.905 pooled). The
  "poisoning" validator fires on plant presence very liberally, so the positive-control dissociation the
  design counted on barely materializes under the engaged denominator. Under **ITT** the S1 gap is larger
  (point 0.094, lower 0.073 > MES) — usefulness-driven differential absorption is present but modest, and
  washed out among engaged rollouts.
- **S2 (unlicensed rejection TOST ±0.05).** ci90 = [0.0, 0.0], **equivalent = true**. Unlicensed rejection
  is at **exact floor (0) in both arms** → equivalence holds trivially. Branch (c) (dissociation-fails)
  does **not** fire (it requires equivalence to FAIL under adequate power; §5.6). Caveat: this is a
  floor-driven equivalence, not a demonstration of adequate S2 power.
- **S3:** see §2.4 (moot).

### 2.6 Surprise-conditioned & effort analyses (§2.2/§2.3)
- **Surprise:** plant-token surprisal is **matched across arms** (usable 3.549 vs inert 3.546; Δ 0.0035,
  σ 0.77). The G-D1-feared confound (surprise ≡ usefulness perfectly separating the arms) is
  **empirically absent** — usefulness did not shift plant surprisal — which *defuses* a prediction-error
  account, though this is moot given the null. NOTE: the S-a off-ramp was folded into every world
  (amendment), which contributes to the matched surprisal; there is no separate S-a-free inert arm in the
  data. The registered *within-matched-surprise-strata* and *effort-conditioned* gap analyses are **not
  implemented** in `e9_analyze.py` (only descriptive per-arm means are produced); given the null primary
  the missing conditioning changes no conclusion, but it is a gap between the analysis plan and the code.
- **Effort:** realized proof length differs by arm — usable μ=10.15 steps vs inert μ=14.43. The
  manipulation *does* change realized effort/engagement even though it does **not** change licensed
  checking or plant surprisal.

### 2.7 ITT sensitivity (§4.4) — required for the headline, run by adjudicator (queue omitted `--itt`)
> a pre-registered **ITT sensitivity** (denominator = all 8 rollouts …) is reported alongside; the two
> must agree in sign for the headline.

The confirmatory queue ran `e9_analyze.py` **without `--itt`**, so the required ITT sensitivity was not
emitted. I ran it (throwaway copy, confirmatory dir untouched): ITT primary point **+0.0204**
(lower_95 0.0124, still < MES; excludes 0.164; does not admit effect > MES). **Sign agrees** with the
engaged headline (+0.0037), both fail MES, both exclude the anchor → **headline is robust across
denominators.** (This missing invocation should be added to the run record.)

### 2.8 Power curve — degenerate, do not rely on
`power_curve.json` reports 1.0 at every δ×n cell. This is an artifact of `power_curve()` shifting every
pair's difference by a **constant** (`delta − base_delta`), collapsing the simulated variance so the
bootstrap lower bound ≈ δ > MES always. The power recompute is **uninformative**; it plays no role in the
verdict (the equivalence test, not power, licenses branch (b)).

---

## 3. Verdict

**BRANCH (b): NULL, properly licensed as a world-side artifact — PROVISIONAL.**

1. Primary licensed-checking gap does not clear MES (lower bound −0.0011 < 0.05).
2. The equivalence/upper-bound test positively excludes the exploratory anchor 0.164 (upper 0.0096),
   and does not admit any effect above MES → this is a licensed NULL, **not** INCONCLUSIVE.
3. Therefore the E5 exploratory 0.187/0.023 split is adjudicated a **world-side artifact** of the EXPD
   construction (freq-2-vs-1 + noun swap), reproduced here as a second registered reversal.
4. S1 (reuse) is **not** co-killed (directionally positive), so this is not branch (b)'s worst case; but
   its magnitude is far below expectation (ceiling in both arms) — the paradigm's positive control is
   weak, which is the main threat to a strong artifact claim.

**PROVISIONAL attaches to (1)–(3)** because every licensed count depends on the E5 classifier whose
transfer to E9 worlds was **never validated** by the registered blind non-author HARD-RULE-5 audit
(`audit_sample.jsonl` emitted, unadjudicated). Until that audit runs and the per-arm classifier
error-difference is shown ≤ 0.05, the null is provisional; if that difference exceeds 0.05 the primary is
reported classifier-confounded, not causal.

**Branches explicitly NOT taken:** (a) CONFIRMS — no (primary fails MES); (c) DISSOCIATION-FAILS — no (S2
equivalence holds); (d) IDENTIFICATION-UNRESOLVED — no (there is no confirmed gap whose attribution is at
issue); INCONCLUSIVE — no (CI excludes the anchor and admits no effect > MES).

---

## 4. Descriptives worth a paper sentence

- On clean single-world goal-flip minimal pairs, licensed derivation-backed checking of a fixed planted
  false statement is **rare and arm-invariant** (~1% engaged, inert 0.011 vs usable 0.007 pooled); the
  E5 exploratory inert rate of 0.187 does not reproduce — the checking baseline, not just the gap,
  collapses on deconfounded worlds.
- The usefulness manipulation is behaviorally real but on the **wrong DV for the claim**: it shortens
  realized proofs (10.2 vs 14.4 steps) and raises goal-engagement (0.95 vs 0.83), while leaving genuine
  checking untouched.
- Plant-token surprisal is **identical across arms** (Δ = 0.0035 nats, σ = 0.77): the presumed
  "surprise ≡ usefulness" identification threat has no empirical differential to ride on in this grammar.
- Reuse/absorption of the plant is at a **~0.90 ceiling in both arms** under the engaged denominator; a
  usefulness-driven reuse dissociation is detectable only under ITT (0.09), far below the ~0.6–0.8
  exploratory expectation — the D11 positive control is weak here.
- Pair yield 0.497 (373/750); G1 "both-arms-solved" is the dominant eligibility filter (352 of 377
  exclusions); the G2 "A-like-B" conditioning event drops only 5 worlds, so the estimand's external
  validity is less selection-narrowed than §1.3 feared.

*Honesty note: this is a reportable null. It licenses the second registered reversal (à la C8) for TMLR,
withholds the ICLR causal headline, and is contingent on the outstanding blind-rater classifier audit.*
