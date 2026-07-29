# ITEM S1 — Strict / hop-sound pass over EXPD + EXPE; H2′ re-adjudicated under hop-soundness

Run 2026-07-23 (UTC) on skampere2, 0 GPU. Reads `results/*` read-only; all new files under
`$EXP/improvement_plan/iclr_exec/s1_strict_hopsound/`. Reuses the **published** estimators
verbatim (checksums verified identical to cluster originals):
`improvement_plan/stage0_strict/strict_metrics.py` (`SM.analyze`, the M1–M4 hop-sound metric that
produced `STRICT_METRICS_REPORT.md`) and `improvement_plan/expd/expd_confirmatory_analysis.py`
(`CA`, the registered EXPD H2′ cohort + paired-family TOST + family-cluster bootstrap).

EXP = `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`.

---

## 0. BOTTOM LINE (what Elyas needs)

**H2′ (polarity equivalence, aff_false vs neg_false) PASSES under the strict hop-sound DV.**
On the exact registered spec (pooled d{1,3}, mirrored cohort, ±0.10 TOST margin, family-cluster),
the equivalence TOST rejects both one-sided nulls at **p = 8e-05 < α/6 = 0.0083** — it passes at
worst-case Holm, *more strongly* than the closure-valid original (p = 0.0012). The polarity
conclusion the ICLR paper leans on (W1: "polarity is not the driver") **survives the very DV the
rewrite demotes.** The point difference flips sign (aff was +0.054 higher on closure; neg is 0.042
higher on hop-sound), which if anything *reinforces* "neither polarity systematically dominates."

Two honest caveats, both disclosed below: (i) equivalence holds at the pre-registered ±0.10 margin
but **not** at a tighter ±0.05 (the aff/neg CI excludes 0, so they are *distinguishable* though
*equivalent within margin*); (ii) both hop-sound rates are low in absolute terms (aff 0.108, neg
0.153) — consistent with the paper's zero-hop thesis, but the equivalence is "both polarities rarely
produce genuine step-by-step derivations, neither more than the other."

No strict metric REVERSES a permissive conclusion the paper relies on. The one sign flip (H2′)
strengthens rather than undermines the claim.

---

## 1. Validation anchors (all PASS before trusting the adaptation)

**A1 — closure recompute agreement.** `SM.analyze` recomputed validator class == stored
`valid_recovery` on **11577/11577 EXPD** and **2104/2104 EXPE** rows (agreement = 1.000000, 0
mismatches). Therefore `hop_sound_valid` is a clean strict subset of the paper's `closure_valid`,
and every published EXPD/EXPE closure-valid cell rate reproduces exactly (spot-checked:
aff_false_attr_d0/d1/d3 = 0.9222/0.9222/0.9456; neg_false_attr_d1/d3 = 0.9044/0.8552;
cat_false_usable_d1 = 0.0933; benign 0.9978; true_interruption 0.9178; EXPE REFUTING_d1@mid 0.3636,
FREQ@mid 0.35, TEMPLATE@mid 0.4045 — all match the published reports).

**A2 — legacy strict reproduction** (re-ran `SM.load_expa`/`load_legacy`, compared to the published
`STRICT_METRICS_REPORT.md`): every number matches to 3 dp.

| family | hop-sound (mine / pub) | closure (mine / pub) | SRR-task (mine / pub) |
|---|---|---|---|
| EXPA:one_hop_falsehood | 0.300 / 0.300 | 0.671 / 0.671 | 0.251 / 0.251 |
| EXPA:global_falsehood | 0.213 / 0.213 | 0.393 / 0.393 | 0.002 / 0.002 |
| EXPA:true_interruption | 0.347 / 0.347 | 0.856 / 0.856 | — |
| EXPA:benign_paraphrase | 0.649 / 0.649 | 0.849 / 0.849 | — |
| legacy:negstep | 0.300 / 0.300 | 0.677 / 0.677 | 0.244 / 0.244 |
| legacy:falsehood | 0.095 / 0.095 | 0.286 / 0.286 | 0.000 / 0.000 |

**A3 — H2′ closure pipeline reproduction.** Running `CA` on `closure_valid` reproduced the
registered H2′ verdict **exactly**: diff = 0.0542, 95% CI [0.0249, 0.0835], p_raw = 0.001202,
paired_families = 163, rows 832/830, aff 0.9351 / neg 0.8783 (cf. `EXPD_hypothesis_verdicts.json`).
The hop-sound run swaps ONLY the DV (`CA.mval` monkeypatched to return `hop_sound_valid`); cohort,
pairing, TOST and bootstrap are byte-for-byte the registered machinery.

---

## 2. Schema-mapping decisions (legacy EXPA schema → EXPD/EXPE)

| # | Decision | Rationale |
|---|---|---|
| M1 | `closure_valid` := stored `valid_recovery` | the paper's registered DV; `CA.mval` reads it; == recomputed class (A1) |
| M2 | `hop_sound_valid` := `closure_valid AND SM.analyze(...).hop_sound_all` | strict subset; identical M2 definition as `STRICT_METRICS_REPORT.md` |
| M3 | audited-false plant: EXPD `audited_truth_status=='false'`; EXPE `injected_statement_truth_status=='false'` | EXPE integrity file: all 2104 plants audited false |
| M4 | cluster unit: EXPD `family_id` (== registered H2′ cluster); EXPE `problem_id` | |
| M5 | cell key: EXPD `condition`; EXPE `arm@injection_position` | |
| M6 | doubt / echo / corrective / targeted / derivational **recomputed by `SM.analyze`** (not stored fields) | SRR / stated-reliance use the EXACT defs of the published strict report; EXPE stored `doubt` is null anyway |
| M7 | EXPD/EXPE `question` passed to `analyze` = manifest `question` | EXPE `question` is the **augmented** question (arm's added rule present) = what the model saw; hop-seed uses closure of that world |

Structural caveat inherited from the legacy metric (not a bug): every non-benign condition omits the
overwritten gold step from the prefix, so hop-soundness also prices in re-deriving that step. The
TRUE-plant control quantifies this burden — EXPD true_interruption closure 0.918 → hop-sound 0.493.

---

## 3. H2′ re-adjudication (the core deliverable, item S1b)

Registered spec: aff_false_attr vs neg_false_attr, closure-valid TOST ±0.10, pooled d{1,3},
jointly-solved + identical-prefix mirrored cohort (MOD-13), positions pooled, cluster = family.
Only the DV changes below.

| DV | diff (aff−neg) | 95% CI | TOST p_raw | pass @0.05 | pass @Holm α/6 | aff / neg | families |
|---|---|---|---|---|---|---|---|
| **closure_valid** (paper, ANCHOR) | +0.0542 | [0.0249, 0.0835] | 0.001202 | ✅ | ✅ | 0.9351 / 0.8783 | 163 |
| **hop_sound_valid** (strict, S1) | **−0.0419** | [−0.0716, −0.0122] | **8e-05** | ✅ | ✅ | 0.1082 / 0.1530 | 163 |

Equivalence holds: the CI lies entirely inside ±0.10; both one-sided nulls rejected (binding side
p_lower = 8e-05: H0 diff ≤ −0.10 rejected). **Verdict: H2′ SURVIVES hop-soundness.**

Sensitivity (same cohort/machinery; disclose all):

| slice | closure diff (pass Holm?) | hop-sound diff (pass Holm?) |
|---|---|---|
| pooled d{1,3} (registered) | +0.0542 (✅) | −0.0419 (✅ p=8e-05) |
| d1-only | +0.0139 (✅) | −0.0602 (⚠️ passes @0.05 p=0.016, **not** @Holm) |
| d3-only | +0.0970 (❌ CI crosses +0.10) | −0.0373 (✅ p=0.00059) |
| pooled, tighter ±0.05 margin | — | −0.0419 (❌ p=0.30; CI excludes 0) |

Reading: at d3 the **closure** DV actually *fails* equivalence (aff 0.948 vs neg 0.851, diff +0.097
sits on the margin) whereas **hop-sound passes cleanly** — the strict re-adjudication is if anything
*more* favorable to the polarity-equivalence claim at the larger distance. d1-only hop-sound is
marginal at Holm but clears 0.05. The pre-registered test is the pooled d{1,3}, which passes
decisively. The tighter-margin failure is the honest boundary: aff and neg are *statistically
distinguishable* (CI excludes 0) but *equivalent within the registered ±0.10* — the paper must say
"equivalent within ±0.10", never "identical".

---

## 4. Strict-vs-permissive accounting — EXPD (feeds the plan's SS2 cuts table)

Positions pooled; denominator = all rows (generation_failed + unparsed retained, house rule).
`hop|val` = share of closure-valid completions that are hop-sound. SRR / stated-reliance over
audited-false plants only. Hop-sound Wilson95 in brackets.

| cell | n | closure-valid | hop-sound-valid | Δpp | goal_jump0 | hop\|valid | SRR-task | stated-reliance |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 450 | 0.9222 | 0.2467 [0.209,0.289] | 67.6 | 0.184 | 0.268 | **0.291** | 0.000 |
| aff_false_attr_d1 | 450 | 0.9222 | 0.1044 [0.080,0.136] | 81.8 | 0.240 | 0.113 | 0.047 | 0.000 |
| aff_false_attr_d2 | 450 | 0.9311 | 0.0956 [0.072,0.126] | 83.6 | 0.320 | 0.103 | 0.047 | 0.000 |
| aff_false_attr_d3 | 441 | 0.9456 | 0.1088 [0.083,0.141] | 83.7 | 0.329 | 0.115 | 0.041 | 0.000 |
| aff_false_attr_d5 | 426 | 0.9624 | 0.1150 [0.088,0.149] | 84.7 | 0.333 | 0.120 | 0.024 | 0.000 |
| neg_false_attr_d0 | 450 | 0.9578 | 0.1800 [0.147,0.218] | 77.8 | 0.307 | 0.188 | 0.033 | 0.000 |
| neg_false_attr_d1 | 450 | 0.9044 | 0.1622 [0.131,0.199] | 74.2 | 0.280 | 0.179 | 0.036 | 0.000 |
| neg_false_attr_d2 | 450 | 0.8533 | 0.1244 [0.097,0.158] | 72.9 | 0.318 | 0.146 | 0.033 | 0.000 |
| neg_false_attr_d3 | 435 | 0.8552 | 0.1471 [0.117,0.184] | 70.8 | 0.315 | 0.172 | 0.028 | 0.000 |
| neg_false_attr_d5 | 414 | 0.8406 | 0.1425 [0.112,0.180] | 69.8 | 0.324 | 0.170 | 0.027 | 0.000 |
| cat_false_inert_d1 | 300 | 0.6233 | 0.1300 [0.097,0.173] | 49.3 | 0.010 | 0.209 | 0.177 | 0.000 |
| cat_false_inert_d3 | 284 | 0.7500 | 0.0493 [0.030,0.081] | 70.1 | 0.042 | 0.066 | 0.035 | 0.000 |
| cat_false_inert_dinf | 300 | 0.7700 | 0.0900 [0.063,0.128] | 68.0 | 0.047 | 0.117 | 0.003 | 0.000 |
| cat_false_usable_d1 | 300 | 0.0933 | **0.0000** [0.000,0.013] | 9.3 | 0.003 | 0.000 | 0.027 | 0.000 |
| cat_false_usable_d3 | 280 | 0.2036 | 0.0107 [0.004,0.031] | 19.3 | 0.039 | 0.053 | 0.000 | 0.000 |
| cat_false_usable_dinf | 300 | 0.2000 | 0.0100 [0.003,0.029] | 19.0 | 0.033 | 0.050 | 0.007 | 0.000 |
| benign_paraphrase (ctrl) | 450 | 0.9978 | 0.9622 [0.940,0.976] | 3.6 | 0.244 | 0.964 | — | — |
| true_interruption (TRUE ctrl) | 450 | 0.9178 | 0.4933 [0.447,0.539] | 42.5 | 0.120 | 0.538 | — | — |
| aff_true_attr_d0..d5 | ~450 | 0.90–0.98 | 0.067–0.176 | ~80 | 0.26–0.38 | 0.07–0.18 | — | — |
| neg_true_attr_d0..d5 | ~450 | 0.93–0.99 | 0.142–0.227 | ~80 | 0.27–0.44 | 0.15–0.23 | — | — |

Strict-side findings the paper can use:
- **Hop-soundness collapses "recovery" everywhere except the benign control** (0.90–0.96 closure →
  0.07–0.25 hop-sound on false-attr cells; ≤0.13 on cat-false). Only ~11–27% of closure-valid
  false-attr completions are genuine step-by-step derivations; the rest reach the target via
  closure-level skips (goal_jump0 ≈ 0.18–0.33). Directly answers W2 on EXPD's flagship grid.
- **The usability dichotomy sharpens under strict:** cat_false_usable hop-sound ≈ 0.00–0.011 vs
  cat_false_inert 0.049–0.130 — usable false plants essentially never yield a hop-sound derivation
  (they are reused, not re-derived). Complements EXPD H4.
- **A strict-only d=0 visibility structure (new, descriptive):** for aff_false, SRR-task is
  **0.291 at d0** then collapses to 0.047 / 0.041 / 0.024 at d1 / d3 / d5, and hop-sound-valid is
  0.247 at d0 vs ~0.10 for d1–d5. Closure-valid was *flat* across d (registered C-0 null, diff
  −0.0024). The strict metrics thus surface the same d=0 cliff the paper argues for — under a DV
  that cannot be dismissed as a parse/skip artifact. (C-0 is non-family/no-verdict; report
  descriptively.)

---

## 5. Strict-vs-permissive accounting — EXPE (all rows audited-false)

| cell | n | closure-valid | hop-sound-valid | hop\|valid | SRR-task | doubt | corrective |
|---|---|---|---|---|---|---|---|
| REFUTING_d1@mid | 220 | 0.3636 | 0.1636 [0.121,0.218] | 0.450 | 0.0591 | 0.073 | 0.114 |
| REFUTING_d2@mid | 220 | 0.3864 | 0.1773 [0.133,0.233] | 0.459 | 0.0545 | 0.023 | 0.132 |
| REFUTING_d3@mid | 220 | 0.3591 | 0.1682 [0.125,0.223] | 0.468 | 0.0409 | 0.014 | 0.136 |
| FREQ_MATCHED_NONREFUTING@mid | 220 | 0.3500 | 0.1227 [0.086,0.173] | 0.351 | 0.0182 | 0.114 | **0.118** |
| TEMPLATE_IRRELEVANT@mid | 220 | 0.4045 | 0.1955 [0.149,0.253] | 0.483 | 0.0091 | 0.023 | 0.000 |
| BASELINE_FILLER@mid | 220 | 0.3591 | 0.1636 [0.121,0.218] | 0.456 | 0.0091 | 0.009 | 0.000 |
| REFUTING_d1@early | 220 | 0.4591 | 0.1864 | 0.406 | 0.0682 | 0.082 | 0.095 |
| REFUTING_d1@late | 172 | 0.3779 | 0.1977 | 0.523 | 0.0698 | 0.064 | 0.140 |
| BASELINE_FILLER@early | 220 | 0.4091 | 0.1864 | 0.456 | 0.0136 | 0.014 | 0.000 |
| BASELINE_FILLER@late | 172 | 0.3140 | 0.1628 | 0.519 | 0.0116 | 0.012 | 0.000 |

EXPE strict findings:
- **The lexical-cueing result (H5′) is corroborated, not reversed, by the strict flags.** On
  recomputed `corrective` (states the licensed complement), **FREQ_MATCHED 0.118 ≈ REFUTING_d1
  0.114**, both ≫ TEMPLATE/BASELINE 0.000; recomputed `doubt` is if anything *higher* for FREQ
  (0.114) than REFUTING_d1 (0.073). Frequency-matched non-refuting overlap triggers the same
  rejection behaviour as genuine refuting evidence — the professor's lexical-priming account. (EXPE's
  registered PRIMARY DV is string-level stated-complement, unchanged by hop-soundness.)
- **No licensed d≥1 hop-sound repair emerges.** SRR-task across REFUTING d1/d2/d3 = 0.059/0.055/0.041
  — flat/slightly declining, no distance gradient; and TEMPLATE hop-sound (0.196) ≥ REFUTING (0.164),
  so strict validity is not evidence-driven. Nothing here resurrects the falsifiability gradient.
- Note: `hop|valid` on EXPE is higher (~0.35–0.52) than on EXPD false-attr cells (~0.11) because
  EXPE traces are shorter re-derivations of a single planted category atom; still a minority-to-half.

---

## 6. Reversal scan (item S1d — does any strict metric flip a permissive conclusion?)

| contrast | permissive | strict | conclusion reversed? |
|---|---|---|---|
| H2′ aff_false vs neg_false (equivalence) | diff +0.054, TOST pass | diff −0.042, TOST pass | **No** — equivalence preserved; point sign flips but stays within ±0.10 (strengthens "polarity not the driver") |
| aff vs neg full-cell ordering, d1 & d3 | aff > neg (closure) | neg > aff (hop-sound) | Sign flip only; not a paper claim (the claim is *equivalence*, which holds) |
| C-0 aff_false d0 vs d1 | closure flat (null, diff −0.002) | hop-sound d0 0.247 ≫ d1 0.104 | Not a reversal — strict *adds* d=0 signal closure hid; descriptive (C-0 non-family) |
| usability dichotomy (cat usable vs inert) | inert > usable (closure) | inert > usable (hop-sound, sharper) | No — direction preserved and strengthened |
| EXPE H5′ FREQ ≈ REFUTING | equal on primary DV | equal on strict corrective/doubt | No — corroborated |

**No conclusion the paper relies on is reversed by the strict standard.** The only sign flip (H2′
point difference) makes the paper's case stronger, not weaker.

---

## 7. Caveats / disclosure checklist

1. H2′ hop-sound equivalence holds at the **pre-registered ±0.10** margin, **not** at ±0.05 (CI
   [−0.072, −0.012] excludes 0). Phrase as "equivalent within ±0.10", not "no difference".
2. Absolute hop-sound rates are low (aff 0.108 / neg 0.153); the equivalence is between two small
   rates. Fully consistent with the zero-hop thesis, but state the base rates.
3. Hop-soundness prices in re-deriving the omitted gold step (true_interruption 0.918→0.493 bounds
   this). This depresses hop-sound uniformly and is a property of the strict standard, not a defect;
   it does not bias the aff-vs-neg *contrast* (both omit one step symmetrically).
4. d1-only hop-sound TOST is marginal at Holm (p=0.016, passes 0.05); the registered test is the
   pooled d{1,3}, which passes at Holm.
5. `SM.analyze`'s derivational-use flag is measurable only for positive-categorical / rule plants;
   negated-attribute cells show echo only (unchanged from the legacy report). SRR uses the recomputed
   doubt regex (backend-independent), so it is directly comparable to `STRICT_METRICS_REPORT.md`.
6. The runner also stored a separate `strict_validation.strict_class` field
   (strict_noncanonical_recovery 10824 / strict_gold_suffix_replay 414 / strict_final_mismatch 339);
   it is a *different* strict notion (step-replay) and was **not** used — S1 uses the stage-0
   hop-sound M2 metric as specified. Worth a future cross-check (S2-adjacent) but out of scope here.

---

## 8. Files produced (all under `$EXP/improvement_plan/iclr_exec/s1_strict_hopsound/`)

- `s1_full.py` — main analysis (anchors A1/A2/A3, H2′ closure+hop-sound, EXPD/EXPE strict cells, accounting, reversal scan).
- `s1_strict_metrics.json` — full machine-readable results.
- `s1_addendum.py` / `s1_h2prime_sensitivity.json` — per-d + tighter-margin H2′ sensitivity.
- `probe2.py` — anchor probe (agreement + published-rate reproduction).
- `REPORT.md` — this report. Local mirror: `exec_reports/s1_strict_hopsound_REPORT.md`.
- `prior_attempt_backup/` — the earlier aborted stub (reached the same H2′ headline; superseded by this independently-verified run).
