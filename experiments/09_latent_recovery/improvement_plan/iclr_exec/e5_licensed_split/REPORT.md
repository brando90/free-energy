# E5 — Licensed-vs-unlicensed stated-complement split

Item E5 of the ICLR revision plan. **Zero GPU.** All inputs read-only; all new files under
`$EXP/improvement_plan/iclr_exec/e5_licensed_split/` (core, validated) and its `v2/` subdir
(authoritative consolidated outputs). `EXP = /lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`.

- Core classifier (validated, reused unchanged): `e5_licensed_split/e5_classify.py :: classify_row`
- Consolidated analysis + CIs: `e5_licensed_split/v2/e5_analyze_v2.py` → `v2/e5_results_v2.json`
- Per-row classifications (current classifier): `v2/classified_rows_v2.jsonl` (449 rows)
- Validation traces: `e5_licensed_split/e5_audit.py`, `v2/e5_trace_d2.py`

> **Provenance flag (important).** The top-level `e5_licensed_split/e5_results.json` and
> `classified_rows.jsonl` are **STALE**: they were written (19:04) *before* the final edit to
> `e5_classify.py` (19:05) and were never regenerated. They encode a more lenient "licensed"
> rule and report inflated licensed counts (e.g. headline 0.795). **Do not cite them.** The
> authoritative numbers are in `v2/` (this report), produced by the current classifier.

---

## Headline verdict

Among the rare rejections that occur at a **designed distance ≥ 1** (i.e. the refuting evidence
is not directly visible in the prefix), **most are genuinely licensed** — the model wrote a
closure-valid derivation that terminates in the complement — but a hard lexical-cueing floor
exists in the controls, and **the large d = 0 rejection signal is 100 % premise-readout, not
derivation.**

**What fraction of all apparent one-hop-or-farther "checking" is licensed?**

| Scope (stated-complement rows) | n | licensed (strict) | 95% CI | licensed (lenient UB) |
|---|---|---|---|---|
| **Complement genuinely world-derivable at ≥1 hop** (measured_d finite ≥1) | 298 | **0.789** | [0.742, 0.833] | 0.916 |
| **Designed d ≥ 1, incl. non-derivable controls** (FREQ/dinf) | 332 | **0.708** | [0.660, 0.755] | 0.822 |
| Non-derivable controls only (measured_d = None; FREQ + cat_dinf) | 34 | **0.000** | [0,0] | 0.000 |
| Requires real search (measured_d ≥ 2, derivable) | 144 | 0.660 | [0.583, 0.734] | 0.924 |
| EXPE REFUTING arms (all d) | 129 | 0.798 | [0.729, 0.863] | 0.876 |

**Answers to the two required questions** (details in Interpretation §):
1. **Does any genuine d ≥ 1 checking exist?** Yes, unambiguously — 235 (strict) complete written
   refuting derivations at designed d ≥ 1, 95 of them requiring ≥2 hops of search. The draft's
   "models never perform even one hop of proof search" is **too strong** and must be softened to
   *"models rarely initiate distant checking, but the few distant checks they write are mostly
   sound."*
2. **Does the d = 0 cliff survive within licensed-only rejections?** **No — it vanishes / inverts.**
   All d = 0 rejections are premise-readout (`d0_visible`, 0 licensed derivations); the genuine
   licensed-derivation rate is low and *non-graded* across d ≥ 1 with **no d = 0 peak**. The d = 0
   cliff is a cliff in *visible readout*, not in *derivational checking* — a sharper statement of
   "seen, not sought."

---

## Method (built on the validated closure machinery — no new parser)

For every row where the model **stated the complement** of the planted falsehood (the rejection
DV), we classify the rejection using only canonical helpers: `validator.{parse_fact,parse_rule,
strip_marker,derivable}`, `expc_polarity_control.{opposite_pred,statement_predicate}`,
`expe_evidence_mover.{question_world,shortest_rule_distance,split_sentences}`. The chain walk is
the same forward-closure state accumulation used by `strict_metrics.analyze()` (add a stated fact
to the true state iff it is closure-derivable from the current true state; the audited-false
plant is never seeded).

- `comp_pred = opposite_pred(statement_predicate(plant))`.
- `measured_d = shortest_rule_distance(comp_pred, prefix_state, direct)` on the **augmented**
  world (EXPE's added rule is in the question) → the world-derivability distance of the
  complement (None ⇒ not forward-derivable at all).

**Four mutually-exclusive classes** (denominator = stated-complement rows):

| class | definition |
|---|---|
| **licensed** | measured_d ≥ 1 **and** the model, before asserting the complement, restated a world rule `lhs → comp_pred` **whose antecedent `("cat",lhs)` was established in the true running state** (premise or a *written* closure-valid step). For d ≥ 2 this forces the model to have written the intermediate hops. |
| **visible_unshown** | measured_d = 1 but no valid written derivation — a bare assertion of a complement that sits one hop from the prefix. |
| **d0_visible** | measured_d = 0 — the complement is a directly-given premise (pure d = 0 visibility / readout). Reported separately; **not** counted as "checking." |
| **unlicensed** | everything else — (a) measured_d = None: complement not world-derivable, so the assertion is lexically cued (all FREQ + cat_dinf); or (b) derivable at ≥2 hops but the chain was not written (silent multi-hop). |

`d0_visible` + `visible_unshown` together form the task's "visible-but-unshown (≤1 hop, no
derivation)" intermediate class; `d0_visible` is broken out because it is the load-bearing d = 0
bucket. A **lenient** upper bound (`klass_lenient`) additionally credits a restated refuting rule
with a *closure-reachable* (not necessarily written) antecedent, or a written valid intermediate
that strictly advances toward the complement.

**Strict vs lenient — where it matters.** Strict = "the model actually wrote the full chain"
(hop-sound-aligned; faithful to the task's *"chain the model actually wrote"*). Lenient credits
compressed/goal-jumping derivations. **At d = 1 the two definitions agree exactly** (licensing is
unambiguous with a single hop); they diverge only at **d ≥ 2**, where some models write the
refuting rule but skip explicitly concluding the antecedent category (verified in traces). Strict
is the reported primary; lenient is the upper bound.

Cluster unit = `problem_id`; 95% CIs by cluster bootstrap (5000 resamples, seed 20260722).

---

## Sanity anchors — reproduced before any split (validation gate)

Stated-complement rate per cell (fraction of all rows), vs published EXPD/EXPE/S1 numbers:

| cell | E5 sc-rate | published | match |
|---|---|---|---|
| EXPD aff_false_attr_d0 | 0.2467 | 0.247 | ✓ |
| EXPD aff_false_attr_d1 | 0.0311 | 0.031 | ✓ |
| EXPD cat_false_inert_d1 | 0.1933 | 0.193 | ✓ |
| EXPD cat_false_usable_d1 | 0.0233 | 0.023 | ✓ |
| EXPE REFUTING_d1 | 0.1144 | 0.114 | ✓ |
| EXPE FREQ_MATCHED | 0.1182 | 0.118 | ✓ |

All six reproduce (matches S1 report table `sc` column and EXPD/EXPE_FULL_REPORT.md). The
loaders and the stated-complement DV are therefore correct.

**Manual audit (HARD RULE 5).** Continuations were eyeballed per (cell × class) via `e5_audit.py`
and `v2/e5_trace_d2.py`. Confirmed: licensed rows write full valid refuting derivations; d0_visible
rows are bare readouts of a given premise; unlicensed rows either (i) assert the complement with no
chain, (ii) list/skip rules without establishing the antecedent (goal-jump), or (iii) **fabricate
a refuting rule not in the world** (all FREQ rows do this). The classifier is **conservative on
licensed**: unparsed refuting-rule surface forms (e.g. "Grimpuses are not a lorpus") downgrade a
genuine derivation to visible_unshown, so strict-licensed is a lower bound inside the refuting cells.

---

## Per-cell table (experiment × condition × d)

`nSC` = stated-complement rows; `lic/vis/d0v/unl` = strict-class counts; `licRate(all)` = licensed
rows / all rows in cell (absolute prevalence); `licFrac(SC)` = licensed / stated-complement. CIs
= cluster bootstrap 95%. True-plant, benign, template and filler cells have 0 stated-complements
(omitted). Full CIs for every class in `v2/e5_results_v2.json`.

| cell | d | nAll | nSC | lic | vis | d0v | unl | licRate(all) 95%CI | licFrac(SC) 95%CI |
|---|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 0 | 450 | 111 | 0 | 0 | 111 | 0 | 0.000 | 0.000 |
| aff_false_attr_d1 | 1 | 450 | 14 | 12 | 2 | 0 | 0 | 0.027 [0.011,0.042] | 0.857 [0.643,1.000] |
| aff_false_attr_d2 | 2 | 450 | 22 | 9 | 0 | 0 | 13 | 0.020 [0.009,0.033] | 0.409 [0.227,0.636] |
| aff_false_attr_d3 | 3 | 441 | 22 | 19 | 0 | 0 | 3 | 0.043 [0.025,0.061] | 0.864 [0.727,1.000] |
| aff_false_attr_d5 | 5 | 426 | 9 | 6 | 0 | 0 | 3 | 0.014 [0.005,0.026] | 0.667 [0.333,1.000] |
| neg_false_attr_d0 | 0 | 450 | 6 | 0 | 0 | 6 | 0 | 0.000 | 0.000 |
| neg_false_attr_d1 | 1 | 450 | 5 | 5 | 0 | 0 | 0 | 0.011 [0.002,0.024] | 1.000 |
| neg_false_attr_d2 | 2 | 450 | 6 | 5 | 0 | 0 | 1 | 0.011 [0.002,0.022] | 0.833 [0.500,1.000] |
| neg_false_attr_d3 | 3 | 435 | 5 | 2 | 0 | 0 | 3 | 0.005 [0.000,0.011] | 0.400 [0.000,0.800] |
| neg_false_attr_d5 | 5 | 414 | 2 | 1 | 0 | 0 | 1 | 0.002 [0.000,0.007] | 0.500 |
| cat_false_inert_d1 | 1 | 300 | 58 | 56 | 2 | 0 | 0 | 0.187 [0.147,0.230] | 0.966 [0.912,1.000] |
| cat_false_inert_d3 | 3 | 284 | 15 | 7 | 0 | 0 | 8 | 0.025 [0.007,0.042] | 0.467 [0.200,0.733] |
| cat_false_inert_dinf | inf | 300 | 7 | 0 | 0 | 0 | 7 | 0.000 | 0.000 |
| cat_false_usable_d1 | 1 | 300 | 7 | 7 | 0 | 0 | 0 | 0.023 [0.007,0.040] | 1.000 |
| cat_false_usable_d3 | 3 | 280 | 4 | 3 | 0 | 0 | 1 | 0.011 [0.000,0.025] | 0.750 [0.250,1.000] |
| cat_false_usable_dinf | inf | 300 | 1 | 0 | 0 | 0 | 1 | 0.000 | 0.000 |
| REFUTING_d1 | 1 | 612 | 70 | 60 | 10 | 0 | 0 | 0.098 [0.073,0.125] | 0.857 [0.761,0.941] |
| REFUTING_d2 | 2 | 220 | 29 | 21 | 0 | 0 | 8 | 0.096 [0.059,0.136] | 0.724 [0.552,0.862] |
| REFUTING_d3 | 3 | 220 | 30 | 22 | 0 | 0 | 8 | 0.100 [0.064,0.141] | 0.733 [0.567,0.867] |
| FREQ_MATCHED_NONREFUTING | inf | 220 | 26 | 0 | 0 | 0 | 26 | 0.000 | 0.000 |
| TEMPLATE_IRRELEVANT | inf | 220 | 0 | — | — | — | — | — | — |

Strict→lenient licensed gains are **entirely at d ≥ 2** (aff_d2 +11, cat_inert_d3 +8,
REFUTING_d2 +5, REFUTING_d3 +5, others ≤3); every d = 1 cell has +0. The aff_false_attr_d2 dip
(licFrac 0.409 strict) is this goal-jump artifact — under lenient it is 20/22 = 0.909.

---

## designed_d × measured_d cross-tab (stated-complement rows)

| designed_d | measured_d distribution |
|---|---|
| 0 | {0: 117} |
| 1 | {1: 154} |
| 2 | {2: 57} |
| 3 | {3: 76} |
| 5 | {5: 11} |
| inf | {None: 34} |

**Perfect agreement, zero d = 0 contamination**: every stated-complement's world-derivability
distance equals its designed distance, and every control-arm complement is genuinely
non-derivable (None). So Framings A and B differ only by the 34 non-derivable control rows.
(This is EXPD/EXPE's matched generated worlds — cleaner than the legacy EXPA cohort, whose 15%
d = 0 contamination is a separate matter handled in E6/RETRO_DISTANCE.)

---

## d = 0 cliff within licensed-only (absolute per-row rates across d)

`sc` = stated-complement rate (all rows); `lic/d0v/vis/unl` = per-class per-row rate.

| ladder | d | sc | licensed | d0_visible | visible_unshown | unlicensed |
|---|---|---|---|---|---|---|
| aff_false_attr | 0 | 0.247 | **0.000** | 0.247 | 0.000 | 0.000 |
| | 1 | 0.031 | 0.027 | 0.000 | 0.004 | 0.000 |
| | 2 | 0.049 | 0.020 | 0.000 | 0.000 | 0.029 |
| | 3 | 0.050 | 0.043 | 0.000 | 0.000 | 0.007 |
| | 5 | 0.021 | 0.014 | 0.000 | 0.000 | 0.007 |
| cat_false_inert | 1 | 0.193 | 0.187 | 0.000 | 0.007 | 0.000 |
| | 3 | 0.053 | 0.025 | 0.000 | 0.000 | 0.028 |
| | inf | 0.023 | 0.000 | 0.000 | 0.000 | 0.023 |
| cat_false_usable | 1 | 0.023 | 0.023 | 0.000 | 0.000 | 0.000 |
| | 3 | 0.014 | 0.011 | 0.000 | 0.000 | 0.004 |
| | inf | 0.003 | 0.000 | 0.000 | 0.000 | 0.003 |
| REFUTING | 1 | 0.114 | 0.098 | 0.000 | 0.016 | 0.000 |
| | 2 | 0.132 | 0.096 | 0.000 | 0.000 | 0.036 |
| | 3 | 0.136 | 0.100 | 0.000 | 0.000 | 0.036 |

The aff_false_attr **licensed** column has **zero** at d = 0 and stays low and flat (0.01–0.04)
across d = 1…5 — no d = 0 peak. The 0.247 at d = 0 is 100 % `d0_visible`.

---

## Interpretation

**Does any genuine d ≥ 1 checking exist?** Yes. Under the strict, hop-sound-aligned definition,
**235** stated-complement rejections at designed d ≥ 1 are backed by a complete written
derivation the model actually produced; **95** of these require ≥2 hops of real proof search
(lenient 133). This is small in *rate* (2–19% of rows within any d ≥ 1 cell) but non-zero and
mostly sound: **78.9%** (strict) / 91.6% (lenient) of rejections whose complement is genuinely
world-derivable at ≥1 hop are licensed. The draft's absolute "models never perform even one hop
of proof search" is therefore contradicted by the data and should be replaced by *"genuine
distance-checking exists but is rare and is not graded by proximity; when a model does state a
distant complement, it usually wrote the derivation."* The one clear positive spike is the
**inert-d1 bump** (cat_false_inert_d1 sc 0.193): E5 shows it is **96.6% licensed genuine 1-hop
checking** — models really do check inert (non-usable) lies when the refutation is one cheap hop
away, and this checking decays with distance (licensed rate 0.187 → 0.025 → 0 at d1/d3/dinf).

**Does the d = 0 cliff survive within licensed-only rejections?** No — it disappears and inverts.
The abundant d = 0 rejection (aff_false_attr sc 0.247) is *entirely* premise-readout: 111/111
`d0_visible`, **zero** licensed derivations. Genuine licensed-derivation prevalence is low and
**non-graded** across d ≥ 1 with no d = 0 advantage. So the d = 0 cliff in the raw rejection DV is
a cliff in *visible readout of a given contradiction*, not in *derivational checking*. This
sharpens the paper's psychophysics: the model rejects abundantly when the contradiction is a
directly-given premise (zero-hop readout) and rarely mounts a genuine derivation when it must be
sought — and never mounts more of them as the evidence gets closer. "Seen, not sought" is exactly
right; the d = 0 signal is *seeing*, not *checking*.

**Lexical cueing (W5 / red-team lexical-confound regress).** The controls give the cleanest
result: **every** FREQ_MATCHED rejection (26/26) and every cat_dinf rejection (8/8) is
**unlicensed and world-non-derivable** — the model fabricates a refuting rule absent from the
world (traces confirm invented `("cat",X)`/`X → not-Y` steps). FREQ's rejection rate (0.118) is
statistically indistinguishable from REFUTING_d1's (0.114) — the registered EXPE H5′ result — but
E5 shows the two are **different in kind**: REFUTING rejections are ~80% licensed genuine
derivations, FREQ rejections are 100% lexical fabrications. The correct, sharper claim for the
rewrite is therefore not "refuting-arm rejections are lexically cued" (they mostly are not) but
**"raw rejection rate cannot distinguish genuine checking from lexical cueing — a
frequency-matched non-refuting rule elicits an equal rate of *fabricated* rejections — and only
the licensed split separates the two."** This both concedes W5 (the rejection/doubt DV is a weak
proxy, contaminated by a real lexical-cueing floor) and supplies the fix (the licensed split).

---

## Caveats / limitations

1. **Qwen2.5-7B only** — same scope as the confirmatory family (EXPD/EXPE). Multi-model
   replication is E1's job.
2. **Small n in many cells.** Several d ≥ 2 attribute/cat/neg cells have nSC ≤ 10, so licFrac(SC)
   CIs are wide (e.g. neg_false_attr_d3 0.40 [0,0.80]). The *aggregate* headline (nSC 298–332) is
   well-powered; per-cell fractions are indicative.
3. **Strict vs lenient at d ≥ 2.** Report both. Strict counts a "wrote the refuting rule but
   skipped concluding the antecedent" goal-jump as unlicensed; lenient credits it. The choice
   moves the designed-d ≥ 1 licensed fraction between 0.708 and 0.822. It does **not** change any
   qualitative conclusion (d = 0 = 0 licensed; controls = 0 licensed; genuine checking exists,
   rare, non-graded). d = 1 is definition-invariant.
4. **Conservative on licensed** (parser misses on unparsed refuting-rule surface forms → strict
   licensed is a lower bound within refuting cells; the 10 visible_unshown in REFUTING_d1 include
   some genuine-but-unparsed derivations).
5. The stated-complement DV is parse-/validity-independent (matches S1's `sc`); this analysis
   inherits its definition exactly and does not re-open it.

## Files produced (cluster, under my slug only)

- `improvement_plan/iclr_exec/e5_licensed_split/v2/e5_analyze_v2.py` (consolidated analysis)
- `improvement_plan/iclr_exec/e5_licensed_split/v2/e5_results_v2.json` (**authoritative**)
- `improvement_plan/iclr_exec/e5_licensed_split/v2/classified_rows_v2.jsonl` (449 rows, current classifier)
- `improvement_plan/iclr_exec/e5_licensed_split/v2/e5_trace_d2.py`, `e5_dump_rows.py`, `run_v2.log`
- `improvement_plan/iclr_exec/e5_licensed_split/REPORT.md` (this file)
- Pre-existing (validated & reused): `e5_licensed_split/e5_classify.py`, `e5_audit.py`,
  `e5_explore.py`, `e5_explore2.py`.
- **STALE, do not cite**: `e5_licensed_split/e5_results.json`, `classified_rows.jsonl` (pre-final-classifier).
