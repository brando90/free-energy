# S1 — Strict / hop-sound pass over EXPD + EXPE, and re-adjudication of H2' under hop-soundness

Item S1 of the ICLR revision plan. Zero GPU. All inputs read-only; all outputs under
`$EXP/improvement_plan/iclr_exec/s1_strict_hopsound/`.

- Analysis script: `improvement_plan/iclr_exec/s1_strict_hopsound/s1_hopsound.py`
- Machine-readable results: `.../s1_strict_metrics.json`
- EXP = `/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery`

## Headline verdict

**H2' (polarity equivalence) SURVIVES the strict standard — and passes more strongly.**
Re-running the registered EXPD H2' TOST (aff_false_attr vs neg_false_attr, ±0.10 margin,
d∈{1,3}, identical mirrored cohort, paired-family estimator) with the DV swapped from the
paper's permissive `closure_valid` to the strict `hop_sound_valid`:

| DV | aff_false | neg_false | diff (aff−neg) | 95% CI | one-sided p_raw | TOST @0.05 | TOST @Holm α/6 |
|---|---|---|---|---|---|---|---|
| closure_valid (paper's registered DV) — **reproduced exactly** | 0.9351 | 0.8783 | **+0.0542** | [0.0249, 0.0835] | 0.001202 | PASS | PASS |
| **hop_sound_valid (strict)** | 0.1082 | 0.1530 | **−0.0419** | [−0.0716, −0.0122] | **8e-05** | **PASS** | **PASS** |

Same 163 paired families, same cohort, same estimator — only the DV changes. The equivalence
holds under both DVs. The point difference even **flips sign** (aff slightly higher under
closure; neg slightly higher under hop-sound), which is the strongest possible evidence
against a systematic polarity artifact: neither polarity consistently wins under either DV,
and both differences sit well inside ±0.10.

This directly closes reviewer **W2**: the family's only confirmatory pass is no longer resting
solely on "closure-valid completion," the very DV the rewrite demotes. It replicates under a
genuinely stepwise hop-sound standard.

**No registered EXPD conclusion is reversed by the strict metric** (task d). The strict lens
changes some *non-registered descriptive* patterns (below), but the load-bearing family pass
and the usability/visibility findings all survive.

---

## Method & schema-mapping decisions (legacy strict metric → EXPD/EXPE)

The strict metric is the byte-identical published cluster original
`improvement_plan/stage0_strict/strict_metrics.py` (verified `diff` == 0 vs the scratchpad
stub). Its `analyze()` re-plays `validator.validate_continuation` and additionally emits
`hop_sound_all` (every stated entity-fact is ONE direct rule-application, or a state member,
from the running state; the injected atom is seeded into the hop-state IFF it is true-derivable
under closure — so never for audited-false plants). I import it unmodified and drive it over the
EXPD/EXPE row schema. Decisions (all in `s1_strict_metrics.json → mapping_decisions`):

- **`closure_valid`** := the stored `valid_recovery` field. This is exactly what the published
  confirmatory estimator reads (`expd_confirmatory_analysis.mval`). Verified identical to
  `analyze()`'s recomputed `valid_rederivation` class (anchor A1).
- **`hop_sound_valid`** := `valid_recovery AND analyze().hop_sound_all`. Because recomputed
  closure == stored closure (agreement 1.0000), this is a clean strict SUBSET of the paper's DV.
- **audited-false plant (`is_false`)**: EXPD `audited_truth_status=='false'`; EXPE
  `injected_statement_truth_status=='false'` (all 2104 EXPE rows, per the integrity guarantee).
- **cluster unit**: EXPD `family_id` (matches the confirmatory family bootstrap); EXPE `problem_id`.
- **cell**: EXPD `condition` (encodes polarity × truth × claim × d); EXPE `arm@position`.
- **SRR / doubt / echo / corrective / targeted / derivational**: recomputed by `analyze()`
  (V.DOUBT regex etc.) — the identical definitions used in the published STRICT_METRICS_REPORT.md,
  NOT the stored per-row fields. (Consequence: the EXPE "deriv" column here is the strict-metric
  M4 derivational-use, which is a *different operationalization* from the stored
  `inj_derivational` used in the EXPE report; do not cross-read them.)

### H2' re-adjudication is a within-cohort DV swap
I import `expd_confirmatory_analysis.py` and call its own `load_rows`, `attr_cohort_maps`,
`annotate_cohort`, `cells_rows(..., cohort_only=True)`, `paired_family_stats`, and `tost`.
For the strict DV I monkeypatch `mval` to return `hop_sound_valid`; nothing else changes. The
mirrored cohort restricts to jointly-eligible pairs with **identical gold prefixes** (MOD-13), so
the hop-soundness re-derivation burden (which strips goal-jumping/shortcut completions) applies
**symmetrically** to the aff and neg members of each pair and largely cancels in the paired
difference — which is why H2' is well suited to the strict DV.

---

## Sanity anchors (all reproduce published numbers before trusting the adaptation)

**A1 — EXPD closure recompute.** `analyze()` recomputed the closure class for all **11,577**
EXPD rows; agreement with stored `valid_recovery` = **1.000000**. Per-cell recomputed
closure-valid rates match the published EXPD_FULL_REPORT.md exactly (aff_false_attr_d0 0.9222,
aff_false_attr_d3 0.9456, neg_false_attr_d1 0.9044, neg_false_attr_d2 0.8533, …).

**A2 — legacy strict.** Re-ran the authoritative strict metric on EXPA + legacy:

| family | hop-sound (mine) | published STRICT_METRICS_REPORT.md | closure (mine) | published |
|---|---|---|---|---|
| EXPA:one_hop_falsehood | 0.300 | 0.300 | 0.671 | 0.671 |
| EXPA:global_falsehood | 0.213 | 0.213 | 0.393 | 0.393 |
| EXPA:benign_paraphrase | 0.649 | 0.649 | 0.849 | 0.849 |
| legacy:negstep | 0.300 | 0.300 | 0.677 | 0.677 |
| legacy:falsehood | 0.095 | 0.095 | 0.286 | 0.286 |

**A3 — EXPE descriptives (bonus).** My EXPE stated-complement (primary rejection DV) and
closure-valid reproduce the published EXPE_FULL_REPORT.md exactly: REFUTING_d1@mid sc 0.114
(pub 0.1136), FREQ@mid sc 0.118 (pub 0.1182), TEMPLATE@mid sc 0.000, REFUTING_d1@mid closure
0.364 (pub 0.3636), FREQ@mid closure 0.350 (pub 0.35).

**H2' closure anchor.** Reproduced diff=+0.0542, aff=0.9351, neg=0.8783, CI [0.0249,0.0835],
p=0.001202, 163 paired families — identical to the registered adjudication.

---

## (c) EXPD strict-vs-permissive accounting table (positions pooled; family-clustered)

`cv` = closure-valid (paper DV); `hs` = hop-sound-valid; `Δpp` = cv−hs; `gjS` = goal-jump-strict
(valid via a real skipped chain); `SRR` = strict-repair rate over audited-false plants;
`rel` = stated-reliance (echo); `use/echo/der` = injection-use split; `sc` = stated-complement
(primary rejection DV, parse-independent, unaffected by hop-soundness); `db` = lexical doubt.

| cell | n | cv | hs | Δpp | gjS | SRR | rel | use | echo | der | sc | db |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 450 | 0.998 | 0.962 | +3.6 | 0.000 | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| true_interruption | 450 | 0.918 | 0.493 | +42.4 | 0.120 | n/a | 0.009 | 0.009 | 0.009 | 0.000 | 0.000 | 0.002 |
| aff_true_attr_d0 | 450 | 0.982 | 0.176 | +80.7 | 0.360 | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 |
| aff_true_attr_d1 | 450 | 0.904 | 0.127 | +77.8 | 0.258 | n/a | 0.027 | 0.029 | 0.029 | 0.000 | 0.000 | 0.002 |
| aff_true_attr_d3 | 447 | 0.940 | 0.072 | +86.8 | 0.378 | n/a | 0.058 | 0.094 | 0.094 | 0.000 | 0.000 | 0.002 |
| neg_true_attr_d0 | 450 | 0.980 | 0.164 | +81.6 | 0.429 | n/a | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| neg_true_attr_d1 | 450 | 0.989 | 0.227 | +76.2 | 0.271 | n/a | 0.024 | 0.024 | 0.024 | 0.000 | 0.000 | 0.007 |
| neg_true_attr_d3 | 450 | 0.927 | 0.171 | +75.6 | 0.318 | n/a | 0.042 | 0.098 | 0.098 | 0.000 | 0.000 | 0.002 |
| **aff_false_attr_d0** | 450 | 0.922 | **0.247** | +67.5 | 0.184 | 0.291 | 0.000 | 0.002 | 0.002 | 0.000 | **0.247** | 0.120 |
| aff_false_attr_d1 | 450 | 0.922 | 0.104 | +81.8 | 0.240 | 0.047 | 0.000 | 0.004 | 0.004 | 0.000 | 0.031 | 0.027 |
| aff_false_attr_d2 | 450 | 0.931 | 0.096 | +83.5 | 0.320 | 0.047 | 0.000 | 0.013 | 0.013 | 0.000 | 0.049 | 0.029 |
| aff_false_attr_d3 | 441 | 0.946 | 0.109 | +83.7 | 0.329 | 0.041 | 0.000 | 0.018 | 0.018 | 0.000 | 0.050 | 0.020 |
| aff_false_attr_d5 | 426 | 0.962 | 0.115 | +84.7 | 0.333 | 0.024 | 0.000 | 0.021 | 0.021 | 0.000 | 0.021 | 0.024 |
| neg_false_attr_d0 | 450 | 0.958 | 0.180 | +77.8 | 0.307 | 0.033 | 0.000 | 0.018 | 0.018 | 0.000 | 0.013 | 0.031 |
| neg_false_attr_d1 | 450 | 0.904 | 0.162 | +74.2 | 0.280 | 0.036 | 0.000 | 0.024 | 0.024 | 0.000 | 0.011 | 0.040 |
| neg_false_attr_d2 | 450 | 0.853 | 0.124 | +72.9 | 0.318 | 0.033 | 0.000 | 0.113 | 0.113 | 0.000 | 0.013 | 0.042 |
| neg_false_attr_d3 | 435 | 0.855 | 0.147 | +70.8 | 0.315 | 0.028 | 0.000 | 0.122 | 0.122 | 0.000 | 0.011 | 0.044 |
| neg_false_attr_d5 | 414 | 0.841 | 0.142 | +69.8 | 0.324 | 0.027 | 0.000 | 0.138 | 0.138 | 0.000 | 0.005 | 0.046 |
| **cat_false_usable_d1** | 300 | 0.093 | **0.000** | +9.3 | 0.003 | 0.027 | 0.000 | **0.863** | 0.000 | 0.863 | 0.023 | 0.020 |
| cat_false_usable_d3 | 280 | 0.204 | 0.011 | +19.3 | 0.039 | 0.000 | 0.000 | 0.771 | 0.004 | 0.771 | 0.014 | 0.004 |
| cat_false_usable_dinf | 300 | 0.200 | 0.010 | +19.0 | 0.033 | 0.007 | 0.000 | 0.787 | 0.000 | 0.787 | 0.003 | 0.007 |
| **cat_false_inert_d1** | 300 | 0.623 | 0.130 | +49.3 | 0.010 | 0.177 | 0.000 | 0.003 | 0.003 | 0.000 | 0.193 | 0.087 |
| cat_false_inert_d3 | 284 | 0.750 | 0.049 | +70.1 | 0.042 | 0.035 | 0.000 | 0.007 | 0.007 | 0.000 | 0.053 | 0.063 |
| cat_false_inert_dinf | 300 | 0.770 | 0.090 | +68.0 | 0.047 | 0.003 | 0.000 | 0.023 | 0.023 | 0.000 | 0.023 | 0.027 |

(full per-cell CIs — Wilson + family-cluster bootstrap — in `s1_strict_metrics.json`.)

### How to read the table
- Hop-soundness cuts every closure-valid rate hard (Δpp +67 to +87 in the attribute cells),
  because the plant overwrites a gold step and most "valid" completions reach the target by
  **goal-jumping / re-deriving through closure-level skips** rather than explicit stepwise
  derivation. `gjS` (goal-jump-strict) is large exactly where Δpp is large. This is the same
  structural burden the legacy report flagged (true_interruption 0.918→0.493 quantifies it with
  a TRUE plant) — it is uniform across cells, not a polarity/claim effect.
- The **primary rejection DV (`sc`, stated-complement) and the reuse split (`der`/`echo`) are
  parse-/validity-independent**, so hop-soundness leaves them untouched. The two flagship
  boundary conditions live on those axes and are therefore robust by construction:
  - **d=0 visibility**: aff_false_attr sc 0.247 @d0 vs 0.031 @d1 (the C-0 cliff) — unchanged.
  - **usability**: cat_false_usable der ≈ 0.86/0.77/0.79 vs cat_false_inert ≈ 0 — unchanged.

---

## (b) H2' re-adjudication — full disclosure

- **Cohort**: 163 mirrored paired families; ~832 aff rows, ~830 neg rows (identical to the
  registered test).
- **Closure (anchor)**: aff ~778/832 = 0.9351; neg ~729/830 = 0.8783; diff +0.0542;
  90% eq-CI [0.0296, 0.0788]; p_raw 0.001202 → PASS.
- **Hop-sound (strict)**: aff ~90/832 = 0.1082; neg ~127/830 = 0.1530; diff −0.0419;
  95% CI [−0.0716, −0.0122]; 90% eq-CI [−0.0668, −0.0171]; p_raw 8e-05 → **PASS at Holm α/6**.
- Family-cluster bootstrap 95%: aff hop-sound [0.0874, 0.1304]; neg hop-sound [0.1283, 0.1774].

**Near-floor caveat (disclose in the paper).** Under the strict DV both rates compress toward a
floor (≈0.11 and ≈0.15), so the ±0.10 equivalence margin is generous relative to the rates. Two
things keep the result substantive rather than a "both ≈ 0" artifact: (i) the counts are
non-trivial (90 and 127 hop-sound-valid completions), and (ii) the sign of the difference
**reverses** between DVs, so there is no consistent polarity advantage under either standard —
which is precisely the claim H2' is meant to support ("polarity is not the driver"). Recommended
paper wording: *"the polarity-equivalence conclusion is not an artifact of the permissive DV: it
replicates under a stepwise hop-sound validity standard (Δ = −4.2pp, 95% CI within ±10pp,
p = 8e-5), with neither polarity favored under either DV."*

---

## (d) Reversal audit — does strict flip any permissive conclusion?

**Registered EXPD verdicts — none reversed:**
- **H2' (PASS on closure_valid)** → still PASS on hop_sound_valid (above). No reversal; strengthened.
- **H4 reuse** (usable derivational reuse, exploratory decreasing trend) → `der` is not a
  validity metric; hop-soundness does not touch it. No reversal.
- **C-0** (d0 vs d1, non-family, no verdict) → stated-complement cliff unchanged. No verdict to reverse.

**Non-registered descriptive pattern changes the strict lens introduces (report as observations,
not verdict reversals):**
1. **aff_false d0 hop-sound validity (0.247) > d1 (0.104).** The registered C-0 found *no* d0–d1
   gap on `closure_valid` (diff −0.0024, p=0.89); the strict DV surfaces a clear d0>d1 gap.
   Interpretation: when the refuting evidence is directly visible (d0) the model's valid
   completions are also *hop-sound*; at d≥1 validity is increasingly carried by hop-unsound
   shortcutting. This **strengthens** the d=0-visibility story rather than contradicting anything.
2. **Within cat_false_inert, the closure-valid d-trend is increasing (0.623→0.750→0.770) but the
   hop-sound d-trend is flat/non-monotonic (0.130→0.049→0.090).** The closure-valid rise with
   distance is a shortcutting artifact (more room to goal-jump at larger d); it disappears under
   the strict DV. Not a registered claim; worth a one-line appendix note.

---

## EXPE strict pass (secondary; EXPE's registered DV is stated-complement, already strict)

EXPE cells (arm@position), positions and arms as in EXPE_FULL_REPORT.md. `sc` reproduces the
published primary DV exactly (anchor A3). Hop-sound validity is uniformly low (0.12–0.20) because
EXPE completions absorb the prefix-planted falsehood heavily (closure-valid itself is only
0.31–0.46). SRR over audited-false plants is small everywhere and highest in the REFUTING arms
(0.041–0.070) vs FREQ 0.018 / TEMPLATE 0.009 / FILLER ≤0.014 — consistent with the EXPE story
that *genuine* strict repair, when it happens at all, tracks the licensed refuting rule, while the
FREQ arm's surface "rejection" (stated-complement 0.118 ≈ REFUTING 0.114) is lexically cued and
does NOT convert into hop-sound repair. Full table in `s1_strict_metrics.json → EXPE_cells`.

Caveat: the EXPE `deriv`/`use` columns here are the strict-metric M4 derivational-use
(analyze(), taint-free twin state), a *different* operationalization from the stored
`inj_derivational` field the EXPE report/H6 use — do not cross-read the two.

---

## Caveats / limitations
1. **Qwen2.5-7B only** (same scope as the confirmatory family). Strict-DV replication across the
   E1 replication models is future work.
2. **Hop-soundness prices in re-deriving the overwritten gold step.** This inflates Δpp uniformly
   and is why absolute hop-sound rates are low; the paired H2' design cancels it, but absolute
   hop-sound cell rates should always be read as "explicit-stepwise" rates, not "the model failed."
3. **Near-floor equivalence** for the strict H2' (above) — disclose the margin-vs-rate ratio.
4. SRR/doubt use the lexical `V.DOUBT` detector (descriptive; backend-sensitive per house rules);
   the strict-repair conjunct also requires closure-valid AND not-goal-jump, so it is far
   stricter than doubt alone.

## Files produced (cluster, under my slug only)
- `improvement_plan/iclr_exec/s1_strict_hopsound/s1_hopsound.py`
- `improvement_plan/iclr_exec/s1_strict_hopsound/s1_strict_metrics.json`
- `improvement_plan/iclr_exec/s1_strict_hopsound/REPORT.md` (this file)
- `improvement_plan/iclr_exec/s1_strict_hopsound/probe.py` (anchor-A1 probe)
