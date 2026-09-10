# EXPD Matched-Gradient FULL CONFIRMATORY Report

Generated: 2026-07-02T21:36:58.248383+00:00
Backend: vllm | model: Qwen/Qwen2.5-7B-Instruct (a09a35458c702b33eeacc393d103063234e8bc28) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5) under the externally timestamped pre-registration.** Gate evidence: git commit `545b35db420d1826b69ca9bdd6e47b5720bb54de` (2026-07-02T14:05:29-07:00), PLAN2.md sha256 `112da1a56a16745181e6ed7d2dac3b1e7d0dd3e7643e78e2bfdf16ca572e2b2e`. Holm family m=6; the three EXPD slots are adjudicated here with raw p-values; final family-level Holm is assembled by the coordinator with the EXPE slots (H1', H5', H6).

## Registered adjudications (PLAN2 v1.1 section B)

| slot | test | estimate | 95% CI | p_raw | passes at alpha/6 (worst-case Holm) | passes at 0.05 |
|---|---|---|---|---|---|---|
| H2' | aff_false vs neg_false closure-valid TOST +/-0.10, d{1,3}, mirrored cohort | diff=0.0542 | [0.0249, 0.0835] (90% eq-CI [0.0296, 0.0788]) | 0.001202 | True | True |
| H3' | conjunction: (i) neg_true~aff_true valid TOST (d{0,1,3}); (ii) stated-comp(neg_false_d1)>stated-comp(neg_true_d1) | i: 0.0271, ii: 0.0093 | i: [0.0088, 0.0454], ii: [-0.0019, 0.0205] | 0.051333 (i: 0.0, ii: 0.051333) | False | False |
| H4 | derivational inj-dep logit trend d{1,3,inf} in cat_false_usable, increasing | slope=-0.2477 | [-0.4323, -0.0632] | 0.99574 | False | False |
| C-0 (stated_complement) | d0 vs d1 two-sided, NON-FAMILY, no verdict | diff=0.2086 | [0.1534, 0.2638] | 0.0 | n/a | n/a |
| C-0 (closure_valid) | d0 vs d1 two-sided, NON-FAMILY, no verdict | diff=-0.0024 | [-0.0377, 0.0329] | 0.893311 | n/a | n/a |

Pooled rates: H2' aff_false valid=0.9351 vs neg_false valid=0.8783 (paired families=163). H3' neg_true valid=0.9672 vs aff_true valid=0.9426; stated-comp neg_false_d1=0.0093 vs neg_true_d1=0.0. H4 derivational by d: {'1': 0.8633, '3': 0.7679, 'inf': 0.7867}. C-0 rates: {'stated_complement': {'d0': 0.2467, 'd1': 0.0311}, 'closure_valid': {'d0': 0.9222, 'd1': 0.9222}}.

## Pre-registered outcome-branch adjudication

- **local_refutability_CONFIRMED**
  - requires: H1'+H4+H5' pass AND H2'+H3' pass (v1.1 carry-over of section 5.1)
  - status: OFF regardless of EXPE results: H4 failed (trend significantly in the OPPOSITE direction) and H3' failed (component ii)
- **professors_confound_account_SUPPORTED**
  - requires: H2'/H3' fail with POLARITY DOMINATING, or H5' fails with FREQ ~= REFUTING (v1.1 section B)
  - status: NOT triggered by EXPD: H2' PASSED (aff/neg equivalent within +/-0.10; point difference +0.054 with aff_false slightly HIGHER closure-valid, the opposite of a negation-driven artifact), and H3'(i) TOST passed emphatically; H3' failed only because its component (ii) hit a FLOOR on the primary DV (stated-complement 0.009 vs 0.000 in attribute cells), not because neg >> aff. EXPE's H5' remains to be scored.
- **H4_reuse_gradient**
  - registered_reading_if_flat: 'reuse gradient was a construction artifact' (PLAN2 section 11)
  - observed: not flat: significantly DECREASING (slope -0.248 logit/step, one-sided-decreasing p ~= 0.004 exploratory); derivational reuse is HIGHEST at d=1 (0.863) where the refuting rule is adjacent - local counterevidence does not suppress reuse in this grammar; if anything the nearest-counterevidence worlds absorb MORE
- **H4 reverse-direction (exploratory):** one-sided-decreasing p = 0.004259. NOT a registered test (direction was registered as increasing); reported so the observed significant DECREASE (reuse highest at d=1) is visible, not just the failure of the registered direction

## Headline per-cell rates (positions pooled; family-cluster bootstrap 95% CIs)

| cell | n | parse | closure-valid | stated-comp | deriv inj-dep | echo | doubt(lex, descriptive) | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 450 | 0.9889 | 0.9222 [0.8956, 0.9467] | 0.2467 [0.1978, 0.2933] | 0.0022 [0.0, 0.0067] | 0.0022 | 0.12 [0.0911, 0.1489] | 0.0467 | 0.0178 | 0.0111 |
| aff_false_attr_d1 | 450 | 0.9911 | 0.9222 [0.8956, 0.9467] | 0.0311 [0.0156, 0.0511] | 0.0044 [0.0, 0.0111] | 0.0044 | 0.0267 [0.0111, 0.0467] | 0.0578 | 0.0067 | 0.0089 |
| aff_false_attr_d2 | 450 | 0.9867 | 0.9311 [0.9044, 0.9556] | 0.0489 [0.0311, 0.0689] | 0.0111 [0.0022, 0.0222] | 0.0133 | 0.0289 [0.0156, 0.0444] | 0.0244 | 0.02 | 0.0133 |
| aff_false_attr_d3 | 441 | 0.9864 | 0.9456 [0.9252, 0.9637] | 0.0499 [0.0317, 0.0703] | 0.0091 [0.0023, 0.0181] | 0.0181 | 0.0204 [0.0091, 0.0363] | 0.0068 | 0.0249 | 0.0136 |
| aff_false_attr_d5 | 426 | 0.9977 | 0.9624 [0.9413, 0.9789] | 0.0211 [0.0094, 0.0352] | 0.0164 [0.0047, 0.0305] | 0.0211 | 0.0235 [0.0094, 0.0376] | 0.0023 | 0.0164 | 0.0023 |
| aff_true_attr_d0 | 450 | 1.0 | 0.9822 [0.9689, 0.9933] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0022 [0.0, 0.0067] | 0.0067 | 0.0111 | 0.0 |
| aff_true_attr_d1 | 450 | 0.9978 | 0.9044 [0.8778, 0.9311] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0289 | 0.0022 [0.0, 0.0067] | 0.0444 | 0.0489 | 0.0022 |
| aff_true_attr_d2 | 450 | 0.9978 | 0.9022 [0.8756, 0.9289] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.1022 | 0.0111 [0.0022, 0.0222] | 0.0467 | 0.0489 | 0.0022 |
| aff_true_attr_d3 | 447 | 1.0 | 0.9396 [0.9172, 0.962] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.094 | 0.0022 [0.0, 0.0067] | 0.0157 | 0.0447 | 0.0 |
| aff_true_attr_d5 | 450 | 0.9978 | 0.9333 [0.9089, 0.9533] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.1067 | 0.0067 [0.0, 0.0156] | 0.0111 | 0.0533 | 0.0022 |
| benign_paraphrase | 450 | 1.0 | 0.9978 [0.9929, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 [0.0, 0.0] | 0.0 | 0.0022 | 0.0 |
| cat_false_inert_d1 | 300 | 0.9467 | 0.6233 [0.5667, 0.6767] | 0.1933 [0.15, 0.2367] | 0.0 [0.0, 0.0] | 0.0033 | 0.0867 [0.05, 0.1233] | 0.3 | 0.0233 | 0.0533 |
| cat_false_inert_d3 | 284 | 0.9789 | 0.75 [0.7007, 0.7958] | 0.0528 [0.0282, 0.081] | 0.0 [0.0, 0.0] | 0.007 | 0.0634 [0.0387, 0.0915] | 0.1761 | 0.0528 | 0.0211 |
| cat_false_inert_dinf | 300 | 0.98 | 0.77 [0.7233, 0.8167] | 0.0233 [0.0067, 0.0433] | 0.0233 [0.0067, 0.04] | 0.0233 | 0.0267 [0.0067, 0.0467] | 0.1467 | 0.04 | 0.02 |
| cat_false_usable_d1 | 300 | 0.9933 | 0.0933 [0.06, 0.13] | 0.0233 [0.01, 0.0433] | 0.8633 [0.82, 0.9033] | 0.0 | 0.02 [0.0067, 0.0367] | 0.0367 | 0.0 | 0.0067 |
| cat_false_usable_d3 | 280 | 1.0 | 0.2036 [0.15, 0.2571] | 0.0143 [0.0036, 0.0286] | 0.7679 [0.7107, 0.8179] | 0.0036 | 0.0036 [0.0, 0.0107] | 0.025 | 0.0036 | 0.0 |
| cat_false_usable_dinf | 300 | 1.0 | 0.2 [0.1533, 0.25] | 0.0033 [0.0, 0.01] | 0.7867 [0.7367, 0.83] | 0.0 | 0.0067 [0.0, 0.0167] | 0.01 | 0.0033 | 0.0 |
| neg_false_attr_d0 | 450 | 0.9889 | 0.9578 [0.9356, 0.9756] | 0.0133 [0.0, 0.0311] | 0.0178 [0.0067, 0.0311] | 0.0178 | 0.0311 [0.0156, 0.0467] | 0.0067 | 0.0067 | 0.0111 |
| neg_false_attr_d1 | 450 | 0.9956 | 0.9044 [0.8756, 0.9311] | 0.0111 [0.0022, 0.0244] | 0.0222 [0.0089, 0.0378] | 0.0244 | 0.04 [0.0178, 0.0644] | 0.0489 | 0.02 | 0.0044 |
| neg_false_attr_d2 | 450 | 0.9956 | 0.8533 [0.8244, 0.8822] | 0.0133 [0.0044, 0.0244] | 0.1 [0.0756, 0.1244] | 0.1133 | 0.0422 [0.0244, 0.06] | 0.02 | 0.0222 | 0.0044 |
| neg_false_attr_d3 | 435 | 0.9931 | 0.8552 [0.823, 0.8851] | 0.0115 [0.0023, 0.023] | 0.0943 [0.069, 0.1241] | 0.1218 | 0.0437 [0.0276, 0.0644] | 0.0092 | 0.0345 | 0.0069 |
| neg_false_attr_d5 | 414 | 0.9903 | 0.8406 [0.8092, 0.8696] | 0.0048 [0.0, 0.0121] | 0.0821 [0.0556, 0.1111] | 0.1377 | 0.0459 [0.0266, 0.0676] | 0.0121 | 0.0556 | 0.0097 |
| neg_true_attr_d0 | 450 | 1.0 | 0.98 [0.9667, 0.9933] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 [0.0, 0.0] | 0.0089 | 0.0111 | 0.0 |
| neg_true_attr_d1 | 450 | 1.0 | 0.9889 [0.9778, 0.9978] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0244 | 0.0067 [0.0, 0.0156] | 0.0022 | 0.0089 | 0.0 |
| neg_true_attr_d2 | 450 | 1.0 | 0.9267 [0.9044, 0.9489] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0867 | 0.0022 [0.0, 0.0067] | 0.0422 | 0.0311 | 0.0 |
| neg_true_attr_d3 | 450 | 0.9911 | 0.9267 [0.9044, 0.9489] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0978 | 0.0022 [0.0, 0.0067] | 0.0244 | 0.04 | 0.0089 |
| neg_true_attr_d5 | 450 | 0.9978 | 0.9356 [0.9133, 0.9556] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.1089 | 0.0067 [0.0, 0.0156] | 0.0067 | 0.0556 | 0.0022 |
| true_interruption | 450 | 1.0 | 0.9178 [0.879, 0.9539] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0089 | 0.0022 [0.0, 0.007] | 0.0511 | 0.0311 | 0.0 |

## Per-position rates (appendix)

| cell@position | n | closure-valid | stated-comp | deriv inj-dep | doubt(lex) | unparsed |
|---|---|---|---|---|---|---|
| aff_false_attr_d0@early | 150 | 0.8667 | 0.2133 | 0.0067 | 0.1067 | 0.0 |
| aff_false_attr_d0@late | 150 | 0.9267 | 0.36 | 0.0 | 0.1 | 0.0333 |
| aff_false_attr_d0@mid | 150 | 0.9733 | 0.1667 | 0.0 | 0.1533 | 0.0 |
| aff_false_attr_d1@early | 150 | 0.8267 | 0.0333 | 0.0133 | 0.02 | 0.0133 |
| aff_false_attr_d1@late | 150 | 0.9533 | 0.0533 | 0.0 | 0.04 | 0.0133 |
| aff_false_attr_d1@mid | 150 | 0.9867 | 0.0067 | 0.0 | 0.02 | 0.0 |
| aff_false_attr_d2@early | 150 | 0.86 | 0.0933 | 0.0333 | 0.0667 | 0.04 |
| aff_false_attr_d2@late | 150 | 0.9467 | 0.0533 | 0.0 | 0.0133 | 0.0 |
| aff_false_attr_d2@mid | 150 | 0.9867 | 0.0 | 0.0 | 0.0067 | 0.0 |
| aff_false_attr_d3@early | 147 | 0.8435 | 0.1088 | 0.0272 | 0.034 | 0.034 |
| aff_false_attr_d3@late | 147 | 0.9932 | 0.0408 | 0.0 | 0.0204 | 0.0068 |
| aff_false_attr_d3@mid | 147 | 1.0 | 0.0 | 0.0 | 0.0068 | 0.0 |
| aff_false_attr_d5@early | 142 | 0.8944 | 0.0423 | 0.0423 | 0.0634 | 0.007 |
| aff_false_attr_d5@late | 142 | 0.993 | 0.0211 | 0.007 | 0.007 | 0.0 |
| aff_false_attr_d5@mid | 142 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d0@early | 150 | 0.96 | 0.0 | 0.0 | 0.0067 | 0.0 |
| aff_true_attr_d0@late | 150 | 0.9933 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d0@mid | 150 | 0.9933 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d1@early | 150 | 0.78 | 0.0 | 0.0 | 0.0 | 0.0067 |
| aff_true_attr_d1@late | 150 | 0.98 | 0.0 | 0.0 | 0.0067 | 0.0 |
| aff_true_attr_d1@mid | 150 | 0.9533 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d2@early | 150 | 0.78 | 0.0 | 0.0 | 0.0333 | 0.0067 |
| aff_true_attr_d2@late | 150 | 0.9533 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d2@mid | 150 | 0.9733 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d3@early | 149 | 0.8389 | 0.0 | 0.0 | 0.0067 | 0.0 |
| aff_true_attr_d3@late | 149 | 0.9866 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d3@mid | 149 | 0.9933 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d5@early | 150 | 0.8333 | 0.0 | 0.0 | 0.02 | 0.0067 |
| aff_true_attr_d5@late | 150 | 0.9867 | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_true_attr_d5@mid | 150 | 0.98 | 0.0 | 0.0 | 0.0 | 0.0 |
| benign_paraphrase@early | 150 | 0.9933 | 0.0 | 0.0 | 0.0 | 0.0 |
| benign_paraphrase@late | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| benign_paraphrase@mid | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| cat_false_inert_d1@early | 150 | 0.42 | 0.3533 | 0.0 | 0.0533 | 0.0267 |
| cat_false_inert_d1@mid | 150 | 0.8267 | 0.0333 | 0.0 | 0.12 | 0.08 |
| cat_false_inert_d3@early | 142 | 0.5423 | 0.1056 | 0.0 | 0.1056 | 0.0352 |
| cat_false_inert_d3@mid | 142 | 0.9577 | 0.0 | 0.0 | 0.0211 | 0.007 |
| cat_false_inert_dinf@early | 150 | 0.62 | 0.04 | 0.0267 | 0.0333 | 0.0333 |
| cat_false_inert_dinf@mid | 150 | 0.92 | 0.0067 | 0.02 | 0.02 | 0.0067 |
| cat_false_usable_d1@early | 150 | 0.04 | 0.0333 | 0.9133 | 0.0133 | 0.0133 |
| cat_false_usable_d1@mid | 150 | 0.1467 | 0.0133 | 0.8133 | 0.0267 | 0.0 |
| cat_false_usable_d3@early | 140 | 0.1714 | 0.0143 | 0.7929 | 0.0071 | 0.0 |
| cat_false_usable_d3@mid | 140 | 0.2357 | 0.0143 | 0.7429 | 0.0 | 0.0 |
| cat_false_usable_dinf@early | 150 | 0.1533 | 0.0067 | 0.8267 | 0.0067 | 0.0 |
| cat_false_usable_dinf@mid | 150 | 0.2467 | 0.0 | 0.7467 | 0.0067 | 0.0 |
| neg_false_attr_d0@early | 150 | 0.94 | 0.0133 | 0.0333 | 0.0267 | 0.0 |
| neg_false_attr_d0@late | 150 | 0.9467 | 0.02 | 0.0067 | 0.0333 | 0.0333 |
| neg_false_attr_d0@mid | 150 | 0.9867 | 0.0067 | 0.0133 | 0.0333 | 0.0 |
| neg_false_attr_d1@early | 150 | 0.82 | 0.0133 | 0.0467 | 0.0867 | 0.0067 |
| neg_false_attr_d1@late | 150 | 0.9067 | 0.02 | 0.0133 | 0.0067 | 0.0067 |
| neg_false_attr_d1@mid | 150 | 0.9867 | 0.0 | 0.0067 | 0.0267 | 0.0 |
| neg_false_attr_d2@early | 150 | 0.6267 | 0.0333 | 0.26 | 0.0467 | 0.0133 |
| neg_false_attr_d2@late | 150 | 0.9467 | 0.0067 | 0.0333 | 0.0133 | 0.0 |
| neg_false_attr_d2@mid | 150 | 0.9867 | 0.0 | 0.0067 | 0.0667 | 0.0 |
| neg_false_attr_d3@early | 145 | 0.6276 | 0.0345 | 0.2414 | 0.0828 | 0.0207 |
| neg_false_attr_d3@late | 145 | 0.9655 | 0.0 | 0.0138 | 0.0069 | 0.0 |
| neg_false_attr_d3@mid | 145 | 0.9724 | 0.0 | 0.0276 | 0.0414 | 0.0 |
| neg_false_attr_d5@early | 138 | 0.6232 | 0.0145 | 0.1884 | 0.0652 | 0.0217 |
| neg_false_attr_d5@late | 138 | 0.9638 | 0.0 | 0.0145 | 0.0145 | 0.0 |
| neg_false_attr_d5@mid | 138 | 0.9348 | 0.0 | 0.0435 | 0.058 | 0.0072 |
| neg_true_attr_d0@early | 150 | 0.9467 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d0@late | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d0@mid | 150 | 0.9933 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d1@early | 150 | 0.9667 | 0.0 | 0.0 | 0.0133 | 0.0 |
| neg_true_attr_d1@late | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d1@mid | 150 | 1.0 | 0.0 | 0.0 | 0.0067 | 0.0 |
| neg_true_attr_d2@early | 150 | 0.7867 | 0.0 | 0.0 | 0.0067 | 0.0 |
| neg_true_attr_d2@late | 150 | 0.9933 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d2@mid | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d3@early | 150 | 0.78 | 0.0 | 0.0 | 0.0067 | 0.0267 |
| neg_true_attr_d3@late | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d3@mid | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d5@early | 150 | 0.82 | 0.0 | 0.0 | 0.02 | 0.0067 |
| neg_true_attr_d5@late | 150 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_true_attr_d5@mid | 150 | 0.9867 | 0.0 | 0.0 | 0.0 | 0.0 |
| true_interruption@early | 150 | 0.9267 | 0.0 | 0.0 | 0.0 | 0.0 |
| true_interruption@late | 150 | 0.9 | 0.0 | 0.0 | 0.0 | 0.0 |
| true_interruption@mid | 150 | 0.9267 | 0.0 | 0.0 | 0.0067 | 0.0 |

## Manipulation checks

- **MC-1 parse matching across contrast cells (gaps):** {"H2prime_aff_vs_neg_false": 0.0056, "H3prime_i_negtrue_vs_afftrue": 0.0022, "H3prime_ii_negfalse_d1_vs_negtrue_d1": 0.0044, "H4_within_usable_max_gap": 0.006700000000000039, "C0_d0_vs_d1": 0.0022}
- **MC-7 identical gold prefix (jointly-eligible mirrored pairs):** overall {"identical": 2116, "checked": 2265, "identical_rate": 0.9342}; by d {"d0": {"identical": 418, "n": 456, "rate": 0.9167}, "d1": {"identical": 462, "n": 480, "rate": 0.9625}, "d2": {"identical": 431, "n": 465, "rate": 0.9269}, "d3": {"identical": 405, "n": 429, "rate": 0.9441}, "d5": {"identical": 400, "n": 435, "rate": 0.9195}}; by position {"early": {"identical": 720, "n": 755, "rate": 0.9536}, "mid": {"identical": 700, "n": 755, "rate": 0.9272}, "late": {"identical": 696, "n": 755, "rate": 0.9219}}
- **Joint-solve mirrored pairs by d:** {"d0": {"joint_eligible": 152, "pairs": 180, "rate": 0.8444}, "d1": {"joint_eligible": 160, "pairs": 180, "rate": 0.8889}, "d2": {"joint_eligible": 155, "pairs": 180, "rate": 0.8611}, "d3": {"joint_eligible": 143, "pairs": 180, "rate": 0.7944}, "d5": {"joint_eligible": 145, "pairs": 180, "rate": 0.8056}}
- **MC-2 anchor benign_paraphrase:** valid TOST {"ci90": [0.1209, 0.1769], "diff": 0.1489, "margin": 0.1, "tost_pass": false}; unparsed TOST {"ci90": [-0.0364, -0.0124], "diff": -0.0244, "margin": 0.1, "tost_pass": true}; doubt diff vs locked-HF -0.08 (backend-confounded, no verdict; gate scoped to valid/unparsed per v1.1 C.2, MOD-12: never gates H2'-H4)
- **MC-2 anchor true_interruption:** valid TOST {"ci90": [0.0276, 0.0968], "diff": 0.0622, "margin": 0.1, "tost_pass": true}; unparsed TOST {"ci90": [-0.0222, -0.0044], "diff": -0.0133, "margin": 0.1, "tost_pass": true}; doubt diff vs locked-HF -0.0178 (backend-confounded, no verdict; gate scoped to valid/unparsed per v1.1 C.2, MOD-12: never gates H2'-H4)

## Funnel (CONSORT, MC-4)

```
{
  "gold_attempted": 2880,
  "gold_eligible": 2479,
  "gold_solved": 2693,
  "gold_valid_rederivation": 2583,
  "injection_audit_rejections": {
    "gold_ineligible:gold_not_solved": 331,
    "gold_ineligible:gold_not_validator_valid": 162,
    "gold_ineligible:too_few_intermediate_entity_steps": 134,
    "measured_d_mismatch": 57,
    "t0_on_gold_path": 216
  },
  "manifest_rows": 11577,
  "validated_rows": 11577,
  "world_audit_failures": 0,
  "worlds_generated": 2880
}
```

World audit: 2880 worlds / 360 families generated, 0 audit failures.

## Integrity

```
{
  "all_anchor_cells_audited_true": true,
  "all_measured_d_match": true,
  "all_planted_false_cells_audited_false": true,
  "duplicate_run_id_count": 0,
  "no_attr_inf_cells": true,
  "prompt_hashes": [
    "f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c"
  ],
  "unique_run_ids": true
}
```

## Analysis decisions & disclosures

- **cat_positions:** ['early', 'mid']
- **cluster_unit:** family_id (mirrored-quadruple/base-chain id) per PLAN2 section 9.1
- **cohort:** H2'/H3' rows conditioned on jointly-eligible mirrored pairs with identical gold prefixes up to si at the row's position (MOD-13); H4 uses all eligible usable-cell rows (the d-trend does not cross the usable/inert mirror); C-0 uses all eligible rows
- **h3i_pooling:** v1.1 B is silent on pooling for H3'(i); the superseded v1.0 H3 slot specified d in {0,1,3}; adjudicated on d in {0,1,3} pooled over all 3 positions (grid gives the true cells 3 positions in v1.1 C.5); mid-only and all-d variants reported as robustness
- **stated_complement:** strict string-level: any continuation sentence whose parse_fact equals (entity, opposite_pred(planted)) - the expe_evidence_mover convention generalized to attributes
- **Generator sizing (disclosed):** FULL_CONFIG runs 180 families/kind so that every cell x d reaches the registered n=150/cell/position after the eligibility filter (each family yields exactly one world per cell x d); the nonce-noun pool was extended with cvcv-prefix forms for capacity (globally unique nouns, audited invariants unchanged). Neither changes any registered quantity.
- **Judge pass:** the Qwen2.5-32B judge is the pre-registered SECONDARY detector (upper bound); it was not run in this pass. The registered PRIMARY rejection DV (strict stated-complement) and all structural DVs are complete. Verbalized doubt is reported descriptively from the lexical detector only (v1.1 A.3: doubt point-rates are backend-sensitive and unregistered).
- **Backend rule (v1.1 C.4):** wholly vLLM 0.24.0, greedy, max_new=192, model revision pinned; no cross-backend row comparisons.

## Stage timings / GPU

```
{
  "generate_passA_attr_anchors_seconds": 262,
  "generate_passB_cat_seconds": 65,
  "gpu": "1x NVIDIA H200 (CUDA_VISIBLE_DEVICES=6)",
  "gpu_stage_total_seconds": 327,
  "note": "prepare (world gen, CPU) ran separately; engine load included in each pass",
  "summarize_seconds": 24,
  "validate_seconds": 2
}
```

Machine-readable verdicts: `hypothesis_verdicts.json`; full analysis document: `confirmatory_analysis.json`; runner mechanical stages: `EXPD_RUNNER_STAGES_REPORT.md`; per-cell summaries: `summary_tables.json`.
