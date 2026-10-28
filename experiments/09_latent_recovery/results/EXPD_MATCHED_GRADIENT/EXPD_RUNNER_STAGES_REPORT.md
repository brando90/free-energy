# EXPD Matched-Gradient FULL RUN (runner stages) Report

Generated: 2026-07-02T21:32:22.332579+00:00
Backend: vllm | model: Qwen/Qwen2.5-7B-Instruct (a09a35458c702b33eeacc393d103063234e8bc28) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the
externally timestamped PLAN2.** Timestamp-gate evidence: git commit `545b35db420d1826b69ca9bdd6e47b5720bb54de`
(2026-07-02T14:05:29-07:00), PLAN2.md sha256 `112da1a56a16745181e6ed7d2dac3b1e7d0dd3e7643e78e2bfdf16ca572e2b2e`.
This file reports the runner's mechanical stages only; the registered adjudications
(H2', H3', H4, C-0) live in EXPD_FULL_REPORT.md + hypothesis_verdicts.json.

## Gate table (PLAN2 section 6)

| gate | value | target | pass |
|---|---|---|---|
| gold greedy solve rate | 0.9351 (n=2880, wilson [0.9255, 0.9435]) | >=0.60 | True |
| gold double filter (solved AND valid AND >=3 sites) | 0.8608 | - | - |
| parse rate: aff_false_attr_d0 | 0.9889 (unparsed n=5/450) | >=0.90 | True |
| parse rate: aff_false_attr_d1 | 0.9911 (unparsed n=4/450) | >=0.90 | True |
| parse rate: aff_false_attr_d2 | 0.9867 (unparsed n=6/450) | >=0.90 | True |
| parse rate: aff_false_attr_d3 | 0.9864 (unparsed n=6/441) | >=0.90 | True |
| parse rate: aff_false_attr_d5 | 0.9977 (unparsed n=1/426) | >=0.90 | True |
| parse rate: aff_true_attr_d0 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: aff_true_attr_d1 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: aff_true_attr_d2 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: aff_true_attr_d3 | 1.0 (unparsed n=0/447) | >=0.90 | True |
| parse rate: aff_true_attr_d5 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: benign_paraphrase | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: cat_false_inert_d1 | 0.9467 (unparsed n=16/300) | >=0.90 | True |
| parse rate: cat_false_inert_d3 | 0.9789 (unparsed n=6/284) | >=0.90 | True |
| parse rate: cat_false_inert_dinf | 0.98 (unparsed n=6/300) | >=0.90 | True |
| parse rate: cat_false_usable_d1 | 0.9933 (unparsed n=2/300) | >=0.90 | True |
| parse rate: cat_false_usable_d3 | 1.0 (unparsed n=0/280) | >=0.90 | True |
| parse rate: cat_false_usable_dinf | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: neg_false_attr_d0 | 0.9889 (unparsed n=5/450) | >=0.90 | True |
| parse rate: neg_false_attr_d1 | 0.9956 (unparsed n=2/450) | >=0.90 | True |
| parse rate: neg_false_attr_d2 | 0.9956 (unparsed n=2/450) | >=0.90 | True |
| parse rate: neg_false_attr_d3 | 0.9931 (unparsed n=3/435) | >=0.90 | True |
| parse rate: neg_false_attr_d5 | 0.9903 (unparsed n=4/414) | >=0.90 | True |
| parse rate: neg_true_attr_d0 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: neg_true_attr_d1 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: neg_true_attr_d2 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: neg_true_attr_d3 | 0.9911 (unparsed n=4/450) | >=0.90 | True |
| parse rate: neg_true_attr_d5 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: true_interruption | 1.0 (unparsed n=0/450) | >=0.90 | True |
| anchor valid TOST(+-0.10): benign_paraphrase | diff=0.1489 ci90=[0.1209, 0.1769] | in margin | False |
| anchor doubt: benign_paraphrase | pilot=0.0 locked-HF=0.08 diff=-0.08 | n/a | backend-confounded, no verdict |
| anchor valid TOST(+-0.10): true_interruption | diff=0.0622 ci90=[0.0276, 0.0968] | in margin | True |
| anchor doubt: true_interruption | pilot=0.0022 locked-HF=0.02 diff=-0.0178 | n/a | backend-confounded, no verdict |
| joint-solve mirrored pairs d0 | 0.8444 (152/180) | - | - |
| joint-solve mirrored pairs d1 | 0.8889 (160/180) | - | - |
| joint-solve mirrored pairs d2 | 0.8611 (155/180) | - | - |
| joint-solve mirrored pairs d3 | 0.7944 (143/180) | - | - |
| joint-solve mirrored pairs d5 | 0.8056 (145/180) | - | - |
| MC-7 identical gold prefix at injection (jointly-solved pairs) | 0.9272 (700/755) | - | - |

## Headline per-cell rates (family-cluster bootstrap 95% CIs)

| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 450 | 0.9222 [0.8956, 0.9467] | 0.12 [0.0911, 0.1489] | 0.0022 [0.0, 0.0067] | 0.0022 | 0.0467 | 0.0178 | 0.0111 |
| aff_false_attr_d1 | 450 | 0.9222 [0.8956, 0.9467] | 0.0267 [0.0111, 0.0467] | 0.0044 [0.0, 0.0111] | 0.0044 | 0.0578 | 0.0067 | 0.0089 |
| aff_false_attr_d2 | 450 | 0.9311 [0.9044, 0.9556] | 0.0289 [0.0156, 0.0444] | 0.0111 [0.0022, 0.0222] | 0.0133 | 0.0244 | 0.02 | 0.0133 |
| aff_false_attr_d3 | 441 | 0.9456 [0.9252, 0.9637] | 0.0204 [0.0091, 0.0363] | 0.0091 [0.0023, 0.0181] | 0.0181 | 0.0068 | 0.0249 | 0.0136 |
| aff_false_attr_d5 | 426 | 0.9624 [0.9413, 0.9789] | 0.0235 [0.0094, 0.0376] | 0.0164 [0.0047, 0.0305] | 0.0211 | 0.0023 | 0.0164 | 0.0023 |
| aff_true_attr_d0 | 450 | 0.9822 [0.9689, 0.9933] | 0.0022 [0.0, 0.0067] | 0.0 [0.0, 0.0] | 0.0 | 0.0067 | 0.0111 | 0.0 |
| aff_true_attr_d1 | 450 | 0.9044 [0.8778, 0.9311] | 0.0022 [0.0, 0.0067] | 0.0 [0.0, 0.0] | 0.0289 | 0.0444 | 0.0489 | 0.0022 |
| aff_true_attr_d2 | 450 | 0.9022 [0.8756, 0.9289] | 0.0111 [0.0022, 0.0222] | 0.0 [0.0, 0.0] | 0.1022 | 0.0467 | 0.0489 | 0.0022 |
| aff_true_attr_d3 | 447 | 0.9396 [0.9172, 0.962] | 0.0022 [0.0, 0.0067] | 0.0 [0.0, 0.0] | 0.094 | 0.0157 | 0.0447 | 0.0 |
| aff_true_attr_d5 | 450 | 0.9333 [0.9089, 0.9533] | 0.0067 [0.0, 0.0156] | 0.0 [0.0, 0.0] | 0.1067 | 0.0111 | 0.0533 | 0.0022 |
| benign_paraphrase | 450 | 0.9978 [0.9929, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0022 | 0.0 |
| cat_false_inert_d1 | 300 | 0.6233 [0.5667, 0.6767] | 0.0867 [0.05, 0.1233] | 0.0 [0.0, 0.0] | 0.0033 | 0.3 | 0.0233 | 0.0533 |
| cat_false_inert_d3 | 284 | 0.75 [0.7007, 0.7958] | 0.0634 [0.0387, 0.0915] | 0.0 [0.0, 0.0] | 0.007 | 0.1761 | 0.0528 | 0.0211 |
| cat_false_inert_dinf | 300 | 0.77 [0.7233, 0.8167] | 0.0267 [0.0067, 0.0467] | 0.0233 [0.0067, 0.04] | 0.0233 | 0.1467 | 0.04 | 0.02 |
| cat_false_usable_d1 | 300 | 0.0933 [0.06, 0.13] | 0.02 [0.0067, 0.0367] | 0.8633 [0.82, 0.9033] | 0.0 | 0.0367 | 0.0 | 0.0067 |
| cat_false_usable_d3 | 280 | 0.2036 [0.15, 0.2571] | 0.0036 [0.0, 0.0107] | 0.7679 [0.7107, 0.8179] | 0.0036 | 0.025 | 0.0036 | 0.0 |
| cat_false_usable_dinf | 300 | 0.2 [0.1533, 0.25] | 0.0067 [0.0, 0.0167] | 0.7867 [0.7367, 0.83] | 0.0 | 0.01 | 0.0033 | 0.0 |
| neg_false_attr_d0 | 450 | 0.9578 [0.9356, 0.9756] | 0.0311 [0.0156, 0.0467] | 0.0178 [0.0067, 0.0311] | 0.0178 | 0.0067 | 0.0067 | 0.0111 |
| neg_false_attr_d1 | 450 | 0.9044 [0.8756, 0.9311] | 0.04 [0.0178, 0.0644] | 0.0222 [0.0089, 0.0378] | 0.0244 | 0.0489 | 0.02 | 0.0044 |
| neg_false_attr_d2 | 450 | 0.8533 [0.8244, 0.8822] | 0.0422 [0.0244, 0.06] | 0.1 [0.0756, 0.1244] | 0.1133 | 0.02 | 0.0222 | 0.0044 |
| neg_false_attr_d3 | 435 | 0.8552 [0.823, 0.8851] | 0.0437 [0.0276, 0.0644] | 0.0943 [0.069, 0.1241] | 0.1218 | 0.0092 | 0.0345 | 0.0069 |
| neg_false_attr_d5 | 414 | 0.8406 [0.8092, 0.8696] | 0.0459 [0.0266, 0.0676] | 0.0821 [0.0556, 0.1111] | 0.1377 | 0.0121 | 0.0556 | 0.0097 |
| neg_true_attr_d0 | 450 | 0.98 [0.9667, 0.9933] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0089 | 0.0111 | 0.0 |
| neg_true_attr_d1 | 450 | 0.9889 [0.9778, 0.9978] | 0.0067 [0.0, 0.0156] | 0.0 [0.0, 0.0] | 0.0244 | 0.0022 | 0.0089 | 0.0 |
| neg_true_attr_d2 | 450 | 0.9267 [0.9044, 0.9489] | 0.0022 [0.0, 0.0067] | 0.0 [0.0, 0.0] | 0.0867 | 0.0422 | 0.0311 | 0.0 |
| neg_true_attr_d3 | 450 | 0.9267 [0.9044, 0.9489] | 0.0022 [0.0, 0.0067] | 0.0 [0.0, 0.0] | 0.0978 | 0.0244 | 0.04 | 0.0089 |
| neg_true_attr_d5 | 450 | 0.9356 [0.9133, 0.9556] | 0.0067 [0.0, 0.0156] | 0.0 [0.0, 0.0] | 0.1089 | 0.0067 | 0.0556 | 0.0022 |
| true_interruption | 450 | 0.9178 [0.879, 0.9539] | 0.0022 [0.0, 0.007] | 0.0 [0.0, 0.0] | 0.0089 | 0.0511 | 0.0311 | 0.0 |

## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)

- aff_false_attr_d1: doubt=0.0267 [0.0111, 0.0467] (n=450, valid=0.9222)
- aff_false_attr_d3: doubt=0.0204 [0.0091, 0.0363] (n=441, valid=0.9456)
- aff_false_attr_d5: doubt=0.0235 [0.0094, 0.0376] (n=426, valid=0.9624)

## MAJOR-2 reuse liveness (H4 MDE input)

- cat_false_inert_d1: derivational=0.0 [0.0, 0.0] echo=0.0033 doubt=0.0867 valid=0.6233 (n=300)
- cat_false_usable_d1: derivational=0.8633 [0.82, 0.9033] echo=0.0 doubt=0.02 valid=0.0933 (n=300)

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

## Disclosures

- Backend confound (measured, improvement_plan/tooling/replay_full/REPLAY_FULL_REPORT.md): doubt absolute rates drop 8-15pp under vLLM vs locked HF numbers while structural metrics (valid_recovery, poisoning, unparsed) are cell-level robust (<=5pp deltas). The doubt-anchor comparison vs EXPA locked HF levels is reported WITHOUT a pass/fail verdict; a vLLM-basis EXPA anchor replay is the pending fix (owned upstream). The anchor TOST gate is evaluated on valid_recovery and unparsed only. Do not fail the pilot go/no-go on a doubt-anchor gap alone.
- Planted-token frequency: ==1 in attribute and inert-categorical cells; the usable categorical cell is structurally 2 (refuting head + MAJOR-2 usable-branch body), held constant across d. See worlds/world_audit.json design_notes.
- cat d=inf worlds (not in pilot cells) carry the refuting-shaped head on an entity-disconnected body: complement underivable from the entity state (canonical PLAN2 section-3 d=inf), lexically matched across d.

## Sample continuations

- `aff_false_attr_d0` [parroted] inj=`Tobias is dusky.` -> Tobias is not dusky. Tobias is not a lotopus. This leads to a contradiction, but the chain of deductions still holds up to the point of being a lotopus. Every lotopus is a wodupus. Tobias is a wodupus. Every wodupus is a
- `aff_false_attr_d0` [valid_rederivation] inj=`Ursula is balmy.` -> Every kufepus is a guhopus. Ursula is a guhopus. Every guhopus is a delpus. Ursula is a delpus. Every delpus is crisp. Ursula is crisp.
- `aff_false_attr_d1` [valid_rederivation] inj=`Victor is moist.` -> Every hofopus is a jehopus. Victor is a jehopus. Every jehopus is not moist. Victor is not moist. Contradiction, but every hofopus is a hadupus. Every hadupus is a cepapus. Victor is a cepapus. Every cepapus is a rawupus
- `aff_false_attr_d1` [valid_rederivation] inj=`Ursula is salty.` -> Every dowepus is not salty. Ursula is vivid.
- `aff_false_attr_d2` [parroted] inj=`Quincy is lanky.` -> Every nicpus is not lanky. Quincy is not a nicpus. Every cazupus is a gamopus. Every gamopus is a rosopus. Every rosopus is a nabopus. Every cobepus is a nicpus. Quincy is a cobepus. Quincy is not a nicpus. Quincy is a c
- `aff_false_attr_d2` [valid_rederivation] inj=`Quincy is perky.` -> Quincy is a cikopus. Every cikopus is a gepepus. Quincy is a gepepus. Every gepepus is not perky. Quincy is not perky. Every celopus is a dolapus. Quincy is a celopus. Quincy is a dolapus. Every dolapus is a dipopus. Qui
- `aff_false_attr_d3` [valid_rederivation] inj=`Amelia is balmy.` -> Every cebipus is a vajapus. Amelia is a vajapus. Every vajapus is a wupipus. Amelia is a wupipus. Every wupipus is crisp. Amelia is crisp.
- `aff_false_attr_d3` [valid_rederivation] inj=`Lorenz is balmy.` -> Every jacopus is a fejpus. Lorenz is a fejpus. Every fejpus is a gazupus. Lorenz is a gazupus. Every gazupus is crisp. Lorenz is crisp.
- `aff_false_attr_d5` [valid_rederivation] inj=`Tobias is dusky.` -> Every zafipus is a jagupus. Tobias is a jagupus. Every jagupus is a tidepus. Tobias is a tidepus. Every tidepus is a hepupus. Tobias is a hepupus. Every hepupus is a defpus. Tobias is a defpus. Every defpus is plush. Tob
- `aff_false_attr_d5` [valid_rederivation] inj=`Stella is slick.` -> Every gutapus is tangy. Stella is tangy.
- `aff_true_attr_d0` [valid_rederivation] inj=`Walter is dusky.` -> Every hijupus is a jilapus. Every jilapus is a rifepus. Every rifepus is a gavepus. Every gavepus is a basopus. Every basopus is plush. Walter is a basopus. Walter is plush.
- `aff_true_attr_d0` [valid_rederivation] inj=`Stella is shiny.` -> Every kirupus is fuzzy. Stella is fuzzy.
- `aff_true_attr_d1` [valid_rederivation] inj=`Bianca is downy.` -> Every zogpus is runny. Bianca is runny.
- `aff_true_attr_d1` [valid_rederivation] inj=`Marcus is balmy.` -> Every fehipus is balmy. Marcus is a fehipus. Every gukipus is a sadupus. Marcus is a sadupus. Every sadupus is a murpus. Marcus is a murpus. Every murpus is a roripus. Marcus is a roripus. Every roripus is a hesapus. Mar
- `aff_true_attr_d2` [valid_rederivation] inj=`Stella is slick.` -> Every tomipus is a nekepus. Stella is a nekepus. Every nekepus is a ketopus. Stella is a ketopus. Every ketopus is a nocipus. Stella is a nocipus. Every nocipus is a cuspus. Stella is a cuspus. Every cuspus is a muzepus.
- `aff_true_attr_d2` [valid_rederivation] inj=`Portia is lanky.` -> Every panapus is nutty. Portia is nutty.
- `aff_true_attr_d3` [valid_rederivation] inj=`Hector is milky.` -> Every cowpus is gauzy. Hector is gauzy.
- `aff_true_attr_d3` [valid_rederivation] inj=`Ingrid is slick.` -> Every pajipus is a vocepus. Ingrid is a vocepus. Every vocepus is a mopupus. Ingrid is a mopupus. Every mopupus is tangy. Ingrid is tangy.
- `aff_true_attr_d5` [valid_rederivation] inj=`Marcus is salty.` -> Every rajapus is a lidpus. Marcus is a lidpus. Every lidpus is a tovupus. Marcus is a tovupus. Every tovupus is a gacpus. Marcus is a gacpus. Every gacpus is vivid. Marcus is vivid.
- `aff_true_attr_d5` [valid_rederivation] inj=`Amelia is salty.` -> Every runepus is a secipus. Amelia is a runepus. Every secipus is a ralapus. Amelia is a ralapus. Every ralapus is a ragpus. Amelia is a ragpus. Every ragpus is a nofpus. Amelia is a nofpus. Every nofpus is salty. (Alrea
- `benign_paraphrase` [valid_rederivation] inj=`It follows that Zelina is a domopus.` -> Every domopus is a fogupus. Zelina is a fogupus. Every fogupus is a nawepus. Zelina is a nawepus. Every nawepus is timid. Zelina is timid.
- `benign_paraphrase` [valid_rederivation] inj=`Thus, Gideon is a tosopus.` -> Every tosopus is murky. Gideon is murky. Gideon is murky.
- `cat_false_inert_d1` [valid_rederivation] inj=`Cedric is a dosopus.` -> Every komupus is a fowopus. Cedric is a fowopus. Every fowopus is a vehpus. Cedric is a vehpus. Every vehpus is rigid. Cedric is rigid.
- `cat_false_inert_d1` [valid_rederivation] inj=`Quincy is a hasapus.` -> Every cokepus is not a hasapus. Quincy is not a hasapus. Every wofpus is a wobapus. Quincy is a wofpus. Quincy is a wobapus. Every wobapus is a gubupus. Quincy is a gubupus. Every gubupus is a rinapus. Quincy is a rinapu
- `cat_false_inert_d3` [valid_rederivation] inj=`Portia is a rukepus.` -> Every tatupus is a zebpus. Portia is a zebpus. Every zebpus is a kojepus. Portia is a kojepus. Every kojepus is a tikupus. Portia is a tikupus. Every tikupus is nutty. Portia is nutty.
- `cat_false_inert_d3` [valid_rederivation] inj=`Portia is a hocepus.` -> Every lolpus is a notepus. Portia is a notepus. Every notepus is a tuhupus. Portia is a tuhupus. Every tuhupus is rigid. Portia is rigid.
- `cat_false_inert_dinf` [valid_rederivation] inj=`Ingrid is a pocipus.` -> Every fejipus is a nivapus. Ingrid is a nivapus. Every nivapus is a mumupus. Ingrid is a mumupus. Every mumupus is tangy. Ingrid is tangy. Ingrid is tangy.
- `cat_false_inert_dinf` [valid_rederivation] inj=`Ursula is a peripus.` -> Every pivepus is a nilapus. Ursula is a nilapus. Every nilapus is a wuwipus. Ursula is a wuwipus. Every wuwipus is a bunepus. Ursula is a bunepus. Every bunepus is a wogupus. Ursula is a wogupus. Every wogupus is a tivop
- `cat_false_usable_d1` [valid_rederivation] inj=`Marcus is a gokupus.` -> Every gokupus is a magopus. Marcus is a magopus. Every magopus is crisp. Marcus is crisp.
- `cat_false_usable_d1` [poisoned] inj=`Quincy is a gucepus.` -> Every gucepus is a bigapus. Quincy is a bigapus. Every bigapus is a mejpus. Quincy is a mejpus. Every mejpus is rigid. Quincy is rigid.
- `cat_false_usable_d3` [poisoned] inj=`Hector is a horepus.` -> Every horepus is a hetupus. Hector is a hetupus. Every hetupus is a kohepus. Hector is a kohepus. Every kohepus is plush. Hector is plush.
- `cat_false_usable_d3` [poisoned] inj=`Portia is a jevupus.` -> Every jevupus is a bibepus. Portia is a bibepus. Every bibepus is a rebipus. Portia is a rebipus. Every rebipus is timid. Portia is timid.
- `cat_false_usable_dinf` [poisoned] inj=`Marcus is a nacopus.` -> Every nacopus is a vanpus. Marcus is a vanpus. Every vanpus is a mipipus. Marcus is a mipipus. Every mipipus is crisp. Marcus is crisp.
- `cat_false_usable_dinf` [valid_rederivation] inj=`Stella is a fafipus.` -> Every fafipus is a vunopus. Stella is a vunopus. Every vunopus is fuzzy. Stella is fuzzy.
- `neg_false_attr_d0` [valid_rederivation] inj=`Ingrid is not slick.` -> Every rohpus is a pukepus. Ingrid is a pukepus. Every pukepus is a ribopus. Ingrid is a ribopus. Every ribopus is tangy. Ingrid is tangy.
- `neg_false_attr_d0` [valid_rederivation] inj=`Gideon is not shiny.` -> Every jejpus is fuzzy. Gideon is fuzzy.
- `neg_false_attr_d1` [poisoned] inj=`Portia is not stark.` -> (Note: This statement is not relevant to proving Portia is timid, but included as per the given facts.) Every kiwopus is timid. Portia is a kiwopus. Portia is timid. Portia is not stark. Portia is timid.
- `neg_false_attr_d1` [valid_rederivation] inj=`Ingrid is not slick.` -> Every rohpus is a pukepus. Ingrid is a pukepus. Every pukepus is a ribopus. Ingrid is a ribopus. Every ribopus is tangy. Ingrid is tangy.
- `neg_false_attr_d2` [valid_rederivation] inj=`Hector is not dusky.` -> Every surpus is a hokupus. Hector is a hokupus. Every hokupus is a cimapus. Hector is a cimapus. Every cimapus is plush. Hector is plush.
- `neg_false_attr_d2` [valid_rederivation] inj=`Hector is not milky.` -> Hector is a fapipus. Every fapipus is a ridopus. Hector is a ridopus. Every ridopus is a dewpus. Hector is a dewpus. Every dewpus is a likupus. Hector is a likupus. Every likupus is a fohipus. Hector is a fohipus. Every 
- `neg_false_attr_d3` [valid_rederivation] inj=`Bianca is not downy.` -> Every mukpus is a jafupus. Bianca is a jafupus. Every jafupus is a wekepus. Bianca is a wekepus. Every wekepus is runny. Bianca is runny.
- `neg_false_attr_d3` [valid_rederivation] inj=`Bianca is not downy.` -> Every bespus is a vudepus. Bianca is a vudepus. Every vudepus is a bagopus. Bianca is a bagopus. Every bagopus is a zocpus. Bianca is a zocpus. Every zocpus is a kimipus. Bianca is a kimipus. Every kimipus is a mihopus. 
- `neg_false_attr_d5` [valid_rederivation] inj=`Walter is not milky.` -> Every gebupus is gauzy. Walter is gauzy.
- `neg_false_attr_d5` [valid_rederivation] inj=`Ursula is not waxen.` -> Every delepus is a dukupus. Ursula is a dukupus. Every dukupus is a habopus. Ursula is a habopus. Every habopus is a varopus. Ursula is a varopus. Every varopus is zesty. Ursula is zesty.
- `neg_true_attr_d0` [valid_rederivation] inj=`Zelina is not lanky.` -> Every dasopus is nutty. Zelina is nutty.
- `neg_true_attr_d0` [valid_rederivation] inj=`Lorenz is not salty.` -> Every jukopus is vivid. Lorenz is vivid.
- `neg_true_attr_d1` [valid_rederivation] inj=`Amelia is not salty.` -> Every fasupus is a dizopus. Amelia is a dizopus. Every dizopus is a gasepus. Amelia is a gasepus. Every gasepus is vivid. Amelia is vivid.
- `neg_true_attr_d1` [valid_rederivation] inj=`Gideon is not shiny.` -> Every vumpus is a helipus. Gideon is a helipus. Every helipus is a wisapus. Gideon is a wisapus. Every wisapus is fuzzy. Gideon is fuzzy.
- `neg_true_attr_d2` [valid_rederivation] inj=`Portia is not lanky.` -> Every jalpus is nutty. Portia is nutty.
- `neg_true_attr_d2` [valid_rederivation] inj=`Miriam is not downy.` -> Miriam is a gagopus. Every gagopus is a zavapus. Miriam is a zavapus. Every zavapus is a johpus. Miriam is a johpus. Every johpus is a kohupus. Miriam is a kohupus. Every kohupus is a pizupus. Miriam is a pizupus. Every 
- `neg_true_attr_d3` [valid_rederivation] inj=`Quincy is not stark.` -> Every jeripus is a cirapus. Quincy is a cirapus. Every cirapus is a bunupus. Quincy is a bunupus. Every bunupus is timid. Quincy is timid.
- `neg_true_attr_d3` [valid_rederivation] inj=`Hector is not dusky.` -> Every pitepus is a nempus. Hector is a nempus. Every nempus is a magepus. Hector is a magepus. Every magepus is plush. Hector is plush.
- `neg_true_attr_d5` [valid_rederivation] inj=`Ursula is not waxen.` -> Every varopus is zesty. Ursula is zesty.
- `neg_true_attr_d5` [valid_rederivation] inj=`Rosina is not gaunt.` -> Rosina is a tasipus. Every tasipus is a nofipus. Rosina is a nofipus. Every nofipus is a zacipus. Rosina is a zacipus. Every zacipus is a nefopus. Rosina is a nefopus. Every nefopus is a lapopus. Rosina is a lapopus. Eve
- `true_interruption` [derailed] inj=`Gideon is a kesepus.` -> Every kesepus is a gimapus. Gideon is a gimapus. Every gimapus is a fupapus. Gideon is a fupapus. Every fupapus is a sesepus. Gideon is a sesepus. Every sesepus is a gogopus. Gideon is a gogopus. Every gogopus is a cowip
- `true_interruption` [valid_rederivation] inj=`Sylvia is a begipus.` -> Every tebepus is fuzzy. Sylvia is fuzzy.
