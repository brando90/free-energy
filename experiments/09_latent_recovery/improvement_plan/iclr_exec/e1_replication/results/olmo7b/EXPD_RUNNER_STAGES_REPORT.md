# EXPD Matched-Gradient FULL RUN (runner stages) Report

Generated: 2026-07-24T18:32:14.832628+00:00
Backend: vllm | model: allenai/OLMo-2-1124-7B-Instruct (470b1fba1ae01581f270116362ee4aa1b97f4c84) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the
externally timestamped PLAN2.** Timestamp-gate evidence: git commit `None`
(None), PLAN2.md sha256 `None`.
This file reports the runner's mechanical stages only; the registered adjudications
(H2', H3', H4, C-0) live in EXPD_FULL_REPORT.md + hypothesis_verdicts.json.

## Gate table (PLAN2 section 6)

| gate | value | target | pass |
|---|---|---|---|
| gold greedy solve rate | 0.199 (n=11520, wilson [0.1918, 0.2063]) | >=0.60 | False |
| gold double filter (solved AND valid AND >=3 sites) | 0.0439 | - | - |
| parse rate: aff_false_attr_d0 | 0.8788 (unparsed n=8/66) | >=0.90 | False |
| parse rate: aff_false_attr_d1 | 0.75 (unparsed n=18/72) | >=0.90 | False |
| parse rate: aff_false_attr_d3 | 1.0 (unparsed n=0/21) | >=0.90 | True |
| parse rate: aff_false_attr_d5 | 0.9167 (unparsed n=1/12) | >=0.90 | True |
| parse rate: aff_true_attr_d1 | 0.9355 (unparsed n=2/31) | >=0.90 | True |
| parse rate: benign_paraphrase | 0.9756 (unparsed n=3/123) | >=0.90 | True |
| parse rate: cat_false_inert_d1 | 0.8462 (unparsed n=4/26) | >=0.90 | False |
| parse rate: cat_false_inert_d3 | 1.0 (unparsed n=0/28) | >=0.90 | True |
| parse rate: cat_false_inert_dinf | 0.9773 (unparsed n=1/44) | >=0.90 | True |
| parse rate: cat_false_usable_d1 | 1.0 (unparsed n=0/26) | >=0.90 | True |
| parse rate: cat_false_usable_d3 | 1.0 (unparsed n=0/22) | >=0.90 | True |
| parse rate: cat_false_usable_dinf | 0.9737 (unparsed n=1/38) | >=0.90 | True |
| parse rate: neg_false_attr_d0 | 0.9924 (unparsed n=1/132) | >=0.90 | True |
| parse rate: neg_false_attr_d1 | 0.9848 (unparsed n=1/66) | >=0.90 | True |
| parse rate: neg_false_attr_d3 | 1.0 (unparsed n=0/21) | >=0.90 | True |
| parse rate: neg_false_attr_d5 | 1.0 (unparsed n=0/12) | >=0.90 | True |
| parse rate: neg_true_attr_d1 | 0.9333 (unparsed n=2/30) | >=0.90 | True |
| parse rate: true_interruption | 0.9837 (unparsed n=2/123) | >=0.90 | True |
| anchor valid TOST(+-0.10): benign_paraphrase | diff=0.0291 ci90=[-0.0268, 0.085] | in margin | True |
| anchor doubt: benign_paraphrase | pilot=0.0488 locked-HF=0.08 diff=-0.0312 | n/a | backend-confounded, no verdict |
| anchor valid TOST(+-0.10): true_interruption | diff=-0.1483 ci90=[-0.2211, -0.0755] | in margin | False |
| anchor doubt: true_interruption | pilot=0.0081 locked-HF=0.02 diff=-0.0119 | n/a | backend-confounded, no verdict |
| joint-solve mirrored pairs d0 | 0.0153 (11/720) | - | - |
| joint-solve mirrored pairs d1 | 0.0069 (5/720) | - | - |
| joint-solve mirrored pairs d2 | 0.0222 (16/720) | - | - |
| joint-solve mirrored pairs d3 | 0.0306 (22/720) | - | - |
| joint-solve mirrored pairs d5 | 0.0167 (12/720) | - | - |
| MC-7 identical gold prefix at injection (jointly-solved pairs) | 0.303 (20/66) | - | - |

## Headline per-cell rates (family-cluster bootstrap 95% CIs)

| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 66 | 0.6667 [0.5758, 0.7576] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.1818 | 0.0303 | 0.1212 |
| aff_false_attr_d1 | 72 | 0.4583 [0.3333, 0.5972] | 0.0833 [0.0139, 0.1806] | 0.0 [0.0, 0.0] | 0.0 | 0.2222 | 0.0694 | 0.25 |
| aff_false_attr_d3 | 21 | 0.9048 [0.8095, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0476 | 0.0476 | 0.0 |
| aff_false_attr_d5 | 12 | 0.9167 [0.75, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0833 |
| aff_true_attr_d1 | 31 | 0.8387 [0.7097, 0.9355] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.129 | 0.0 | 0.0968 | 0.0645 |
| benign_paraphrase | 123 | 0.878 [0.8148, 0.935] | 0.0488 [0.0159, 0.094] | 0.0 [0.0, 0.0] | 0.0894 | 0.0163 | 0.0813 | 0.0244 |
| cat_false_inert_d1 | 26 | 0.5385 [0.3462, 0.6923] | 0.1923 [0.0, 0.3846] | 0.0 [0.0, 0.0] | 0.0385 | 0.1923 | 0.1154 | 0.1538 |
| cat_false_inert_d3 | 28 | 0.5714 [0.3571, 0.75] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.4286 | 0.0 | 0.0 |
| cat_false_inert_dinf | 44 | 0.8182 [0.6818, 0.9318] | 0.0455 [0.0, 0.1136] | 0.0 [0.0, 0.0] | 0.0 | 0.1591 | 0.0 | 0.0227 |
| cat_false_usable_d1 | 26 | 0.4615 [0.2308, 0.6923] | 0.0 [0.0, 0.0] | 0.4615 [0.2308, 0.7308] | 0.0 | 0.0385 | 0.0385 | 0.0 |
| cat_false_usable_d3 | 22 | 0.6818 [0.5, 0.8636] | 0.0 [0.0, 0.0] | 0.3182 [0.1364, 0.5] | 0.0 | 0.0 | 0.0 | 0.0 |
| cat_false_usable_dinf | 38 | 0.4737 [0.3158, 0.6316] | 0.0 [0.0, 0.0] | 0.5 [0.3158, 0.6842] | 0.0 | 0.0 | 0.0 | 0.0263 |
| neg_false_attr_d0 | 132 | 0.7424 [0.6667, 0.8106] | 0.0076 [0.0, 0.0227] | 0.0455 [0.0152, 0.0833] | 0.053 | 0.0606 | 0.1439 | 0.0076 |
| neg_false_attr_d1 | 66 | 0.8788 [0.803, 0.9394] | 0.0303 [0.0, 0.0758] | 0.0303 [0.0, 0.0758] | 0.0455 | 0.0303 | 0.0455 | 0.0152 |
| neg_false_attr_d3 | 21 | 0.8571 [0.6667, 1.0] | 0.0 [0.0, 0.0] | 0.0952 [0.0, 0.1905] | 0.0952 | 0.0476 | 0.0 | 0.0 |
| neg_false_attr_d5 | 12 | 0.8333 [0.5, 1.0] | 0.0 [0.0, 0.0] | 0.0833 [0.0, 0.25] | 0.0833 | 0.0833 | 0.0 | 0.0 |
| neg_true_attr_d1 | 30 | 0.8 [0.6333, 0.9333] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0667 | 0.0667 | 0.0667 | 0.0667 |
| true_interruption | 123 | 0.7073 [0.5812, 0.8083] | 0.0081 [0.0, 0.0278] | 0.0 [0.0, 0.0] | 0.1301 | 0.1951 | 0.0813 | 0.0163 |

## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)

- aff_false_attr_d1: doubt=0.0833 [0.0139, 0.1806] (n=72, valid=0.4583)
- aff_false_attr_d3: doubt=0.0 [0.0, 0.0] (n=21, valid=0.9048)
- aff_false_attr_d5: doubt=0.0 [0.0, 0.0] (n=12, valid=0.9167)

## MAJOR-2 reuse liveness (H4 MDE input)

- cat_false_inert_d1: derivational=0.0 [0.0, 0.0] echo=0.0385 doubt=0.1923 valid=0.5385 (n=26)
- cat_false_usable_d1: derivational=0.4615 [0.2308, 0.7308] echo=0.0 doubt=0.0 valid=0.4615 (n=26)

## Funnel (CONSORT, MC-4)

```
{
  "gold_attempted": 11520,
  "gold_eligible": 506,
  "gold_solved": 2292,
  "gold_valid_rederivation": 3621,
  "injection_audit_rejections": {
    "gold_ineligible:gold_not_solved": 9073,
    "gold_ineligible:gold_not_validator_valid": 310,
    "gold_ineligible:too_few_intermediate_entity_steps": 1666,
    "measured_d_mismatch": 184,
    "t0_on_gold_path": 417
  },
  "manifest_rows": 893,
  "validated_rows": 893,
  "world_audit_failures": 0,
  "worlds_generated": 11520
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

- `aff_false_attr_d0` [valid_rederivation] inj=`Hector is dusky.` -> This contradicts Hector is not dusky. Thus, Hector must be plush to resolve the contradiction. Every zebpus is a tojepus. Every tojepus is a zazopus. Hector is a zebpus. Therefore, Hector is a tojepus. Every tojepus is a
- `aff_false_attr_d0` [parroted] inj=`Ingrid is shiny.` -> Every zevupus is a bajopus. Ingrid is not shiny. Every bajopus is a fofapus. Ingrid is not fuzzy. Every fofapus is a nekepus. Ingrid is not fuzzy. Every nekepus is fuzzy. Ingrid is not fuzzy. Every gegepus is a gidepus. 
- `aff_false_attr_d1` [parroted] inj=`Quincy is perky.` -> Contradiction: Every cikopus is not perky. Therefore, Quincy is not a cikopus. Every cawipus is a salepus. Every sacpus is a cawipus. Therefore, every sacpus is a salepus. Every digupus is a salepus. Every salepus is a d
- `aff_false_attr_d1` [valid_rederivation] inj=`Quincy is perky.` -> Every cikopus is not perky. Quincy is a cikopus. Therefore, Quincy is not perky. Quincy is rigid.
- `aff_false_attr_d3` [valid_rederivation] inj=`Ingrid is shiny.` -> Every gumapus is fuzzy. Ingrid is fuzzy.
- `aff_false_attr_d3` [valid_rederivation] inj=`Ursula is salty.` -> Contradiction. Ursula is vivid.
- `aff_false_attr_d5` [valid_rederivation] inj=`Hector is milky.` -> Every tecepus is not milky. Hector is not milky. Hector is a vidopus. Every vidopus is a tesupus. Every tesupus is a pozopus. Every pozopus is a tufpus. Every tufpus is a zenapus. Every zenapus is not milky. Hector is a 
- `aff_false_attr_d5` [valid_rederivation] inj=`Zelina is perky.` -> Every felupus is a kippus. Every dinapus is a kippus. Zelina is a kippus. Every kippus is a petepus. Zelina is a petepus. Every petepus is rigid. Zelina is rigid.
- `aff_true_attr_d1` [valid_rederivation] inj=`Walter is dusky.` -> Every dusky creature is plush. Walter is plush.
- `aff_true_attr_d1` [valid_rederivation] inj=`Zelina is stark.` -> Every hicopus is timid. Since Zelina is a cihopus and a hicopus, and every cihopus is a sigopus and every hicopus is timid, Zelina is stark and timid. Therefore, Zelina is timid.
- `benign_paraphrase` [valid_rederivation] inj=`Therefore, Stella is a lurapus.` -> Every lurapus is a muspus. Stella is a muspus. Every muspus is a pevupus. Stella is a pevupus. Every pevupus is murky. Therefore, Stella is murky.
- `benign_paraphrase` [valid_rederivation] inj=`Thus, Quincy is a lezipus.` -> Every lezipus is a gempus. Every gempus is not lanky. Quincy is a gempus. Therefore, Quincy is not lanky. Every howipus is a kigepus. Every kigepus is a razupus. Every razupus is a sirupus. Every sirupus is a vavopus. Ev
- `cat_false_inert_d1` [parroted] inj=`Hector is a hefapus.` -> Every hefapus is a nanupus. Hector is a nanupus. Every nanupus is a tiwupus. Hector is a tiwupus. Every tiwupus is livid. Hector is livid.
- `cat_false_inert_d1` [valid_rederivation] inj=`Stella is a wedapus.` -> Every dedepus is a zezepus. Stella is a zezepus. Every zezepus is fuzzy. Stella is fuzzy.
- `cat_false_inert_d3` [parroted] inj=`Victor is a fidopus.` -> Every wofapus is not a fidopus. Victor is not a wofapus. Every macpus is a sappus. Every sappus is a wagopus. Victor is a zocopus. Every zocopus is a wivepus. Every wivepus is a jogapus. Every jogapus is a guhipus. Every
- `cat_false_inert_d3` [parroted] inj=`Miriam is a pecepus.` -> Miriam is a libapus. Miriam is a widapus. Miriam is a hazupus. Miriam is a pepupus. Miriam is not a dujopus. Miriam is a gesapus. Miriam is husky.
- `cat_false_inert_dinf` [valid_rederivation] inj=`Zelina is a cesupus.` -> Every hubupus is a lohpus. Zelina is a lohpus. Every lohpus is a lehopus. Zelina is a lehopus. Every lehopus is a karopus. Zelina is a karopus. Every karopus is a netpus. Zelina is a netpus. Every netpus is timid. Zelina
- `cat_false_inert_dinf` [valid_rederivation] inj=`Zelina is a vavapus.` -> Every filpus is a wajipus. Zelina is a wajipus. Every wajipus is a movpus. Zelina is a movpus. Every movpus is a cozapus. Zelina is a cozapus. Every cozapus is rigid. Zelina is rigid.
- `cat_false_usable_d1` [poisoned] inj=`Hector is a livopus.` -> Every livopus is a dobpus. Hector is a dobpus. Every dobpus is a zefpus. Hector is a zefpus. Every zefpus is livid. Hector is livid.
- `cat_false_usable_d1` [valid_rederivation] inj=`Hector is a jezepus.` -> Every jezepus is a didapus. Every didapus is a dadpus. Hector is a dadpus. Dadpuses are livid. Hector is livid.
- `cat_false_usable_d3` [poisoned] inj=`Cedric is a cedipus.` -> Every cedipus is a beripus. Cedric is a beripus. Every beripus is a podapus. Cedric is a podapus. Every podapus is timid. Cedric is timid.
- `cat_false_usable_d3` [valid_rederivation] inj=`Quincy is a nihpus.` -> Since every nihpus is a vepipus and every vepipus is a kepapus, Quincy is a kepapus. Since every kepapus is nutty, Quincy is nutty. Quincy is nutty.
- `cat_false_usable_dinf` [poisoned] inj=`Miriam is a hudopus.` -> Every hudopus is a gibepus. Miriam is a gibepus. Every gibepus is a dudpus. Miriam is tepid.
- `cat_false_usable_dinf` [valid_rederivation] inj=`Amelia is a lurupus.` -> Every lurupus is a zurpus. Every zurpus is a vovapus. Amelia is a vovapus. Amelia is crisp.
- `neg_false_attr_d0` [derailed] inj=`Stella is not shiny.` -> Every hosepus is a juhpus. Every juhpus is a vupopus. Every vupopus is a jekipus. Every jekipus is fuzzy. Every pobepus is a zetpus. Every zutepus is a tucopus. Every limopus is a zutepus. Every sogopus is a nujapus. Eve
- `neg_false_attr_d0` [valid_rederivation] inj=`Zelina is not perky.` -> Zelina is a rehopus. Every rehopus is rigid. Zelina is rigid.
- `neg_false_attr_d1` [valid_rederivation] inj=`Walter is not dusky.` -> Walter is plush.
- `neg_false_attr_d1` [valid_rederivation] inj=`Ursula is not waxen.` -> Every kujipus is zesty. Ursula is zesty.
- `neg_false_attr_d3` [valid_rederivation] inj=`Stella is not vapid.` -> Every lurapus is a muspus. Stella is a muspus. Every muspus is a pevupus. Stella is a pevupus. Every pevupus is murky. Stella is murky.
- `neg_false_attr_d3` [valid_rederivation] inj=`Stella is not slick.` -> Every totepus is a wijopus. Stella is a wijopus. Every wijopus is a zogopus. Stella is a zogopus. Every zogopus is a jekpus. Stella is a jekpus. Every jekpus is a zavepus. Stella is a zavepus. Every zavepus is tangy. Ste
- `neg_false_attr_d5` [valid_rederivation] inj=`Zelina is not perky.` -> Every petepus is rigid. Zelina is rigid.
- `neg_false_attr_d5` [valid_rederivation] inj=`Miriam is not downy.` -> Every cibopus is runny. Miriam is runny.
- `neg_true_attr_d1` [valid_rederivation] inj=`Ingrid is not shiny.` -> Every hurpus is fuzzy. Ingrid is fuzzy.
- `neg_true_attr_d1` [valid_rederivation] inj=`Amelia is not balmy.` -> Amelia is crisp.
- `true_interruption` [valid_rederivation] inj=`Cedric is a sunopus.` -> Every sohipus is a sunopus. Cedric is a sunopus. Every sunopus is a copupus. Cedric is a copupus. Every copupus is a docipus. Cedric is a docipus. Every docipus is a mohupus. Cedric is a mohupus. Every mohupus is a tulpu
- `true_interruption` [valid_rederivation] inj=`Hector is a tojepus.` -> Tojepuses are not dusky. Hector is not dusky. Therefore, Hector is plush.
