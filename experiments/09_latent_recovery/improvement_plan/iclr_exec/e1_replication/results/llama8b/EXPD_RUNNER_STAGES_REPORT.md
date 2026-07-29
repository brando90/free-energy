# EXPD Matched-Gradient FULL RUN (runner stages) Report

Generated: 2026-07-24T18:23:11.442359+00:00
Backend: vllm | model: NousResearch/Meta-Llama-3.1-8B-Instruct (d10aef7999a2b5ba950ab3974312feeedbfe0b77) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the
externally timestamped PLAN2.** Timestamp-gate evidence: git commit `None`
(None), PLAN2.md sha256 `None`.
This file reports the runner's mechanical stages only; the registered adjudications
(H2', H3', H4, C-0) live in EXPD_FULL_REPORT.md + hypothesis_verdicts.json.

## Gate table (PLAN2 section 6)

| gate | value | target | pass |
|---|---|---|---|
| gold greedy solve rate | 0.9816 (n=5760, wilson [0.9778, 0.9848]) | >=0.60 | True |
| gold double filter (solved AND valid AND >=3 sites) | 0.5408 | - | - |
| parse rate: aff_false_attr_d0 | 0.9933 (unparsed n=3/450) | >=0.90 | True |
| parse rate: aff_false_attr_d1 | 0.9956 (unparsed n=2/450) | >=0.90 | True |
| parse rate: aff_false_attr_d3 | 0.9822 (unparsed n=8/450) | >=0.90 | True |
| parse rate: aff_false_attr_d5 | 0.9911 (unparsed n=4/450) | >=0.90 | True |
| parse rate: aff_true_attr_d1 | 1.0 (unparsed n=0/150) | >=0.90 | True |
| parse rate: benign_paraphrase | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: cat_false_inert_d1 | 0.9733 (unparsed n=8/300) | >=0.90 | True |
| parse rate: cat_false_inert_d3 | 0.92 (unparsed n=24/300) | >=0.90 | True |
| parse rate: cat_false_inert_dinf | 0.9467 (unparsed n=16/300) | >=0.90 | True |
| parse rate: cat_false_usable_d1 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_usable_d3 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_usable_dinf | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: neg_false_attr_d0 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: neg_false_attr_d1 | 0.9911 (unparsed n=4/450) | >=0.90 | True |
| parse rate: neg_false_attr_d3 | 0.9844 (unparsed n=7/450) | >=0.90 | True |
| parse rate: neg_false_attr_d5 | 0.9733 (unparsed n=12/450) | >=0.90 | True |
| parse rate: neg_true_attr_d1 | 0.98 (unparsed n=3/150) | >=0.90 | True |
| parse rate: true_interruption | 0.9911 (unparsed n=4/450) | >=0.90 | True |
| anchor valid TOST(+-0.10): benign_paraphrase | diff=0.1489 ci90=[0.1209, 0.1769] | in margin | False |
| anchor doubt: benign_paraphrase | pilot=0.0 locked-HF=0.08 diff=-0.08 | n/a | backend-confounded, no verdict |
| anchor valid TOST(+-0.10): true_interruption | diff=0.1022 ci90=[0.0708, 0.1336] | in margin | False |
| anchor doubt: true_interruption | pilot=0.0 locked-HF=0.02 diff=-0.02 | n/a | backend-confounded, no verdict |
| joint-solve mirrored pairs d0 | 0.5472 (197/360) | - | - |
| joint-solve mirrored pairs d1 | 0.4917 (177/360) | - | - |
| joint-solve mirrored pairs d2 | 0.475 (171/360) | - | - |
| joint-solve mirrored pairs d3 | 0.4861 (175/360) | - | - |
| joint-solve mirrored pairs d5 | 0.475 (171/360) | - | - |
| MC-7 identical gold prefix at injection (jointly-solved pairs) | 0.9349 (833/891) | - | - |

## Headline per-cell rates (family-cluster bootstrap 95% CIs)

| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 450 | 0.9311 [0.9044, 0.9533] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0089 | 0.0533 | 0.0067 |
| aff_false_attr_d1 | 450 | 0.76 [0.7178, 0.7978] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0022 | 0.1267 | 0.1089 | 0.0044 |
| aff_false_attr_d3 | 450 | 0.7933 [0.7556, 0.8333] | 0.0 [0.0, 0.0] | 0.0067 [0.0, 0.0133] | 0.0089 | 0.06 | 0.1222 | 0.0178 |
| aff_false_attr_d5 | 450 | 0.82 [0.7844, 0.8533] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0133 | 0.0422 | 0.1289 | 0.0089 |
| aff_true_attr_d1 | 150 | 0.8733 [0.82, 0.9267] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.04 | 0.0133 | 0.1133 | 0.0 |
| benign_paraphrase | 450 | 0.9978 [0.9924, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0044 | 0.0 | 0.0 | 0.0022 |
| cat_false_inert_d1 | 300 | 0.67 [0.6167, 0.7267] | 0.0 [0.0, 0.0] | 0.03 [0.01, 0.0567] | 0.03 | 0.2467 | 0.0267 | 0.0267 |
| cat_false_inert_d3 | 300 | 0.5833 [0.5267, 0.64] | 0.0 [0.0, 0.0] | 0.0133 [0.0033, 0.0267] | 0.0167 | 0.2633 | 0.06 | 0.08 |
| cat_false_inert_dinf | 300 | 0.6633 [0.6033, 0.7233] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0033 | 0.2333 | 0.05 | 0.0533 |
| cat_false_usable_d1 | 300 | 0.0433 [0.02, 0.07] | 0.0 [0.0, 0.0] | 0.9367 [0.9033, 0.9667] | 0.0067 | 0.02 | 0.0 | 0.0 |
| cat_false_usable_d3 | 300 | 0.05 [0.0233, 0.08] | 0.0 [0.0, 0.0] | 0.93 [0.8967, 0.96] | 0.0033 | 0.0133 | 0.0067 | 0.0 |
| cat_false_usable_dinf | 300 | 0.0433 [0.02, 0.07] | 0.0 [0.0, 0.0] | 0.9467 [0.9133, 0.9733] | 0.0 | 0.0067 | 0.0033 | 0.0 |
| neg_false_attr_d0 | 450 | 0.8311 [0.7956, 0.8644] | 0.0 [0.0, 0.0] | 0.1022 [0.0778, 0.1267] | 0.1044 | 0.0044 | 0.06 | 0.0022 |
| neg_false_attr_d1 | 450 | 0.4311 [0.3867, 0.4756] | 0.0 [0.0, 0.0] | 0.0733 [0.0511, 0.0978] | 0.0889 | 0.3644 | 0.1222 | 0.0089 |
| neg_false_attr_d3 | 450 | 0.5178 [0.4711, 0.5644] | 0.0 [0.0, 0.0] | 0.2044 [0.1711, 0.2378] | 0.26 | 0.0978 | 0.1644 | 0.0156 |
| neg_false_attr_d5 | 450 | 0.4978 [0.4533, 0.5422] | 0.0 [0.0, 0.0] | 0.2156 [0.1756, 0.2578] | 0.3133 | 0.0711 | 0.1889 | 0.0267 |
| neg_true_attr_d1 | 150 | 0.9533 [0.92, 0.98] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.1067 | 0.0 | 0.0267 | 0.02 |
| true_interruption | 450 | 0.9578 [0.9303, 0.9821] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.1622 | 0.02 | 0.0133 | 0.0089 |

## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)

- aff_false_attr_d1: doubt=0.0 [0.0, 0.0] (n=450, valid=0.76)
- aff_false_attr_d3: doubt=0.0 [0.0, 0.0] (n=450, valid=0.7933)
- aff_false_attr_d5: doubt=0.0 [0.0, 0.0] (n=450, valid=0.82)

## MAJOR-2 reuse liveness (H4 MDE input)

- cat_false_inert_d1: derivational=0.03 [0.01, 0.0567] echo=0.03 doubt=0.0 valid=0.67 (n=300)
- cat_false_usable_d1: derivational=0.9367 [0.9033, 0.9667] echo=0.0067 doubt=0.0 valid=0.0433 (n=300)

## Funnel (CONSORT, MC-4)

```
{
  "gold_attempted": 5760,
  "gold_eligible": 3115,
  "gold_solved": 5654,
  "gold_valid_rederivation": 5551,
  "injection_audit_rejections": {
    "gold_ineligible:gold_not_solved": 92,
    "gold_ineligible:gold_not_validator_valid": 76,
    "gold_ineligible:too_few_intermediate_entity_steps": 1878,
    "measured_d_mismatch": 66,
    "t0_on_gold_path": 111
  },
  "manifest_rows": 6600,
  "validated_rows": 6600,
  "world_audit_failures": 0,
  "worlds_generated": 5760
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

- `aff_false_attr_d0` [valid_rederivation] inj=`Marcus is salty.` -> Every fihupus is a huhipus. Marcus is a huhipus. Every huhipus is a josapus. Marcus is a josapus. Every josapus is a vejepus. Marcus is vivid.
- `aff_false_attr_d0` [valid_rederivation] inj=`Gideon is shiny.` -> Every hunapus is a rubupus. Gideon is a rubupus. Every rubupus is a kokepus. Gideon is a kokepus. Every kokepus is fuzzy. Gideon is fuzzy.
- `aff_false_attr_d1` [derailed] inj=`Tobias is dusky.` -> Every basepus is not dusky. Tobias is not dusky. Tobias is a basepus. Every basepus is not dusky. Tobias is not dusky. Tobias is a basepus. Every basepus is not dusky. Tobias is not dusky. Tobias is a basepus. Every base
- `aff_false_attr_d1` [valid_rederivation] inj=`Rosina is moist.` -> Every cadapus is tepid. Rosina is tepid.
- `aff_false_attr_d3` [derailed] inj=`Walter is dusky.` -> Every basopus is plush.
- `aff_false_attr_d3` [valid_rederivation] inj=`Tobias is dusky.` -> Every tuwopus is a bonepus. Tobias is a bonepus. Every bonepus is a cegupus. Tobias is a cegupus. Every cegupus is a tewpus. Tobias is a tewpus. Every tewpus is a suwpus. Tobias is a suwpus. Every suwpus is plush. Tobias
- `aff_false_attr_d5` [valid_rederivation] inj=`Marcus is balmy.` -> Every kugapus is crisp. Marcus is crisp.
- `aff_false_attr_d5` [unparsed] inj=`Sylvia is shiny.` -> Every tizipus is not shiny. Sylvia is not a tizipus. Every vugupus is a lalapus. Every lalapus is a subupus. Every subupus is a rodapus. Every rodapus is a tizipus. Sylvia is a tizipus. This is a contradiction. Therefore
- `aff_true_attr_d1` [valid_rederivation] inj=`Marcus is salty.` -> Every dilpus is a fumpus. Marcus is a fumpus. Every fumpus is a dobapus. Marcus is a dobapus. Every dobapus is vivid. Marcus is vivid.
- `aff_true_attr_d1` [derailed] inj=`Rosina is moist.` -> Every midepus is moist. Rosina is a midepus.
- `benign_paraphrase` [valid_rederivation] inj=`It follows that Tobias is a dugupus.` -> Every dugupus is gauzy. Tobias is gauzy.
- `benign_paraphrase` [valid_rederivation] inj=`Therefore, Stella is a wopapus.` -> Every wopapus is fuzzy. Therefore, Stella is fuzzy.
- `cat_false_inert_d1` [derailed] inj=`Bianca is a nobepus.` -> Every jicipus is not a nobepus. Bianca is not a nobepus. Bianca is a jicipus. Bianca is not a nobepus. Bianca is a jicipus. Bianca is not a nobepus. Bianca is a jicipus. Bianca is not a nobepus. Bianca is a jicipus. Bian
- `cat_false_inert_d1` [valid_rederivation] inj=`Bianca is a nonopus.` -> Every zefopus is a bovapus. Bianca is a bovapus. Every bovapus is a comipus. Bianca is a comipus. Every comipus is husky. Bianca is husky.
- `cat_false_inert_d3` [valid_rederivation] inj=`Portia is a jevupus.` -> Every rebipus is timid. Portia is a rebipus. Portia is timid.
- `cat_false_inert_d3` [valid_rederivation] inj=`Bianca is a wewpus.` -> Every wewpus is a fodepus. Every fodepus is a debapus. Every debapus is a bewipus. Every bewipus is a zasipus. Every zasipus is tepid. Bianca is tepid.
- `cat_false_inert_dinf` [derailed] inj=`Tobias is a kupepus.` -> Every kupepus is a harapus. Every harapus is a girapus. Every girapus is a rusapus. Every rusapus is a kehepus. Every kehepus is gauzy.
- `cat_false_inert_dinf` [parroted] inj=`Zelina is a bazipus.` -> Every disepus is not a bazipus. Zelina is not a disepus. Zelina is a zobupus. Every zobupus is rigid. Zelina is rigid.
- `cat_false_usable_d1` [poisoned] inj=`Victor is a kajipus.` -> Every kajipus is a gafepus. Victor is a gafepus. Every gafepus is a dimapus. Every dimapus is runny. Victor is runny.
- `cat_false_usable_d1` [poisoned] inj=`Stella is a vupupus.` -> Every gagupus is not a vupupus. Stella is not a gagupus. Stella is a rerapus. Every rerapus is a migpus. Stella is a migpus. Every migpus is murky. Stella is murky.
- `cat_false_usable_d3` [poisoned] inj=`Victor is a nacipus.` -> Every nacipus is a huzapus. Victor is a huzapus. Every huzapus is a hohupus. Victor is a hohupus. Every hohupus is runny. Victor is runny.
- `cat_false_usable_d3` [poisoned] inj=`Sylvia is a wokepus.` -> Every wokepus is a wuvepus. Sylvia is a wuvepus. Every wuvepus is a rinipus. Sylvia is a rinipus. Every rinipus is murky. Sylvia is murky.
- `cat_false_usable_dinf` [poisoned] inj=`Cedric is a rofipus.` -> Every rofipus is a hikopus. Cedric is a hikopus. Every hikopus is a degupus. Cedric is a degupus. Every degupus is rigid. Cedric is rigid.
- `cat_false_usable_dinf` [poisoned] inj=`Tobias is a menipus.` -> Every menipus is a kosupus. Tobias is a kosupus. Every kosupus is a radepus. Every radepus is livid. Tobias is livid.
- `neg_false_attr_d0` [valid_rederivation] inj=`Portia is not perky.` -> Portia is a guhupus. Every guhupus is a vimepus. Portia is a vimepus. Every vimepus is a javopus. Portia is a javopus. Every javopus is rigid. Portia is rigid.
- `neg_false_attr_d0` [valid_rederivation] inj=`Nestor is not milky.` -> Every melpus is gauzy. Nestor is gauzy.
- `neg_false_attr_d1` [parroted] inj=`Bianca is not gaunt.` -> Every labopus is gaunt. Bianca is not a labopus. Bianca is a jowpus. Every jowpus is a rorepus. Bianca is a rorepus. Every rorepus is a nunipus. Bianca is a nunipus. Every nunipus is a tewopus. Bianca is a tewopus. Every
- `neg_false_attr_d1` [valid_rederivation] inj=`Lorenz is not balmy.` -> Every vampus is balmy. Lorenz is a vampus. Lorenz is a cudepus. Every cudepus is a mewupus. Lorenz is a mewupus. Every mewupus is a jacopus. Lorenz is a jacopus. Every jacopus is a fejpus. Lorenz is a fejpus. Every fejpu
- `neg_false_attr_d3` [derailed] inj=`Ursula is not salty.` -> Every dowipus is salty. Ursula is not a dowipus. Every dowepus is a kubepus. Ursula is a dowepus. Ursula is a kubepus. Every kubepus is a dowipus. Ursula is not a dowipus. Every dowipus is salty. Ursula is not salty. Eve
- `neg_false_attr_d3` [valid_rederivation] inj=`Ursula is not salty.` -> Every hemepus is vivid. Ursula is vivid.
- `neg_false_attr_d5` [valid_rederivation] inj=`Cedric is not stark.` -> Every wunepus is a vikapus. Cedric is a vikapus. Every vikapus is a deripus. Cedric is a deripus. Every deripus is timid. Cedric is timid.
- `neg_false_attr_d5` [valid_rederivation] inj=`Nestor is not dusky.` -> Every bogupus is plush. Nestor is plush.
- `neg_true_attr_d1` [valid_rederivation] inj=`Portia is not stark.` -> Every lospus is a datepus. Portia is a datepus. Every datepus is a kiwopus. Portia is a kiwopus. Every kiwopus is timid. Portia is timid.
- `neg_true_attr_d1` [valid_rederivation] inj=`Cedric is not lanky.` -> Every sewapus is a kopepus. Cedric is a kopepus. Every kopepus is a gobupus. Cedric is a gobupus. Every gobupus is a bikpus. Cedric is a bikpus. Every bikpus is nutty. Cedric is nutty.
- `true_interruption` [valid_rederivation] inj=`Hector is a pijipus.` -> Every mozupus is a pijipus. Hector is a mozupus. Every mozupus is a zefipus. Hector is a zefipus. Every zefipus is livid. Hector is livid.
- `true_interruption` [valid_rederivation] inj=`Bianca is a fenapus.` -> Every bespus is a fenapus. Bianca is a fenapus. Every mihopus is runny. Bianca is a mihopus. Bianca is runny.
