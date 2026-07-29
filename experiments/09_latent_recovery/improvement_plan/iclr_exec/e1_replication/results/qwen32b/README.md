# EXPD Matched-Gradient FULL RUN (runner stages) Report

Generated: 2026-07-24T18:50:08.456940+00:00
Backend: vllm | model: Qwen/Qwen2.5-32B-Instruct (5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the
externally timestamped PLAN2.** Timestamp-gate evidence: git commit `None`
(None), PLAN2.md sha256 `None`.
This file reports the runner's mechanical stages only; the registered adjudications
(H2', H3', H4, C-0) live in EXPD_FULL_REPORT.md + hypothesis_verdicts.json.

## Gate table (PLAN2 section 6)

| gate | value | target | pass |
|---|---|---|---|
| gold greedy solve rate | 1.0 (n=5760, wilson [0.9993, 1.0]) | >=0.60 | True |
| gold double filter (solved AND valid AND >=3 sites) | 0.9951 | - | - |
| parse rate: aff_false_attr_d0 | 0.9933 (unparsed n=3/450) | >=0.90 | True |
| parse rate: aff_false_attr_d1 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: aff_false_attr_d3 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: aff_false_attr_d5 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: aff_true_attr_d1 | 0.9933 (unparsed n=1/150) | >=0.90 | True |
| parse rate: benign_paraphrase | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: cat_false_inert_d1 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_inert_d3 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_inert_dinf | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_usable_d1 | 0.98 (unparsed n=6/300) | >=0.90 | True |
| parse rate: cat_false_usable_d3 | 0.9733 (unparsed n=8/300) | >=0.90 | True |
| parse rate: cat_false_usable_dinf | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: neg_false_attr_d0 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: neg_false_attr_d1 | 0.9956 (unparsed n=2/450) | >=0.90 | True |
| parse rate: neg_false_attr_d3 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: neg_false_attr_d5 | 0.9956 (unparsed n=2/450) | >=0.90 | True |
| parse rate: neg_true_attr_d1 | 1.0 (unparsed n=0/150) | >=0.90 | True |
| parse rate: true_interruption | 1.0 (unparsed n=0/450) | >=0.90 | True |
| anchor valid TOST(+-0.10): benign_paraphrase | diff=0.1511 ci90=[0.1233, 0.1789] | in margin | False |
| anchor doubt: benign_paraphrase | pilot=0.0 locked-HF=0.08 diff=-0.08 | n/a | backend-confounded, no verdict |
| anchor valid TOST(+-0.10): true_interruption | diff=0.14 ci90=[0.1123, 0.1677] | in margin | False |
| anchor doubt: true_interruption | pilot=0.0 locked-HF=0.02 diff=-0.02 | n/a | backend-confounded, no verdict |
| joint-solve mirrored pairs d0 | 0.9972 (359/360) | - | - |
| joint-solve mirrored pairs d1 | 0.9944 (358/360) | - | - |
| joint-solve mirrored pairs d2 | 0.9944 (358/360) | - | - |
| joint-solve mirrored pairs d3 | 0.9917 (357/360) | - | - |
| joint-solve mirrored pairs d5 | 0.9861 (355/360) | - | - |
| MC-7 identical gold prefix at injection (jointly-solved pairs) | 0.9989 (1785/1787) | - | - |

## Headline per-cell rates (family-cluster bootstrap 95% CIs)

| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 450 | 0.8933 [0.8578, 0.9244] | 0.0911 [0.0622, 0.1244] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.1 | 0.0067 |
| aff_false_attr_d1 | 450 | 0.8756 [0.8333, 0.9111] | 0.1222 [0.0889, 0.1622] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.1244 | 0.0 |
| aff_false_attr_d3 | 450 | 0.9733 [0.9533, 0.9911] | 0.0156 [0.0044, 0.0289] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0244 | 0.0022 |
| aff_false_attr_d5 | 450 | 0.9733 [0.9533, 0.9889] | 0.0222 [0.0089, 0.04] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0267 | 0.0 |
| aff_true_attr_d1 | 150 | 0.9933 [0.98, 1.0] | 0.0067 [0.0, 0.02] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0067 |
| benign_paraphrase | 450 | 1.0 [1.0, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| cat_false_inert_d1 | 300 | 0.9633 [0.9367, 0.9867] | 0.0433 [0.0167, 0.0733] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0367 | 0.0 |
| cat_false_inert_d3 | 300 | 0.9767 [0.9567, 0.9933] | 0.0233 [0.0067, 0.0433] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0233 | 0.0 |
| cat_false_inert_dinf | 300 | 1.0 [1.0, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| cat_false_usable_d1 | 300 | 0.6167 [0.55, 0.6767] | 0.3167 [0.2567, 0.3767] | 0.0167 [0.0033, 0.0333] | 0.0033 | 0.0233 | 0.3233 | 0.02 |
| cat_false_usable_d3 | 300 | 0.65 [0.58, 0.7133] | 0.28 [0.22, 0.3433] | 0.0133 [0.0033, 0.0267] | 0.0 | 0.0033 | 0.3067 | 0.0267 |
| cat_false_usable_dinf | 300 | 0.8933 [0.8567, 0.9267] | 0.0767 [0.05, 0.1067] | 0.01 [0.0, 0.0233] | 0.0 | 0.0 | 0.0967 | 0.0 |
| neg_false_attr_d0 | 450 | 0.8222 [0.7822, 0.8622] | 0.1067 [0.08, 0.1378] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.1756 | 0.0022 |
| neg_false_attr_d1 | 450 | 0.9622 [0.94, 0.98] | 0.02 [0.0067, 0.0356] | 0.0067 [0.0, 0.0133] | 0.0067 | 0.0 | 0.0267 | 0.0044 |
| neg_false_attr_d3 | 450 | 0.9311 [0.9022, 0.9578] | 0.0333 [0.0156, 0.0533] | 0.0222 [0.0089, 0.0378] | 0.0222 | 0.0 | 0.0444 | 0.0022 |
| neg_false_attr_d5 | 450 | 0.9378 [0.9067, 0.9644] | 0.0356 [0.0156, 0.0622] | 0.0089 [0.0, 0.0222] | 0.0089 | 0.0 | 0.0489 | 0.0044 |
| neg_true_attr_d1 | 150 | 1.0 [1.0, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| true_interruption | 450 | 0.9956 [0.9865, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0244 | 0.0022 | 0.0022 | 0.0 |

## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)

- aff_false_attr_d1: doubt=0.1222 [0.0889, 0.1622] (n=450, valid=0.8756)
- aff_false_attr_d3: doubt=0.0156 [0.0044, 0.0289] (n=450, valid=0.9733)
- aff_false_attr_d5: doubt=0.0222 [0.0089, 0.04] (n=450, valid=0.9733)

## MAJOR-2 reuse liveness (H4 MDE input)

- cat_false_inert_d1: derivational=0.0 [0.0, 0.0] echo=0.0 doubt=0.0433 valid=0.9633 (n=300)
- cat_false_usable_d1: derivational=0.0167 [0.0033, 0.0333] echo=0.0033 doubt=0.3167 valid=0.6167 (n=300)

## Funnel (CONSORT, MC-4)

```
{
  "gold_attempted": 5760,
  "gold_eligible": 5732,
  "gold_solved": 5760,
  "gold_valid_rederivation": 5760,
  "injection_audit_rejections": {
    "gold_ineligible:too_few_intermediate_entity_steps": 4
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

- `aff_false_attr_d0` [valid_rederivation] inj=`Bianca is downy.` -> Every zocpus is a kimipus. Bianca is a kimipus. Every kimipus is a mihopus. Bianca is a mihopus. Every mihopus is runny. Bianca is runny.
- `aff_false_attr_d0` [valid_rederivation] inj=`Nestor is dusky.` -> Every cuzepus is a putpus. Nestor is a putpus. Every putpus is a kojpus. Nestor is a kojpus. Every kojpus is plush. Nestor is plush.
- `aff_false_attr_d1` [valid_rederivation] inj=`Hector is dusky.` -> Every batopus is a losupus. Hector is a losupus. Every losupus is a pitepus. Hector is a pitepus. Every pitepus is a nempus. Hector is a nempus. Every nempus is a magepus. Hector is a magepus. Every magepus is plush. Hec
- `aff_false_attr_d1` [valid_rederivation] inj=`Bianca is moist.` -> Every sigepus is tepid. Bianca is tepid.
- `aff_false_attr_d3` [valid_rederivation] inj=`Amelia is balmy.` -> Every hejapus is crisp. Amelia is crisp.
- `aff_false_attr_d3` [valid_rederivation] inj=`Cedric is lanky.` -> Every jonepus is a hevopus. Cedric is a hevopus. Every hevopus is a dorapus. Cedric is a dorapus. Every dorapus is a cofupus. Cedric is a cofupus. Every cofupus is a satipus. Cedric is a satipus. Every satipus is a kejpu
- `aff_false_attr_d5` [valid_rederivation] inj=`Miriam is downy.` -> Every pizupus is runny. Miriam is runny.
- `aff_false_attr_d5` [valid_rederivation] inj=`Cedric is perky.` -> Every kuhpus is a megapus. Cedric is a megapus. Every megapus is a kemepus. Cedric is a kemepus. Every kemepus is a cetopus. Cedric is a cetopus. Every cetopus is a muropus. Cedric is a muropus. Every muropus is rigid. C
- `aff_true_attr_d1` [valid_rederivation] inj=`Marcus is salty.` -> Every huhipus is a josapus. Marcus is a josapus. Every josapus is a vejepus. Marcus is a vejepus. Every vejepus is vivid. Marcus is vivid.
- `aff_true_attr_d1` [valid_rederivation] inj=`Tobias is milky.` -> Tobias is a kifepus. Every kifepus is a sesapus. Tobias is a sesapus. Every sesapus is a hamipus. Tobias is a hamipus. Every hamipus is gauzy. Tobias is gauzy.
- `benign_paraphrase` [valid_rederivation] inj=`It follows that Nestor is a revepus.` -> Every revepus is gauzy. Nestor is gauzy.
- `benign_paraphrase` [valid_rederivation] inj=`Therefore, Stella is a wopapus.` -> Every wopapus is fuzzy. Stella is fuzzy.
- `cat_false_inert_d1` [valid_rederivation] inj=`Ingrid is a pefopus.` -> Every cecupus is a gecipus. Ingrid is a gecipus. Every gecipus is a judapus. Ingrid is a judapus. Every judapus is tangy. Ingrid is tangy.
- `cat_false_inert_d1` [valid_rederivation] inj=`Nestor is a beropus.` -> Every mumpus is a wivapus. Nestor is a wivapus. Every wivapus is a jezopus. Nestor is a jezopus. Every jezopus is gauzy. Nestor is gauzy.
- `cat_false_inert_d3` [valid_rederivation] inj=`Lorenz is a senipus.` -> Every nanpus is a safepus. Lorenz is a safepus. Every safepus is a lepipus. Lorenz is a lepipus. Every lepipus is vivid. Lorenz is vivid.
- `cat_false_inert_d3` [valid_rederivation] inj=`Gideon is a dilepus.` -> Every vohapus is a rinipus. Gideon is a rinipus. Every rinipus is a latpus. Gideon is a latpus. Every latpus is a zepupus. Gideon is a zepupus. Every zepupus is a bicapus. Gideon is a bicapus. Every bicapus is a fedepus.
- `cat_false_inert_dinf` [valid_rederivation] inj=`Gideon is a wanopus.` -> Every jajapus is a vohpus. Gideon is a vohpus. Every vohpus is a forepus. Gideon is a forepus. Every forepus is a cumapus. Gideon is a cumapus. Every cumapus is a besepus. Gideon is a besepus. Every besepus is a socupus.
- `cat_false_inert_dinf` [valid_rederivation] inj=`Rosina is a kudupus.` -> Every warpus is a kolupus. Rosina is a kolupus. Every kolupus is a kavupus. Rosina is a kavupus. Every kavupus is husky. Rosina is husky.
- `cat_false_usable_d1` [valid_rederivation] inj=`Rosina is a zekupus.` -> Every zekupus is a rofepus. Every rofepus is a bakapus. Every bakapus is runny. Rosina is a momipus. Every momipus is a zilupus. Rosina is a zilupus. Every zilupus is a pufopus. Rosina is a pufopus. Every pufopus is a za
- `cat_false_usable_d1` [valid_rederivation] inj=`Lorenz is a tebopus.` -> Every ladpus is a tafepus. Lorenz is a tafepus. Every tafepus is a sihapus. Lorenz is a sihapus. Every sihapus is a zugapus. Lorenz is a zugapus. Every zugapus is zesty. Lorenz is zesty.
- `cat_false_usable_d3` [derailed] inj=`Amelia is a gokapus.` -> Every vuvopus is a bampus. Amelia is a bampus. Every bampus is a hekpus. Amelia is a hekpus. Every hekpus is a dacupus. Amelia is a dacupus. Every dacupus is a cotopus. Amelia is a cotopus. Every cotopus is a nogopus. Am
- `cat_false_usable_d3` [valid_rederivation] inj=`Ursula is a kizapus.` -> Ursula is a zumopus. Every zumopus is a josipus. Ursula is a josipus. Every josipus is a tibpus. Ursula is a tibpus. Every tibpus is a hospus. Ursula is a hospus. Every hospus is a pozepus. Ursula is a pozepus. Every poz
- `cat_false_usable_dinf` [valid_rederivation] inj=`Amelia is a cispus.` -> Every megopus is a zagapus. Amelia is a zagapus. Every zagapus is a tarepus. Amelia is a tarepus. Every tarepus is crisp. Amelia is crisp.
- `cat_false_usable_dinf` [valid_rederivation] inj=`Zelina is a wiwapus.` -> Every jewepus is a naspus. Zelina is a naspus. Every naspus is a riwapus. Zelina is a riwapus. Every riwapus is rigid. Zelina is rigid.
- `neg_false_attr_d0` [valid_rederivation] inj=`Quincy is not perky.` -> Quincy is a jodupus. Every jodupus is a sirpus. Quincy is a sirpus. Every sirpus is a mefepus. Quincy is a mefepus. Every mefepus is rigid. Quincy is rigid.
- `neg_false_attr_d0` [valid_rederivation] inj=`Quincy is not lanky.` -> Quincy is a sufopus. Every sufopus is nutty. Quincy is nutty.
- `neg_false_attr_d1` [valid_rederivation] inj=`Lorenz is not waxen.` -> Lorenz is a binpus. Every binpus is a coropus. Lorenz is a coropus. Every coropus is a vovupus. Lorenz is a vovupus. Every vovupus is a sifpus. Lorenz is a sifpus. Every sifpus is a jopepus. Lorenz is a jopepus. Every jo
- `neg_false_attr_d1` [valid_rederivation] inj=`Cedric is not lanky.` -> Every dodepus is a johapus. Cedric is a johapus. Every johapus is a nepipus. Cedric is a nepipus. Every nepipus is a nepapus. Cedric is a nepapus. Every nepapus is a golapus. Cedric is a golapus. Every golapus is a tugap
- `neg_false_attr_d3` [poisoned] inj=`Hector is not dusky.` -> Every tawapus is a zirapus. Hector is a zirapus. Every zirapus is a surpus. Hector is a surpus. Every surpus is a hokupus. Hector is a hokupus. Every hokupus is a cimapus. Hector is a cimapus. Every cimapus is plush. Hec
- `neg_false_attr_d3` [valid_rederivation] inj=`Bianca is not downy.` -> Bianca is a mihopus. Every mihopus is runny. Bianca is runny.
- `neg_false_attr_d5` [valid_rederivation] inj=`Portia is not perky.` -> Portia is a tiwopus. Every tiwopus is a fivapus. Portia is a fivapus. Every fivapus is a holipus. Portia is a holipus. Every holipus is rigid. Portia is rigid.
- `neg_false_attr_d5` [valid_rederivation] inj=`Victor is not moist.` -> Victor is a mocupus. Every mocupus is tepid. Victor is tepid.
- `neg_true_attr_d1` [valid_rederivation] inj=`Zelina is not perky.` -> Zelina is a jinipus. Every jinipus is a hakopus. Zelina is a hakopus. Every hakopus is a pippus. Zelina is a pippus. Every pippus is rigid. Zelina is rigid.
- `neg_true_attr_d1` [valid_rederivation] inj=`Zelina is not stark.` -> Zelina is a domopus. Every domopus is a fogupus. Zelina is a fogupus. Every fogupus is a nawepus. Zelina is a nawepus. Every nawepus is timid. Zelina is timid.
- `true_interruption` [valid_rederivation] inj=`Ingrid is a numpus.` -> Every puhipus is a numpus. Ingrid is a jawopus. Every jawopus is a sapopus. Ingrid is a sapopus. Every sapopus is a cijpus. Ingrid is a cijpus. Every cijpus is a pajipus. Ingrid is a pajipus. Every pajipus is a vocepus. 
- `true_interruption` [valid_rederivation] inj=`Zelina is a pefepus.` -> Every domopus is a fogupus. Zelina is a fogupus. Every fogupus is a nawepus. Zelina is a nawepus. Every nawepus is timid. Zelina is timid.
