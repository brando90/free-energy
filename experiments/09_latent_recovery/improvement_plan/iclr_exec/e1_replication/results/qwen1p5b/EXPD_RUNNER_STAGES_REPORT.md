# EXPD Matched-Gradient FULL RUN (runner stages) Report

Generated: 2026-07-24T18:15:36.511838+00:00
Backend: vllm | model: Qwen/Qwen2.5-1.5B-Instruct (989aa7980e4cf806f80c7fef2b1adb7bc71aa306) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: FULL confirmatory grid (PLAN2 v1.1 section C.5), generated under the
externally timestamped PLAN2.** Timestamp-gate evidence: git commit `None`
(None), PLAN2.md sha256 `None`.
This file reports the runner's mechanical stages only; the registered adjudications
(H2', H3', H4, C-0) live in EXPD_FULL_REPORT.md + hypothesis_verdicts.json.

## Gate table (PLAN2 section 6)

| gate | value | target | pass |
|---|---|---|---|
| gold greedy solve rate | 0.8828 (n=11520, wilson [0.8768, 0.8886]) | >=0.60 | True |
| gold double filter (solved AND valid AND >=3 sites) | 0.2829 | - | - |
| parse rate: aff_false_attr_d0 | 0.9977 (unparsed n=1/441) | >=0.90 | True |
| parse rate: aff_false_attr_d1 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: aff_false_attr_d3 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: aff_false_attr_d5 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: aff_true_attr_d1 | 1.0 (unparsed n=0/150) | >=0.90 | True |
| parse rate: benign_paraphrase | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: cat_false_inert_d1 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_inert_d3 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_inert_dinf | 0.9967 (unparsed n=1/300) | >=0.90 | True |
| parse rate: cat_false_usable_d1 | 0.9967 (unparsed n=1/300) | >=0.90 | True |
| parse rate: cat_false_usable_d3 | 1.0 (unparsed n=0/300) | >=0.90 | True |
| parse rate: cat_false_usable_dinf | 0.9967 (unparsed n=1/300) | >=0.90 | True |
| parse rate: neg_false_attr_d0 | 0.997 (unparsed n=1/336) | >=0.90 | True |
| parse rate: neg_false_attr_d1 | 1.0 (unparsed n=0/450) | >=0.90 | True |
| parse rate: neg_false_attr_d3 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: neg_false_attr_d5 | 0.9978 (unparsed n=1/450) | >=0.90 | True |
| parse rate: neg_true_attr_d1 | 0.9933 (unparsed n=1/150) | >=0.90 | True |
| parse rate: true_interruption | 0.9956 (unparsed n=2/450) | >=0.90 | True |
| anchor valid TOST(+-0.10): benign_paraphrase | diff=0.1378 ci90=[0.1086, 0.167] | in margin | False |
| anchor doubt: benign_paraphrase | pilot=0.0 locked-HF=0.08 diff=-0.08 | n/a | backend-confounded, no verdict |
| anchor valid TOST(+-0.10): true_interruption | diff=-0.0578 ci90=[-0.0992, -0.0164] | in margin | True |
| anchor doubt: true_interruption | pilot=0.0 locked-HF=0.02 diff=-0.02 | n/a | backend-confounded, no verdict |
| joint-solve mirrored pairs d0 | 0.1264 (91/720) | - | - |
| joint-solve mirrored pairs d1 | 0.3 (216/720) | - | - |
| joint-solve mirrored pairs d2 | 0.2431 (175/720) | - | - |
| joint-solve mirrored pairs d3 | 0.2569 (185/720) | - | - |
| joint-solve mirrored pairs d5 | 0.2597 (187/720) | - | - |
| MC-7 identical gold prefix at injection (jointly-solved pairs) | 0.9649 (824/854) | - | - |

## Headline per-cell rates (family-cluster bootstrap 95% CIs)

| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d0 | 441 | 0.5782 [0.5283, 0.6259] | 0.0 [0.0, 0.0] | 0.0045 [0.0, 0.0113] | 0.0045 | 0.1293 | 0.2857 | 0.0023 |
| aff_false_attr_d1 | 450 | 0.6156 [0.5644, 0.6644] | 0.0 [0.0, 0.0] | 0.0044 [0.0, 0.0111] | 0.0067 | 0.0733 | 0.3067 | 0.0 |
| aff_false_attr_d3 | 450 | 0.5844 [0.54, 0.6311] | 0.0 [0.0, 0.0] | 0.0178 [0.0067, 0.0311] | 0.0267 | 0.1067 | 0.2911 | 0.0 |
| aff_false_attr_d5 | 450 | 0.6089 [0.5622, 0.6578] | 0.0 [0.0, 0.0] | 0.02 [0.0067, 0.0333] | 0.04 | 0.0756 | 0.2956 | 0.0 |
| aff_true_attr_d1 | 150 | 0.8733 [0.82, 0.9267] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.08 | 0.0667 | 0.06 | 0.0 |
| benign_paraphrase | 450 | 0.9867 [0.9701, 0.9979] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0067 | 0.0 | 0.0111 | 0.0022 |
| cat_false_inert_d1 | 300 | 0.46 [0.3967, 0.5233] | 0.0 [0.0, 0.0] | 0.0167 [0.0033, 0.0333] | 0.02 | 0.1667 | 0.3567 | 0.0 |
| cat_false_inert_d3 | 300 | 0.4567 [0.3933, 0.5233] | 0.0 [0.0, 0.0] | 0.0033 [0.0, 0.01] | 0.01 | 0.2133 | 0.3267 | 0.0 |
| cat_false_inert_dinf | 300 | 0.4167 [0.3533, 0.4767] | 0.0 [0.0, 0.0] | 0.0133 [0.0033, 0.0267] | 0.0233 | 0.28 | 0.2867 | 0.0033 |
| cat_false_usable_d1 | 300 | 0.23 [0.1833, 0.2833] | 0.0 [0.0, 0.0] | 0.7133 [0.66, 0.7633] | 0.03 | 0.0167 | 0.0367 | 0.0033 |
| cat_false_usable_d3 | 300 | 0.2267 [0.18, 0.2767] | 0.0 [0.0, 0.0] | 0.7333 [0.6867, 0.78] | 0.0233 | 0.02 | 0.02 | 0.0 |
| cat_false_usable_dinf | 300 | 0.2567 [0.2133, 0.3] | 0.0 [0.0, 0.0] | 0.6867 [0.6433, 0.7267] | 0.0067 | 0.02 | 0.0333 | 0.0033 |
| neg_false_attr_d0 | 336 | 0.5327 [0.497, 0.5685] | 0.0 [0.0, 0.0] | 0.0833 [0.0565, 0.1131] | 0.0923 | 0.253 | 0.128 | 0.003 |
| neg_false_attr_d1 | 450 | 0.5 [0.4622, 0.5356] | 0.0 [0.0, 0.0] | 0.0978 [0.0711, 0.1267] | 0.12 | 0.2378 | 0.1644 | 0.0 |
| neg_false_attr_d3 | 450 | 0.5778 [0.5467, 0.6067] | 0.0 [0.0, 0.0] | 0.1467 [0.1133, 0.18] | 0.1733 | 0.1489 | 0.1244 | 0.0022 |
| neg_false_attr_d5 | 450 | 0.5867 [0.5489, 0.62] | 0.0 [0.0, 0.0] | 0.1022 [0.0756, 0.1311] | 0.1289 | 0.14 | 0.1689 | 0.0022 |
| neg_true_attr_d1 | 150 | 0.94 [0.9, 0.9733] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0533 | 0.0333 | 0.02 | 0.0067 |
| true_interruption | 450 | 0.7978 [0.7303, 0.8578] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0756 | 0.1 | 0.0978 | 0.0044 |

## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)

- aff_false_attr_d1: doubt=0.0 [0.0, 0.0] (n=450, valid=0.6156)
- aff_false_attr_d3: doubt=0.0 [0.0, 0.0] (n=450, valid=0.5844)
- aff_false_attr_d5: doubt=0.0 [0.0, 0.0] (n=450, valid=0.6089)

## MAJOR-2 reuse liveness (H4 MDE input)

- cat_false_inert_d1: derivational=0.0167 [0.0033, 0.0333] echo=0.02 doubt=0.0 valid=0.46 (n=300)
- cat_false_usable_d1: derivational=0.7133 [0.66, 0.7633] echo=0.03 doubt=0.0 valid=0.23 (n=300)

## Funnel (CONSORT, MC-4)

```
{
  "gold_attempted": 11520,
  "gold_eligible": 3259,
  "gold_solved": 10170,
  "gold_valid_rederivation": 9926,
  "injection_audit_rejections": {
    "gold_ineligible:gold_not_solved": 855,
    "gold_ineligible:gold_not_validator_valid": 308,
    "gold_ineligible:too_few_intermediate_entity_steps": 5131,
    "measured_d_mismatch": 58,
    "t0_on_gold_path": 45
  },
  "manifest_rows": 6477,
  "validated_rows": 6477,
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

- `aff_false_attr_d0` [valid_rederivation] inj=`Miriam is gaunt.` -> Every wihupus is a gatopus. Miriam is a gatopus. Every gatopus is a tazpus. Miriam is a tazpus. Every tazpus is husky. Miriam is husky.
- `aff_false_attr_d0` [valid_rederivation] inj=`Victor is downy.` -> Every cepupus is a zukipus. Victor is a zukipus. Every zukipus is a bapupus. Victor is a bapupus. Every bapupus is runny. Victor is runny.
- `aff_false_attr_d1` [valid_rederivation] inj=`Zelina is lanky.` -> Every gopapus is a kecipus. Zelina is a kecipus. Every kecipus is a wudupus. Zelina is a wudupus. Every wudupus is a vudpus. Zelina is a vudpus. Every vudpus is nutty. Zelina is nutty.
- `aff_false_attr_d1` [valid_rederivation] inj=`Stella is vapid.` -> Every vapid is not murky. Stella is murky.
- `aff_false_attr_d3` [valid_rederivation] inj=`Miriam is downy.` -> Every nudpus is runny. Miriam is runny.
- `aff_false_attr_d3` [valid_rederivation] inj=`Bianca is downy.` -> Every docpus is a ladepus. Bianca is a ladepus. Every ladepus is a hizupus. Bianca is a hizupus. Every hizupus is a rakpus. Bianca is a rakpus. Every rakpus is a zocpus. Bianca is a zocpus. Every zocpus is runny. Bianca 
- `aff_false_attr_d5` [derailed] inj=`Tobias is dusky.` -> Every tigipus is plush. Tobias is not plush.
- `aff_false_attr_d5` [parroted] inj=`Bianca is downy.` -> Every vudepus is a bagopus. Bianca is not a bagopus. Bianca is runny.
- `aff_true_attr_d1` [valid_rederivation] inj=`Gideon is vapid.` -> Every vapid is murky. Gideon is murky.
- `aff_true_attr_d1` [valid_rederivation] inj=`Stella is vapid.` -> Every vapid is murky. Stella is murky.
- `benign_paraphrase` [valid_rederivation] inj=`Therefore, Lorenz is a vumepus.` -> Every vumepus is crisp. Lorenz is crisp.
- `benign_paraphrase` [valid_rederivation] inj=`It follows that Portia is not stark.` -> Portia is not stark. Portia is timid.
- `cat_false_inert_d1` [derailed] inj=`Rosina is a dilopus.` -> Every dilopus is not a husky. Rosina is not husky.
- `cat_false_inert_d1` [valid_rederivation] inj=`Gideon is a bovpus.` -> Every bovpus is murky. Gideon is murky.
- `cat_false_inert_d3` [valid_rederivation] inj=`Rosina is a kudupus.` -> Every kudupus is husky. Rosina is husky.
- `cat_false_inert_d3` [derailed] inj=`Zelina is a dabapus.` -> Every dabapus is not a decupus. Zelina is not nutty.
- `cat_false_inert_dinf` [parroted] inj=`Ingrid is a vokpus.` -> Every vokpus is a guppus. Ingrid is a guppus. Every guppus is a bolapus. Ingrid is a bolapus. Every bolapus is fuzzy. Ingrid is fuzzy.
- `cat_false_inert_dinf` [derailed] inj=`Lorenz is a rabpus.` -> Lorenz is not crisp.
- `cat_false_usable_d1` [valid_rederivation] inj=`Rosina is a jejpus.` -> Every jejpus is a mokopus. Every mokopus is a morapus. Rosina is not a jejpus. Rosina is runny.
- `cat_false_usable_d1` [poisoned] inj=`Zelina is a ciwipus.` -> Every ciwipus is a tavupus. Zelina is a tavupus. Every tavupus is a sapipus. Zelina is a sapipus. Every sapipus is nutty. Zelina is nutty.
- `cat_false_usable_d3` [poisoned] inj=`Sylvia is a gelapus.` -> Sylvia is a budepus. Sylvia is a nojepus. Sylvia is tangy.
- `cat_false_usable_d3` [valid_rederivation] inj=`Ingrid is a jazapus.` -> Every jazapus is a kasupus. Ingrid is a vahapus. Every vahapus is tangy. Ingrid is tangy.
- `cat_false_usable_dinf` [poisoned] inj=`Hector is a jujapus.` -> Hector is a wibipus. Hector is a pohpus. Hector is livid.
- `cat_false_usable_dinf` [poisoned] inj=`Victor is a jewpus.` -> Victor is a nitupus. Every nitupus is a digepus. Victor is a digepus. Every digepus is husky. Victor is husky.
- `neg_false_attr_d0` [parroted] inj=`Sylvia is not slick.` -> Every turpus is a cupepus. Sylvia is not a cupepus. Every cupepus is a hufopus. Sylvia is not a hufopus. Every hufopus is a rusopus. Sylvia is not a rusopus. Every rusopus is a sawipus. Sylvia is not a sawipus. Every saw
- `neg_false_attr_d0` [poisoned] inj=`Hector is not dusky.` -> Hector is a batopus. Every batopus is a losupus. Hector is not dusky. Hector is a losupus. Every losupus is a pitepus. Hector is not dusky. Hector is a pitepus. Every pitepus is a nempus. Hector is not dusky. Hector is a
- `neg_false_attr_d1` [derailed] inj=`Lorenz is not salty.` -> Lorenz is a fafopus. Every fafopus is salty. Lorenz is not vivid.
- `neg_false_attr_d1` [valid_rederivation] inj=`Zelina is not perky.` -> Every mabapus is rigid. Zelina is rigid.
- `neg_false_attr_d3` [valid_rederivation] inj=`Nestor is not dusky.` -> Nestor is a sasapus. Every sasapus is a dizopus. Nestor is a dizopus. Every dizopus is plush. Nestor is plush.
- `neg_false_attr_d3` [valid_rederivation] inj=`Portia is not stark.` -> Every vocepus is a dibopus. Portia is a dibopus. Every dibopus is a sugipus. Portia is a sugipus. Every sugipus is timid. Portia is timid.
- `neg_false_attr_d5` [poisoned] inj=`Hector is not dusky.` -> Hector is a sohapus. Every sohapus is a muwpus. Hector is not dusky. Hector is a muwpus. Every muwpus is a mupapus. Hector is not dusky. Hector is a mupapus. Every mupapus is a habipus. Hector is not dusky. Hector is a h
- `neg_false_attr_d5` [derailed] inj=`Nestor is not milky.` -> Every talipus is a ledepus. Nestor is not gauzy. Nestor is a mekpus. Every mekpus is a witupus. Nestor is not gauzy. Nestor is a fapepus. Every fapepus is a wizepus. Nestor is not gauzy. Nestor is a wizepus. Every wizepu
- `neg_true_attr_d1` [valid_rederivation] inj=`Portia is not stark.` -> Every balepus is a pocopus. Portia is a pocopus. Every pocopus is a husipus. Portia is a husipus. Every husipus is timid. Portia is timid.
- `neg_true_attr_d1` [valid_rederivation] inj=`Rosina is not moist.` -> Every radipus is a ketapus. Rosina is a ketapus. Every ketapus is a cahepus. Rosina is a cahepus. Every cahepus is a cuvpus. Rosina is tepid.
- `true_interruption` [derailed] inj=`Walter is a macpus.` -> Every macpus is a koppus. Walter is koppus. Every koppus is a redipus. Walter is a redipus. Every redipus is a larepus. Walter is a larepus. Every larepus is a lubopus. Walter is a lubopus. Every lubopus is a cohipus. Wa
- `true_interruption` [valid_rederivation] inj=`Ursula is a gecpus.` -> Every gecpus is vivid. Ursula is vivid.
