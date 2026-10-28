# EXPD Matched-Gradient PILOT Report

Generated: 2026-07-02T20:11:22.221784+00:00
Backend: vllm | model: Qwen/Qwen2.5-7B-Instruct (a09a35458c702b33eeacc393d103063234e8bc28) | prompt sha256: f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c

**Status: 200-world pilot, pre-registration-exempt (PLAN2 section 8).** The full
confirmatory run is gated on the externally timestamped PLAN2 (section 9.10) and was NOT run.

## Gate table (PLAN2 section 6)

| gate | value | target | pass |
|---|---|---|---|
| gold greedy solve rate | 0.935 (n=200, wilson [0.892, 0.9616]) | >=0.60 | True |
| gold double filter (solved AND valid AND >=3 sites) | 0.9 | - | - |
| parse rate: aff_false_attr_d1 | 1.0 (unparsed n=0/35) | >=0.90 | True |
| parse rate: aff_false_attr_d3 | 1.0 (unparsed n=0/17) | >=0.90 | True |
| parse rate: aff_false_attr_d5 | 1.0 (unparsed n=0/15) | >=0.90 | True |
| parse rate: benign_paraphrase | 1.0 (unparsed n=0/35) | >=0.90 | True |
| parse rate: cat_false_inert_d1 | 0.913 (unparsed n=2/23) | >=0.90 | True |
| parse rate: cat_false_usable_d1 | 1.0 (unparsed n=0/23) | >=0.90 | True |
| parse rate: neg_false_attr_d1 | 1.0 (unparsed n=0/34) | >=0.90 | True |
| parse rate: true_interruption | 1.0 (unparsed n=0/35) | >=0.90 | True |
| anchor valid TOST(+-0.10): benign_paraphrase | diff=0.1511 ci90=[0.1233, 0.1789] | in margin | False |
| anchor doubt: benign_paraphrase | pilot=0.0 locked-HF=0.08 diff=-0.08 | n/a | backend-confounded, no verdict |
| anchor valid TOST(+-0.10): true_interruption | diff=0.0301 ci90=[-0.0625, 0.1227] | in margin | False |
| anchor doubt: true_interruption | pilot=0.0 locked-HF=0.02 diff=-0.02 | n/a | backend-confounded, no verdict |
| joint-solve mirrored pairs d1 | 0.9714 (34/35) | - | - |
| joint-solve mirrored pairs d3 | 0.8 (16/20) | - | - |
| joint-solve mirrored pairs d5 | 0.75 (15/20) | - | - |
| MC-7 identical gold prefix at injection (jointly-solved pairs) | 0.9077 (59/65) | - | - |

## Headline per-cell rates (family-cluster bootstrap 95% CIs)

| cell | n | valid | doubt | derivational inj-dep | echo | parroted | derailed | unparsed |
|---|---|---|---|---|---|---|---|---|
| aff_false_attr_d1 | 35 | 0.9714 [0.9143, 1.0] | 0.0 [0.0, 0.0] | 0.0286 [0.0, 0.0857] | 0.0286 | 0.0 | 0.0 | 0.0 |
| aff_false_attr_d3 | 17 | 1.0 [1.0, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| aff_false_attr_d5 | 15 | 1.0 [1.0, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| benign_paraphrase | 35 | 1.0 [1.0, 1.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| cat_false_inert_d1 | 23 | 0.8696 [0.6957, 1.0] | 0.2174 [0.087, 0.3913] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0435 | 0.087 |
| cat_false_usable_d1 | 23 | 0.2174 [0.087, 0.3913] | 0.0 [0.0, 0.0] | 0.7826 [0.6087, 0.913] | 0.0 | 0.0 | 0.0 | 0.0 |
| neg_false_attr_d1 | 34 | 1.0 [1.0, 1.0] | 0.0294 [0.0, 0.0882] | 0.0 [0.0, 0.0] | 0.0 | 0.0 | 0.0 | 0.0 |
| true_interruption | 35 | 0.8857 [0.7879, 0.9714] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 | 0.0571 | 0.0571 | 0.0 |

## Descriptive doubt gradient (aff_false_attr, negation-free planted sentence)

- aff_false_attr_d1: doubt=0.0 [0.0, 0.0] (n=35, valid=0.9714)
- aff_false_attr_d3: doubt=0.0 [0.0, 0.0] (n=17, valid=1.0)
- aff_false_attr_d5: doubt=0.0 [0.0, 0.0] (n=15, valid=1.0)

## MAJOR-2 reuse liveness (H4 MDE input)

- cat_false_inert_d1: derivational=0.0 [0.0, 0.0] echo=0.0 doubt=0.2174 valid=0.8696 (n=23)
- cat_false_usable_d1: derivational=0.7826 [0.6087, 0.913] echo=0.0 doubt=0.0 valid=0.2174 (n=23)

## Funnel (CONSORT, MC-4)

```
{
  "gold_attempted": 200,
  "gold_eligible": 180,
  "gold_solved": 187,
  "gold_valid_rederivation": 182,
  "injection_audit_rejections": {
    "gold_ineligible:gold_not_solved": 7,
    "gold_ineligible:gold_not_validator_valid": 3,
    "gold_ineligible:too_few_intermediate_entity_steps": 2,
    "measured_d_mismatch": 1,
    "t0_on_gold_path": 14
  },
  "manifest_rows": 217,
  "validated_rows": 217,
  "world_audit_failures": 0,
  "worlds_generated": 200
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

- `aff_false_attr_d1` [valid_rederivation] inj=`Rosina is gaunt.` -> Every weppus is a campus. Rosina is a campus. Every campus is a haspus. Rosina is a haspus. Every haspus is husky. Rosina is husky.
- `aff_false_attr_d1` [valid_rederivation] inj=`Stella is shiny.` -> Every canpus is a judpus. Stella is a judpus. Every judpus is a gugpus. Stella is a gugpus. Every gugpus is fuzzy. Stella is fuzzy.
- `aff_false_attr_d3` [valid_rederivation] inj=`Zelina is stark.` -> Every nunpus is a kavpus. Zelina is a kavpus. Every kavpus is a zunpus. Zelina is a zunpus. Every zunpus is timid. Zelina is timid.
- `aff_false_attr_d3` [valid_rederivation] inj=`Marcus is salty.` -> Every cetpus is a hekpus. Marcus is a hekpus. Every hekpus is a jofpus. Marcus is a jofpus. Every jofpus is vivid. Marcus is vivid.
- `aff_false_attr_d5` [valid_rederivation] inj=`Cedric is perky.` -> Every pewpus is a pacpus. Cedric is a pacpus. Every pacpus is a litpus. Cedric is a litpus. Every litpus is rigid. Cedric is rigid.
- `aff_false_attr_d5` [valid_rederivation] inj=`Rosina is gaunt.` -> Every weppus is a campus. Rosina is a campus. Every campus is a haspus. Rosina is a haspus. Every haspus is husky. Rosina is husky.
- `benign_paraphrase` [valid_rederivation] inj=`It follows that Portia is a jimpus.` -> Every jimpus is a dospus. Portia is a dospus. Every dospus is a tocpus. Portia is a tocpus. Every tocpus is timid. Portia is timid.
- `benign_paraphrase` [valid_rederivation] inj=`Therefore, Hector is a sefpus.` -> Every sefpus is a hempus. Therefore, Hector is a hempus. Every hempus is a cinpus. Therefore, Hector is a cinpus. Every cinpus is plush. Therefore, Hector is plush.
- `cat_false_inert_d1` [unparsed] inj=`Sylvia is a ribpus.` -> Every ribpus is not a bijpus. Sylvia is a bijpus. This is a contradiction, but we continue to find more information. Every tohpus is a nujpus. Sylvia is a nujpus. Every nujpus is a vappus. Sylvia is a vappus. Every vappu
- `cat_false_inert_d1` [valid_rederivation] inj=`Quincy is a febpus.` -> Every nonpus is a vukpus. Quincy is a vukpus. Every vukpus is a cajpus. Quincy is a cajpus. Every cajpus is rigid. Quincy is rigid.
- `cat_false_usable_d1` [poisoned] inj=`Nestor is a kukpus.` -> Every kukpus is a dezpus. Nestor is a dezpus. Every dezpus is a ralpus. Nestor is a ralpus. Every ralpus is livid. Nestor is livid.
- `cat_false_usable_d1` [poisoned] inj=`Cedric is a gatpus.` -> Every gatpus is a menpus. Cedric is a menpus. Every menpus is a leppus. Cedric is a leppus. Every leppus is timid. Cedric is timid.
- `neg_false_attr_d1` [valid_rederivation] inj=`Tobias is not milky.` -> Every bigpus is a wuzpus. Tobias is a wuzpus. Every wuzpus is a rimpus. Tobias is a rimpus. Every rimpus is gauzy. Tobias is gauzy.
- `neg_false_attr_d1` [valid_rederivation] inj=`Quincy is not lanky.` -> Every fuspus is a mujpus. Quincy is a mujpus. Every mujpus is a hospus. Quincy is a hospus. Every hospus is a neppus. Quincy is a neppus. Every neppus is nutty. Quincy is nutty.
- `true_interruption` [valid_rederivation] inj=`Rosina is a despus.` -> Every despus is a campus. Rosina is a campus. Every campus is a haspus. Rosina is a haspus. Every haspus is husky. Rosina is husky.
- `true_interruption` [valid_rederivation] inj=`Cedric is a bispus.` -> Every lefpus is a pewpus. Cedric is a pewpus. Every pewpus is a pacpus. Cedric is a pacpus. Every pacpus is a litpus. Cedric is a litpus. Every litpus is rigid. Cedric is rigid.
