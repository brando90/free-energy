# EXPE Evidence-Mover Report (SMOKE)

Generated: 2026-07-02T20:03:23.449726+00:00

Protocol disclosure (MOD-11): the own-trace was generated under the
unaugmented question and all arms replay it under an augmented question --
equally off-policy across arms. EXPE<->EXPB comparisons are cross-protocol.
Doubt below is the LEXICAL regex DV; the judge pass is not run at smoke scale.

## Availability (eligibility ladder)

- Cohort ladder: {"gold_not_solved": 19, "gold_not_validator_valid": 32, "gold_rollouts_total": 219, "gold_validated_cohort": 168, "too_few_injection_points": 38}
- Per-position ladders: {"mid": {"admits_all_arms": 27, "admits_all_arms_B_invocab_reused": 5, "admits_all_arms_C_fresh_nouns": 22, "admits_all_arms_measurable_poisoning": 27, "no_falsehood_candidate": 95, "not_all_arms_available": 8}}
- PAIRED-AVAILABILITY (problems admitting ALL six arms under one F): {"mid": 27}
- Arm available under some F: {"mid|BASELINE_FILLER": 35, "mid|FREQ_MATCHED_NONREFUTING": 31, "mid|REFUTING_d1": 35, "mid|REFUTING_d2": 35, "mid|REFUTING_d3": 30, "mid|TEMPLATE_IRRELEVANT": 35}
- Vocabulary tiers of selected problems: {"B_invocab_reused": 2, "C_fresh_nouns": 18}

Vocabulary-tier disclosure: the legacy cohort's unentailed-category pools
are tiny (95/130 injectable worlds have zero), so control-arm nonce nouns
use a preference ladder -- A: in-vocab distinct, B: in-vocab reused across
arms, C: fresh nonce nouns for TEMPLATE-head/FILLER slots only. The FREQ
antecedent is always in-question-vocabulary (audited).

## Per-arm rates (pooled)

| arm | n | valid | doubt(lex) | stated-not-F | inj-dep | parroted | derailed | unparsed | re-solve |
|---|---|---|---|---|---|---|---|---|---|
| REFUTING_d1 | 20 | 0.35 (n=20) | 0.0 (n=20) | 0.2 (n=20) | 0.35 (n=20) | 0.2 (n=20) | 0.1 (n=20) | 0.0 (n=20) | 1.0 (n=20) |
| REFUTING_d2 | 20 | 0.3 (n=20) | 0.0 (n=20) | 0.15 (n=20) | 0.4 (n=20) | 0.25 (n=20) | 0.0 (n=20) | 0.05 (n=20) | 1.0 (n=20) |
| REFUTING_d3 | 20 | 0.4 (n=20) | 0.0 (n=20) | 0.1 (n=20) | 0.4 (n=20) | 0.05 (n=20) | 0.1 (n=20) | 0.05 (n=20) | 1.0 (n=20) |
| FREQ_MATCHED_NONREFUTING | 20 | 0.45 (n=20) | 0.1 (n=20) | 0.05 (n=20) | 0.25 (n=20) | 0.2 (n=20) | 0.05 (n=20) | 0.05 (n=20) | 1.0 (n=20) |
| TEMPLATE_IRRELEVANT | 20 | 0.4 (n=20) | 0.05 (n=20) | 0.0 (n=20) | 0.35 (n=20) | 0.15 (n=20) | 0.05 (n=20) | 0.05 (n=20) | 1.0 (n=20) |
| BASELINE_FILLER | 20 | 0.25 (n=20) | 0.0 (n=20) | 0.0 (n=20) | 0.4 (n=20) | 0.1 (n=20) | 0.2 (n=20) | 0.05 (n=20) | 1.0 (n=20) |

## Paired contrasts (descriptive at smoke n)

- H5a_doubt_REFUTING_d1_vs_FREQ: diff=-0.1, CI=[-0.25, 0.0], discordant=0/2, McNemar p=0.5 (n_pairs=20)
- H5b_doubt_REFUTING_d1_vs_TEMPLATE: diff=-0.05, CI=[-0.15, 0.0], discordant=0/1, McNemar p=1.0 (n_pairs=20)
- H6_injdep_REFUTING_d1_vs_FILLER: diff=-0.05, CI=[-0.2, 0.1], discordant=1/2, McNemar p=1.0 (n_pairs=20)
- explor_doubt_d1_vs_d3: diff=0.0, CI=[0.0, 0.0], discordant=0/0, McNemar p=1.0 (n_pairs=20)
- explor_statedref_REFUTING_d1_vs_FREQ: diff=0.15, CI=[-0.05, 0.4], discordant=4/1, McNemar p=0.375 (n_pairs=20)
- explor_statedref_d1_vs_d3: diff=0.1, CI=[-0.1, 0.35], discordant=4/2, McNemar p=0.6875 (n_pairs=20)
- explor_valid_REFUTING_d1_vs_FILLER: diff=0.1, CI=[-0.05, 0.3], discordant=3/1, McNemar p=0.625 (n_pairs=20)

## Audit pass rates (fail-closed; selection requires 1.0)

- BASELINE_FILLER: fully_audited=1.0, gold_revalidation=1.0, measured_d={"None": 20}
- FREQ_MATCHED_NONREFUTING: fully_audited=1.0, gold_revalidation=1.0, measured_d={"None": 20}
- REFUTING_d1: fully_audited=1.0, gold_revalidation=1.0, measured_d={"1": 20}
- REFUTING_d2: fully_audited=1.0, gold_revalidation=1.0, measured_d={"2": 20}
- REFUTING_d3: fully_audited=1.0, gold_revalidation=1.0, measured_d={"3": 20}
- TEMPLATE_IRRELEVANT: fully_audited=1.0, gold_revalidation=1.0, measured_d={"None": 20}

## Sanity

- {"added_rule_char_len_spread": {"max": 3, "mean": 1.6}, "failures": [], "trace_and_slot_identical_across_arms": true}

## Integrity

- [x] all_falsehoods_audited_false
- [x] all_gold_traces_revalidate
- [x] every_selected_row_fully_audited
- [x] failed_generations_logged
- [x] no_target_shortcut_all_rows
- [x] unique_run_ids

## Limitations

- Smoke run: n <= 20/arm at mid; estimation only, no confirmatory claims.
- Lexical doubt regex is the DV here; PLAN2's primary doubt DV is the
  32B judge (doubt_judge_v2), to be run over these continuations later.
- 'stated-not-F' (continuation explicitly derives the complement of F)
  is a smoke-added secondary DV: refutation is frequently derivational
  with zero doubt markers, which the lexical regex cannot see. If kept
  for the full run it must be added to PLAN2 by timestamped addendum.
- vLLM backend: locked HF results are not row-level reproducible under
  vLLM (VLLM_PORT_REPORT.md); new suites run wholly under vLLM.
