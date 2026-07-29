# EXPE Gold-Cohort Expansion Report

Generated: 2026-07-02T20:14:06.761145+00:00

Cohort prep only: NO perturbation arms were run; the legacy
results/gold/ directory was not touched. Gold protocol identical to
collect_gold.py (no-prefill greedy chat prompt, max_new=256) but under
the vLLM backend -- new suites run wholly under vLLM (VLLM_PORT_REPORT).

## Screen

- {"dataset_cot_not_validator_valid": 4, "duplicate_question": 139, "excluded_legacy_pilot_question": 500, "no_entity_or_unparsable_target": 2437, "pool_rows": 19100, "question_not_fully_parsable": 3629, "screened_in": 470, "too_few_dataset_derived_steps": 10491, "too_few_unentailed_categories": 1430}
- Hard screen: unentailed categories >= 2, dataset derived steps >= 4, target parses, question not in legacy pilot.
- Ordering: unentailed-pool size DESC (maximizes tier-A/B vocabulary).

## Rollouts and double filter

- Rollouts generated: 470
- Gold-valid (solved AND validator valid_rederivation): 334 (0.71)
- GPU seconds (generation batches): 30.5
- Model: Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28

## Availability (EXPE six-arm ladder at mid, same audits as the runner)

- Ladder: {"admits_all_arms": 220, "admits_all_arms_measurable_poisoning": 220, "gold_valid_cohort": 334, "not_all_arms_available": 19, "too_few_injection_points": 95}
- PAIRED AVAILABILITY (new cohort): 220 (target 150; reached: True)
- Including the 27 legacy problems: 247
- Vocabulary tiers: {"B_invocab_reused": 26, "C_fresh_nouns": 194}
- By source file: {"3hop_ProofsOnly_random_noadj.json": 118, "4hop_ProofsOnly_random_noadj.json": 102}

## Notes

- cohort.jsonl holds the expanded own-trace gold cohort (question, target,
  entity, model trace); the EXPE runner can consume it after the PLAN2
  amendment that re-scopes the EXPE cohort is timestamped.
- The screen conditions are structural necessary conditions for the
  six-arm set plus a disclosed richness ordering; they do not condition
  on model behavior beyond the pre-registered double filter.
