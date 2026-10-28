# EXPE Evidence-Mover Report (FULL)

Generated: 2026-07-02T21:17:51.128098+00:00

- Cohort source: expanded (/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/results/EXPE_GOLD_EXPANSION/cohort.jsonl)
  (runner change for the full run: the gold cohort is read from the
  PLAN2 v1.1 C.1 expansion, results/EXPE_GOLD_EXPANSION/cohort.jsonl,
  via --cohort expanded; --cohort legacy preserves the original
  pilot.jsonl + results/gold/rollouts.jsonl path.)
- Position-arm grid (v1.1 C.5): {"early": ["REFUTING_d1", "BASELINE_FILLER"], "late": ["REFUTING_d1", "BASELINE_FILLER"], "mid": ["REFUTING_d1", "REFUTING_d2", "REFUTING_d3", "FREQ_MATCHED_NONREFUTING", "TEMPLATE_IRRELEVANT", "BASELINE_FILLER"]}
- Pre-registration evidence: PLAN2.md v1.1 external timestamp: git commit 545b35db420d1826b69ca9bdd6e47b5720bb54de (2026-07-02T14:05:29-07:00), file sha256 112da1a56a16745181e6ed7d2dac3b1e7d0dd3e7643e78e2bfdf16ca572e2b2e

Protocol disclosure (MOD-11): the own-trace was generated under the
unaugmented question and all arms replay it under an augmented question --
equally off-policy across arms. EXPE<->EXPB comparisons are cross-protocol.
Doubt below is the LEXICAL regex DV; the judge pass is not run at smoke scale.

## Availability (eligibility ladder)

- Cohort ladder: {"expanded_cohort_rows": 334, "gold_validated_cohort": 334, "too_few_injection_points": 95}
- Per-position ladders: {"early": {"admits_all_arms": 239, "admits_all_arms_B_invocab_reused": 31, "admits_all_arms_C_fresh_nouns": 208, "admits_all_arms_measurable_poisoning": 239}, "late": {"admits_all_arms": 172, "admits_all_arms_B_invocab_reused": 16, "admits_all_arms_C_fresh_nouns": 156, "admits_all_arms_measurable_poisoning": 172, "not_all_arms_available": 67}, "mid": {"admits_all_arms": 220, "admits_all_arms_B_invocab_reused": 26, "admits_all_arms_C_fresh_nouns": 194, "admits_all_arms_measurable_poisoning": 220, "not_all_arms_available": 19}}
- PAIRED-AVAILABILITY (problems admitting ALL six arms under one F): {"early": 239, "late": 172, "mid": 220}
- Arm available under some F: {"early|BASELINE_FILLER": 239, "early|FREQ_MATCHED_NONREFUTING": 239, "early|REFUTING_d1": 239, "early|REFUTING_d2": 239, "early|REFUTING_d3": 239, "early|TEMPLATE_IRRELEVANT": 239, "late|BASELINE_FILLER": 239, "late|FREQ_MATCHED_NONREFUTING": 239, "late|REFUTING_d1": 239, "late|REFUTING_d2": 239, "late|REFUTING_d3": 172, "late|TEMPLATE_IRRELEVANT": 239, "mid|BASELINE_FILLER": 239, "mid|FREQ_MATCHED_NONREFUTING": 239, "mid|REFUTING_d1": 239, "mid|REFUTING_d2": 239, "mid|REFUTING_d3": 220, "mid|TEMPLATE_IRRELEVANT": 239}
- Vocabulary tiers of selected problems: {"B_invocab_reused": 69, "C_fresh_nouns": 543}

Vocabulary-tier disclosure: the legacy cohort's unentailed-category pools
are tiny (95/130 injectable worlds have zero), so control-arm nonce nouns
use a preference ladder -- A: in-vocab distinct, B: in-vocab reused across
arms, C: fresh nonce nouns for TEMPLATE-head/FILLER slots only. The FREQ
antecedent is always in-question-vocabulary (audited).

## Per-arm rates (pooled)

| arm | n | valid | doubt(lex) | stated-not-F | inj-dep | parroted | derailed | unparsed | re-solve |
|---|---|---|---|---|---|---|---|---|---|
| REFUTING_d1 | 612 | 0.402 (n=612) | 0.0735 (n=612) | 0.1144 (n=612) | 0.3333 (n=612) | 0.1291 (n=612) | 0.085 (n=612) | 0.0507 (n=612) | 1.0 (n=612) |
| REFUTING_d2 | 220 | 0.3864 (n=220) | 0.0227 (n=220) | 0.1318 (n=220) | 0.3773 (n=220) | 0.1591 (n=220) | 0.0591 (n=220) | 0.0182 (n=220) | 1.0 (n=220) |
| REFUTING_d3 | 220 | 0.3591 (n=220) | 0.0136 (n=220) | 0.1364 (n=220) | 0.3955 (n=220) | 0.1682 (n=220) | 0.0591 (n=220) | 0.0182 (n=220) | 1.0 (n=220) |
| FREQ_MATCHED_NONREFUTING | 220 | 0.35 (n=220) | 0.1136 (n=220) | 0.1182 (n=220) | 0.3455 (n=220) | 0.2 (n=220) | 0.0636 (n=220) | 0.0409 (n=220) | 1.0 (n=220) |
| TEMPLATE_IRRELEVANT | 220 | 0.4045 (n=220) | 0.0227 (n=220) | 0.0 (n=220) | 0.3636 (n=220) | 0.1591 (n=220) | 0.0682 (n=220) | 0.0045 (n=220) | 1.0 (n=220) |
| BASELINE_FILLER | 612 | 0.3644 (n=612) | 0.0114 (n=612) | 0.0 (n=612) | 0.3693 (n=612) | 0.1879 (n=612) | 0.0637 (n=612) | 0.0147 (n=612) | 1.0 (n=612) |

## Paired contrasts (descriptive at smoke n)

- H5a_doubt_REFUTING_d1_vs_FREQ: diff=-0.0409, CI=[-0.0955, 0.0136], discordant=13/22, McNemar p=0.175465 (n_pairs=220)
- H5b_doubt_REFUTING_d1_vs_TEMPLATE: diff=0.05, CI=[0.0136, 0.0864], discordant=14/3, McNemar p=0.012726 (n_pairs=220)
- H6_injdep_REFUTING_d1_vs_FILLER: diff=-0.0359, CI=[-0.0766, 0.0049], discordant=59/81, McNemar p=0.075551 (n_pairs=612)
- explor_doubt_d1_vs_d3: diff=0.0591, CI=[0.0273, 0.0955], discordant=14/1, McNemar p=0.000977 (n_pairs=220)
- explor_statedref_REFUTING_d1_vs_FREQ: diff=-0.0045, CI=[-0.0591, 0.0455], discordant=18/19, McNemar p=1.0 (n_pairs=220)
- explor_statedref_d1_vs_d3: diff=-0.0227, CI=[-0.0818, 0.0318], discordant=20/25, McNemar p=0.551484 (n_pairs=220)
- explor_valid_REFUTING_d1_vs_FILLER: diff=0.0376, CI=[-0.0113, 0.0842], discordant=96/73, McNemar p=0.090291 (n_pairs=612)

## Audit pass rates (fail-closed; selection requires 1.0)

- BASELINE_FILLER: fully_audited=1.0, gold_revalidation=1.0, measured_d={"None": 612}
- FREQ_MATCHED_NONREFUTING: fully_audited=1.0, gold_revalidation=1.0, measured_d={"None": 220}
- REFUTING_d1: fully_audited=1.0, gold_revalidation=1.0, measured_d={"1": 612}
- REFUTING_d2: fully_audited=1.0, gold_revalidation=1.0, measured_d={"2": 220}
- REFUTING_d3: fully_audited=1.0, gold_revalidation=1.0, measured_d={"3": 220}
- TEMPLATE_IRRELEVANT: fully_audited=1.0, gold_revalidation=1.0, measured_d={"None": 220}

## Sanity

- {"added_rule_char_len_spread": {"max": 3, "mean": 1.07}, "failures": [], "trace_and_slot_identical_across_arms": true}

## Integrity

- [x] all_falsehoods_audited_false
- [x] all_gold_traces_revalidate
- [x] every_selected_row_fully_audited
- [x] failed_generations_logged
- [x] no_target_shortcut_all_rows
- [x] unique_run_ids

## Notes (full run)

- Primary rejection DV (PLAN2 v1.1 section B): strict stated-complement
  ('stated-not-F' column). Judge explicit-rejection is the registered
  secondary (doubt_judge_v2 pass); verbalized doubt (lexical + judge)
  is descriptive only.
- Registered adjudications (H1'/H5'/H6) are computed by
  expe_full_analysis.py into hypothesis_verdicts.json /
  EXPE_FULL_REPORT.md; this file is the runner-level view.
- vLLM backend: locked HF results are not row-level reproducible under
  vLLM (VLLM_PORT_REPORT.md); new suites run wholly under vLLM.
