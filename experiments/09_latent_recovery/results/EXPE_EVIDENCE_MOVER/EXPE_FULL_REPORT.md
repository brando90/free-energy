# EXPE_EVIDENCE_MOVER — full confirmatory run report

- Pre-registration: PLAN2.md v1.1 external timestamp: git commit 545b35db420d1826b69ca9bdd6e47b5720bb54de (2026-07-02T14:05:29-07:00), file sha256 112da1a56a16745181e6ed7d2dac3b1e7d0dd3e7643e78e2bfdf16ca572e2b2e
- Model: Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28 (backend vllm, greedy, max_new=192)
- Cohort: expanded (results/EXPE_GOLD_EXPANSION/cohort.jsonl, PLAN2 v1.1 C.1; the runner was patched to read it via --cohort expanded; --cohort legacy preserves the smoke-era path)
- Grid (v1.1 C.5): {"early": ["REFUTING_d1", "BASELINE_FILLER"], "late": ["REFUTING_d1", "BASELINE_FILLER"], "mid": ["REFUTING_d1", "REFUTING_d2", "REFUTING_d3", "FREQ_MATCHED_NONREFUTING", "TEMPLATE_IRRELEVANT", "BASELINE_FILLER"]}
- GPU seconds: generate=50.0, judge(32B)=153.0

DV registry (v1.1 B): PRIMARY = strict stated-complement; SECONDARY = judge explicit-rejection (32B, doubt_judge_v2); verbalized doubt (lexical + judge) DESCRIPTIVE ONLY.

Protocol disclosure (MOD-11): own-trace generated under the unaugmented question; all arms replay it under an augmented question -- equally off-policy across arms. Re-solve rate (MC-6) reported per cell below.

## Registered adjudications (raw p; Holm m=6 assembled by coordinator)

### H1_prime_stated_complement_trend_d123

- Test: Cochran-Armitage trend over d in {1,2,3}, decreasing; clustered on problem via within-problem permutation (20000 draws)
- DV: strict stated-complement (primary); scope: mid
- statistic: {"perm_trend_stat": 5.0, "ca_z_unclustered": 0.7152}
- p_raw (one-sided): **0.789711**
- estimate: {"slope_per_hop": 0.01136, "rates_by_d": {"1": 0.1136, "2": 0.1318, "3": 0.1364}}
- ci95: [-0.01818, 0.04091]
- ns: {"problems": 220, "rows": 660}

### H5_prime_derivability_vs_lexical_conjunction

- Test: conjunction (intersection-union): stated-complement REFUTING_d1 > FREQ_MATCHED_NONREFUTING and > TEMPLATE_IRRELEVANT; exact one-sided McNemar within problem; slot p = max(component p)
- DV: strict stated-complement (primary); scope: mid
- statistic: {"vs_FREQ_discordant": [18, 19], "vs_TEMPLATE_discordant": [25, 0]}
- p_raw (one-sided): **0.628585**
- estimate: {"diff_vs_FREQ": -0.0045, "diff_vs_TEMPLATE": 0.1136}
- ci95: {"vs_FREQ": [-0.0591, 0.0455], "vs_TEMPLATE": [0.0727, 0.1545]}
- ns: {"pairs_vs_FREQ": 220, "pairs_vs_TEMPLATE": 220}
  - component vs_FREQ_MATCHED_NONREFUTING: rate_hi=0.1136 rate_lo=0.1182 diff=-0.0045 CI=[-0.0591, 0.0455] discordant=18/19 p=0.628585 (n=220)
  - component vs_TEMPLATE_IRRELEVANT: rate_hi=0.1136 rate_lo=0.0 diff=0.1136 CI=[0.0727, 0.1545] discordant=25/0 p=2.98023e-08 (n=220)

### H6_within_problem_evidence_flip_injdep

- Test: derivational inj-dep REFUTING_d1 < BASELINE_FILLER; exact one-sided McNemar within problem
- DV: derivational injection-dependence (validator 'poisoned'); scope: mid (primary; early/late/pooled sensitivity in detail)
- statistic: {"discordant_R1_F0": 25, "discordant_R0_F1": 27}
- p_raw (one-sided): **0.444942**
- estimate: {"diff_R_minus_F": -0.0091, "rate_REFUTING_d1": 0.3591, "rate_BASELINE_FILLER": 0.3682}
- ci95: [-0.0727, 0.0545]
- ns: {"pairs_mid": 220}
  - sensitivity early: diff=-0.0364 CI=[-0.1, 0.0273] p=0.161118 (n=220)
  - sensitivity late: diff=-0.0698 CI=[-0.1395, 0.0] p=0.0364757 (n=172)
  - sensitivity pooled: diff=-0.0359 CI=[-0.0765, 0.005] p=0.0377754 (n=612)

## Secondary DV (judge explicit-rejection, 32B)

- H1_prime_SECONDARY_judge_explicit_rejection: p_raw=0.523274 {"slope_per_hop": 0.0, "rates_by_d": {"1": 0.3182, "2": 0.2545, "3": 0.3182}}
- H5_prime_SECONDARY_judge_explicit_rejection: p_raw=0.702856 ""
- H6_SECONDARY_judge_explicit_rejection: p_raw=1.0 ""

## Descriptives per arm x position

| cell | n | stated-comp | judge-reject | closure-valid | inj-dep(deriv) | echo | judge-doubt | lex-doubt | unparsed | re-solve | tiers |
|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE_FILLER|early | 220 | 0.0 | 0.1182 | 0.4091 | 0.2955 | 0.2045 | 0.0227 | 0.0136 | 0.0136 | 1.0 | {"C_fresh_nouns": 193, "B_invocab_reused": 27} |
| REFUTING_d1|early | 220 | 0.0955 | 0.3636 | 0.4591 | 0.2591 | 0.1136 | 0.0955 | 0.0818 | 0.0591 | 1.0 | {"C_fresh_nouns": 193, "B_invocab_reused": 27} |
| BASELINE_FILLER|late | 172 | 0.0 | 0.1047 | 0.314 | 0.4651 | 0.1512 | 0.0349 | 0.0116 | 0.0233 | 1.0 | {"C_fresh_nouns": 156, "B_invocab_reused": 16} |
| REFUTING_d1|late | 172 | 0.1395 | 0.314 | 0.3779 | 0.3953 | 0.1279 | 0.0698 | 0.064 | 0.0349 | 1.0 | {"C_fresh_nouns": 156, "B_invocab_reused": 16} |
| BASELINE_FILLER|mid | 220 | 0.0 | 0.1409 | 0.3591 | 0.3682 | 0.2 | 0.0045 | 0.0091 | 0.0091 | 1.0 | {"C_fresh_nouns": 194, "B_invocab_reused": 26} |
| FREQ_MATCHED_NONREFUTING|mid | 220 | 0.1182 | 0.3364 | 0.35 | 0.3455 | 0.2 | 0.1409 | 0.1136 | 0.0409 | 1.0 | {"C_fresh_nouns": 194, "B_invocab_reused": 26} |
| REFUTING_d1|mid | 220 | 0.1136 | 0.3182 | 0.3636 | 0.3591 | 0.1455 | 0.0727 | 0.0727 | 0.0545 | 1.0 | {"C_fresh_nouns": 194, "B_invocab_reused": 26} |
| REFUTING_d2|mid | 220 | 0.1318 | 0.2545 | 0.3864 | 0.3773 | 0.1591 | 0.0182 | 0.0227 | 0.0182 | 1.0 | {"C_fresh_nouns": 194, "B_invocab_reused": 26} |
| REFUTING_d3|mid | 220 | 0.1364 | 0.3182 | 0.3591 | 0.3955 | 0.1682 | 0.0273 | 0.0136 | 0.0182 | 1.0 | {"C_fresh_nouns": 194, "B_invocab_reused": 26} |
| TEMPLATE_IRRELEVANT|mid | 220 | 0.0 | 0.1727 | 0.4045 | 0.3636 | 0.1591 | 0.0409 | 0.0227 | 0.0045 | 1.0 | {"C_fresh_nouns": 194, "B_invocab_reused": 26} |

## Six-class multinomial per cell (unparsed visible, house rule)

- BASELINE_FILLER|early: {"valid_rederivation": 90, "poisoned": 65, "parroted": 45, "derailed": 17, "unparsed": 3, "other": 0}
- REFUTING_d1|early: {"valid_rederivation": 101, "poisoned": 57, "parroted": 25, "derailed": 24, "unparsed": 13, "other": 0}
- BASELINE_FILLER|late: {"valid_rederivation": 54, "poisoned": 80, "parroted": 26, "derailed": 8, "unparsed": 4, "other": 0}
- REFUTING_d1|late: {"valid_rederivation": 65, "poisoned": 68, "parroted": 22, "derailed": 11, "unparsed": 6, "other": 0}
- BASELINE_FILLER|mid: {"valid_rederivation": 79, "poisoned": 81, "parroted": 44, "derailed": 14, "unparsed": 2, "other": 0}
- FREQ_MATCHED_NONREFUTING|mid: {"valid_rederivation": 77, "poisoned": 76, "parroted": 44, "derailed": 14, "unparsed": 9, "other": 0}
- REFUTING_d1|mid: {"valid_rederivation": 80, "poisoned": 79, "parroted": 32, "derailed": 17, "unparsed": 12, "other": 0}
- REFUTING_d2|mid: {"valid_rederivation": 85, "poisoned": 83, "parroted": 35, "derailed": 13, "unparsed": 4, "other": 0}
- REFUTING_d3|mid: {"valid_rederivation": 79, "poisoned": 87, "parroted": 37, "derailed": 13, "unparsed": 4, "other": 0}
- TEMPLATE_IRRELEVANT|mid: {"valid_rederivation": 89, "poisoned": 80, "parroted": 35, "derailed": 15, "unparsed": 1, "other": 0}

## Integrity and audits (runner summary)

- [x] all_falsehoods_audited_false
- [x] all_gold_traces_revalidate
- [x] every_selected_row_fully_audited
- [x] failed_generations_logged
- [x] no_target_shortcut_all_rows
- [x] unique_run_ids
- sanity: {"added_rule_char_len_spread": {"max": 3, "mean": 1.07}, "failures": [], "trace_and_slot_identical_across_arms": true}
- availability: {"early": 239, "late": 172, "mid": 220}

## Deviations and notes

- late-position availability is 172 < 220: the full available late cohort was taken (v1.1 C.5 'full available cohort' rule); early had 239 available, 220 selected by seeded shuffle.
- Eligibility still requires all six arms constructible under one shared F at every position (paired-availability constraint held constant); only the GENERATED arm set is restricted at early/late per v1.1 C.5.
- MC-1 (parse rates): unparsed differs across compared cells at mid (REFUTING_d1 0.0545 vs REFUTING_d2/d3 0.0182, FREQ 0.0409, TEMPLATE 0.0045, FILLER 0.0091). The primary stated-complement DV is a string-level check computed on EVERY generated row (parse-independent), and the judge pass covered all rows, so no parse-conditioned imputation is required; unparsed rows are retained in all denominators (house rule) and visible per cell in the six-class table.
- Registered outcome branch (v1.1 section B, carried from 5.1): H5' failed specifically in the pre-stated confound direction -- FREQ ~= REFUTING on the primary DV (diff -0.0045, CI [-0.0591, +0.0455]) while REFUTING >> TEMPLATE (p ~ 3e-8). This is the 'professor's confound account SUPPORTED (lexical priming)' condition on the EXPE side; the family-level call belongs to the coordinator once EXPD's H2'/H3'/H4 slots land.
- Qualitative check on the DV (disclosed): sampled FREQ-arm stated-complement rows reach the complement via UNLICENSED chains (e.g. treating 'Every brimpus is not a gorpus' as if it refuted 'Stella is a gorpus'), whereas REFUTING_d1 rows derive it through the licensed added rule -- consistent with lexical/template priming producing the same surface statement.
- Holm (m=6) is assembled by the coordinator over {H1', H2', H3', H4, H5', H6}; only raw p-values are reported here.
