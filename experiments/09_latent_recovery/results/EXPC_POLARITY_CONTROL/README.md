# EXPC_POLARITY_CONTROL README

Status: **complete**
Generated: 2026-06-18T20:05:39.734895+00:00

## Design

- 2x2 factors: locality (local/global) x polarity (positive/negative).
- Positions: early, mid, late; selected problems are fully paired across all four cells at all positions.
- The prompt instruction and few-shot block are identical across all conditions.
- Prompt SHA-256: `f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c`.

## Sample Size

- Target fully matched problems: 7
- Generated fully matched problems: 7
- Validated rows: 84

## Main Results

- LOCAL_FALSE_POSITIVE: n=21, valid_recovery=0.7143, doubt=0.2857, poisoning=None, parroted=0.0952, derailed=0.0952, unparsed=0.0952
- LOCAL_FALSE_NEGATIVE: n=21, valid_recovery=0.5238, doubt=0.3333, poisoning=None, parroted=0.0952, derailed=0.3333, unparsed=0.0476
- GLOBAL_FALSE_POSITIVE: n=21, valid_recovery=0.6667, doubt=0.0, poisoning=0.2381, parroted=0.0, derailed=0.0952, unparsed=0.0
- GLOBAL_FALSE_NEGATIVE: n=21, valid_recovery=0.7143, doubt=0.1429, poisoning=None, parroted=0.0, derailed=0.2381, unparsed=0.0

## Token And Grammar Balance

- LOCAL_FALSE_POSITIVE: token_mean=4.762, token_range=[4, 5], templates={'entity_is_attribute': 21}, predicate_families={'attribute': 21}
- LOCAL_FALSE_NEGATIVE: token_mean=8.476, token_range=[7, 9], templates={'entity_is_not_a_category': 21}, predicate_families={'category': 21}
- GLOBAL_FALSE_POSITIVE: token_mean=7.238, token_range=[6, 8], templates={'entity_is_a_category': 21}, predicate_families={'category': 21}
- GLOBAL_FALSE_NEGATIVE: token_mean=7.667, token_range=[5, 9], templates={'entity_is_not_a_category': 14, 'entity_is_not_attribute': 7}, predicate_families={'attribute': 7, 'category': 14}

## Primary Analysis

- valid_recovery: model_status=ok; local_minus_global=-0.0714 CI=[-0.2143, 0.0714]; positive_minus_negative=0.0714 CI=[-0.0952, 0.1905]
- doubt: model_status=ok; local_minus_global=0.2381 CI=[0.0714, 0.4048]; positive_minus_negative=-0.0952 CI=[-0.2381, 0.0238]
- poisoning: model_status=insufficient_data; local_minus_global=None CI=None; positive_minus_negative=None CI=None

## Interpretation

- Contrasts are noisy or overlapping; report the null/ambiguous result and avoid strengthening the thesis from this run.
- If larger runs overturn this pattern, revise the thesis rather than filtering or rewording conditions.

## Integrity Checklist

- [x] Every run has a unique run_id
- [x] Every result has problem_id, condition, model, injection_position, and seed
- [x] Every injected statement has audited truth status
- [x] Every original proof was validated before perturbation
- [x] No duplicate examples are accidentally counted as independent
- [x] All failed generations are logged
- [x] All unparsed generations are logged
- [x] All unavailable perturbations are logged
- [x] Exact model revisions are pinned
- [x] Decoding settings are saved
- [x] Random seeds are saved
- [x] Git commit hash is saved
- [x] Result directory immutability guard is enabled
- [x] Tables are regenerated from artifacts
- [x] Validator unit tests cover required cases
- [x] Truth-status audit is tested independently of model outputs
- [x] Strict validator does not overwrite closure validator
- [x] Manual inspection of at least 20 random examples per new condition is saved
- [x] Confidence intervals are reported
- [x] Problem-clustered bootstrap or paired model is used
- [x] Paired designs are analyzed as paired
- [x] Multiple comparisons are labeled
- [x] No exclusion rule was changed after seeing results
- [x] No hand-entered result numbers
- [x] Null results are included
- [x] Limitations are updated
- [x] Claims are weakened where necessary
- [x] Figures do not hide sample-size differences

## Limitations

- Local affirmative contradictions in this PrOntoQA grammar are often affirmative attributes rather than category nouns; the grammar-template table reports this imbalance explicitly.
- Poisoning is only interpretable when the planted predicate can participate in downstream rules; rows where it is not measurable are excluded from poisoning denominators.
- Position-level cells are reported in `summary_tables.json`; pooled values should not be used if those cells diverge.
