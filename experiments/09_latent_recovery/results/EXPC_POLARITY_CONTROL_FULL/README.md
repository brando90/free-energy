# EXPC_POLARITY_CONTROL README

Status: **complete**
Generated: 2026-07-02T00:19:31.286014+00:00

## Design

- 2x2 factors: locality (local/global) x polarity (positive/negative).
- Positions: early, mid, late; selected problems are fully paired across all four cells at all positions.
- The prompt instruction and few-shot block are identical across all conditions.
- Prompt SHA-256: `f668299fe738b58ba81a6b8435bf284ac5eed682b99f5e7d4cccfee62c7ab93c`.

## Sample Size

- Target fully matched problems: 127
- Generated fully matched problems: 127
- Validated rows: 1524

## Main Results

- LOCAL_FALSE_POSITIVE: n=381, valid_recovery=0.7349, doubt=0.4357, poisoning=None, parroted=0.1312, derailed=0.0971, unparsed=0.0367
- LOCAL_FALSE_NEGATIVE: n=381, valid_recovery=0.5722, doubt=0.378, poisoning=None, parroted=0.1601, derailed=0.2047, unparsed=0.042
- GLOBAL_FALSE_POSITIVE: n=381, valid_recovery=0.3675, doubt=0.0131, poisoning=0.2887, parroted=0.21, derailed=0.0866, unparsed=0.0472
- GLOBAL_FALSE_NEGATIVE: n=381, valid_recovery=0.6063, doubt=0.1365, poisoning=None, parroted=0.1654, derailed=0.1365, unparsed=0.0499

## Token And Grammar Balance

- LOCAL_FALSE_POSITIVE: token_mean=4.816, token_range=[4, 6], templates={'entity_is_attribute': 381}, predicate_families={'attribute': 381}
- LOCAL_FALSE_NEGATIVE: token_mean=8.304, token_range=[7, 9], templates={'entity_is_not_a_category': 381}, predicate_families={'category': 381}
- GLOBAL_FALSE_POSITIVE: token_mean=7.315, token_range=[6, 8], templates={'entity_is_a_category': 381}, predicate_families={'category': 381}
- GLOBAL_FALSE_NEGATIVE: token_mean=7.465, token_range=[5, 9], templates={'entity_is_not_a_category': 255, 'entity_is_not_attribute': 126}, predicate_families={'attribute': 126, 'category': 255}

## Primary Analysis

- valid_recovery: model_status=ok; local_minus_global=0.1667 CI=[0.122, 0.2113]; positive_minus_negative=-0.0381 CI=[-0.0853, 0.0118]
- doubt: model_status=ok; local_minus_global=0.332 CI=[0.29, 0.3727]; positive_minus_negative=-0.0328 CI=[-0.0787, 0.0066]
- poisoning: model_status=ok; local_minus_global=None CI=None; positive_minus_negative=None CI=None

## Interpretation

- Locality remains the clearer recovery contrast after controlling for polarity; this supports the accessibility thesis for this run.
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
