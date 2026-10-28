# EXPA_GLOBAL_EXPANSION README

Status: **complete**
Generated: 2026-06-18T19:51:27.800603+00:00

## Sample Size

- Target global-falsehood examples per position: 2
- Achieved global-falsehood examples per position: 2
- Availability rate among candidate-audit rows marked candidate eligible: 2 generated/validated problems; see `eligibility_audit.jsonl` for all unavailable cases.

## Main Result

- global_falsehood: n=6, valid=0.6667, poisoned=0.3333, parroted=0.0, derailed=0.0, unparsed=0.0, doubt=0.1667
- benign_paraphrase: n=6, valid=0.3333, poisoned=0.0, parroted=0.0, derailed=0.3333, unparsed=0.3333, doubt=0.3333
- one_hop_falsehood: n=6, valid=0.6667, poisoned=0.0, parroted=0.0, derailed=0.1667, unparsed=0.1667, doubt=0.1667
- true_interruption: n=6, valid=1.0, poisoned=0.0, parroted=0.0, derailed=0.0, unparsed=0.0, doubt=0.3333

## Eligibility Counts

- too_few_dataset_steps: 11389
- candidate_falsehood_available: 5819
- no_global_falsehood_candidate: 1753
- duplicate_question: 139
- gold_not_validator_valid: 12
- gold_valid_global_falsehood_available: 2
- gold_not_solved: 2

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
- [x] Tables are regenerated from artifacts
- [x] Validator unit tests cover required cases
- [x] Manual inspection note saved
- [x] Confidence intervals are reported
- [x] Problem-clustered bootstrap or paired model used
- [x] Multiple comparisons labeled

## Notes

- Null and contradictory findings should be read directly from `summary_tables.json`; no result numbers are hand-entered in downstream tables.
- If status is partial, the run did not reach the requested n>=150 per position and the maximum achieved sample is reported above.
