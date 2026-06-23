# EXPA Global Expansion Report

Status: **complete**
Generated: 2026-06-18T21:42:20.811444+00:00

## Sample Size

- Target global-falsehood examples per position: 150
- Achieved global-falsehood examples per position: 150
- Pre-model global-falsehood candidate availability: 5819/18961 (0.3069) unique candidate worlds.
- Observed post-generation eligibility before stopping at target: 150/3032 (0.0495) gold-generation attempts.
- All unavailable perturbations and failed filters are preserved in `eligibility_audit.jsonl`.

## Main Result

- global_falsehood: n=450, valid=0.3933, poisoned=0.3422, parroted=0.2, derailed=0.0467, unparsed=0.0178, doubt=0.0133
- benign_paraphrase: n=450, valid=0.8489, poisoned=0.0, parroted=0.0222, derailed=0.1044, unparsed=0.0244, doubt=0.08
- one_hop_falsehood: n=450, valid=0.6711, poisoned=0.0156, parroted=0.1489, derailed=0.1267, unparsed=0.0378, doubt=0.3356
- true_interruption: n=450, valid=0.8556, poisoned=0.0, parroted=0.1089, derailed=0.0222, unparsed=0.0133, doubt=0.02

## Eligibility Counts

- too_few_dataset_steps: 11389
- candidate_falsehood_available: 5819
- gold_not_validator_valid: 1892
- no_global_falsehood_candidate: 1753
- gold_not_solved: 926
- gold_valid_global_falsehood_available: 150
- duplicate_question: 139
- too_few_model_intermediate_steps: 64

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
- [x] Result directory is made read-only after completion
- [x] Tables are regenerated from artifacts
- [x] Validator unit tests cover required cases
- [x] Manual inspection note saved
- [x] Confidence intervals are reported
- [x] Problem-clustered bootstrap or paired model used
- [x] Multiple comparisons labeled

## Notes

- Null and contradictory findings should be read directly from `summary_tables.json`; no result numbers are hand-entered in downstream tables.
- If status is partial, the run did not reach the requested n>=150 per position and the maximum achieved sample is reported above.
