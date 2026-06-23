# EXPB Local-Certificate Flip Report

Status: **complete**
Generated: 2026-06-18T20:16:22.709372+00:00

## Sample Size

- Target paired triples per position: 100
- Achieved paired triples by position: {'early': 100, 'late': 100, 'mid': 100}
- Problem clusters: 183
- Manifest rows: 900; validated rows: 900

## Primary Metrics

- GLOBAL_BASELINE: n=300, valid=0.3767, poisoned=0.1533, doubt=0.01
- LOCAL_CERTIFICATE: n=300, valid=0.4667, poisoned=0.0267, doubt=0.38
- IRRELEVANT_CERTIFICATE_CONTROL: n=300, valid=0.5, poisoned=0.24, doubt=0.0233

## Paired Tests

- poisoned LOCAL_CERTIFICATE_vs_GLOBAL_BASELINE: diff=-0.1267, CI=[-0.1751, -0.08], McNemar p=0.0
- poisoned LOCAL_CERTIFICATE_vs_IRRELEVANT_CERTIFICATE_CONTROL: diff=-0.2133, CI=[-0.2718, -0.1579], McNemar p=0.0
- valid_rederivation LOCAL_CERTIFICATE_vs_GLOBAL_BASELINE: diff=0.09, CI=[0.0032, 0.1758], McNemar p=0.019677
- valid_rederivation LOCAL_CERTIFICATE_vs_IRRELEVANT_CERTIFICATE_CONTROL: diff=-0.0333, CI=[-0.1287, 0.0561], McNemar p=0.46255
- verbalized_doubt LOCAL_CERTIFICATE_vs_GLOBAL_BASELINE: diff=0.37, CI=[0.3095, 0.4309], McNemar p=0.0
- verbalized_doubt LOCAL_CERTIFICATE_vs_IRRELEVANT_CERTIFICATE_CONTROL: diff=0.3567, CI=[0.2928, 0.4184], McNemar p=0.0

## Sanity Checks

- C true rate in LOCAL_CERTIFICATE: 1.0
- F false rate: 1.0
- C_irrel true rate: 1.0
- C_irrel locally falsifies F count: 0
- Prompt formatting identical except planned manipulation: True
- Certificate char length delta local-control: {'max': 1, 'mean': -0.023, 'min': -1}

## Integrity Checklist

- [x] Every run has a unique run_id
- [x] Every result has problem_id, condition, model, injection_position, and seed
- [x] Every injected statement has audited truth status
- [x] Every original proof was validated before perturbation
- [x] No duplicate examples are accidentally counted as independent - distinct falsehood triples are clustered by problem_id
- [x] All failed generations are logged
- [x] All unparsed generations are logged
- [x] All unavailable perturbations are logged
- [x] Exact model revisions are pinned - a09a35458c702b33eeacc393d103063234e8bc28
- [x] Decoding settings are saved
- [x] Random seeds are saved
- [x] Git commit hash is saved - d860dc734e285ac148100b3d8da74fc2721dbcaa
- [x] Result directories are immutable - finalize writes RUN_COMPLETE and removes write bits
- [x] Tables and figures are regenerated from artifacts - no figures specified for EXPB
- [x] Unit tests cover true, false, local, global, poisoned, parroted, skipped, and unparsed cases
- [x] Truth-status audit is tested independently of model outputs
- [x] Strict validator does not overwrite closure validator
- [x] Manual inspection of at least 20 random examples per new condition is saved
- [x] Confidence intervals are reported
- [x] Problem-clustered bootstrap or mixed-effects models are used
- [x] Paired designs are analyzed as paired
- [x] Multiple comparisons are labeled exploratory unless pre-specified
- [x] No exclusion rule was changed after seeing results
- [x] No hand-entered result numbers
- [x] Null results are included - in EXPB_REPORT and summary_tables.json
- [x] Limitations are updated - in EXPB_REPORT; paper not edited by instruction
- [x] Claims are weakened where necessary - in EXPB_REPORT; paper not edited by instruction
- [x] Figures do not hide sample-size differences - no EXPB figures generated
- [x] C is true in 100% of LOCAL_CERTIFICATE cases
- [x] F is false in 100% of all cases
- [x] C_irrel does not locally falsify F
- [x] Token length differences across conditions are reported
- [x] Prompt formatting is identical except for planned certificate manipulation
- [x] Target sample size reached or availability reported

## Limitations

- The local certificate uses the repository's closed-world category-falsehood convention: an unentailed category claim is audited false, and its negated category certificate is audited true.
- Multiple falsehoods can come from the same problem; all inferential comparisons are paired by triple and clustered by problem_id.
- The strict stepwise validator is not present in this repo, so stepwise-valid recovery is reported as unavailable rather than substituted for the closure validator.
- The paper was not edited, per instruction. This report includes null results and limitations without changing manuscript claims.
