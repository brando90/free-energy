# Stage-0 re-analysis (zero GPU)

## Neghop dose-response, 6-class multinomial

| k | pos | n | unparsed | valid (all runs) | valid (parsed-only) | valid [worst, best] | doubt |
|---|-----|---|----------|------------------|--------------------|---------------------|-------|
| 1 | early | 130 | 0.115 | 0.646 | 0.730 | [0.646, 0.761] | 0.392 |
| 1 | mid | 130 | 0.238 | 0.638 | 0.838 | [0.638, 0.877] | 0.254 |
| 1 | late | 130 | 0.123 | 0.746 | 0.851 | [0.746, 0.869] | 0.185 |
| 2 | early | 130 | 0.262 | 0.561 | 0.760 | [0.561, 0.823] | 0.323 |
| 2 | mid | 130 | 0.277 | 0.646 | 0.894 | [0.646, 0.923] | 0.154 |
| 3 | early | 130 | 0.323 | 0.531 | 0.784 | [0.531, 0.854] | 0.231 |
| 3 | mid | 69 | 0.304 | 0.493 | 0.708 | [0.493, 0.797] | 0.217 |
| 4 | early | 96 | 0.240 | 0.594 | 0.781 | [0.594, 0.833] | 0.229 |
| 4 | mid | 38 | 0.263 | 0.526 | 0.714 | [0.526, 0.789] | 0.158 |
| 5 | early | 69 | 0.246 | 0.594 | 0.788 | [0.594, 0.841] | 0.246 |
| 5 | mid | 22 | 0.364 | 0.455 | 0.714 | [0.455, 0.818] | 0.227 |

Reading: the plotted closure-valid decline across k is carried by unparsed rows; parsed-only validity is flat-to-increasing. The doubt gradient (computed on raw text before parsing) survives. Per MASTER_PLAN C1: retire the legacy validity curve, keep doubt.

## EXPB closure-valid imputation bounds

```json
{
  "GLOBAL_BASELINE": {
    "n": 300,
    "valid_count": 113,
    "unparsed_count": 12,
    "valid_rate_worst_case": 0.3767,
    "valid_rate_best_case": 0.4167,
    "unparsed_rate": 0.04
  },
  "IRRELEVANT_CERTIFICATE_CONTROL": {
    "n": 300,
    "valid_count": 150,
    "unparsed_count": 11,
    "valid_rate_worst_case": 0.5,
    "valid_rate_best_case": 0.5367,
    "unparsed_rate": 0.0367
  },
  "LOCAL_CERTIFICATE": {
    "n": 300,
    "valid_count": 140,
    "unparsed_count": 95,
    "valid_rate_worst_case": 0.4667,
    "valid_rate_best_case": 0.7833,
    "unparsed_rate": 0.3167
  }
}
```
