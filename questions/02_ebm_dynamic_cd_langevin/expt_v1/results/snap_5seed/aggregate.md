# Aggregate Results

Input: `expt_v1/results/snap_5seed`
Reports: 5
Seeds: 0, 1, 2, 3, 4

| Method | Coverage | Entropy | TV to Uniform | Nearest Dist | Radial Error | Train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cd1 | 8.00±0.00 | 0.996±0.002 | 0.053±0.017 | 0.902±0.010 | 0.756±0.013 | 2.31±0.46 |
| dynamic_weighted_cd | 8.00±0.00 | 0.996±0.001 | 0.051±0.017 | 0.783±0.022 | 0.630±0.025 | 5.21±0.28 |

## Verdict

Dynamic weighted CD improved mean nearest-mode distance by 0.120 when positive, and improved mode-balance TV by 0.001 when positive. Interpret this as a heuristic negative-phase result, not an unbiased likelihood result.
