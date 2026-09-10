# E11 DEPTH x OPACITY report -- Qwen/Qwen2.5-7B-Instruct

Generated 2026-07-28T09:34:39. Flag channel: judge_reject if judged else doubt_lex (frozen lexicon).

## Cells (3-way DV distribution; absorbed rate is the headline)

| cell | n | prog_n | power | absorbed | silently_corr | flagged | unresolved |
|---|---|---|---|---|---|---|---|
| k0_bare | 1350 | 150 | ok | 0.596 [0.57,0.62] | 0.290 [0.27,0.31] | 0.013 [0.01,0.02] | 0.101 [0.09,0.12] |
| k1_bare | 1350 | 150 | ok | 0.986 [0.98,0.99] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.014 [0.01,0.02] |
| k1_full | 1350 | 150 | ok | 0.993 [0.99,1.00] | 0.001 [0.00,0.00] | 0.000 [0.00,0.00] | 0.007 [0.00,0.01] |
| k1_partial | 1350 | 150 | ok | 0.988 [0.98,0.99] | 0.001 [0.00,0.00] | 0.000 [0.00,0.00] | 0.011 [0.01,0.02] |
| k2_bare | 1350 | 150 | ok | 0.997 [0.99,1.00] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.003 [0.00,0.01] |
| k2_full | 1350 | 150 | ok | 0.999 [1.00,1.00] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.001 [0.00,0.00] |
| k2_partial | 1350 | 150 | ok | 0.991 [0.98,0.99] | 0.000 [0.00,0.00] | 0.001 [0.00,0.00] | 0.008 [0.00,0.01] |
| k3_bare | 1350 | 150 | ok | 0.987 [0.98,0.99] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.013 [0.01,0.02] |
| k3_full | 1350 | 150 | ok | 0.990 [0.98,0.99] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.010 [0.01,0.02] |
| k3_partial | 1350 | 150 | ok | 0.987 [0.98,0.99] | 0.001 [0.00,0.00] | 0.000 [0.00,0.00] | 0.012 [0.01,0.02] |
| k5_bare | 504 | 56 | ci_widened_flagged | 0.946 [0.92,0.96] | 0.000 [-0.00,0.01] | 0.000 [-0.00,0.01] | 0.054 [0.04,0.08] |
| k5_full | 504 | 56 | ci_widened_flagged | 0.992 [0.98,1.00] | 0.000 [-0.00,0.01] | 0.004 [0.00,0.01] | 0.004 [0.00,0.01] |
| k5_partial | 504 | 56 | ci_widened_flagged | 0.950 [0.93,0.97] | 0.000 [-0.00,0.01] | 0.002 [0.00,0.01] | 0.048 [0.03,0.07] |

## Contrasts

```json
{
  "k0_vs_k1_gap": {
    "absorbed_k0": 0.5963,
    "absorbed_k1_bare": 0.9859,
    "gap_k1_minus_k0": 0.3896
  },
  "monotonic_trend": {
    "bare_rate_by_k": {
      "0": 0.5963,
      "1": 0.9859,
      "2": 0.997,
      "3": 0.9867,
      "5": 0.9464
    },
    "isotonic_nondecreasing": false,
    "k5_minus_k0": 0.3501,
    "spearman_rho": 0.3
  },
  "opacity_restoration": {
    "per_k": {
      "k1": {
        "bare": 0.9859,
        "full": 0.9926,
        "restoration_bare_minus_full": -0.0067
      },
      "k2": {
        "bare": 0.997,
        "full": 0.9993,
        "restoration_bare_minus_full": -0.0023
      },
      "k3": {
        "bare": 0.9867,
        "full": 0.9896,
        "restoration_bare_minus_full": -0.0029
      },
      "k5": {
        "bare": 0.9464,
        "full": 0.9921,
        "restoration_bare_minus_full": -0.0457
      }
    },
    "pooled_kge1": {
      "bare": 0.9851,
      "full": 0.9936,
      "restoration": -0.0086
    }
  }
}
```

Gold cohort (solve rates / eligible per k): `{"0": {"eligible": 373, "kept": 150, "solve_rate": 0.2072, "tested": 1800}, "1": {"eligible": 367, "kept": 150, "solve_rate": 0.2039, "tested": 1800}, "2": {"eligible": 434, "kept": 150, "solve_rate": 0.2411, "tested": 1800}, "3": {"eligible": 391, "kept": 150, "solve_rate": 0.2172, "tested": 1800}, "5": {"eligible": 56, "kept": 56, "solve_rate": 0.0311, "tested": 1800}}`

Notes: Wilson95 + program-cluster bootstrap95 (n_boot=1000); cluster=program family (program_id)
