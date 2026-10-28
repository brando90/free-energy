# E11 DEPTH x OPACITY report -- Qwen/Qwen2.5-7B-Instruct

Generated 2026-07-28T08:49:14. Flag channel: judge_reject if judged else doubt_lex (frozen lexicon).

## Cells (3-way DV distribution; absorbed rate is the headline)

| cell | n | prog_n | power | absorbed | silently_corr | flagged | unresolved |
|---|---|---|---|---|---|---|---|
| k0_bare | 1350 | 150 | ok | 0.713 [0.69,0.74] | 0.204 [0.18,0.23] | 0.000 [0.00,0.00] | 0.083 [0.07,0.10] |
| k1_bare | 1350 | 150 | ok | 0.784 [0.76,0.81] | 0.006 [0.00,0.01] | 0.000 [0.00,0.00] | 0.210 [0.19,0.23] |
| k1_full | 1350 | 150 | ok | 0.936 [0.92,0.95] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.064 [0.05,0.08] |
| k1_partial | 1350 | 150 | ok | 0.926 [0.91,0.94] | 0.004 [0.00,0.01] | 0.000 [0.00,0.00] | 0.070 [0.06,0.09] |
| k2_bare | 1350 | 150 | ok | 0.943 [0.93,0.95] | 0.001 [0.00,0.00] | 0.000 [0.00,0.00] | 0.056 [0.05,0.07] |
| k2_full | 1350 | 150 | ok | 0.953 [0.94,0.96] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.047 [0.04,0.06] |
| k2_partial | 1350 | 150 | ok | 0.951 [0.94,0.96] | 0.001 [0.00,0.00] | 0.000 [0.00,0.00] | 0.048 [0.04,0.06] |
| k3_bare | 1350 | 150 | ok | 0.933 [0.92,0.94] | 0.000 [0.00,0.00] | 0.000 [0.00,0.00] | 0.067 [0.06,0.08] |
| k3_full | 1350 | 150 | ok | 0.964 [0.95,0.97] | 0.006 [0.00,0.01] | 0.000 [0.00,0.00] | 0.030 [0.02,0.04] |
| k3_partial | 1350 | 150 | ok | 0.956 [0.94,0.97] | 0.001 [0.00,0.00] | 0.000 [0.00,0.00] | 0.043 [0.03,0.06] |
| k5_bare | 198 | 22 | INVALID_underpowered | 0.944 [0.90,0.97] | 0.000 [0.00,0.02] | 0.000 [0.00,0.02] | 0.056 [0.03,0.10] |
| k5_full | 198 | 22 | INVALID_underpowered | 1.000 [0.98,1.00] | 0.000 [0.00,0.02] | 0.000 [0.00,0.02] | 0.000 [0.00,0.02] |
| k5_partial | 198 | 22 | INVALID_underpowered | 0.884 [0.83,0.92] | 0.000 [0.00,0.02] | 0.000 [0.00,0.02] | 0.116 [0.08,0.17] |

## Contrasts

```json
{
  "k0_vs_k1_gap": {
    "absorbed_k0": 0.7126,
    "absorbed_k1_bare": 0.7844,
    "gap_k1_minus_k0": 0.0718
  },
  "monotonic_trend": {
    "bare_rate_by_k": {
      "0": 0.7126,
      "1": 0.7844,
      "2": 0.943,
      "3": 0.9326,
      "5": 0.9444
    },
    "isotonic_nondecreasing": false,
    "k5_minus_k0": 0.2318,
    "spearman_rho": 0.9
  },
  "opacity_restoration": {
    "per_k": {
      "k1": {
        "bare": 0.7844,
        "full": 0.9363,
        "restoration_bare_minus_full": -0.1519
      },
      "k2": {
        "bare": 0.943,
        "full": 0.9526,
        "restoration_bare_minus_full": -0.0096
      },
      "k3": {
        "bare": 0.9326,
        "full": 0.9644,
        "restoration_bare_minus_full": -0.0318
      },
      "k5": {
        "bare": 0.9444,
        "full": 1.0,
        "restoration_bare_minus_full": -0.0556
      }
    },
    "pooled_kge1": {
      "bare": 0.8894,
      "full": 0.9534,
      "restoration": -0.064
    }
  }
}
```

Gold cohort (solve rates / eligible per k): `{"0": {"eligible": 258, "kept": 150, "solve_rate": 0.1433, "tested": 1800}, "1": {"eligible": 255, "kept": 150, "solve_rate": 0.1417, "tested": 1800}, "2": {"eligible": 241, "kept": 150, "solve_rate": 0.1339, "tested": 1800}, "3": {"eligible": 162, "kept": 150, "solve_rate": 0.09, "tested": 1800}, "5": {"eligible": 22, "kept": 22, "solve_rate": 0.0122, "tested": 1800}}`

Notes: Wilson95 + program-cluster bootstrap95 (n_boot=1000); cluster=program family (program_id)
