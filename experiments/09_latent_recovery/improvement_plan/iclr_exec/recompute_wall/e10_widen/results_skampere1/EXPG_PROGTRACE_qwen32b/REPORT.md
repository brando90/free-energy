# EXPG_PROGTRACE pilot report

Generated 2026-07-28T07:52:40. Model Qwen/Qwen2.5-32B-Instruct @ 5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd (pinned=False), backend vllm, greedy.

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.9958}, "kc5": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.9417}, "kr1": {"n": 240, "parse_rate": 1.0, "solve_rate": 1.0}, "kr8": {"n": 240, "parse_rate": 1.0, "solve_rate": 1.0}}, "failure_taxonomy": {"wrong_value": 15}, "n_programs": 960, "parse_rate_line_level": 1.0, "parse_rate_program_level": 1.0, "solve_rate": 0.9844}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": true} (pass=True)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 150 | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.020 [0.00,0.05] (3/150) | 0.020 [0.00,0.05] (3/150) | 0.000 [0.00,0.00] (0/150) |
| true_interruption | 150 | 0.973 [0.94,0.99] (146/150) | 1.000 [1.00,1.00] (150/150) | n.m. | n.m. | n.m. | 0.000 | 0.013 [0.00,0.03] (2/150) | 0.020 [0.00,0.05] (3/150) | 0.020 [0.00,0.05] (3/150) | 0.020 [0.00,0.04] (3/150) | 0.020 [0.00,0.05] (3/150) |
| adjacent_contradiction | 150 | 0.920 [0.87,0.96] (138/150) | 0.940 [0.90,0.97] (141/150) | 0.060 [0.03,0.10] (9/150) | 0.060 [0.03,0.10] (9/150) | 0.080 [0.04,0.13] (12/150) | 0.000 | 0.007 [0.00,0.02] (1/150) | 0.027 [0.01,0.05] (4/150) | 0.033 [0.01,0.07] (5/150) | 0.847 [0.79,0.90] (127/150) | 0.020 [0.00,0.05] (3/150) |
| opfree_kr1 | 150 | 0.180 [0.11,0.24] (27/150) | 0.180 [0.11,0.24] (27/150) | 0.787 [0.72,0.85] (118/150) | 0.707 [0.63,0.78] (106/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.020 [0.00,0.05] (3/150) | 0.167 [0.11,0.23] (25/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr8 | 150 | 0.313 [0.24,0.39] (47/150) | 0.320 [0.25,0.39] (48/150) | 0.620 [0.54,0.69] (93/150) | 0.493 [0.41,0.57] (74/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.027 [0.01,0.05] (4/150) | 0.253 [0.18,0.32] (38/150) | 0.000 [0.00,0.00] (0/150) |
| onehop_kc1 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.007 [0.00,0.02] (1/150) | 0.007 [0.00,0.02] (1/150) | 0.000 [0.00,0.00] (0/150) |
| deep_kc5 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.007 [0.00,0.02] (1/150) | 0.067 [0.03,0.11] (10/150) | 0.000 [0.00,0.00] (0/150) |

## Gate G-B (floor gate)

```json
{
  "branch": "full_grid",
  "criterion": "doubt >= 0.10 OR next-read absorption <= 0.85 on adjacent_contradiction or opfree_kr1",
  "doubt_dv": "judge",
  "full_grid_viable": true,
  "inputs": {
    "adjacent_contradiction": {
      "doubt_lex": 0.0067,
      "doubt_used": 0.0333,
      "judge_doubt": 0.0333,
      "next_read_absorbed": 0.06
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.02,
      "judge_doubt": 0.02,
      "next_read_absorbed": 0.7867
    }
  }
}
```

## k_r manipulation probe (opfree_kr1 vs opfree_kr8; exploratory at pilot n)

```json
{
  "doubt_broad": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  },
  "doubt_lex": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  },
  "final_output_absorbed": {
    "delta_kr1_minus_kr8": 0.2134,
    "kr1": 0.7067,
    "kr8": 0.4933
  },
  "judge_doubt": {
    "delta_kr1_minus_kr8": -0.0067,
    "kr1": 0.02,
    "kr8": 0.0267
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": -0.0866,
    "kr1": 0.1667,
    "kr8": 0.2533
  },
  "next_read_absorbed": {
    "delta_kr1_minus_kr8": 0.1667,
    "kr1": 0.7867,
    "kr8": 0.62
  },
  "repair_event": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  },
  "trace_valid": {
    "delta_kr1_minus_kr8": -0.1333,
    "kr1": 0.18,
    "kr8": 0.3133
  }
}
```

Notes: unparsed and generation-failed runs retained in denominators (house convention); *_parsed_only columns give the dual denominator. wilson95 + program-cluster bootstrap95 (n_boot=1000).
