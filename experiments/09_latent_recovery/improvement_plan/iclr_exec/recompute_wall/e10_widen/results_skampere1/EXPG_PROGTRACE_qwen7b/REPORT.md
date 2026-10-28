# EXPG_PROGTRACE pilot report

Generated 2026-07-28T07:52:21. Model Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28 (pinned=True), backend vllm, greedy.

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.9708}, "kc5": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.4375}, "kr1": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.9875}, "kr8": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.9417}}, "failure_taxonomy": {"wrong_value": 159}, "n_programs": 960, "parse_rate_line_level": 1.0, "parse_rate_program_level": 1.0, "solve_rate": 0.8344}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": true} (pass=True)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 150 | 0.993 [0.98,1.00] (149/150) | 0.993 [0.98,1.00] (149/150) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.020 [0.00,0.05] (3/150) | 0.020 [0.00,0.05] (3/150) | 0.000 [0.00,0.00] (0/150) |
| true_interruption | 150 | 0.987 [0.97,1.00] (148/150) | 0.993 [0.98,1.00] (149/150) | n.m. | n.m. | n.m. | 0.001 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.013 [0.00,0.03] (2/150) | 0.020 [0.00,0.05] (3/150) | 0.000 [0.00,0.00] (0/150) |
| adjacent_contradiction | 150 | 0.240 [0.17,0.31] (36/150) | 0.240 [0.17,0.31] (36/150) | 0.753 [0.67,0.82] (113/150) | 0.727 [0.65,0.79] (109/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.200 [0.14,0.27] (30/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr1 | 150 | 0.000 [0.00,0.00] (0/150) | 0.007 [0.00,0.02] (1/150) | 0.960 [0.93,0.99] (144/150) | 0.840 [0.78,0.89] (126/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.027 [0.01,0.05] (4/150) | 0.060 [0.03,0.10] (9/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr8 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.987 [0.97,1.00] (148/150) | 0.913 [0.87,0.95] (137/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.020 [0.00,0.05] (3/150) | 0.013 [0.00,0.03] (2/150) | 0.000 [0.00,0.00] (0/150) |
| onehop_kc1 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.960 [0.93,0.99] (144/150) | 0.880 [0.83,0.93] (132/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.007 [0.00,0.02] (1/150) | 0.033 [0.01,0.07] (5/150) | 0.000 [0.00,0.00] (0/150) |
| deep_kc5 | 105 | 0.000 [0.00,0.00] (0/105) | 0.000 [0.00,0.00] (0/105) | 0.991 [0.97,1.00] (104/105) | 0.991 [0.97,1.00] (104/105) | 0.000 [0.00,0.00] (0/105) | 0.000 | 0.000 [0.00,0.00] (0/105) | 0.000 [0.00,0.00] (0/105) | 0.029 [0.00,0.07] (3/105) | 0.038 [0.01,0.08] (4/105) | 0.000 [0.00,0.00] (0/105) |

## Gate G-B (floor gate)

```json
{
  "branch": "full_grid",
  "criterion": "doubt >= 0.10 OR next-read absorption <= 0.85 on adjacent_contradiction or opfree_kr1",
  "doubt_dv": "judge",
  "full_grid_viable": true,
  "inputs": {
    "adjacent_contradiction": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0,
      "judge_doubt": 0.0,
      "next_read_absorbed": 0.7533
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0267,
      "judge_doubt": 0.0267,
      "next_read_absorbed": 0.96
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
    "delta_kr1_minus_kr8": -0.0733,
    "kr1": 0.84,
    "kr8": 0.9133
  },
  "judge_doubt": {
    "delta_kr1_minus_kr8": 0.0067,
    "kr1": 0.0267,
    "kr8": 0.02
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": 0.0467,
    "kr1": 0.06,
    "kr8": 0.0133
  },
  "next_read_absorbed": {
    "delta_kr1_minus_kr8": -0.0267,
    "kr1": 0.96,
    "kr8": 0.9867
  },
  "repair_event": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  },
  "trace_valid": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  }
}
```

Notes: unparsed and generation-failed runs retained in denominators (house convention); *_parsed_only columns give the dual denominator. wilson95 + program-cluster bootstrap95 (n_boot=1000).
