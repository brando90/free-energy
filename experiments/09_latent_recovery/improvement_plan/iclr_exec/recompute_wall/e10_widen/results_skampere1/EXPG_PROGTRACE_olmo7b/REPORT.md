# EXPG_PROGTRACE pilot report

Generated 2026-07-28T07:52:26. Model allenai/OLMo-2-1124-7B-Instruct @ 470b1fba1ae01581f270116362ee4aa1b97f4c84 (pinned=False), backend vllm, greedy.

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.275}, "kc5": {"n": 240, "parse_rate": 0.975, "solve_rate": 0.0625}, "kr1": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.6875}, "kr8": {"n": 240, "parse_rate": 0.9917, "solve_rate": 0.45}}, "failure_taxonomy": {"parse": 1, "wrong_output": 5, "wrong_value": 600}, "n_programs": 960, "parse_rate_line_level": 0.9974, "parse_rate_program_level": 0.9917, "solve_rate": 0.3688}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": false} (pass=False)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 150 | 0.967 [0.93,0.99] (145/150) | 0.967 [0.93,0.99] (145/150) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.040 [0.01,0.07] (6/150) | 0.013 [0.00,0.03] (2/150) | 0.000 [0.00,0.00] (0/150) |
| true_interruption | 150 | 0.400 [0.32,0.49] (60/150) | 0.887 [0.83,0.93] (133/150) | n.m. | n.m. | n.m. | 0.003 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.013 [0.00,0.03] (2/150) | 0.007 [0.00,0.02] (1/150) | 0.540 [0.46,0.62] (81/150) |
| adjacent_contradiction | 150 | 0.133 [0.09,0.19] (20/150) | 0.147 [0.09,0.20] (22/150) | 0.820 [0.76,0.88] (123/150) | 0.653 [0.57,0.73] (98/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.133 [0.08,0.19] (20/150) | 0.233 [0.17,0.30] (35/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr1 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.987 [0.97,1.00] (148/150) | 0.827 [0.77,0.89] (124/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.013 [0.00,0.03] (2/150) | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr8 | 108 | 0.000 [0.00,0.00] (0/108) | 0.000 [0.00,0.00] (0/108) | 0.944 [0.90,0.98] (102/108) | 0.759 [0.68,0.83] (82/108) | 0.000 [0.00,0.00] (0/108) | 0.000 | 0.000 [0.00,0.00] (0/108) | 0.000 [0.00,0.00] (0/108) | 0.009 [0.00,0.04] (1/108) | 0.056 [0.02,0.10] (6/108) | 0.000 [0.00,0.00] (0/108) |
| onehop_kc1 | 66 | 0.000 [0.00,0.00] (0/66) | 0.015 [0.00,0.05] (1/66) | 1.000 [1.00,1.00] (66/66) | 0.727 [0.62,0.83] (48/66) | 0.000 [0.00,0.00] (0/66) | 0.003 | 0.000 [0.00,0.00] (0/66) | 0.000 [0.00,0.00] (0/66) | 0.015 [0.00,0.05] (1/66) | 0.000 [0.00,0.00] (0/66) | 0.015 [0.00,0.05] (1/66) |
| deep_kc5 | 15 | 0.000 [0.00,0.00] (0/15) | 0.000 [0.00,0.00] (0/15) | 1.000 [1.00,1.00] (15/15) | 0.867 [0.67,1.00] (13/15) | 0.000 [0.00,0.00] (0/15) | 0.000 | 0.000 [0.00,0.00] (0/15) | 0.000 [0.00,0.00] (0/15) | 0.000 [0.00,0.00] (0/15) | 0.067 [0.00,0.20] (1/15) | 0.000 [0.00,0.00] (0/15) |

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
      "doubt_used": 0.1333,
      "judge_doubt": 0.1333,
      "next_read_absorbed": 0.82
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0133,
      "judge_doubt": 0.0133,
      "next_read_absorbed": 0.9867
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
    "delta_kr1_minus_kr8": 0.0674,
    "kr1": 0.8267,
    "kr8": 0.7593
  },
  "judge_doubt": {
    "delta_kr1_minus_kr8": 0.004,
    "kr1": 0.0133,
    "kr8": 0.0093
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": -0.0556,
    "kr1": 0.0,
    "kr8": 0.0556
  },
  "next_read_absorbed": {
    "delta_kr1_minus_kr8": 0.0423,
    "kr1": 0.9867,
    "kr8": 0.9444
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
