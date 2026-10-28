# EXPG_PROGTRACE pilot report

Generated 2026-07-02T14:07:16. Model Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28 (pinned=True), backend vllm, greedy.

## Stage-0 (grammar gate)

Gates: {"parse_gate_0.95": true, "solve_gate_0.70": true} (pass=True). Gold eval: {"by_shape": {"kc1": {"n": 25, "parse_rate": 1.0, "solve_rate": 1.0}, "kc5": {"n": 25, "parse_rate": 1.0, "solve_rate": 0.68}, "kr1": {"n": 25, "parse_rate": 1.0, "solve_rate": 1.0}, "kr8": {"n": 25, "parse_rate": 1.0, "solve_rate": 1.0}}, "failure_taxonomy": {"wrong_value": 8}, "n_programs": 100, "parse_rate_line_level": 1.0, "parse_rate_program_level": 1.0, "solve_rate": 0.92}

Final knobs: `{"const_max": 45, "const_min": 2, "deep_const_max": 9, "filler_const_prob": 0.35, "forward_use_gap": 2, "loop_prob": 0.0, "max_attempts_per_program": 400, "mult_prob": 0.15, "read_const_max": 20, "value_max": 999, "value_min": 1}`

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 60, "parse_rate": 1.0, "solve_rate": 0.9667}, "kc5": {"n": 60, "parse_rate": 1.0, "solve_rate": 0.45}, "kr1": {"n": 60, "parse_rate": 1.0, "solve_rate": 0.9833}, "kr8": {"n": 60, "parse_rate": 1.0, "solve_rate": 0.9167}}, "failure_taxonomy": {"wrong_value": 41}, "n_programs": 240, "parse_rate_line_level": 1.0, "parse_rate_program_level": 1.0, "solve_rate": 0.8292}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": true} (pass=True)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 35 | 0.971 [0.89,1.00] (34/35) | 0.971 [0.89,1.00] (34/35) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.029 [0.00,0.09] (1/35) | 0.057 [0.00,0.14] (2/35) | 0.000 [0.00,0.00] (0/35) |
| true_interruption | 35 | 0.971 [0.89,1.00] (34/35) | 0.971 [0.89,1.00] (34/35) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.029 [0.00,0.09] (1/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) |
| adjacent_contradiction | 35 | 0.229 [0.09,0.37] (8/35) | 0.229 [0.09,0.37] (8/35) | 0.743 [0.60,0.89] (26/35) | 0.686 [0.51,0.83] (24/35) | 0.000 [0.00,0.00] (0/35) | 0.000 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.200 [0.09,0.34] (7/35) | 0.000 [0.00,0.00] (0/35) |
| opfree_kr1 | 35 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 1.000 [1.00,1.00] (35/35) | 0.829 [0.69,0.94] (29/35) | 0.000 [0.00,0.00] (0/35) | 0.000 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) |
| opfree_kr8 | 35 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 1.000 [1.00,1.00] (35/35) | 0.914 [0.83,1.00] (32/35) | 0.000 [0.00,0.00] (0/35) | 0.000 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) |
| onehop_kc1 | 35 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.943 [0.86,1.00] (33/35) | 0.857 [0.71,0.97] (30/35) | 0.000 [0.00,0.00] (0/35) | 0.000 | 0.000 [0.00,0.00] (0/35) | 0.000 [0.00,0.00] (0/35) | 0.029 [0.00,0.09] (1/35) | 0.029 [0.00,0.09] (1/35) | 0.000 [0.00,0.00] (0/35) |
| deep_kc5 | 27 | 0.000 [0.00,0.00] (0/27) | 0.000 [0.00,0.00] (0/27) | 1.000 [1.00,1.00] (27/27) | 1.000 [1.00,1.00] (27/27) | 0.000 [0.00,0.00] (0/27) | 0.000 | 0.000 [0.00,0.00] (0/27) | 0.000 [0.00,0.00] (0/27) | 0.000 [0.00,0.00] (0/27) | 0.111 [0.00,0.26] (3/27) | 0.000 [0.00,0.00] (0/27) |

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
      "next_read_absorbed": 0.7429
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0,
      "judge_doubt": 0.0,
      "next_read_absorbed": 1.0
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
    "delta_kr1_minus_kr8": -0.0857,
    "kr1": 0.8286,
    "kr8": 0.9143
  },
  "judge_doubt": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 0.0,
    "kr8": 0.0
  },
  "next_read_absorbed": {
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 1.0,
    "kr8": 1.0
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
