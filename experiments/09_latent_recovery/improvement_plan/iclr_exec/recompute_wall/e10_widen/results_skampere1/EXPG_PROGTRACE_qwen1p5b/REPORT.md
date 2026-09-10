# EXPG_PROGTRACE pilot report

Generated 2026-07-28T07:52:14. Model Qwen/Qwen2.5-1.5B-Instruct @ 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 (pinned=False), backend vllm, greedy.

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 240, "parse_rate": 0.9792, "solve_rate": 0.125}, "kc5": {"n": 240, "parse_rate": 0.9833, "solve_rate": 0.0042}, "kr1": {"n": 240, "parse_rate": 0.9833, "solve_rate": 0.2542}, "kr8": {"n": 240, "parse_rate": 0.9583, "solve_rate": 0.0375}}, "failure_taxonomy": {"parse": 5, "wrong_output": 1, "wrong_value": 853}, "n_programs": 960, "parse_rate_line_level": 0.9971, "parse_rate_program_level": 0.976, "solve_rate": 0.1052}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": false} (pass=False)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 61 | 0.836 [0.74,0.92] (51/61) | 0.853 [0.75,0.93] (52/61) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/61) | 0.000 [0.00,0.00] (0/61) | 0.016 [0.00,0.05] (1/61) | 0.016 [0.00,0.05] (1/61) | 0.000 [0.00,0.00] (0/61) |
| true_interruption | 61 | 0.672 [0.54,0.79] (41/61) | 0.738 [0.61,0.84] (45/61) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/61) | 0.000 [0.00,0.00] (0/61) | 0.066 [0.02,0.13] (4/61) | 0.000 [0.00,0.00] (0/61) | 0.016 [0.00,0.05] (1/61) |
| adjacent_contradiction | 61 | 0.000 [0.00,0.00] (0/61) | 0.000 [0.00,0.00] (0/61) | 1.000 [1.00,1.00] (61/61) | 0.541 [0.41,0.67] (33/61) | 0.000 [0.00,0.00] (0/61) | 0.009 | 0.000 [0.00,0.00] (0/61) | 0.000 [0.00,0.00] (0/61) | 0.049 [0.00,0.10] (3/61) | 0.016 [0.00,0.05] (1/61) | 0.033 [0.00,0.08] (2/61) |
| opfree_kr1 | 61 | 0.000 [0.00,0.00] (0/61) | 0.000 [0.00,0.00] (0/61) | 0.803 [0.70,0.90] (49/61) | 0.410 [0.30,0.54] (25/61) | 0.000 [0.00,0.00] (0/61) | 0.008 | 0.000 [0.00,0.00] (0/61) | 0.000 [0.00,0.00] (0/61) | 0.016 [0.00,0.05] (1/61) | 0.082 [0.02,0.16] (5/61) | 0.016 [0.00,0.05] (1/61) |
| opfree_kr8 | 9 | 0.000 [0.00,0.00] (0/9) | 0.000 [0.00,0.00] (0/9) | 1.000 [1.00,1.00] (9/9) | 0.444 [0.11,0.78] (4/9) | 0.000 [0.00,0.00] (0/9) | 0.000 | 0.000 [0.00,0.00] (0/9) | 0.000 [0.00,0.00] (0/9) | 0.000 [0.00,0.00] (0/9) | 0.000 [0.00,0.00] (0/9) | 0.000 [0.00,0.00] (0/9) |
| onehop_kc1 | 30 | 0.000 [0.00,0.00] (0/30) | 0.000 [0.00,0.00] (0/30) | 0.933 [0.83,1.00] (28/30) | 0.500 [0.33,0.70] (15/30) | 0.000 [0.00,0.00] (0/30) | 0.000 | 0.000 [0.00,0.00] (0/30) | 0.000 [0.00,0.00] (0/30) | 0.033 [0.00,0.10] (1/30) | 0.000 [0.00,0.00] (0/30) | 0.000 [0.00,0.00] (0/30) |
| deep_kc5 | 1 | 0.000 [0.00,0.00] (0/1) | 0.000 [0.00,0.00] (0/1) | 1.000 [1.00,1.00] (1/1) | 1.000 [1.00,1.00] (1/1) | 0.000 [0.00,0.00] (0/1) | 0.000 | 0.000 [0.00,0.00] (0/1) | 0.000 [0.00,0.00] (0/1) | 0.000 [0.00,0.00] (0/1) | 0.000 [0.00,0.00] (0/1) | 0.000 [0.00,0.00] (0/1) |

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
      "doubt_used": 0.0492,
      "judge_doubt": 0.0492,
      "next_read_absorbed": 1.0
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0164,
      "judge_doubt": 0.0164,
      "next_read_absorbed": 0.8033
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
    "delta_kr1_minus_kr8": -0.0346,
    "kr1": 0.4098,
    "kr8": 0.4444
  },
  "judge_doubt": {
    "delta_kr1_minus_kr8": 0.0164,
    "kr1": 0.0164,
    "kr8": 0.0
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": 0.082,
    "kr1": 0.082,
    "kr8": 0.0
  },
  "next_read_absorbed": {
    "delta_kr1_minus_kr8": -0.1967,
    "kr1": 0.8033,
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
