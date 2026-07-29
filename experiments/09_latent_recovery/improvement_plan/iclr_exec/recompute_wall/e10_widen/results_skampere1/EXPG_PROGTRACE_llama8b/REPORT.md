# EXPG_PROGTRACE pilot report

Generated 2026-07-28T07:52:33. Model NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef7999a2b5ba950ab3974312feeedbfe0b77 (pinned=False), backend vllm, greedy.

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 240, "parse_rate": 1.0, "solve_rate": 1.0}, "kc5": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.2667}, "kr1": {"n": 240, "parse_rate": 1.0, "solve_rate": 1.0}, "kr8": {"n": 240, "parse_rate": 1.0, "solve_rate": 0.9958}}, "failure_taxonomy": {"wrong_value": 177}, "n_programs": 960, "parse_rate_line_level": 1.0, "parse_rate_program_level": 1.0, "solve_rate": 0.8156}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": true} (pass=True)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 150 | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.020 [0.00,0.05] (3/150) | 0.020 [0.00,0.05] (3/150) | 0.000 [0.00,0.00] (0/150) |
| true_interruption | 150 | 0.880 [0.83,0.93] (132/150) | 1.000 [1.00,1.00] (150/150) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.013 [0.00,0.03] (2/150) | 0.013 [0.00,0.03] (2/150) | 0.120 [0.07,0.17] (18/150) |
| adjacent_contradiction | 150 | 0.387 [0.31,0.47] (58/150) | 0.393 [0.31,0.47] (59/150) | 0.553 [0.47,0.63] (83/150) | 0.580 [0.50,0.66] (87/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.067 [0.03,0.11] (10/150) | 0.393 [0.31,0.47] (59/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr1 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.027 [0.01,0.05] (4/150) | 0.007 [0.00,0.02] (1/150) | 0.000 [0.00,0.00] (0/150) |
| opfree_kr8 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.020 [0.00,0.04] (3/150) | 0.013 [0.00,0.03] (2/150) | 0.000 [0.00,0.00] (0/150) |
| onehop_kc1 | 150 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 1.000 [1.00,1.00] (150/150) | 1.000 [1.00,1.00] (150/150) | 0.000 [0.00,0.00] (0/150) | 0.000 | 0.000 [0.00,0.00] (0/150) | 0.000 [0.00,0.00] (0/150) | 0.007 [0.00,0.02] (1/150) | 0.007 [0.00,0.02] (1/150) | 0.000 [0.00,0.00] (0/150) |
| deep_kc5 | 64 | 0.000 [0.00,0.00] (0/64) | 0.000 [0.00,0.00] (0/64) | 1.000 [1.00,1.00] (64/64) | 1.000 [1.00,1.00] (64/64) | 0.000 [0.00,0.00] (0/64) | 0.000 | 0.000 [0.00,0.00] (0/64) | 0.000 [0.00,0.00] (0/64) | 0.016 [0.00,0.05] (1/64) | 0.031 [0.00,0.08] (2/64) | 0.000 [0.00,0.00] (0/64) |

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
      "doubt_used": 0.0667,
      "judge_doubt": 0.0667,
      "next_read_absorbed": 0.5533
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0267,
      "judge_doubt": 0.0267,
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
    "delta_kr1_minus_kr8": 0.0,
    "kr1": 1.0,
    "kr8": 1.0
  },
  "judge_doubt": {
    "delta_kr1_minus_kr8": 0.0067,
    "kr1": 0.0267,
    "kr8": 0.02
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": -0.0066,
    "kr1": 0.0067,
    "kr8": 0.0133
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
