# EXPG_PROGTRACE pilot report

Generated 2026-07-02T13:53:36. Model Qwen/Qwen2.5-7B-Instruct @ a09a35458c702b33eeacc393d103063234e8bc28 (pinned=True), backend vllm, greedy.

## Cohort funnel + gate G-A

{"by_shape": {"kc1": {"n": 8, "parse_rate": 1.0, "solve_rate": 1.0}, "kc5": {"n": 8, "parse_rate": 1.0, "solve_rate": 1.0}, "kr1": {"n": 8, "parse_rate": 1.0, "solve_rate": 1.0}, "kr8": {"n": 8, "parse_rate": 1.0, "solve_rate": 1.0}}, "failure_taxonomy": {}, "n_programs": 32, "parse_rate_line_level": 1.0, "parse_rate_program_level": 1.0, "solve_rate": 1.0}, gate G-A: {"parse_gate_0.90": true, "solve_gate_0.60": true} (pass=True)

## Stage-1 pilot cells (mid position, n<=35/cell)

| cell | n | trace_valid | final_output_valid | next_read_absorbed | final_output_absorbed | repair_event | line_skip | doubt_lex | doubt_broad | judge_doubt | judge_reject | unparsed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| benign_paraphrase | 5 | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |
| true_interruption | 5 | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | n.m. | n.m. | n.m. | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |
| adjacent_contradiction | 5 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | 0.000 [0.00,0.00] (0/5) | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |
| opfree_kr1 | 5 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | 0.000 [0.00,0.00] (0/5) | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |
| opfree_kr8 | 5 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | 0.000 [0.00,0.00] (0/5) | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |
| onehop_kc1 | 5 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | 0.000 [0.00,0.00] (0/5) | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |
| deep_kc5 | 5 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | 1.000 [1.00,1.00] (5/5) | 1.000 [1.00,1.00] (5/5) | 0.000 [0.00,0.00] (0/5) | 0.000 | 0.000 [0.00,0.00] (0/5) | 0.000 [0.00,0.00] (0/5) | n.m. | n.m. | 0.000 [0.00,0.00] (0/5) |

## Gate G-B (floor gate)

```json
{
  "branch": "floor_documentation",
  "criterion": "doubt >= 0.10 OR next-read absorption <= 0.85 on adjacent_contradiction or opfree_kr1",
  "doubt_dv": "lexical",
  "full_grid_viable": false,
  "inputs": {
    "adjacent_contradiction": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0,
      "judge_doubt": null,
      "next_read_absorbed": 1.0
    },
    "opfree_kr1": {
      "doubt_lex": 0.0,
      "doubt_used": 0.0,
      "judge_doubt": null,
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
    "delta_kr1_minus_kr8": null,
    "kr1": null,
    "kr8": null
  },
  "judge_reject": {
    "delta_kr1_minus_kr8": null,
    "kr1": null,
    "kr8": null
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
