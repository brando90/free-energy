# E12 FIXES-LADDER report -- claude-haiku-4-5 (TRUE prefill)

Does an explicit verify instruction breach the recompute wall (E10/E11: computed-value plants absorbed ~1.0)? Arms A0 baseline / A1 generic / A2 targeted / A3 few-shot. Instruction prepended to the user block; prefill unchanged; golds regenerated per arm. Arms present: A0, A1, A2, A3.

## Ladder: absorbed rate by arm (Wilson95), per cell

### adjacent_contradiction (readable control)

| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |
|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | 0.017 | [0.01,0.03] | 0.982 | 0.002 | 187 | 0.985 | 540(60) | ci_widened |
| A1 (generic) | 0.009 | [0.00,0.02] | 0.987 | 0.002 | 196 | 0.991 | 540(60) | ci_widened |
| A2 (targeted) | 0.030 | [0.02,0.05] | 0.915 | 0.006 | 242 | 0.993 | 540(60) | ci_widened |
| A3 (few-shot) | 0.035 | [0.02,0.05] | 0.902 | 0.006 | 225 | 0.987 | 540(60) | ci_widened |

### onehop_kc1 (shallow computed)

| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |
|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 164 | 0.117 | 540(60) | ci_widened |
| A1 (generic) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 163 | 0.100 | 540(60) | ci_widened |
| A2 (targeted) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 163 | 0.100 | 540(60) | ci_widened |
| A3 (few-shot) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 163 | 0.100 | 540(60) | ci_widened |

### deep_kc5 (deep computed)

| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |
|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 163 | 0.067 | 540(60) | ci_widened |
| A1 (generic) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 163 | 0.067 | 540(60) | ci_widened |
| A2 (targeted) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 163 | 0.067 | 540(60) | ci_widened |
| A3 (few-shot) | 1.000 | [0.99,1.00] | 0.000 | 0.000 | 164 | 0.067 | 540(60) | ci_widened |

## Instruction-breach contrasts (vs A0 baseline)

Breach = absorbed drop >= 0.15 AND Wilson95 intervals disjoint.

| cell | arm | absorbed A0 -> arm | drop | CIs disjoint | verdict |
|---|---|---|---|---|---|
| k0_bare | A1 | 0.017 -> 0.009 | 0.007 | False | **wall_holds** |
| k0_bare | A2 | 0.017 -> 0.030 | -0.013 | False | **wall_holds** |
| k0_bare | A3 | 0.017 -> 0.035 | -0.018 | False | **wall_holds** |
| k1_bare | A1 | 1.000 -> 1.000 | 0.000 | False | **wall_holds** |
| k1_bare | A2 | 1.000 -> 1.000 | 0.000 | False | **wall_holds** |
| k1_bare | A3 | 1.000 -> 1.000 | 0.000 | False | **wall_holds** |
| k5_bare | A1 | 1.000 -> 1.000 | 0.000 | False | **wall_holds** |
| k5_bare | A2 | 1.000 -> 1.000 | 0.000 | False | **wall_holds** |
| k5_bare | A3 | 1.000 -> 1.000 | 0.000 | False | **wall_holds** |

## Notes

- Readable control (k0 adjacent_contradiction): a prompt should not need to fix what is already re-readable; large A0 movement here would indicate the instruction changes readable behaviour too (interpret computed-cell drops against this baseline).
- Compliance cost = out_tok/cont column; recompute signature (shows_recompute) is the mechanical grep defined in each arm's run_manifest.json.
- Flag channel: doubt_lex frozen lexicon (no 32B judge on the API arm).
- Cost: {"A0": 3.8029, "A1": 3.8499, "A2": 4.0291, "A3": 4.2388} (per arm, this run). Cumulative hard stop enforced by api_gen ledger.
- Gold cohort (eligible/kept) per arm: `{"A0": {"0": {"eligible": 91, "kept": 60}, "1": {"eligible": 85, "kept": 60}, "5": {"eligible": 91, "kept": 60}}, "A1": {"0": {"eligible": 91, "kept": 60}, "1": {"eligible": 83, "kept": 60}, "5": {"eligible": 90, "kept": 60}}, "A2": {"0": {"eligible": 91, "kept": 60}, "1": {"eligible": 82, "kept": 60}, "5": {"eligible": 77, "kept": 60}}, "A3": {"0": {"eligible": 90, "kept": 60}, "1": {"eligible": 79, "kept": 60}, "5": {"eligible": 88, "kept": 60}}}`

