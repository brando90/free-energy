# E12 FIXES-LADDER report -- claude-haiku-4-5 (TRUE prefill)

Does an explicit verify instruction breach the recompute wall (E10/E11: computed-value plants absorbed ~1.0)? Arms A0 baseline / A1 generic / A2 targeted / A3 few-shot. Instruction prepended to the user block; prefill unchanged; golds regenerated per arm. Arms present: A0, A1, A2, A3.

## Ladder: absorbed rate by arm (Wilson95), per cell

### adjacent_contradiction (readable control)

| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |
|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | 0.008 | [0.00,0.03] | 0.988 | 0.004 | 184 | 0.992 | 240(60) | ci_widened |
| A1 (generic) | 0.008 | [0.00,0.03] | 0.992 | 0.000 | 196 | 0.992 | 240(60) | ci_widened |
| A2 (targeted) | 0.025 | [0.01,0.05] | 0.929 | 0.000 | 241 | 0.996 | 240(60) | ci_widened |
| A3 (few-shot) | 0.033 | [0.02,0.06] | 0.908 | 0.008 | 224 | 0.988 | 240(60) | ci_widened |

### onehop_kc1 (shallow computed)

| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |
|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.100 | 240(60) | ci_widened |
| A1 (generic) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.100 | 240(60) | ci_widened |
| A2 (targeted) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.100 | 240(60) | ci_widened |
| A3 (few-shot) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.100 | 240(60) | ci_widened |

### deep_kc5 (deep computed)

| arm | absorbed | Wilson95 | silently_corr | flagged | out_tok/cont | shows_recompute | n(prog) | power |
|---|---|---|---|---|---|---|---|---|
| A0 (baseline) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.067 | 240(60) | ci_widened |
| A1 (generic) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.067 | 240(60) | ci_widened |
| A2 (targeted) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 163 | 0.067 | 240(60) | ci_widened |
| A3 (few-shot) | 1.000 | [0.98,1.00] | 0.000 | 0.000 | 164 | 0.067 | 240(60) | ci_widened |

## Instruction-breach contrasts (vs A0 baseline)

Breach = absorbed drop >= 0.15 AND Wilson95 intervals disjoint.

| cell | arm | absorbed A0 -> arm | drop | CIs disjoint | verdict |
|---|---|---|---|---|---|
| k0_bare | A1 | 0.008 -> 0.008 | 0.000 | False | **wall_holds** |
| k0_bare | A2 | 0.008 -> 0.025 | -0.017 | False | **wall_holds** |
| k0_bare | A3 | 0.008 -> 0.033 | -0.025 | False | **wall_holds** |
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
- Cost: {"A0": 2.1115, "A1": 2.1552, "A2": 2.2399, "A3": 2.3588} (per arm, this run). Cumulative hard stop enforced by api_gen ledger.
- Gold cohort (eligible/kept) per arm: `{"A0": {"0": {"eligible": 91, "kept": 60}, "1": {"eligible": 85, "kept": 60}, "5": {"eligible": 94, "kept": 60}}, "A1": {"0": {"eligible": 91, "kept": 60}, "1": {"eligible": 83, "kept": 60}, "5": {"eligible": 90, "kept": 60}}, "A2": {"0": {"eligible": 90, "kept": 60}, "1": {"eligible": 82, "kept": 60}, "5": {"eligible": 81, "kept": 60}}, "A3": {"0": {"eligible": 90, "kept": 60}, "1": {"eligible": 80, "kept": 60}, "5": {"eligible": 88, "kept": 60}}}`

