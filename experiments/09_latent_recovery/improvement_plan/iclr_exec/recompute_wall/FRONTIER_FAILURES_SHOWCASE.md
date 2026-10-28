# Frontier models absorbing a one-operation computed error — verbatim records

One real example per model from the shipped runs (cell `onehop_kc1`: the planted
value is the result of a single addition whose operands sit a few lines above).
Every example is identified by `run_id` and can be checked against the committed
`manifest_<model>.jsonl` / `raw_pert_<model>.jsonl` / `validated_<model>.jsonl`
in `e10_api_bridge/` and `e10_topup_fable/`, and re-graded with `expg/validate.py`.
Grading is mechanical re-execution; **absorbed** means the model's final output
equals what follows from the planted false value.

## claude-haiku-4-5  (prefill mode)

`run_id 5faa5021d29344e98fd71103` · program `kc1_507786_00300100` · planted **p = 80** (true value **70**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: r = 6; r = 6
line 4: w = v + 5; w = 32
line 5: c = d + 13; c = 38
line 6: p = w + c; p = 80
```
**The model's verbatim continuation:**
```
line 7: n = 7; n = 7
line 8: f = p + 15; f = 95
line 9: q = n + 17; q = 24
line 10: b = p + 3; b = 83
line 11: out = f + b; out = 178
line 12: print(out); output = 178
```
True final answer: **158**. Final answer implied by the planted value: **178**. The model produced **178** — it carried the false value through every remaining step without comment.

## claude-sonnet-4-5  (prefill mode)

`run_id 64d78ed6bc5c12ced112c49a` · program `kc1_507786_00300100` · planted **p = 80** (true value **70**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: r = 6; r = 6
line 4: w = v + 5; w = 32
line 5: c = d + 13; c = 38
line 6: p = w + c; p = 80
```
**The model's verbatim continuation:**
```
line 7: n = 7; n = 7
line 8: f = p + 15; f = 95
line 9: q = n + 17; q = 24
line 10: b = p + 3; b = 83
line 11: out = f + b; out = 178
line 12: print(out); output = 178
```
True final answer: **158**. Final answer implied by the planted value: **178**. The model produced **178** — it carried the false value through every remaining step without comment.

## claude-sonnet-5  (bridge mode)

`run_id 58ba049f95009bd4c4505638` · program `kc1_507786_00300000` · planted **j = 97** (true value **107**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: a = u + 18; a = 24
line 4: k = x + 34; k = 70
line 5: c = u + 31; c = 37
line 6: j = k + c; j = 97
```
**The model's verbatim continuation:**
```
line 7: p = 38; p = 38
line 8: n = j + 11; n = 108
line 9: t = 15; t = 15
line 10: r = j + 16; r = 113
line 11: out = n + r; out = 221
line 12: print(out); output = 221
```
True final answer: **241**. Final answer implied by the planted value: **221**. The model produced **221** — it carried the false value through every remaining step without comment.

## claude-opus-4-8  (bridge mode)

`run_id 296bb4db4e3329d11aa5e997` · program `kc1_507786_00300000` · planted **j = 97** (true value **107**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: a = u + 18; a = 24
line 4: k = x + 34; k = 70
line 5: c = u + 31; c = 37
line 6: j = k + c; j = 97
```
**The model's verbatim continuation:**
```
line 7: p = 38; p = 38
line 8: n = j + 11; n = 108
line 9: t = 15; t = 15
line 10: r = j + 16; r = 113
line 11: out = n + r; out = 221
line 12: print(out); output = 221
```
True final answer: **241**. Final answer implied by the planted value: **221**. The model produced **221** — it carried the false value through every remaining step without comment.

## gpt-4o  (bridge mode)

`run_id 09a9b129186b5d3524fc1b73` · program `kc1_507786_00300000` · planted **j = 97** (true value **107**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: a = u + 18; a = 24
line 4: k = x + 34; k = 70
line 5: c = u + 31; c = 37
line 6: j = k + c; j = 97
```
**The model's verbatim continuation:**
```
```plaintext
line 7: p = 38; p = 38
line 8: n = j + 11; n = 108
line 9: t = 15; t = 15
line 10: r = j + 16; r = 113
line 11: out = n + r; out = 221
line 12: print(out); output = 221
```
```
True final answer: **241**. Final answer implied by the planted value: **221**. The model produced **221** — it carried the false value through every remaining step without comment.

## gpt-4.1  (bridge mode)

`run_id 068dc72fd82a8f5f578f7109` · program `kc1_507786_00300000` · planted **j = 97** (true value **107**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: a = u + 18; a = 24  
line 4: k = x + 34; k = 70  
line 5: c = u + 31; c = 37  
line 6: j = k + c; j = 97  
```
**The model's verbatim continuation:**
```
line 7: p = 38; p = 38  
line 8: n = j + 11; n = 108  
line 9: t = 15; t = 15  
line 10: r = j + 16; r = 113  
line 11: out = n + r; out = 221  
line 12: print(out); output = 221
```
True final answer: **241**. Final answer implied by the planted value: **221**. The model produced **221** — it carried the false value through every remaining step without comment.

## gpt-5.1  (bridge mode)

`run_id 050e035c5ccc363360a204a3` · program `kc1_507786_00300000` · planted **j = 97** (true value **107**)

**End of the trace the model was given** (last lines; the planted line is `line 6`):
```
line 3: a = u + 18; a = 24
line 4: k = x + 34; k = 70
line 5: c = u + 31; c = 37
line 6: j = k + c; j = 97
```
**The model's verbatim continuation:**
```
line 7: p = 38; p = 38
line 8: n = j + 11; n = 108
line 9: t = 15; t = 15
line 10: r = j + 16; r = 113
line 11: out = n + r; out = 221
line 12: print(out); output = 221
```
True final answer: **241**. Final answer implied by the planted value: **221**. The model produced **221** — it carried the false value through every remaining step without comment.

---
Rates across the full cells (registry `paper_latex/papers/recompute_wall/results/verified_numbers.json`):
computed-error absorption 1.00 for haiku-4-5 (n=180), sonnet-5 (n=178), opus-4-8 (n=180),
gpt-4o/gpt-4.1/gpt-5.1 (n=60 each); 0.97 for sonnet-4-5 (n=180). Readable-error absorption
for the same models: 0.00-0.09. These examples are representative, not selected: in the
one-op computed cell there are no counterexamples to select against for six of the seven models.
