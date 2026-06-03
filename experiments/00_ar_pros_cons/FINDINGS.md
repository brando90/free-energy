# FINDINGS — living results log

Update this as runs complete. One entry per probe/grid run. The point is to record
**what actually happened**, especially where it contradicts the pre-registered
expectation in `probes/PROBE_SPECS.md`. Surprises are the most valuable rows.

Each entry: date, config (smoke / full, model scale, seeds), the measured effect
with CI, the verdict, and a one-line interpretation. Keep the pre-registered
prediction visible next to the result so drift is obvious.

---

## Status board

| Claim | Pre-registered | Measured verdict | Effect (CI) | Layer | Notes |
|---|---|---|---|---|---|
| 1 softmax bottleneck | CONFIRMED toy; real ? | — | — | arch+data | not yet run |
| 2 mode-covering | CONFIRMED toy; realized ? | — | — | objective | not yet run |
| 3 rank collapse | CONFIRMED pure; MASKED trained | — | — | architecture | not yet run |
| 4 partition removable | likely CONFIRMED | — | — | arch (ablation) | not yet run |
| 5 fixed compute | CONFIRMED | — | — | arch+complexity | not yet run |
| 6 error compounding | NOT-SUPPORTED (verifier) | — | — | trained+verifier | the key test |
| 7 reversal curse | CONFIRMED | — | — | trained behavior | not yet run |
| 8 Lipschitz–margin | CONFIRMED | — | — | trained behavior | not yet run |
| data wall | descriptive | — | — | external | not yet run |

---

## Entry template (copy per run)

```
### [DATE] probe_NN — <name>  (config: smoke|full, model=<...>, seeds=[...])
- Pre-registered prediction: <...>
- Positive control: PASS|FAIL  (<one line on the control result>)
- Measurement: <effect size> [CI low, CI high]
- Verdict: CONFIRMED | PARTIAL | MASKED | INTERACTION-DRIVEN | NOT-SUPPORTED
- Compute-matched vs param-matched: <agree? differ how?>
- Interpretation: <one or two lines>
- Surprise vs pre-registration: <none | describe>
- Artifacts: probes/out/probe_NN/figure.png, stats.json
```

---

## Integrated grid entries

```
### [DATE] grid run — <design: full|fractional, cells=N, mode=from_scratch|finetune>
- Cells completed: <...> ; seeds per cell: <...>
- Bridge plot: integrated/out/bridge_scatter.png
- Load-bearing mechanisms (CI excludes 0): <...>
- Masked mechanisms (strong isolated, ~0 realized): <...> ; masker: <...>
- Interaction terms: objective×head=<coef,CI>, attn×residual=<...>, objective×compute=<...>
- Conjunctive EBM hypothesis: SUPPORTED | REJECTED  (<which interaction carried it>)
- Emergence order (baseline): <which of rank / e / pass-rate moved first>
- Surprises: <...>
```

---

## Running narrative

_(Write the story here as it develops — what the suite is teaching you, which
mechanisms turned out load-bearing vs masked, and whether the EBM motivation is
additive or conjunctive in this setting. This is the section that becomes the blog
post / talk.)_
