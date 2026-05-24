# Integrated harness — all mechanisms, trained together

The isolated probes measure each mechanism's *latent strength*. The thing you
actually ship is one autoregressive model where every mechanism coexists,
interacts, and partially masks the others. This harness trains real AR models
end-to-end on VeriBench, logs every probe as an **online metric**, and runs a
**factorial ablation grid** so each mechanism's *realized* downstream effect is
**causally attributable** — not merely correlated.

It complements the isolated probes; it does not replace them.

---

## Why one trained model isn't enough (the attribution trap)

Train one model, measure eight things, and they all correlate with each other and
with the training step — so nothing is attributable. The fix is a **factorial
ablation grid**: train the *same* architecture with each suspected mechanism
independently toggled, and measure all probes on every cell. The *difference* a
toggle makes on a downstream metric, with everything else held fixed, is the
**causal effect** of that mechanism.

One run gives anecdote. The grid gives attribution.

---

## The toggles

Defined in `grid.yaml`, each with a control level and a treatment level, all else
identical:

| Toggle | Levels | Tests mechanism |
|---|---|---|
| `attn` | `softmax` / `sigmoid` / `linear` | partition-function tax (#4) in situ |
| `residual` | `on` / `off` | rank collapse (#3) |
| `mlp` | `on` / `off` | rank collapse (#3) |
| `head` | `single_softmax` / `mixture_of_softmax` | bottleneck (#1) |
| `objective` | `mle` / `margin_energy` | mode-covering (#2); energy variant = unnormalized head + margin loss, **Z paid only at eval** |
| `compute` | `fixed` / `scratchpad` | fixed-compute ceiling (#5) |

**Design size:** run the full cross-product only if cheap. Otherwise use a
**fractional factorial** — one-toggle-at-a-time off a strong baseline, **plus** the
2–3 interaction pairs most likely to couple:

- `objective × head` (does the energy objective only help with a higher-rank head?)
- `attn × residual` (does removing softmax interact with the residual path?)
- `objective × compute` (does the energy case need adaptive compute?)

`grid.yaml` makes the design matrix explicit; `run_grid.py` executes any subset.

---

## Online instrumentation

Every `eval_every` steps, against the same checkpoints, log all probes to one tidy
dataframe (schema in `docs/METHODOLOGY.md` §6):

- effective rank of (a) final hidden states, (b) the target-relevant subspace seen
  by the head — #1, #3 in situ
- per-step probability mass on verifier-rejected tokens — #2 realized `e`
- VeriBench / VeriBench-FTP **pass@1 and pass@k** via the Lean verifier — the
  ground-truth downstream metric
- `P(compiles)` vs proof length, refit each checkpoint to
  {geometric `(1−e)ⁿ`, constant, recoverable-Markov} — #6 over training time
- proof success vs proof depth, fixed vs scratchpad — #5 in situ
- output margin and `∏_i ‖W_i‖₂`, plus minimal-perturbation flip rate — #8
- reversal-curse asymmetry on held-out bidirectional lemmas — #7

Trajectories, not endpoints — so we see *when* each effect emerges.

---

## Analyses

| Script | Output |
|---|---|
| `train_integrated.py` | trains one grid cell, logs all online metrics |
| `run_grid.py` | executes the design matrix (`--smoke` for the CPU subset) |
| `bridge.py` | the headline scatter: isolated strength ↔ realized effect (see `BRIDGE.md`) |
| `interactions.py` | fits `pass_rate ~ toggleA * toggleB`; reports interaction terms + CIs |
| `dashboard.py` | one figure: per-mechanism verdict + realized effect size + CI |

### Interaction analysis
For each interaction pair, fit downstream pass rate `~ toggleA * toggleB` and report
whether the interaction term is significant (bootstrap CI). **Headline question:**
does the energy/margin objective help *only when* paired with the non-softmax head
or adaptive compute — i.e. is the EBM-style case **conjunctive**?

### Emergence-order plot
Overlay normalized trajectories of (rank, realized `e`, VeriBench pass rate) on a
shared step axis for the baseline run. Which moves first is evidence of causal
direction (e.g. does rank collapse *precede* or *follow* the pass-rate plateau?).

---

## Controls (enforced)

- same data, same token budget, **same compute per cell**
- report **compute-matched and param-matched** comparisons (toggles change param
  counts — report both so neither side hides behind FLOPs)
- **≥ 3 seeds** per cell; all deltas with **bootstrap CIs**
- pre-register the predicted **sign** of each toggle's effect in `grid.yaml`; flag
  surprises in `FINDINGS.md`
- a **CPU smoke config** (tiny model, 20-example VeriBench subset, 2 toggles) must
  run the *entire* grid + bridge end-to-end in minutes before any GPU spend

---

## Scope decision: from-scratch vs fine-tuned

| Mode | Pro | Con | Answers |
|---|---|---|---|
| **from-scratch** (default) | clean attribution; toggles act on a blank model | tiny models → weak absolute VeriBench numbers | tests the **mechanisms** |
| **fine-tuned** (one baseline) | realistic pass rates | toggles interact with frozen pretrained structure → muddier attribution | tests the **deployment** |

Default to **from-scratch** for the causal grid, and add **one fine-tuned
small-pretrained baseline** as a reality check. They mean different things: the
from-scratch grid is where the bridge plot is valid; the fine-tuned run tells you
whether the mechanism effects you found still matter on top of pretraining.
Set `mode: from_scratch | finetune` in `grid.yaml`.

---

## Two pre-registered expectations (check the harness against these)

1. **Rank collapse → "real but masked."** Its isolated strength (probe 03) will
   over-predict its realized effect; residuals neutralize most of the downstream
   damage. The size of that gap is the finding.
2. **The EBM-style case is likely conjunctive.** `objective=margin_energy` probably
   won't beat MLE on VeriBench pass rate *alone*, but may show a positive
   interaction with `head=mixture_of_softmax` or `compute=scratchpad`. If the
   interaction term is insignificant, that's a genuine strike against the integrated
   EBM motivation — and exactly the thing you want to learn.
