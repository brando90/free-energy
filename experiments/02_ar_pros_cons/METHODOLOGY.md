# Methodology — how not to fool yourself

This suite is adversarial against its own conclusions. Every probe states a
**prediction**, a **null**, and a **measurement** before any plotting, and every
claim is tested at the correct layer with a positive control. This document is the
shared protocol all probes and the integrated harness must follow.

---

## 1. Layer separation (non-negotiable)

A claim tested at the wrong layer produces a confident, wrong answer. The three
layers and what each requires:

- **architecture** — randomly initialized model, **no training, no data**. The
  effect is a property of the function class. Probes here are near-proofs: if the
  positive control fails, the bug is in the code, not the theory.
- **objective** — a controlled training run isolating the loss function, with the
  architecture held fixed.
- **trained behavior** — a trained model plus the Lean verifier for ground-truth
  pass/fail. No soft proxy metrics where a hard verifier is available.
- **external** — a resource/scaling curve; no model claim at all.

Each figure must print its layer in the title. The integrated harness logs the
layer alongside every metric row.

---

## 2. Positive controls (every probe has one)

For each probe, implement a setting where the effect **must** appear if the theory
is true, with known ground truth:

- bottleneck → a synthetic target log-prob matrix of *known* rank `r > d`
- rank collapse → pure attention, no residual/MLP, random init
- fixed-compute → parity / graph-connectivity at growing `n`
- mode-covering → an explicit bimodal target

If the positive control does not reproduce the effect, **halt and fix the probe**.
Do not proceed to the real test. A green control is the license to trust the real
measurement.

---

## 3. Ablation, not cross-model comparison

Claims of the form "component X is removable / X causes Y" are tested by **toggling
X with everything else held fixed**, never by comparing to a different published
model (which confounds X with scale, data, and a hundred other choices).

- "softmax is removable" → same architecture, swap only the attention nonlinearity.
- "the MLE objective causes mode-covering" → same architecture, swap only the loss.

The causal effect of a toggle is the *difference* it makes on a downstream metric
with all else equal.

---

## 4. Statistical protocol

- **Seeds:** seed everything (Python, NumPy, Torch, CUDA). Log the seed in every
  output JSON. Every cell/probe runs on **≥ 3 seeds**.
- **Uncertainty:** report **bootstrap confidence intervals** on every effect size,
  not point estimates. Plots show CIs; the dashboard shows effect ± CI.
- **Pre-registration:** the predicted *sign* of every toggle's effect is written in
  `integrated/grid.yaml` before running. Surprises (wrong sign, or zero where an
  effect was predicted) are flagged in `FINDINGS.md` — surprises are results, not
  failures.
- **Effect sizes over p-values:** report the magnitude and CI of the realized delta;
  significance is a side note.

---

## 5. Compute- and param-matched comparisons

Toggles change parameter counts (a mixture-of-softmaxes head has more params; an
MLP-off model has fewer). Any comparison must be reported **both ways**:

- **compute-matched** — equal training FLOPs / token budget.
- **param-matched** — equal parameter count.

Report both so neither side of an argument can hide behind a FLOPs or parameter
advantage. If a toggle only helps under one matching, say so explicitly.

---

## 6. Online instrumentation (integrated harness)

In the integrated runs, every probe is logged as an **online metric** every
`eval_every` steps against the same checkpoints, producing trajectories, not
endpoints. This lets us see *when* in training each effect emerges. Order of
emergence (e.g. does rank collapse precede or follow the VeriBench plateau?) is
itself evidence of causal direction.

All metrics land in one tidy dataframe:

```
run_id, step, <toggle levels...>, layer, metric, value, ci_low, ci_high, seed
```

One schema for everything makes the bridge and interaction analyses trivial joins.

---

## 7. The bridge: latent strength vs realized effect

The headline analysis. For each mechanism:

- **x-axis** = isolated latent strength (from the standalone probe)
- **y-axis** = realized downstream effect (ablation-grid causal delta on VeriBench
  pass rate)

Read the scatter as:

- **on the diagonal** → load-bearing (strength predicts effect)
- **high x, low y** → masked in practice; name the masker
- **low x, high y** → an interaction is responsible; identify the pair

A mechanism is only "confirmed for deployment" if its realized effect is nonzero
with a CI that excludes zero — independent of how strong it looks in isolation.

---

## 8. Honesty checklist (per run)

- [ ] layer tagged on every figure
- [ ] positive control green before real test
- [ ] ≥ 3 seeds, bootstrap CIs on all deltas
- [ ] predicted sign pre-registered; surprises flagged
- [ ] both compute-matched and param-matched numbers reported
- [ ] smoke config reproduced end-to-end before GPU spend
- [ ] verifier (Lean) used for pass/fail, not a proxy, wherever available
