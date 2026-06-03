# Bridge analysis — latent strength vs realized effect

This is the headline deliverable that ties the two levels together. The isolated
probes tell you how strong each mechanism is *in principle*; the integrated grid
tells you how much it actually *moves the downstream metric* once everything trains
together. The bridge plot puts them on the same axes so you can see, per mechanism,
whether theory predicts practice.

---

## The plot

A single scatter. Each mechanism is one labeled point.

- **x-axis — isolated latent strength.** A normalized [0, 1] score from the
  standalone probe (e.g. for the bottleneck: how far the data rank exceeds the
  head's `d`, normalized; for rank collapse: collapse rate in the pure-attention
  control).
- **y-axis — realized downstream effect.** The ablation-grid causal delta on
  VeriBench pass@1 from toggling that mechanism, with everything else fixed
  (bootstrap CI as error bars).

Both axes carry uncertainty; show CIs on both where available.

---

## Reading the three regions

| Region | Pattern | Interpretation | Example we expect |
|---|---|---|---|
| **diagonal** | high x, high y | **load-bearing** — strength predicts effect | softmax bottleneck (#1), if data rank > d |
| **lower-right** | high x, low y | **masked in practice** — name the masker | rank collapse (#3), masked by residuals |
| **upper-left** | low x, high y | **interaction-driven** — a pair does the work | energy objective (#2), only with #1 or #5 |

Each point gets a one-line causal note in the output (`bridge_notes.md`) explaining
which region it fell in and why.

---

## How each axis is computed

### Isolated strength (x)
From `probes/out/probe_NN/stats.json`, the `effect_size` field, normalized per
mechanism to [0, 1] using the probe's own scale (documented in `PROBE_SPECS.md`).
For mechanisms with a positive control, strength is measured *in the control*, where
the effect is guaranteed to be visible if real.

### Realized effect (y)
From the ablation grid. For a toggle `T` with control level `c` and treatment `t`:

```
realized_effect(T) = E[ pass@1 | T = c ] − E[ pass@1 | T = t ]
```

averaged over all other toggle settings (marginal effect), with a bootstrap CI over
seeds and cells. For mechanisms tested by *removing* a component (residual, mlp),
the sign convention is "effect of having the mechanism present."

---

## Why a mechanism can be strong but not matter (and vice versa)

- **Strong but masked:** rank collapse is doubly-exponential in *pure* attention,
  but residual connections inject full-rank signal at every layer, so the trained
  model keeps usable rank. The mechanism is real; the architecture already
  defuses it. Reporting this prevents over-claiming from the isolated probe.
- **Weak but impactful via interaction:** the energy/margin objective may do little
  on its own (MLE is a strong baseline) but unlock value *only* when the head is
  higher-rank (#1) or compute is adaptive (#5). The marginal effect looks small; the
  conditional effect is large. `interactions.py` is what catches this, and the
  bridge plot flags the point as interaction-driven.

---

## Companion: interaction matrix

Alongside the scatter, emit a small heatmap of the fitted interaction terms for the
pre-registered pairs (`objective×head`, `attn×residual`, `objective×compute`), each
cell = interaction coefficient with a significance marker (CI excludes zero). This
is where the **conjunctive EBM hypothesis** is accepted or rejected:

- significant positive `objective×head` and/or `objective×compute` → the EBM-style
  case is **conjunctive** (drop-softmax + drop-MLE together), not additive.
- all interaction terms insignificant → no synergy; the EBM motivation must stand on
  the marginal effects alone, which is a genuine strike if those are also small.

---

## Output files

- `integrated/out/bridge_scatter.png` — the headline figure
- `integrated/out/interaction_heatmap.png` — interaction coefficients
- `integrated/out/bridge_notes.md` — per-mechanism region + causal note
- `integrated/out/bridge_table.json` — `{mechanism: {x, x_ci, y, y_ci, region, note}}`

---

## Verdict vocabulary (feeds the dashboard)

Each mechanism resolves to one of:

- **CONFIRMED** — strong in isolation *and* realized effect CI excludes zero.
- **MASKED** — strong in isolation, realized effect CI includes zero; masker named.
- **INTERACTION-DRIVEN** — weak marginal effect, significant interaction term.
- **PARTIAL** — mixed across matchings (compute- vs param-matched disagree) or across
  toy vs real.
- **NOT-SUPPORTED in this setup** — neither isolated nor realized effect materializes
  (with caveats: scale, sample size, what wasn't controlled).

The dashboard (`dashboard.py`) renders one row per mechanism with verdict, realized
effect ± CI, and the caveats column.
