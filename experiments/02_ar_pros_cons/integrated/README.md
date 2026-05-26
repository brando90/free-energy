# Integrated harness

The isolated probes test mechanisms one at a time. The integrated harness is the
next stage: train actual autoregressive models on the VeriBench split and log the
same probe metrics online.

The integrated question is different:

```text
When all mechanisms coexist in one trained system, which ones actually explain
downstream verifier pass rate?
```

## Planned grid

The minimum useful factorial grid toggles:

| Axis | Baseline | Alternative | Mechanism |
|---|---|---|---|
| decoding/inference | blind AR | verifier-guided retry/search | LeCun unrecoverability |
| objective | MLE | margin/energy-style objective | mode-covering vs compatibility |
| compute | fixed token budget | scratchpad / iterative refinement | fixed compute per token |

For each cell, record:

- pass@1 and pass@k on the VeriBench test split;
- success probability vs proof/program length;
- geometric vs recoverable-Markov fit for probe 06;
- proof-depth curves for probe 05;
- any verifier-rejected intermediate states that later recover.

## Not implemented yet

This folder currently documents the target harness. The concrete first merged
pieces are:

- `toy/toy_error_process.py` for the synthetic positive control;
- `data/setup.py` for deterministic VeriBench train/val/test manifests;
- `probes/probe_06_error_compounding.py` for the model-comparison smoke control.

The next PR should add a small `integrated/run_grid.py` that consumes
`data/splits/*.jsonl` and emits a first pass@k/length dashboard.
