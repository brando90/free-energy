# H01 Expt v1 - Scalar Energy Bottleneck Benchmark

**TLDR:** This experiment tests whether a compressed scalar sequence energy is
too lossy compared with structured token-position energy contributions. It uses
a synthetic sequence task with known responsible positions, then compares
classification, calibration, localization, and intervention effects.

## Scientific Question

Is one scalar energy output a bad bottleneck for sequence EBMs, and do
structured/decomposed energy outputs fix anything measurable?

## Files

- `src/run_experiment.py` - standard-library benchmark driver.
- `results/seed_metrics.csv` - per-seed metrics.
- `results/aggregate_metrics.csv` - mean/std summary by model.
- `results/results.json` - full machine-readable output.
- `results/verdict.md` - short interpretation.

## Run

```bash
python3 experiments/hypothesis/01_h_interpretable_energy_tags/expt_v1/src/run_experiment.py
```

## Status

| Step | Status | Notes |
| --- | --- | --- |
| Local run | Done | Three seeds with pure Python SGD. |
| SNAP run | Not needed | The benchmark is intentionally small and synthetic. |
| Verdict | Done | See `results/verdict.md`. |
