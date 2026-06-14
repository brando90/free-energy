# H00 Expt v1 - Boltzmann Exponential Hardware Benchmark

**TLDR:** This experiment benchmarks whether `exp(-E)`, `logsumexp`, and
softmax normalization are material runtime bottlenecks for EBM-style candidate
scoring. It runs with standard-library Python only, so the local Mac smoke/full
run is reproducible without a working NumPy/PyTorch install.

## Scientific Question

Is `exp(-E)` essential to the useful EBM story, or can ranking-oriented
alternatives preserve the important behavior while avoiding meaningful
normalization cost?

## Files

- `src/run_experiment.py` - benchmark driver.
- `results/primitive_timings.csv` - primitive timing sweep.
- `results/task_metrics.csv` - synthetic ranking/calibration metrics.
- `results/network_split.csv` - energy-network compute versus normalization
  split.
- `results/results.json` - full machine-readable output.
- `results/verdict.md` - short interpretation.

## Run

```bash
python3 experiments/hypothesis/00_h_boltzmann_surjection_object/expt_v1/src/run_experiment.py
```

## Status

| Step | Status | Notes |
| --- | --- | --- |
| Local run | Done | Standard-library Python on local Mac. |
| SNAP run | Not needed | Local run covers the requested 128 through 64K candidate sweep. |
| Verdict | Done | See `results/verdict.md`. |
