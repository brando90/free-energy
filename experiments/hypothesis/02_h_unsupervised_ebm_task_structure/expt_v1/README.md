# H02 Expt v1 - Transformer Input-Coverage Benchmark

**TLDR:** This experiment tests whether attention-style full-context lookup
already solves the input-coverage failure mode that EBMs are sometimes credited
with solving. It compares recurrent last-window memory, exact attention lookup,
and an EBM-style candidate reranker on synthetic long-context key-value
retrieval.

## Scientific Question

Do EBMs solve a real long-context input-coverage problem that transformers do
not already solve, or is attention the main mechanism?

## Files

- `src/run_experiment.py` - standard-library benchmark driver.
- `results/length_metrics.csv` - metrics by context length and model.
- `results/aggregate_metrics.csv` - aggregate metrics by model.
- `results/results.json` - full machine-readable output.
- `results/verdict.md` - short interpretation.

## Run

```bash
python3 experiments/hypothesis/02_h_unsupervised_ebm_task_structure/expt_v1/src/run_experiment.py
```

## Status

| Step | Status | Notes |
| --- | --- | --- |
| Local run | Done | Synthetic long-context retrieval, 5 seeds. |
| SNAP run | Not needed | The benchmark is algorithmic and CPU-bound. |
| Verdict | Done | See `results/verdict.md`. |
