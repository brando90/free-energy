# Hypothesis Experiment Results Summary

**TLDR:** All three `expt_v1` experiments were implemented and run locally with
standard-library Python because the local compiled NumPy/PyTorch environment is
broken. SNAP was not needed: the experiments are controlled synthetic
benchmarks and completed to verdict on the Mac.

## Run Environment

- Host: `DNa81a129.SUNet`
- Platform: `macOS-26.5-arm64-arm-64bit`
- Python: `3.11.6`
- Branch: `codex/execute-hypothesis-experiments`
- SNAP: not used; local CPU runs completed all designed sweeps.

## H00 - `exp(-E)` Hardware Primitive

- Folder: `00_h_boltzmann_surjection_object/expt_v1/`
- Script: `src/run_experiment.py`
- Result files: `results/primitive_timings.csv`, `results/task_metrics.csv`,
  `results/network_split.csv`, `results/results.json`, `results/verdict.md`
- Verdict: partially supported. Normalization is measurable at large candidate
  counts, but in this pure-Python benchmark `logsumexp` was only about `2.7%`
  of the synthetic energy-network plus normalization time at 64K candidates.
- Key numbers: batch-8, 64K candidates took `36.980 ms` for raw argmin,
  `112.473 ms` for `logsumexp`, and `166.394 ms` for full softmax.

## H01 - Scalar Energy Bottleneck

- Folder: `01_h_interpretable_energy_tags/expt_v1/`
- Script: `src/run_experiment.py`
- Result files: `results/seed_metrics.csv`, `results/aggregate_metrics.csv`,
  `results/results.json`, `results/verdict.md`
- Verdict: supports a narrow version of the hypothesis. A heavily compressed
  scalar energy underperformed, while a structured token-position energy
  recovered both task accuracy and localization.
- Key numbers: scalar compressed EBM reached `0.583` accuracy and `0.000`
  localization F1; decomposed EBM reached `1.000` accuracy and `0.867`
  localization F1 across three seeds.

## H02 - Transformer Input Coverage

- Folder: `02_h_unsupervised_ebm_task_structure/expt_v1/`
- Script: `src/run_experiment.py`
- Result files: `results/length_metrics.csv`, `results/aggregate_metrics.csv`,
  `results/results.json`, `results/verdict.md`
- Verdict: supports the hypothesis. Full-context attention and a full-context
  EBM reranker both solve the synthetic input-coverage task; recurrence and
  recent-only energy scoring fail from forgetting.
- Key numbers: transformer-style attention and full-context EBM reranker both
  reached `1.000` mean accuracy; last-16 recurrence reached `0.353`; recent-only
  EBM reranking reached `0.076`.

## Caveats

- These are controlled synthetic experiments, not final paper-grade evidence.
- H00 is pure-Python timing, so absolute wall-clock is not hardware-kernel
  representative; the relative pass-count behavior is still informative.
- H01 intentionally contrasts compressed scalar features against structured
  position-token features; it does not prove all scalar-valued neural energies
  are intrinsically too lossy.
- H02 is algorithmic rather than trained-neural; it isolates input access and
  forgetting, but does not measure transformer training dynamics.
