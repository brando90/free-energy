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
- Verdict: consistent with the hypothesis, largely by construction. Full-context
  lookup and a full-context EBM reranker both solve the synthetic
  input-coverage task; recurrence and recent-only energy scoring fail from
  forgetting.
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

## Review Notes

Added during second-model review; result files are unchanged.

- H00: softmax probabilities here are normalized, not calibrated (task ECE
  `0.229` at T=1). The `2.7%` normalization fraction is close to `1/(dims + 1)`
  for the 32-dim toy network, so it reflects the chosen network cost. The
  prompt's dtype sweep, peak-memory and numerical-failure measurements were not
  done. The 64K-candidate rows at batch 8 and 16 are single repetitions.
- H01: `decomposed_ebm` and `full_position_classifier` are the same logistic
  model on the same features (identical rows), so there is no separate
  classifier baseline yet. Both models emit one scalar logit; the contrast is
  between featurizations. The scalar model's `0.000` localization F1 is set by
  its uniform attribution.
- H02: the "transformer" baseline is an exact key scan and the full-context EBM
  reranker calls it, so both `1.000` accuracies hold by construction. The
  full-context reranker is poorly calibrated (ECE `0.919`).
