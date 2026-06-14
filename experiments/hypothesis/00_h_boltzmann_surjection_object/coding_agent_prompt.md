# Coding Agent Prompt - H00: Is `exp(-E)` A Bad Hardware Primitive?

**TLDR:** Build a real benchmark deciding whether the Boltzmann exponential and
partition-function path are a meaningful hardware/inference bottleneck for EBMs.
Start with a local smoke test, then use a single SNAP GPU only if the local
benchmark is too small to expose the scaling behavior.

## Context

- GitHub issue: https://github.com/brando90/free-energy/issues/46
- Hypothesis notes: `pre_prompt.md`
- Source image: `assets/source_photo.jpg`

## Compact Hypothesis

- **Goal:** Test whether the Boltzmann exponential `exp(-E)` is a bad
  hardware/inference primitive, and whether another monotone or normalized score
  can keep the useful ordering without paying exponential/partition costs.
- **Confidence:** ~75%.
- **Importance:** Very high, roughly 9.5/10.

## Your Task

Create `expt_v1/` under this folder and implement a reproducible benchmark.
Do not only write a literature note. The deliverable must include runnable code,
plots or tables, and a short verdict.

## Required Experiment

1. Benchmark core primitives on CPU and, if available, one GPU:
   - raw energy ranking with no normalization
   - `exp(-E)`
   - `logsumexp`
   - softmax probabilities
   - at least one alternative bounded or monotone scoring path, such as
     temperature-scaled softmax, sparsemax/entmax if easy, or rank-only
     energy selection
2. Sweep tensor shapes that mimic EBM candidate scoring:
   - batch sizes: small, medium, large
   - candidate counts/classes: at least 128, 1K, 8K, and 64K if memory permits
   - dtypes: fp32, bf16/fp16 if the hardware supports them
3. Add one task-level toy benchmark:
   - generate synthetic candidate energies and labels
   - compare top-1/top-k ranking, calibration, and wall-clock cost for each
     scoring path
   - separate ranking quality from calibrated probability quality
4. Report:
   - latency, throughput, peak memory, numerical failures, and calibration
   - the fraction of time spent in normalization versus energy-network compute
     for at least one small learned energy model

## SNAP Cluster Escalation

Run locally first. Escalate to SNAP only if local CPU/GPU smoke tests cannot
answer the scaling question.

If using SNAP:

1. Pick exactly one GPU for the main run.
2. Set `CUDA_VISIBLE_DEVICES=<chosen_id>` before running Python.
3. Record `hostname`, `nvidia-smi`, GPU model, CUDA version, PyTorch/JAX
   version, and wall-clock start/end.
4. Save logs under `expt_v1/results/`.
5. Do not use raw LLM API calls or API keys; this experiment should be ordinary
   Python benchmarking.

## Deliverables

- `expt_v1/README.md` with setup, commands, and status.
- `expt_v1/src/` with benchmark code.
- `expt_v1/results/` with machine-readable results and plots/tables.
- `expt_v1/results/verdict.md` answering the hypothesis in one paragraph.

## Verdict Criteria

The hypothesis is strengthened if `exp`/`logsumexp`/normalization is a material
share of runtime or memory at realistic candidate counts and alternatives retain
ranking quality at lower cost. It is weakened if normalization overhead is
small relative to the learned energy network or if alternatives destroy the
calibration/search behavior EBMs need.
