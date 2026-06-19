# Folder 00 - PROTOCOL (LOCKED before any expensive run)
**TLDR:** This protocol freezes the first experiment for testing whether smooth and gated activations justify their hardware cost. Any future expensive run should start from this file, then document deviations in a report rather than silently changing the question.

This file freezes the protocol before any LLM-budget-spending or GPU run begins.
It is written prior to seeing any results. Any deviation observed during
execution should be documented in a future `REPORT.md`, not by editing this
file after the fact.

## Scientific Question (locked)

If hardware efficiency is important, why do successful ML models still use
smooth or gated activations such as softplus, GELU, GEGLU, and SwiGLU?

Operationally: measure whether the extra activation cost is material relative
to matrix multiplication and memory movement, and whether any quality/training
gain justifies that cost in controlled tasks.

## Locked Metrics And Thresholds

- **Primary cost metrics:** activation latency, end-to-end feed-forward latency,
  throughput, peak memory, and numerical failures.
- **Primary quality metrics:** training loss, validation loss or accuracy, and
  calibration if classification is used.
- **Strengthens hardware objection:** smooth/gated activations cost at least
  10% end-to-end latency or memory at matched dimensions and fail to improve
  quality beyond noise.
- **Weakens hardware objection:** marginal end-to-end cost is under 5% after
  fusion/approximation, or quality gains are large enough to improve the
  speed-quality frontier.
- **Ambiguous:** cost and quality both move materially; report Pareto frontier
  rather than forcing a yes/no answer.

## Compared Activations

- ReLU
- Squared ReLU
- Softplus
- SiLU/Swish
- GELU exact
- GELU tanh approximation
- ReGLU
- GEGLU
- SwiGLU

## Scale Verification

Use matched tensor shapes and matched parameter budgets when comparing gated
and non-gated feed-forward blocks. Gated activations can change hidden width and
parameter count; do not compare an unadjusted wider gated block to a narrower
ReLU block without reporting the mismatch.

## Pre-Conditions For Run

- Python environment with PyTorch installed.
- CPU benchmark always required.
- GPU benchmark optional and must start with one visible GPU only.
- No external LLM API calls or paid API keys.
- Record exact framework version, device name, dtype, and command.

## Estimated Wall-Clock

- Local CPU smoke benchmark: under 10 minutes.
- Single-GPU smoke benchmark: under 10 minutes if a GPU is already available.
- Tiny controlled training task: 15-60 minutes depending on hardware.

## Pass/Fail And Abort Rules

- Smoke passes only if every activation returns finite outputs and benchmark
  results are saved as machine-readable files.
- Abort or downscale if the benchmark OOMs twice at the same shape.
- Do not launch a multi-GPU run for this question.
- If GPU utilization is low and wall-clock is dominated by Python overhead,
  report that and redesign the benchmark before scaling.

## Deliverables

- `expt_v1/README.md`
- `expt_v1/src/` benchmark and tiny-task code
- `expt_v1/results/` machine-readable results and plots/tables
- `expt_v1/results/verdict.md` with one paragraph answering the question
