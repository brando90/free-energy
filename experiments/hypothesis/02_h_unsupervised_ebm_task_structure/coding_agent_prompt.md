# Coding Agent Prompt - H02: Do Transformers Already Solve Input Coverage?

**TLDR:** Build a controlled long-context experiment testing whether
transformers/attention already solve the alleged full-input or unbounded-length
conditioning advantage sometimes attributed to EBMs. The key comparison is
attention versus recurrence/forgetting, with EBM-style scoring included only if
it tests a real alternative.

## Context

- GitHub issue: https://github.com/brando90/free-energy/issues/48
- Hypothesis notes: `pre_prompt.md`
- Source image: `assets/source_photo.jpg`

## Compact Hypothesis

- **Goal:** Test whether transformers/attention already solve the alleged
  unbounded-length/full-input conditioning issue, leaving little special
  advantage for EBMs on this axis.
- **Confidence:** High.
- **Importance:** Very high, 9.8/10.

## Your Task

Create `expt_v1/` under this folder and implement a controlled experiment.
Do not argue abstractly that attention sees all tokens. Measure the failure
mode: length generalization, global dependency capture, recurrence forgetting,
calibration, and inference compute.

## Required Experiment

1. Build one or more synthetic long-context tasks with known dependency
   structure:
   - key-value retrieval from a long sequence
   - copy/select the token paired with a query key
   - parity or equality over far-apart spans
   - optional proof/trace validity if simple enough
2. Compare:
   - LSTM/GRU recurrent baseline
   - transformer encoder or decoder baseline
   - EBM-style reranker or energy scorer over candidate answers, if cheap
3. Sweep train and test lengths:
   - train on shorter contexts
   - evaluate on in-distribution and longer out-of-distribution contexts
4. Control for:
   - parameter count
   - training examples
   - wall-clock or FLOP proxy
   - inference-time candidate count for the EBM/reranker path
5. Report:
   - accuracy by length
   - calibration or confidence by length
   - runtime and memory by length
   - whether failures are from forgetting, attention scaling, search, or
     insufficient training

## SNAP Cluster Escalation

Run a local smoke test first. Escalate to SNAP for length/model sweeps if the
smoke test is sound.

If using SNAP:

1. Use one GPU for initial full runs.
2. Set `CUDA_VISIBLE_DEVICES=<chosen_id>`.
3. Save logs, `nvidia-smi`, hardware metadata, and exact commands under
   `expt_v1/results/`.
4. Do not use raw LLM API calls.

## Deliverables

- `expt_v1/README.md` with task definitions, commands, and status.
- `expt_v1/src/` with data generation, models, training, and evaluation.
- `expt_v1/results/` with metrics, plots/tables, and logs.
- `expt_v1/results/verdict.md` deciding whether attention already explains the
  input-coverage advantage.

## Verdict Criteria

The hypothesis is strengthened if transformers match or beat the EBM/reranker
path under matched compute while recurrent baselines fail from forgetting. It is
weakened if an EBM-style scorer handles global dependencies or length shifts
that the transformer baseline misses at comparable compute.
