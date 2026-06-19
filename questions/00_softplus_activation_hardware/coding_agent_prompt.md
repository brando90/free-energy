# Coding Agent Prompt - Q00 Smooth Activations vs Hardware Cost
**TLDR:** Build `expt_v1/` to benchmark whether softplus, GELU, GEGLU, and SwiGLU justify their hardware cost relative to simpler activations. Start with local CPU smoke tests, use at most one GPU if available, save machine-readable results, and end with a short verdict tied to `PROTOCOL.md`.

## Context

- Question packet: `questions/00_softplus_activation_hardware/`
- Source photos: `assets/photo_1.jpg`, `assets/photo_2.jpg`
- Transcription: `transcription.md`
- Locked protocol: `PROTOCOL.md`
- GitHub issue: https://github.com/brando90/free-energy/issues/55

## Your Task

Create `expt_v1/` under this folder and implement a reproducible benchmark.
Do not only write a literature note. The deliverable must include runnable code,
tables or plots, saved raw results, and a verdict.

## Required Experiment

1. Benchmark activation primitives:
   - ReLU
   - squared ReLU
   - softplus
   - SiLU/Swish
   - GELU exact
   - GELU tanh approximation
2. Benchmark feed-forward blocks:
   - non-gated MLP with ReLU, squared ReLU, GELU, and SiLU
   - GLU-family blocks: ReGLU, GEGLU, and SwiGLU
   - matched parameter-count setting and a clearly labeled unmatched setting
3. Sweep practical tensor shapes:
   - batch/sequence combinations that mimic transformer MLP calls
   - hidden sizes at least 512, 2048, and 8192 if memory permits
   - dtypes `float32` and `bfloat16` or `float16` when supported
4. Add one tiny controlled training task:
   - synthetic classification or a small public dataset available without
     credentials
   - identical optimizer, parameter budget, and training steps across
     activations
   - report quality and speed together
5. Report:
   - latency, throughput, peak memory, finite-output checks, and framework
     version
   - whether activation cost is visible once embedded in a feed-forward block
   - whether any quality gain shifts the speed-quality frontier

## Hardware Discipline

Run locally first. If using a GPU, use exactly one visible GPU and record the
device name. Do not use raw LLM API calls or API keys.

## Deliverables

- `expt_v1/README.md` with setup, commands, and status.
- `expt_v1/src/` with benchmark code.
- `expt_v1/results/` with raw results and plots/tables.
- `expt_v1/results/verdict.md` answering the question in one paragraph.

## Verdict Criteria

The hardware objection is strengthened if smooth/gated activations materially
increase end-to-end latency or memory without a controlled quality gain. It is
weakened if fused kernels make their marginal cost small or if quality gains
move the speed-quality frontier enough to justify the cost.
