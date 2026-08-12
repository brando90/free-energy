# H00 Expt v1 - `exp(-E)` Hardware Primitive Attempt

**TLDR:** Draft PR scaffold for the first real benchmark of whether the
Boltzmann exponential and partition-function path are a meaningful EBM
hardware/inference bottleneck. The coding agent should run
`../coding_agent_prompt.md`, implement the benchmark, and replace this scaffold
with results.

## Assignment

- GitHub issue: https://github.com/brando90/free-energy/issues/46
- Coding-agent prompt: `../coding_agent_prompt.md`
- Suggested assignees: @brando90, @eobbad
- Collaborators to loop in: @srivatsavad, @kirillacharya
- Priority: high, because the hypothesis is marked ~9.5/10 importance.

## First Agent Action

Read `../coding_agent_prompt.md` and create the runnable benchmark under this
`expt_v1/` folder. Do not stop at literature review or discussion. The PR is
done only when it includes code, a local smoke test, results artifacts, and a
short verdict.

## Expected Implementation

- `src/bench_normalizers.py` or equivalent benchmark entrypoint.
- `results/` containing machine-readable benchmark data.
- `results/verdict.md` with a one-paragraph answer.
- Optional SNAP run logs if local hardware is insufficient.

## SNAP Note

Escalate to SNAP only after a local smoke test. If using SNAP, run one GPU
first, set `CUDA_VISIBLE_DEVICES`, record `nvidia-smi`, and save the exact
launch command and logs in `results/`.
