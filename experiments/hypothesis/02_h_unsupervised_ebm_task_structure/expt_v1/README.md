# H02 Expt v1 - Transformer Input-Coverage Attempt

**TLDR:** Draft PR scaffold for testing whether transformers/attention already
solve the alleged full-input or long-context conditioning advantage sometimes
attributed to EBMs. The coding agent should run `../coding_agent_prompt.md`,
implement the controlled long-context benchmark, and replace this scaffold with
real results.

## Assignment

- GitHub issue: https://github.com/brando90/free-energy/issues/48
- Coding-agent prompt: `../coding_agent_prompt.md`
- Suggested assignees: @brando90, @kirillacharya
- Collaborators to loop in: @eobbad, @srivatsavad
- Priority: highest, because the hypothesis is marked 9.8/10 importance and
  high confidence.

## First Agent Action

Read `../coding_agent_prompt.md` and implement the long-context benchmark under
this `expt_v1/` folder. The PR should compare recurrent, transformer, and
EBM-style scorer/reranker paths under matched data, capacity, and inference
compute.

## Expected Implementation

- `src/` with synthetic long-context data, model definitions, training, and
  evaluation.
- `results/` with accuracy-by-length, calibration, runtime, memory, and logs.
- `results/verdict.md` deciding whether attention already explains the
  input-coverage advantage.
- Optional SNAP logs for length/model sweeps after local smoke tests pass.

## SNAP Note

Run a local smoke test first. Escalate to SNAP only for length/model sweeps.
Use one GPU for the first full run, set `CUDA_VISIBLE_DEVICES`, record
`nvidia-smi`, and save exact commands and logs in `results/`.
