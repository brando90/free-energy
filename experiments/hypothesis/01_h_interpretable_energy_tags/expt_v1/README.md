# H01 Expt v1 - Scalar Energy Bottleneck Attempt

**TLDR:** Draft PR scaffold for testing whether one scalar sequence energy is
too lossy compared with decomposed token/tag energies. The coding agent should
run `../coding_agent_prompt.md`, implement the synthetic sequence benchmark,
and replace this scaffold with real results.

## Assignment

- GitHub issue: https://github.com/brando90/free-energy/issues/47
- Coding-agent prompt: `../coding_agent_prompt.md`
- Suggested assignee: @srivatsavad
- Collaborators to loop in: @eobbad, @kirillacharya
- Priority: medium, because the hypothesis is marked 5/10 importance and ~50%
  confidence.

## First Agent Action

Read `../coding_agent_prompt.md` and implement the controlled synthetic
sequence experiment under this `expt_v1/` folder. The PR should compare scalar
energy, decomposed energy, and an autoregressive or classifier baseline.

## Expected Implementation

- `src/` with dataset generation, model definitions, training, and evaluation.
- `results/` with JSON/CSV metrics, plots/tables, and logs.
- `results/verdict.md` answering whether scalar energy is the actual bottleneck.
- At least 3 seeds after the local smoke test passes.

## SNAP Note

Prefer local smoke tests first. Use SNAP only for the seed/model sweep if local
hardware is too slow. If using SNAP, set `CUDA_VISIBLE_DEVICES`, record
`nvidia-smi`, and save exact launch commands and logs in `results/`.
