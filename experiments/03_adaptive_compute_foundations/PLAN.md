# Plan - Are EBMs Distinctively Adaptive-Compute?

**TLDR:** The experiment should test whether EBM inference is distinctively adaptive, or whether the same benefit is already obtained by autoregressive chain-of-thought, hidden thinking tokens, verifier-guided retry, best-of-N, and search.

## 1. Claim Under Audit

Current repo-level claim:

> Adaptive-compute inference (energy descent, iterative refinement) beats
> equivalent additional training compute on variable-difficulty tasks.

The weak version is almost certainly true but not distinctive:

> Some systems can spend more inference-time compute on hard instances.

Call the attractive term **adaptive**: a system uses more inference-time work
when the instance needs it. The foundations question is whether EBMs are
distinctively adaptive, or whether adaptivity is a broader property of any
iterative inference/search system.

CoT/thinking models already do that by emitting more reasoning tokens, running
internal reasoning steps, retrying, using tools, or searching. The experiment
must therefore test a stronger claim:

> EBM-style iterative inference allocates compute to hard instances in a way
> that improves a global compatibility objective more efficiently than
> compute-matched AR thinking/search baselines.

## 2. Foundations Distinctions

Separate these axes before running anything:

| Axis | EBM-style route | AR/thinking/search route |
|---|---|---|
| State being optimized | candidate output, latent representation, or relaxed variables | token prefix, scratchpad, sampled candidate set, tool/search state |
| Objective | lower energy / improve global compatibility | maximize next-token likelihood, verifier score, reward, or search heuristic |
| Compute knob | energy-descent steps, MCMC/Langevin steps, refinement iterations, candidates rescored | CoT length, hidden thinking budget, self-refinement rounds, best-of-N, verifier calls |
| Visibility | often latent or candidate-level | often visible tokens, but may be hidden in "thinking" models |
| Failure mode | hard inference, bad energy landscape, mixing, local minima | verbose reasoning, search blowup, verifier overfitting, local token objective |

The difference cannot be "variable compute" alone. It has to be something like:

- whole-object scoring rather than local next-token scoring;
- iterative improvement of a shared candidate rather than serially extending a prefix;
- lower marginal cost for extra inference steps at fixed output length;
- better compute allocation when difficulty is not known in advance;
- better use of a verifier because the score is global rather than local.

## 3. Operational Definitions

Define **adaptive** before comparing models:

- weak adaptivity: compute used increases with instance difficulty;
- useful adaptivity: extra compute improves hard-instance success without
  wasting much compute on easy instances;
- distinctive EBM adaptivity: the gain comes from optimizing/scoring global
  compatibility, not merely from generic search or longer reasoning traces.

Define instance difficulty using at least two independent signals:

1. oracle difficulty: known task size/depth in a synthetic problem;
2. realized difficulty: baseline success probability or verifier pass rate;
3. search difficulty: number of candidates/steps needed for a simple solver;
4. optional Lean difficulty: tactic-step depth or proof-search depth from
   VeriBench/VeriBench-FTP.

Define adaptive compute metrics:

- `compute_used`: FLOPs, wall-clock, tokens, verifier calls, or model forwards;
- `difficulty_bin`: easy/medium/hard by the signals above;
- `elasticity`: slope of `compute_used` versus `difficulty_bin`;
- `efficiency`: improvement in success per extra unit compute;
- `overcompute`: compute wasted on easy instances;
- `hard-case lift`: success gain on hard bins at fixed average compute.

## 4. Reuse Existing Work

Start from `experiments/00_ar_pros_cons`:

- Probe 05 already tests fixed compute per token and says scratchpad/CoT moves
  the threshold outward.
- The integrated harness already has a `compute` axis (`fixed` vs
  `scratchpad`), which can become the AR-thinking baseline.
- `PROBE_SPECS.md` already names the null: if CoT gives no lift, fixed compute
  may not be the relevant bottleneck.

Reuse `experiments/00_start_off` and `experiments/01_toy_ebm_training`:

- finite candidate sets allow exact energy scoring and controlled best-of-N;
- MCMC chain length is an EBM-side adaptive compute knob;
- exact enumeration can separate "energy score is good" from "sampler found it."

Use `experiments/02_fisher_div_grad_cost` only for cost-accounting lessons if
continuous EBM variants enter the picture. It is not the core experiment.

## 5. Minimal Experimental Protocol

### Stage A - Toy Foundations

Build or reuse a synthetic task with known difficulty, e.g. parity,
graph connectivity, constrained sequence repair, or a small proof/search puzzle.

Compare at matched average inference compute:

1. one-shot AR / fixed-depth transformer baseline;
2. AR + visible scratchpad / CoT with variable max tokens;
3. AR + best-of-N or verifier-guided retry;
4. EBM exact scorer over finite candidates, where possible;
5. EBM iterative refinement / MCMC / local-search steps;
6. optional diffusion/iterative baseline as a non-EBM iterative comparator.

### Stage B - VeriBench-Style Pilot

If Stage A gives a nontrivial distinction, run a small VeriBench/Lean pilot:

- bin tasks by proof depth or baseline pass rate;
- compare fixed AR, AR+CoT/search, and EBM-style scoring/refinement;
- report compute, verifier calls, pass rate, and hard-bin lift.

Do not claim foundations success from Stage B alone; it is a domain sanity check.

## 6. Outcomes And Interpretations

| Outcome | Interpretation |
|---|---|
| AR+CoT/search matches EBM under compute matching | "adaptive compute" is not an EBM-specific advantage; the repo claim must be narrowed. |
| EBM helps only with exact enumeration but not approximate inference | the energy score may be useful, but the sampler/refiner is the bottleneck. |
| EBM gives higher hard-bin lift at equal average compute | evidence for a distinctive inference-as-optimization advantage. |
| All iterative methods help similarly | the important variable is iterative/search compute, not EBM specifically. |
| None help after strict verifier/compute matching | fixed-compute objection is weak in this setting. |

## 7. Deliverables

1. `FOUNDATIONS.md` - conceptual answer to "what is the difference from CoT?"
2. `PROTOCOL.md` - pre-registered compute-matching and difficulty-binning rules.
3. `QUESTIONS.md` - concise research questions around "adaptive EBMs" versus
   "adaptive inference" generally.
4. toy script or adapted Probe 05 run showing the comparison.
5. results table with success vs difficulty and compute used.
6. recommendation: keep, narrow, or drop the repo-level adaptive-compute claim.

## 8. Non-Goals

- Do not argue "EBM good, AR bad" in general.
- Do not compare unmatched compute budgets.
- Do not treat visible CoT length as the only AR adaptive-compute mechanism;
  hidden thinking, self-refinement, search, and verifier calls count too.
- Do not conflate a good global score with a tractable inference method.
