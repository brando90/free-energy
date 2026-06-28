# Roadmap - From Toy CD to the Z/MCMC Question

Claude's transcription is useful as a broader agenda, but several claims need
tightening before they become experiments.

## Keep

- Treat the umbrella question as: where does `Z` cancel, where does it merely
  relocate, and where is it genuinely load-bearing?
- Split theory from experiments:
  - theory: argmin/ranking invariance, score invariance, sampling-vs-density
    asymmetry;
  - experiments: parallel chains, step-weighted CD, AR-vs-EBM relocation costs,
    and Lean/VeriBench proxy tasks.
- Make each item falsifiable, matched-compute, and tied to an artifact.

## Correct

- Pointwise tractable density evaluation does not automatically imply an
  efficient exact sampler. The safer statement is: many directed normalized
  factorizations are designed to provide ancestral sampling, while generic
  unnormalized EBMs do not.
- Tractable approximate sampling does not imply tractable density or tractable
  partition-function evaluation.
- Attention softmax, output softmax, and EBM partition functions are all
  normalizers, but they are not literally the same `Z`; each normalizes over a
  different axis and serves a different role.
- In likelihood-style EBM training, generated chain states remain negative-phase
  samples. Later states can receive higher negative-phase weights, but calling
  them positive examples changes the objective.

## Recommended Next Work

1. **Q02 v2 empirical sweep:** run multi-seed CD-1 vs dynamic weighted CD on
   SNAP and aggregate metrics. Extend to `T in {1, 2, 4, 8, 16}` and ramps
   `{0, 1, 2, 4}`.
2. **Q02 v3 ablation:** separate CD failure causes by holding data fixed and
   varying chain length/noise, then holding chain length fixed and varying
   dataset size.
3. **Theory packet A1:** prove/document that adding an `x`-dependent constant
   `log Z(x)` to all candidate energies leaves `argmin_y E(x, y)` invariant,
   and that `Z(theta)` drops from `grad_x log p_theta(x)`.
4. **Parallel MCMC packet:** benchmark one long chain vs many short chains vs
   parallel tempering at matched energy evaluations.
5. **Lean/VeriBench packet:** compare AR scoring, energy ranking, and verifier
   loop inference on a small frozen proxy before touching full VeriBench.

## Current Status

`expt_v1/` implements item 1 for one dynamic weighting schedule. The initial
local CPU run is positive on one seed, but this should be treated as an
existence proof only until the SNAP multi-seed sweep is aggregated.
