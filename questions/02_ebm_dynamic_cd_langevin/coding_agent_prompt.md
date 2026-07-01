# Coding Agent Prompt - Q02 Dynamic Weighted CD for Continuous EBMs

You are an AI researcher and ML engineer specializing in energy-based models, MCMC, Langevin dynamics, and PyTorch.

## Context

- Question packet: `questions/02_ebm_dynamic_cd_langevin/`
- Source photos: `assets/photo_1.jpg` through `assets/photo_4.jpg`
- Research brief: `research_brief.md`
- Locked protocol: `PROTOCOL.md`
- Prototype: `expt_v1/src/dynamic_weighted_cd.py`

## Task

Implement and extend the toy EBM experiment comparing standard CD-1 with Dynamic Weighted CD on a 2D continuous distribution. Keep the theory honest: do not claim the partition function disappears; claim only that explicit evaluation of `Z(theta)` can be avoided by alternative objectives or approximate negative phases.

## Required Deliverables

1. Maintain a runnable PyTorch script under `expt_v1/src/`.
2. Train both:
   - CD-1 with one Langevin step.
   - Dynamic Weighted CD with a weighted trajectory of Langevin states.
3. Save:
   - `report.json`,
   - `energy_surfaces.png`,
   - `samples.png`,
   - `verdict.md`.
4. Add or update a short explanation of whether dynamic weighting improved:
   - mode coverage,
   - sample distance to data modes,
   - energy-surface quality,
   - compute/runtime.

## Key Theory Constraints

- `Z(theta)` is needed to define normalized probabilities, but MLE gradients can avoid explicit `Z(theta)` evaluation by replacing it with a model expectation.
- Langevin dynamics is still MCMC.
- `p0(x)` is fixed with respect to `theta` unless parameterized; it may still affect the score through `grad_x log p0(x)`.
- Early and late trajectory states are negative-phase samples under the likelihood-gradient view.
- If adaptive weights depend on model outputs, note whether their gradients are included or intentionally stopped.

## Suggested Extensions

- Start by reading `ROADMAP.md`; it incorporates the useful parts of the
  Claude follow-up while correcting overclaims about tractable density,
  sampling, and `Z`.
- Add PCD with a replay buffer.
- Add score matching or denoising score matching baseline.
- Add a no-data-start sampler initialized from Gaussian noise.
- Sweep `T`, weight schedule, Langevin step size, and replay-buffer ratio.
- Replace eight-Gaussians with Swiss roll or a molecular toy potential.
