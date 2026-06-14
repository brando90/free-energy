# Pre-Prompt - Hypothesis 01: Single Scalar Energy Bottleneck

**TLDR:** Test whether reducing a rich sequence or representation to one scalar
real-valued energy is too lossy for EBMs, unless the energy is structured or
decomposed into token, hidden-state, concept, or tag components.

## Compact Hypothesis Summary

- **Goal:** Test whether reducing a rich sequence or representation to one
  scalar real-valued energy is too lossy for EBMs, unless the energy is
  structured or decomposed.
- **Confidence:** ~50%.
- **Importance:** Medium, 5/10.
- **Key uncertainty:** The handwritten note says "I like why but I'm 50/50 if
  this is right"; treat this as a plausible but not central hypothesis.

## Source

- Saved source photo: `assets/source_photo.jpg`
- Saved summary update photo: `assets/summary_update_photo.jpg`
- Original upload path: `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/0d7a30d9-1e1b-47e1-9e9f-644e9753f2ff/2-Photo-2.jpg`
- Summary update upload path: `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/ea880cd6-79b2-4fda-ad8e-23ca65ea9c0a/2-Photo-2.jpg`
- Visible formulas include a Boltzmann conditional form and a sequence-energy
  expression resembling `E_theta(x_{1:i})`.
- Transcript confidence: medium-low. Re-read the image before finalizing.

## Rough User Seed

The summary update states:

> Hypothesis: one single real-valued output is a bad idea; maybe it guarantees
> failure of EBMs. Importance 5/10, confidence roughly 50/50.

The notebook seems to ask:

> To me, mapping energy to a sequence or real tag component seems like a real
> tag/component question. Can we map the energy to `x_i`, `h_i`, concepts, or
> tags? How could we study it?

## Prompt To Future Agent

Do not assume interpretability follows automatically from using an energy.
The task is to test whether the one-scalar-energy bottleneck is actually the
problem, and whether decomposition fixes it or merely creates post-hoc stories.

Plan:

1. Re-transcribe both photos and identify the intended objects: sequence
   prefix, token `x_i`, hidden state `h_i`, concept/tag `c_j`, and scalar
   energy.
2. Define candidate decompositions:
   - additive token energy: `E(x) = sum_i e_i(x)`
   - prefix energy: `E(x_{1:i})`
   - hidden-state energy contributions from `h_i`
   - concept/tag energy from probes or learned concept directions
   - attention/head/layer contribution attribution
3. Choose one small, controlled domain with ground-truth tags:
   - synthetic grammar with known latent tags
   - arithmetic/proof traces with known step types
   - classification text with known concepts
4. Train or attach an energy head and compare against AR log-probability,
   classifier probes, and attribution baselines.
5. Test interventions:
   - remove or edit a tagged token
   - swap concept-bearing spans
   - clamp hidden-state directions
   - measure whether predicted energy changes in the expected direction
6. Define metrics:
   - localization accuracy
   - rank correlation between energy contribution and known tag importance
   - intervention effect size
   - stability across seeds and model sizes
7. Decide what would falsify the idea. If decompositions are unstable,
   non-identifiable, or no better than AR log-prob attribution, report that.

Deliverable: an experiment plan plus a minimal synthetic benchmark deciding
whether sequence energies expose real interpretable components.
