**Source artifacts.** Original notebook photo saved at `assets/source_photo.jpg`;
summary update photo saved at `assets/summary_update_photo.jpg`.

**Coding-agent prompt.** `coding_agent_prompt.md` specifies the real synthetic
sequence experiment, optional SNAP escalation, deliverables, and verdict
criteria.

**Compact hypothesis.**
- **Goal:** Test whether reducing a rich sequence or representation to one
  scalar real-valued energy is too lossy for EBMs, unless the energy is
  structured or decomposed.
- **Confidence:** ~50%.
- **Importance:** Medium, 5/10.
- **Key uncertainty:** This is plausible but not yet central; the note says the
  idea feels reasonable but is only 50/50.

**Context.** The source notebook photo appears to ask how to map an EBM energy
for a sequence or prefix onto real components: tokens, hidden states, concepts,
or tags. The visible math includes a Boltzmann conditional form and an
`E_theta(x_{1:i})`-style sequence energy. The question is not whether an energy
is a scalar, but whether useful component-level structure can be recovered from
that scalar or from the network computing it.

**Question.** Is one scalar energy output a bad design bottleneck for sequence
EBMs, and do structured/decomposed energy outputs fix anything measurable?

**Tasks**
- [ ] Re-transcribe both source images and refine the intended variables,
  confidence, and importance.
- [ ] Define why a single real-valued output might be too lossy: optimization,
  credit assignment, interpretability, calibration, or expressivity.
- [ ] Define candidate decompositions: token-additive energy, prefix energy,
  hidden-state attribution, concept-probe energy, and layer/head contribution.
- [ ] Pick one controlled domain with known tags or concepts.
- [ ] Train or attach an energy head and compare against AR log-probability,
  classifier probes, and standard attribution baselines.
- [ ] Run interventions: token removal, span swap, hidden-state direction
  clamp, and concept edit.
- [ ] Measure localization accuracy, intervention effect size, rank
  correlation with known tag importance, and seed stability.
- [ ] Decide whether any decomposition is scientifically meaningful or only a
  post-hoc visualization.

**Deliverable.** A minimal benchmark and analysis note in
`experiments/hypothesis/01_h_interpretable_energy_tags/` that reports whether
EBM sequence energies produce stable interpretable components.
