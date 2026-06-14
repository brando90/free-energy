**Source artifact.** Notebook photo saved at `assets/source_photo.jpg`.

**Coding-agent prompt.** `coding_agent_prompt.md` specifies the real
long-context experiment, optional SNAP escalation, deliverables, and verdict
criteria.

**Compact hypothesis.**
- **Goal:** Test whether transformers/attention already solve the alleged
  unbounded-length/full-input conditioning issue, leaving little special
  advantage for EBMs on this axis.
- **Confidence:** High.
- **Importance:** Very high, 9.8/10.
- **Key uncertainty:** The strongest baseline may be attention versus
  recurrence/forgetting, not only AR likelihood versus EBM energy.

**Context.** The source notebook photo shows unconditional/conditional EBM
normalization and appears to ask how an unsupervised EBM would solve a task.
The concern is that attention-based or autoregressive models may already learn
the useful input structure, so any "unsupervised EBM learns task structure"
claim needs a controlled baseline rather than a slogan.

**Question.** Do EBMs solve a real input-coverage or long-context problem that
transformers/attention do not already solve, or is attention the main
mechanism?

**Tasks**
- [ ] Re-transcribe the source image and refine the exact one-sentence goal,
  confidence, and importance.
- [ ] Define the alleged "unbounded length" or full-input conditioning issue
  precisely: context length, recurrence forgetting, global dependency capture,
  compute scaling, or calibration.
- [ ] Define measurable outcomes: downstream accuracy, sample efficiency,
  calibration, OOD robustness, and reranking improvement.
- [ ] Build a small controlled benchmark with latent factors or known tags.
- [ ] Compare unconditional EBM, conditional EBM, AR transformer, LSTM, and a
  cheap contrastive or masked-model baseline.
- [ ] Control for data, parameters, training budget, and inference compute.
- [ ] Test transfer: frozen energy features, light fine-tune, and energy-based
  reranking.
- [ ] Run kill tests: label permutation, latent-factor ablation, and
  compute-matched attention/AR comparison.

**Deliverable.** A falsifiable experiment plan and minimal prototype in
`experiments/hypothesis/02_h_unsupervised_ebm_task_structure/` deciding whether
the unsupervised EBM task-structure claim survives strong baselines.
