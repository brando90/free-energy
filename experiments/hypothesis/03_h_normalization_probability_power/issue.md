**Source artifact.** Notebook photo saved at `assets/source_photo.jpg`.

**Coding-agent prompt.** `coding_agent_prompt.md` specifies a finite-world
experiment/proof suite, deliverables, optional SNAP escalation, and verdict
criteria.

**Compact hypothesis.**
- **Goal:** Test whether probability normalization gives concrete mathematical
  or computational power that unnormalized scores cannot reproduce.
- **Confidence:** Answer unknown; roughly 80% confidence that this is an
  important question to settle.
- **Importance:** High, roughly 8/10.
- **Key uncertainty:** Ranking/search may only need score differences, while
  calibration, sampling, entropy, marginalization, and Bayesian composition may
  genuinely need normalized probabilities.

**Transcription.**

User-provided transcription seed:

> Does the normalization actually give us concrete mathematical or
> computational power we could not get without it being a probability?

Rough image-only read, low confidence:

> Q / Hypothesis: only probability / normalization? I don't think we know.
> Maybe the short question name is something like "probability power." Prod
> importance or risk: 8/10. Confidence: maybe 80%.

**Context.** H00 asks whether the Boltzmann exponential and partition path are
computationally expensive. This hypothesis asks the complementary question: if
normalization is expensive, what capability do we lose by dropping it? The
answer may depend on the downstream operation. Argmax search and top-k ranking
may only need an energy ordering, while calibrated abstention, entropy,
sampling, marginalization, likelihood comparison, and Bayesian updates may need
a real probability measure or at least well-calibrated ratios.

**Question.** Does converting scores/energies into normalized probabilities
give concrete mathematical or computational power that cannot be reproduced by
unnormalized energies, monotone score transforms, or ratio-only methods?

**Tasks**
- [ ] Re-transcribe the source image and refine the one-sentence hypothesis,
  confidence, and importance.
- [ ] Write a capability matrix for common operations: argmax, top-k, MCMC
  ratios, calibrated thresholding, expected utility, entropy, sampling,
  marginalization, and Bayesian composition.
- [ ] For each operation, specify the minimal object required: ordering,
  score difference, unnormalized density, normalized probability, or calibrated
  probability.
- [ ] Build a finite candidate-world experiment where multiple monotone score
  transforms preserve ranking but change normalized probabilities.
- [ ] Measure which downstream tasks are invariant to normalization and which
  fail without it.
- [ ] Include at least one latent-variable or hidden-state marginalization case
  where the need for normalization is explicit.
- [ ] Decide whether probability normalization provides a real capability or
  only a convenient interface for tasks that could use unnormalized scores.

**Deliverable.** A note and runnable finite-world experiment under
`experiments/hypothesis/03_h_normalization_probability_power/` with a compact
verdict: ranking-only workflows can drop normalization, or probability-level
workflows cannot.
