**Source artifact.** Notebook photo saved at `assets/source_photo.jpg`.

**Coding-agent prompt.** `coding_agent_prompt.md` specifies the real benchmark,
local smoke test, optional SNAP escalation, deliverables, and verdict criteria.

**Compact hypothesis.**
- **Goal:** Test whether the Boltzmann exponential `exp(-E)` is a bad
  hardware/inference primitive, and whether another monotone or normalized
  score can keep the useful ordering without paying exponential/partition
  costs.
- **Confidence:** ~75%.
- **Importance:** Very high, roughly 9.5/10.
- **Key uncertainty:** Maybe any learnable scoring function is fine once the
  model and data are strong enough.

**Context.** The source notebook photo shows the conditional EBM form
`p_theta(y | x) = exp(-E_theta(x, y)) / Z_theta(x)` and appears to ask whether
there is a "Boltzmann/surjection-like" object behind the formula. The
handwriting is ambiguous, but the core research question seems to be whether
the exponential/partition-function form is a real computational liability,
especially for hardware and inference, or merely one convenient monotone
parameterization among many.

**Question.** Is `exp(-E)` essential to the useful EBM story, or can EBMs use a
more hardware-friendly score/normalization while preserving the ordering,
search, and calibration properties that matter?

**Tasks**
- [ ] Re-transcribe the source image and refine the one-sentence hypothesis,
  confidence, and importance.
- [ ] Write the standard EBM derivation for `p_theta(y | x)`,
  `Z_theta(x)`, and `log p_theta(y | x)`.
- [ ] Identify the exact hardware/inference costs caused by `exp`, `logsumexp`,
  and estimating `Z`.
- [ ] Compare alternative scoring/normalization choices that preserve ranking
  but avoid or reduce exponential/partition-function costs.
- [ ] Formalize at least three interpretations of the "surjection-like"
  phrase: finite Boltzmann distribution, latent-variable marginalization, and
  pushforward/quotient over many derivations mapping to one answer.
- [ ] Build a toy finite example where several latent paths map to the same
  output; show how energy, multiplicity, and normalization change the output
  probabilities.
- [ ] Catalog which issues are mathematical and which are numerical:
  intractable sum, overflow/underflow, Monte Carlo variance, and approximate
  negative-sample bias.
- [ ] Decide whether this yields a new hypothesis for EBMs or collapses to
  standard partition-function language.

**Deliverable.** A short note in
`experiments/hypothesis/00_h_boltzmann_surjection_object/` plus a toy finite
calculation that either preserves or kills the hypothesis.
