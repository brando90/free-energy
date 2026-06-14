# Pre-Prompt - Hypothesis 03: Does Probability Normalization Buy Real Power?

**TLDR:** Test whether turning energies or scores into normalized probabilities
gives concrete mathematical or computational capabilities that cannot be
recovered from unnormalized scores alone.

## Compact Hypothesis Summary

- **Goal:** Test whether probability normalization gives concrete mathematical
  or computational power that unnormalized scores cannot reproduce.
- **Confidence:** Answer unknown; roughly 80% confidence that this is an
  important question to settle.
- **Importance:** High, roughly 8/10.
- **Key uncertainty:** Rank-only search may only need score differences, while
  calibrated decisions, sampling, entropy, marginalization, and Bayesian
  composition may genuinely require normalized probabilities.

## Source

- Saved source photo: `assets/source_photo.jpg`
- Original upload path:
  `/tmp/codex-remote-attachments/019ec76d-55f8-7671-aed0-2667ac416b8b/161e8121-79ce-4e05-b40d-ca02b4834505/1-Photo-1.jpg`
- Transcript confidence: low. The image is rotated/cropped and only part of
  the note is visible.

## Transcription

User-provided transcription seed:

> Does the normalization actually give us concrete mathematical or
> computational power we could not get without it being a probability?

Rough image-only read, low confidence:

> Q / Hypothesis: only probability / normalization? I don't think we know.
> Maybe the short question name is something like "probability power." Prod
> importance or risk: 8/10. Confidence: maybe 80%.

## Prompt To Future Agent

Do not collapse this into the earlier hardware-cost question. This hypothesis is
about what probability normalization *buys*, not only what it costs.

Plan:

1. Re-transcribe the source photo and keep the user-provided sentence as the
   authoritative seed unless the image clearly contradicts it.
2. Formalize the finite candidate setting:
   - scores or energies `s(y)` / `E(y)`
   - normalized probabilities
     `p(y) = exp(s(y)) / sum_y' exp(s(y'))`
   - unnormalized scores known only up to additive or monotone transforms
3. Build a capability matrix separating operations that need only ordering or
   score differences from operations that need a probability measure:
   - argmax and top-k search
   - MCMC acceptance ratios
   - calibrated thresholding and abstention
   - expected utility
   - entropy and uncertainty
   - exact sampling
   - marginalization over latent variables
   - Bayes updates and product/composition of evidence
4. For each operation, state the minimal mathematical object required:
   ordering, log-ratio, unnormalized density, normalized probability, or
   calibrated probability.
5. Build a toy finite-world example where ranking is identical under several
   score transforms, but probability-dependent quantities change. Use it to
   test which downstream decisions break without normalization.
6. Decide whether normalization gives real power, or whether the workflow only
   needed an energy/ranking oracle.

Deliverable: a short note plus a runnable finite-world experiment that produces
a capability matrix and a go/no-go verdict.
