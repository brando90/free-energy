# Pre-Prompt - Why Use Smooth Activations If Hardware Matters?
**TLDR:** Sharpen the screenshot question into a testable hardware-aware ML hypothesis. The core issue is whether smooth/gated activations buy enough optimization, calibration, or representation quality to justify their kernel cost compared with simpler hardware-friendly alternatives.

## Compact Question Summary

- **Goal:** Test why smooth or gated activations such as softplus, GELU,
  GEGLU, and SwiGLU are used if hardware efficiency is so important.
- **Confidence:** Handwritten question high; `Noam S.` as Noam Shazeer medium.
- **Importance:** High, roughly 8/10.
- **Key uncertainty:** The page is about softplus/sigmoid identities, but the
  handwritten question also points at modern transformer activations.

## Rough User Seed

The notebook question appears to ask:

> If hardware is so important, why are GELU or the unusual activations Noam S.
> came up with used in machine learning?

## Prompt To Future Agent

Do not answer with a slogan like "accuracy matters more." Determine when that
is actually true.

Plan:

1. Re-read `transcription.md` and the source photos.
2. Formalize the candidate activations:
   - ReLU and squared ReLU as hardware-friendly baselines.
   - Softplus as a smooth positive-part function.
   - SiLU/Swish, GELU exact, GELU tanh approximation.
   - GLU-family activations such as ReGLU, GEGLU, and SwiGLU.
3. Separate three questions:
   - Primitive cost: how expensive are `exp`, `erf`, `tanh`, multiply gates,
     and memory traffic on actual hardware?
   - Kernel reality: are these activations fused, approximated, or hidden under
     GEMM cost in modern frameworks?
   - Model value: do they improve optimization, calibration, perplexity,
     sample efficiency, or stability enough to pay the cost?
4. Compare the textbook softplus story with modern transformer feed-forward
   activations. Softplus can be mathematically convenient without being the
   best production activation.
5. Make falsification criteria explicit. If smooth/gated activations are slower
   and do not improve a controlled task, the hardware objection strengthens. If
   fused kernels make their marginal cost small or quality gains dominate, the
   objection weakens.

Deliverable: a short note plus a reproducible benchmark plan that answers the
question with measured tradeoffs rather than vibes.
