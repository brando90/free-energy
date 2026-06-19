# Transcription - Softplus Activation Hardware Question
**TLDR:** The handwritten question asks why hardware-costly smooth or gated activations are used if hardware matters so much. The printed page is a softplus/sigmoid reference; this file records equations and summarizes the textbook prose without copying the full page.

## Source Images

- `assets/photo_1.jpg`
- `assets/photo_2.jpg`

## Handwritten Note

Best reading:

> Q: I don't understand: if hardware is so important, then why does GELU or the
> crazy activations Noam S. came up with are used in machine learning?

Cleaned grammatical version:

> If hardware is so important, why are GELU or the unusual activations Noam S.
> came up with used in machine learning?

## Transcription Uncertainty

- `GELU` is the likely reading. The handwriting could also be read as `GeLU` or
  `Golu`, but the machine-learning context strongly suggests GELU.
- `Noam S.` likely refers to Noam Shazeer, especially GLU-family activations
  such as GEGLU or SwiGLU. This is an interpretation, not certain from the
  handwriting alone.
- The final words appear to be `machine learning`.

## Printed Page Context

The visible textbook page is headed `PROBABILITY AND INFORMATION THEORY` and
discusses the softplus function, sigmoid identities, and a graph labeled
`Figure 3.4: The softplus function.`

The printed prose says, in summary, that softplus is useful for positive
parameters such as distribution scale/rate parameters, that it appears when
manipulating sigmoid expressions, and that it is a smooth version of the
positive-part function.

Visible equations:

```tex
x^+ = max(0, x)

\sigma(x) = \frac{\exp(x)}{\exp(x) + \exp(0)}

\frac{d}{dx}\sigma(x) = \sigma(x)(1 - \sigma(x))

1 - \sigma(x) = \sigma(-x)

\log \sigma(x) = -\zeta(-x)

\frac{d}{dx}\zeta(x) = \sigma(x)

\forall x \in (0, 1), \quad \sigma^{-1}(x) =
\log\left(\frac{x}{1-x}\right)

\forall x > 0, \quad \zeta^{-1}(x) = \log(\exp(x) - 1)

\zeta(x) = \int_{-\infty}^{x} \sigma(y)\,dy

\zeta(x) - \zeta(-x) = x
```

## Research-Question Interpretation

The note is not just asking what softplus is. It asks a broader hardware-aware
ML question: if exponentials, smooth nonlinearities, and gated activations are
more expensive than ReLU-like primitives, why do successful models still use
them?
