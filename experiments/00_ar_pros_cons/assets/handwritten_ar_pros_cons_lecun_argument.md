# Handwritten note — Exponential Error Compounding Argument (Pros & Cons of ARs / Transformers)

**TLDR:** Best-effort transcription of a handwritten one-page note that names the headline objection to autoregressive language models — the **Exponential Error Compounding Argument** — derives `(1 − ε)^{T_y}`, makes the independence assumption explicit, and flags the question of whether the assumption survives in trained, verifier-guided systems. Image: [`handwritten_ar_pros_cons_lecun_argument.jpg`](handwritten_ar_pros_cons_lecun_argument.jpg) in this folder. Notation: `T_y` is the **output** sequence length the AR model is asked to generate; `T_x` is the input/prompt length (recorded for symmetry, not the primary axis of the argument).

---

## Provenance

- Source: photo of a handwritten one-page note, latest upload by Brando 2026-05-27.
- This page **supersedes** an earlier handwritten note on the same topic (same image filename) that did not yet name the argument and used `T_x` for the generated-sequence length. Notation here: **`T_y` = output sequence length**, **`T_x` = input/prompt length**.
- Original ephemeral upload path: `~/.claude/uploads/160bb7b7-4117-4302-9ab1-d0590abd0948/b236d69d-19827.jpg`.
- Saved here: `assets/handwritten_ar_pros_cons_lecun_argument.jpg`. Mirror copies live in `website/brandomiranda/experiments/10_autoregressive_llm_pros_cons/images/photo-1-ar-transformer-pros-cons.jpg` and the brandomiranda site repo (`experiments/10_autoregressive_llm_pros_cons/images/`).
- Related experiment context: this experiment (`00_ar_pros_cons`) tests *which* of the standard theoretical objections to autoregressive LLMs actually bite empirically. The Exponential Error Compounding Argument is one of the headline claims under test — see [`../README.md`](../README.md), [`../CLAIMS.md`](../CLAIMS.md), [`../PROBE_SPECS.md`](../PROBE_SPECS.md) (Probe 06), [`../FINDINGS.md`](../FINDINGS.md).

Handwriting is at times rushed/blurred; passages marked `[unclear: ...]` are best guesses. Words crossed out in the original are noted as `~~struck~~`.

---

## Transcription

### Title (new, top of page)

> **This has a name: Exponential Error Compounding Argument**

*Editorial note: the page now opens with this naming, recognizing that the `(1 − ε)^N` objection has a formal label. The blog post in [`../blog/2026-05-26-ar-error-compounding-real-or-fiction.md`](../blog/2026-05-26-ar-error-compounding-real-or-fiction.md) is titled from this naming.*

### Subtitle

**Pros & Cons of ~~ARs~~ / Transformers**  *(margin: "in the case `T_y = N`, and `T_y ≠ T_x` (with `T_x` = input prompt length)")*

### Top-right margin (next to title)

> Idea: do tools `Δ` the probability `ε` of an error? The model AR/LLM might produce output. The output is `[unclear: irrecoverable]` wrong but tools can catch it… *(reading: a verifier / tool can reduce the effective error probability that survives — this is exactly the recoverable-Markov vs geometric distinction in [`../PROBE_SPECS.md`](../PROBE_SPECS.md) Probe 06.)*

### (1) LeCun's `(1 − ε)^{T_y}` argument

→ **(1)** LeCun's `(1 − ε)^{T_y}` argument → detailed pros/cons, assumptions.

Margin annotation: `T_y = N = output seq len` (the chain-rule length of the generated sequence; **not** the prompt length `T_x`).

### (2) EBM "advantage" remark

→ **(2)** "Advantage" of ~~ARs~~ **EBMs** is taking a score on an entire **sequence** instead of a per-token prediction / inference. *(PS: what about — its training cost too bad lol.)*

### (3) Confront / clarify what this actually means

→ **(3)** I confront the issue. Clarify what this actually means.

*(This is the framing the blog post adopts: take the argument seriously, but separate the algebra `(1 − ε)^{T_y} → 0` from the empirical claim that real verifier-guided systems behave like that.)*

### Exponential Error Compounding Argument — definition box

> **Exponential Error Compounding Argument**: LeCun's argument that ARs are "compounding error."
>
> Per-step probability `ε` for a generated token takes you **off the manifold** of correct answers — probably because tokens give us info (errors carry past the step they occur in).
>
> The actual mathematics is very similar [via a central-limit / sum-of-Bernoullis style result] to the standard product-of-correctness argument: on the right manifold = (correct).

### Independence + indicator setup

> Assume errors are **independent**.

Indicator:

$$
\mathbb{1}\!\left[\hat X^{(t)} = X^{(t)}_*\right]
=
\begin{cases}
1 & \hat X^{(t)} = X^{(t)}_* \\
0 & \hat X^{(t)} \neq X^{(t)}_*
\end{cases}
\quad \text{(is "pred right here")}
$$

(`X^{(t)}_*` denotes the correct/reference token at step `t`; `\hat X^{(t)}` is the model's sampled token.)

### Per-token probabilities

$$
\Pr\!\big[\hat X^{(t)} = X^{(t)}_*\big] \;=\; \text{Prob. any given generated token is correct} \;=\; 1 - \varepsilon
$$

$$
\Pr\!\big[\hat X^{(t)} \neq X^{(t)}_*\big] \;=\; \text{Prob. any given generated token is incorrect} \;=\; \varepsilon
$$

### Sequence-level (assuming token-error independence)

$$
\Pr\!\big[\text{Sequence of length } T_y \text{ is correct}\big]
\;=\; \Pr\!\Big[\textstyle\bigcap_{t=1}^{T_y}\, \hat X^{(t)} = X^{(t)}_*\Big]
\;=\; \prod_{t=1}^{T_y} \Pr\!\big[\hat X^{(t)} = X^{(t)}_*\big]
\;=\; \prod_{t=1}^{T_y} (1 - \varepsilon)
\;=\; (1 - \varepsilon)^{T_y}
\;=\; e^{\,T_y \log(1 - \varepsilon)} \;\xrightarrow[T_y \to \infty]{}\; 0.
$$

*(The exponential form `e^{T_y \log(1-\varepsilon)}` is the new derivation step on this page and makes the decay rate explicit: the half-life in `T_y` is `log 2 / |log(1 − ε)| ≈ log 2 / ε` for small `ε`.)*

### Margin annotations on the derivation

- "if independence is **true**, this example shows as `T_y` gets large is an upper bound to LeCun (but we can prob fix that)" — *reading: independence makes the bound tight; relaxing independence (e.g., recovery, self-correction) loosens it in the right direction.*
- "if `T_y` = number of tokens (or all tokens in seq)" — *the exponent is the length of the generated sequence, not the prompt.*

### Conclusion (bottom of page)

→ So **for any fixed** `ε > 0`, the probability of generating a fully correct sequence goes to **zero exponentially fast** in `T_y` (the generated-sequence length).

### Bottom-right margin Q (handwriting more rushed)

> Q: Is this true? How big do depend on `T_y` (& bigness)... if it's not, what's so... or it will be brighter... *(reading: the bound is mathematically valid; the open question is whether the independence + constant-ε assumption survives in trained, verifier-guided AR models in practice — exactly Probe 06 in [`../PROBE_SPECS.md`](../PROBE_SPECS.md).)*

---

## Editor's notes (not on the page)

1. **What this version of the page adds vs. the previous one:**
   - Names the argument: **Exponential Error Compounding Argument**.
   - Switches the generated-sequence-length symbol from `T_x` (incorrect, that's input/prompt length) to **`T_y`** (output length).
   - Adds the exponential reformulation `(1 − ε)^{T_y} = e^{T_y \log(1 − \varepsilon)}`, which makes the decay rate explicit and connects the geometric model to the Bernoulli / log-likelihood form.
   - Adds the "independence true ⟹ upper bound to LeCun" margin note, which is exactly the experimental hypothesis: dropping independence (recovery, verifier resampling) should loosen the bound.

2. **The algebra is correct under its assumptions; the assumptions are what the experiment stress-tests.** The argument as written assumes:
   - Per-token error probability `ε` is **constant** across positions.
   - Per-token errors are **independent**.
   Both assumptions are exactly what Probe 06 in this folder is designed to falsify on real trained AR models + Lean verifiers, where `ε` is highly position- and context-dependent and errors are *not* independent (a single bad token can either be self-corrected by the model or amplified, and a verifier can reject before the error survives).

3. **The EBM aside (item 2) is the standard counter-pitch:** score the *whole sequence* with an energy function rather than factor over tokens. The cost objection ("training cost too bad") is also real and is part of why `00_ar_pros_cons` separates *isolated* probes from *integrated* end-to-end runs. See also Probe 06 PS#2 in [`../PROBE_SPECS.md`](../PROBE_SPECS.md) on whether the "full-sequence EBM needs a fixed-dim x" objection actually bites — transformers/LSTMs already handle variable-length inputs.

4. **Naming:** "Exponential Error Compounding Argument" is the umbrella name; "LeCun's `(1 − ε)^N` argument" is one instance of it. The blog post adopts the umbrella name to keep the discussion technical rather than personal.

5. See [`../CLAIMS.md`](../CLAIMS.md) (Claim 6) for the formal version of the `(1 − ε)^{T_y}` claim and how this experiment plans to falsify or confirm its bite end-to-end.
