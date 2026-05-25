# Handwritten note — Pros & Cons of ARs / Transformers (LeCun's (1-ε)^N argument)

**TLDR:** Best-effort transcription of a handwritten one-page note that sketches LeCun's exponential error-compounding argument against autoregressive transformers, plus a side remark on EBM scoring of full sequences. Image: `handwritten_ar_pros_cons_lecun_argument.jpg` in this folder.

---

## Provenance

- Source: photo of a handwritten note, uploaded by Brando 2026-05-24.
- Original upload path (ephemeral): `~/.claude/uploads/c12acaee-.../6610023b-19238.jpg`
- Saved here: `assets/handwritten_ar_pros_cons_lecun_argument.jpg`
- Related experiment context: this experiment (`02_ar_pros_cons`) tests *which* of the standard theoretical objections to autoregressive LLMs actually bite empirically. LeCun's exponential error-compounding bound is one of the headline claims under test — see `../README.md`, `../CLAIMS.md`, `../FINDINGS.md`.

Handwriting is at times rushed/blurred; passages marked `[unclear: ...]` are best guesses. Words crossed out in the original are noted as `~~struck~~`.

---

## Transcription

### Title

**Pros & Cons of ARs / Transformers**  *(small annotation above title: "use the same paper [unclear]")*

### Top-right margin note (next to title)

> Idea: to predict[?] the probability ε of an error? The model AR/LLM [unclear] produces next True[?] which is `x^(t+1)`, being best can't contradict… [blurry, but its [unclear: irreproducible/inreproducible]]

### (1) LeCun's `(1-ε)^N` argument

→ **(1)** LeCun's `(1-ε)^N` argument → detailed pros/cons, assumptions.

Margin annotation: `T_x i.e. N = T_x = seq len` (i.e. sequence length).

### (2) EBM "advantage" remark

→ **(2)** "Advantage" of EBMs is taking score on an entire **sequence** instead of a per-token prediction / inference. *(PS: what about — it's training cost too bad).*

### Exponential Error Compounding Argument

→ **Exponential Error Compounding Argument** — *seems weird; why an EBM doesn't[?] do the same for prob.*

- generated token takes you **outside the manifold** of "correct" answers, e.g. entity/gradient[?] exams → tokens `[...]` why we are doing (serious) adult mathematics or wars[?], takes is more reasonable (correct).

  *(Reading: a single bad sampled token lands you off the correct-completion manifold; later tokens are conditioned on the off-manifold prefix, so errors compound.)*

### Independence + product step

> Above errors are independent.

Indicator setup:

$$
\mathbb{1}[\hat X^{(t)} = X^{(t)}] \;=\; \begin{cases} 1 & \text{if } \hat X^{(t)} = X^{(t)} \\ 0 & \text{if } \hat X^{(t)} \neq X^{(t)} \end{cases}
\quad\text{(is "pred right here")}
$$

Per-token probabilities:

$$
\Pr\big[\hat X^{(t)} = X^{(t)}\big] \;=\; \text{Prob. any single gen[erated token] is correct} \;=\; 1-\varepsilon
$$

$$
\Pr\big[\hat X^{(t)} \neq X^{(t)}\big] \;=\; \text{Prob. any given generated token is incorrect} \;=\; \varepsilon
$$

Sequence-level (assuming token-error independence):

$$
\Pr\big[\text{Sequence } N\,(T_x) \text{ is correct}\big]
\;=\; \Pr\big[\text{all are correct}\big]
\;=\; \Pr\!\left[\bigcap_{t=1}^{T_x} \hat X^{(t)} = X^{(t)}\right]
$$

$$
\;=\; \prod_{t=1}^{T_x} \Pr\big[\hat X^{(t)} = X^{(t)}\big]
\;=\; \prod_{t=1}^{T_x} (1-\varepsilon)
\;=\; (1-\varepsilon)^{T_x}.
$$

### Conclusion (bottom of page)

→ So **for any fixed** [non-zero error rate ε], the probability of generating a fully correct sequence goes to **zero exponentially fast** in `T_x` (the sequence length).

### Bottom-right marginal Q (handwriting more rushed)

> Q: but this is true — this is a property [unclear]. has a way? or not? This lemma can be lifted for higher `ε^2`[?] and even for **trained** ML models. *(reading: the bound is mathematically valid; question is whether the independence + i.i.d.-ε assumption survives in trained autoregressive models in practice.)*

---

## Editor's notes (not on the page)

1. The argument as written assumes:
   - Per-token error probability ε is **constant** across positions.
   - Per-token errors are **independent**.
   Both assumptions are exactly what an integrated experiment in this folder is designed to stress-test — in real trained AR models on real Lean sequences, ε is highly position- and context-dependent, and errors are *not* independent (a single bad token can either be self-corrected by the model or amplified).
2. The EBM aside (item 2) is the standard counter-pitch: score the *whole sequence* with an energy function rather than factor over tokens. The cost objection ("training cost too bad") is also real and is part of why `02_ar_pros_cons` separates *isolated* probes from *integrated* end-to-end runs.
3. See `../CLAIMS.md` for the formal version of the (1-ε)^N claim and how this experiment plans to falsify or confirm its bite end-to-end.
