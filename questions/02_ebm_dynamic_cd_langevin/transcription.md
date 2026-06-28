# Transcription - Dynamic Weighted CD and Langevin EBM Notes

This transcription preserves equations and brief context from the printed material, but summarizes long printed textbook passages instead of copying them verbatim.

## Photo 1

Printed context:

- Topic: Monte Carlo methods for EBMs.
- Highlighted point: In an EBM, the chicken-and-egg problem is avoided by sampling with a Markov chain.
- A Markov chain has a state `x` initialized at an arbitrary value and repeatedly updated until it is approximately a fair sample from `p(x)`.
- Transition distribution notation: `T(x' | x)`.
- Running the chain means repeatedly updating `x` to a sampled `x'`.
- The text reparameterizes countable many states by a positive integer `x`.
- The distribution of all chains at time `t` is `q^(t)`.
- Goal: make `q^(t)` converge to `p(x)`.
- Recursion shown:

```text
q^(t+1)(x') = sum_x q^(t)(x) T(x' | x)
```

Handwritten notes and questions, best effort:

- "What the chicken egg problem is caused by graph being undirected (many?) + have no good estimate of p(x)?"
- "Q: Converse, when it supposedly can we of course in a probabilistic setting only if the Markov property + [stationary distribution?]?"
- "chain?"
- "Run Markov: x' <- T(x' | x); x <- x'?"
- "If q^(t)(x) = p(x) if [linear?]"
- "Maybe it tells you what p(x) looks like but not exactly how to weight?"
- "Q: in simulations maybe with hidden Markov latent states as updates + weight them with alignment and Zipf fit + use CD (counter?)"
- Margin note: "p0 needs to be constant for weighting / good theory?"

Uncertain fragments:

- Several notes around "Converse" and "stationary" are partially obscured.
- The bottom line appears to ask whether hidden/latent Markov states can be weighted using an alignment or Zipf-style fit.

## Photo 2

Printed context:

- Topic: Markov chain Monte Carlo methods.
- The page contrasts ancestral sampling in directed models with MCMC in undirected models.
- Highlighted point: EBMs make drawing direct samples difficult because the graph is undirected and contains zero-probability states; theoretical guarantees for MCMC are case-by-case.
- Highlighted point: MCMC applies to many probability distributions that cannot be analyzed with simple tools.

Handwritten notes and questions, best effort:

- Header: "chicken egg problem when sampling from undirected graphs."
- Parenthetical: "since arrows have no direction?"
- "Q: can we only compute p_model(x) at this point?"
- "T -> need to compute p_model(x) actually this is available?"
- "If p_model(x) = tractable(constant) -> p_model(x) tractable(sample), [so] estimate Z out?"
- "Parallel: can we update transition chains to sample faster?"
- Left margin: "Q: I'm thinking if the state is something... then p_model(x) is good. But what about too many states where we cannot predict them; EBM?"
- Lower note: "If possible, continue saying MCMCs from random samples as searches with data."
- "Q: is possible; somehow."

Uncertain fragments:

- The lower-left handwriting is mostly legible as a concern about state count and predicting/covering many possible states, but exact wording is uncertain.

## Photo 3

Printed context:

- Topic: Langevin sampling.
- The page defines the score:

```text
s(x, w) = grad_x log p(x | w)
```

- Substituting the EBM form gives:

```text
s(x, w) = -grad_x E(x, w)
```

when the base distribution is constant or absorbed and `Z(w)` is independent of `x`.

- Langevin step:

```text
x^(tau+1) = x^(tau) + eta grad_x log p(x^(tau), w) + sqrt(2 eta) epsilon^(tau)
epsilon^(tau) ~ N(0, I)
```

- In the limits `eta -> 0` and `T -> infinity`, the chain samples from `p(x)`.
- Contrastive divergence starts the chain from a training data point and uses only a few steps.
- Highlighted warning: the resulting sample is biased and close to the data manifold; this is useful for discrimination but less effective for generative modeling.

Handwritten notes and questions, best effort:

- "Q: Score the x? ... p0 prior or base distribution? Why necessary?"
- "Need for sampling from EBM in undirected graphical models?"
- "Ooh it's to next keep distribution, all we can see is energy output? It is a map, so we estimated if this is an EBM?"
- Top margin: "But if we believe learning the manifold hypothesis, all we need to learn is the energy surface, then why Z or p(x)?"
- Right margin: "fundamentals question is: what about it?"
- Bottom:
  - "Q1 why? Is it because not enough data / too noisy due to 1 single step?"
  - "Q2 I was thinking x^(T) can have some CD weight alpha^(T) for having CD, earlier T's more 'negative examples' (low), further T's -> higher weight for CD as 'positive examples' + parallelization for MD to work/scale."

Uncertain fragments:

- The top margin is a paraphrase; several words are partly cut off.
- The "positive examples" phrase is legible, but mathematically it needs correction in the research brief.

## Photo 4

Screenshot text:

```text
Transcribe everything here especially all my questions and handwriting and then make a list of all their research questions and questions I have so that I can create so that you can create an awesome agent coding prompt to solve everything and figure out what the answers are
```

## Consolidated Research Questions

1. If the manifold hypothesis is right and the energy surface is the key object, why is `Z(theta)` needed at all?
2. Is the partition function only problematic because it is unobserved/intractable, or because normalized probabilities are mathematically required for likelihood training?
3. Is MCMC necessary for EBMs, or merely the standard practical sampling tool?
4. Can EBM training bypass MCMC with scalable heuristics, as autoregressive models and other large models often scale with imperfect objectives?
5. Are textbook PMF/PDF restrictions unnecessary if we only care about energy landscapes?
6. Can many Markov chains or semi-sequential transition updates be parallelized to improve mixing or wall-clock efficiency?
7. If the graph is undirected and direct ancestral sampling is unavailable, does moving to a continuous state space plus Langevin dynamics remove the chicken-and-egg problem?
8. In score matching or Langevin sampling, what exactly is the role of the base distribution `p0(x)`?
9. Is the model "sampling itself" during Langevin dynamics, or is `p0(x)` fixed during parameter updates?
10. Why does CD-k fail for generation when `k` is small: insufficient manifold coverage, gradient noise, or systematic bias?
11. Can samples at Langevin time `T` be assigned dynamic weights `alpha^(T)` during CD?
12. Can early trajectory states be used as low-weight hard negatives while later states receive higher weight?
13. Can this weighting be aligned with an empirical law such as a Zipf fit?
14. Can weighted short trajectories make molecular/Langevin dynamics massively parallel for EBM training?
