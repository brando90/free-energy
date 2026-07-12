# Transcription — Monte Carlo / Markov Chains / Langevin / Contrastive Divergence

Source: three handwritten textbook pages (Goodfellow et al., *Deep Learning*,
Ch. 17–18 territory: Monte Carlo methods, the partition function, Langevin/SGLD
sampling, Contrastive Divergence). Original photo filenames as provided:
`26344.jpg`, `26345.jpg`, `26343.jpg`. The photos themselves were not committed
(only the transcription was supplied); this file is the authoritative record.

The handwriting is partly ambiguous. Per the convention of
`experiments/02_fisher_div_grad_cost`, the transcription is treated as a
hypothesis to sharpen, not settled doctrine.

---

## `26344.jpg` — Monte Carlo Methods / Intractability

**Top margin (logic & definitions):**

- `if P_model(x) ∈ Tractable(compute) ⟹ P_model(x) ∈ Tractable(sample)`
- `if P_model(x) ∈ Tractable(sample) ⇏ P_model(x) ∈ Tractable(compute)`
- Example: `P_model(x)` leads to samples via SGLD but `Z(θ)` is still hard to
  compute.

**Inline annotations:**

- Next to "Obviously you can't compute `P_model(x)` at this point."
- Next to "When no tractable method to sample `P_model(x)` exists": *"(2) no
  low-variance `q(x)` is available."*
- "Chicken-and-egg problem when sampling from undirected graphs (since arrows
  have no directions)."

**Left margin:**

- Q: "I'm thinking, for the chicken-and-egg problem, what if we make the process
  continuous to try pure EBMs which we know how to do…" (partly obscured)
- Q: "Can we…" (obscured)

**Right margin:** "Do it anyway: F. This"

---

## `26345.jpg` — Monte Carlo Methods / Markov Chains

**Top margin:** "Note that the chicken-and-egg problem is caused by the graph
being undirected (mainly), and not by not having a good estimate of `p0(x)`."

**Right margin:**

- Q: "Can we do sampling in parallel / semi-sequential s.t. mixing improves &
  cost goes down?"
- "Markov chain makes it sequential."
- Chain: run Markov `x' ← T(x' | x)` (sample), `x ← x'` (update).
- "Are `x ∈ X = N_0 = {0,1,2,…}`?"
- `lim_{t→∞} q^(t)(x) = p(x)`
- Q: "we decided to be careful to not weight one against good samples."

**Left margin:** Q: "is parallel; continuous sampling MCMCs from previous
models, or something…" (partly obscured)

**Inline / bottom:**

- "Note: it's talking about PMFs/PDFs but I want EBMs, so I wonder if this is an
  unnecessary limitation?"
- "use the 'inefficiency' or go forever!"
- Q: "Can we not use 'bad' samples as bad examples & weight them with alignment
  (e.g. Zipf fit) + use CD (Contrastive Divergence)?"

---

## `26343.jpg` — Langevin Sampling & Contrastive Divergence

**Top/left margin (long):**

- RQ: "If we believe the manifold hypothesis / energy is all we need to
  traverse the world — then why the heck are we trying to deal with the
  partition function at all? Maybe it only seems intractable since we don't
  really *see* `Z(θ)`?"
- Q: "true to the `p(x|h)`, `p(h|x)` — about understanding if/why MCMC is a
  fundamental tool / thing we need for sampling from EBMs (undirected
  PGMs/models)."
- RQ: "if it's not really fundamental, can we somehow bypass it or do it in a
  messy way so we succeed in training an EBM? → SAGE/GPTs/…"

**Right margin:** "≈ needed if fundamental. Question: is it?"

**Inline (next to Eq. 14.62):** "Wait? but we are sampling the model…?! wait
`p0(x)` is constant?"

**Bottom (Contrastive Divergence steps):**

- Q[1]: "why? is it because not enough data, or too noisy due to 1 simple step?"
- Q[2]: "I was thinking `X^(T)` can have some (CD) weight `α^(T)` for training
  CD; earlier `T`'s → more 'negative examples', further `T`'s → ∞ higher weight
  for CD as 'positive examples' + parallelization for MD to work / scale."

---

## Consolidated research questions (the 9 the experiments answer)

**Core theory — EBMs & partition functions**

1. **RQ1 (manifold / why Z at all):** If the manifold hypothesis holds (energy
   is all we need to traverse the world), why deal with `Z(θ)` at all? Does it
   only *seem* intractable because we never actually observe `Z(θ)`?
2. **RQ2 (is MCMC fundamental):** Is MCMC a fundamental requirement for sampling
   from EBMs (undirected PGMs), or just the current default? If not fundamental,
   can we bypass it — or do it "messily" — and still train EBMs at scale (à la
   SAGE/GPTs)?
3. **Q3 (PMF/PDF limitation):** Textbooks restrict to PMFs/PDFs (countable
   discrete states). For pure EBMs, is the restriction to standard PMFs/PDFs an
   unnecessary limitation we can discard?

**Algorithms — MCMC & Langevin**

4. **Q4 (parallel / semi-sequential chains):** Can we run Markov-chain sampling
   in a parallel or semi-sequential way so mixing improves while cost drops?
5. **Q5 (continuous to dodge chicken-and-egg):** For the undirected chicken-and-
   egg problem, what if we strictly use a *continuous* setting and train pure
   EBMs we already know how to sample via Langevin dynamics?
6. **Q6 (Eq. 14.62 — sampling the model; `p0` constant?):** In the log-
   likelihood gradient, are we sampling the *model* itself, and is the base
   `p0(x)` being treated as a constant?

**Contrastive Divergence & scaling**

7. **Q7 (why is short-run CD bad):** Running the chain for only a few (or one)
   steps gives a biased sample. Why exactly is that problematic — is it because
   it doesn't cover the data manifold, or because the single-step gradient is
   too noisy (bias vs variance)?
8. **Q8 (dynamic CD weighting):** Can we assign a dynamic weight `α^(T)` to
   chain samples `X^(T)` — early steps as "negative" with one weight, late
   steps (`T→∞`) with higher weight as "positive" — and, with alignment (Zipf
   fit), use "bad" samples as informative negatives?
9. **Q9 (weighting → parallel scaling):** Can that dynamic weighting be combined
   with parallel/persistent chains so Langevin/"MD" EBM training actually
   scales?
