# EBM MCMC Notes for VeriBench

TL;DR: The paper is Yang Song and Diederik P. Kingma, "How to Train Your Energy-Based Models", arXiv:2101.03288. The useful research question is not "is MCMC inefficient?" The useful question is: for Lean/VeriBench candidate scoring, can short-run or biased sampling still be good enough to train an energy function that ranks correct formal artifacts above bad ones?

## Source Paper

- Yang Song and Diederik P. Kingma. "How to Train Your Energy-Based Models." arXiv:2101.03288, submitted January 9, 2021; revised February 17, 2021.
- The photographed pages are from the introductory EBM setup and the section "Maximum Likelihood Training with MCMC."
- The paper frames EBMs as unnormalized probabilistic models with density

```math
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta},
\qquad
Z_\theta = \int \exp(-E_\theta(x))\,dx.
```

The photos are about the core pain point: exact likelihood and exact sample synthesis are generally intractable, so maximum-likelihood EBM training uses approximate model samples, often from MCMC.

## Photo Transcription

### Photo 2

Printed content, paraphrased:

- The energy function does not need to integrate to one, so it can be any nonlinear regression function as long as it is normalizable in principle.
- Exact likelihood computation and exact sampling are generally intractable for EBMs.
- The paper lists three main training routes: maximum likelihood with MCMC, score matching, and noise contrastive estimation.
- Section 2 defines the EBM density using energy `E_theta(x)` and partition function `Z_theta`.
- Section 3 starts maximum likelihood training with MCMC and writes the expected log-likelihood over the data distribution.

Handwritten notes:

- "Q: if not normalizable then what?"
- "QC" / question marker next to the statement that density estimation is reduced to nonlinear regression.
- Question marker next to "exact synthesis of samples ... generally intractable."
- Near Section 2: "like x in R^n?" / question about what kind of variable `x` is.
- Near the EBM density:

```math
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z_\theta}
            = \frac{e^{-E_\theta(x)}}{Z_\theta}
```

- "NN" near the energy, reading `E_theta` as a neural-network scalar scorer.
- Right margin, partly unclear: question about whether EBMs are good for something like reasoning / proof / goal-conditioned objects.
- Bottom: boxed objective

```math
\mathbb{E}_{x \sim p_\mathrm{data}(x)}[\log p_\theta(x)]
```

- Right of `p_data(x)`: note that `p_data(x)` is the underlying data distribution, plausibly `p^*(x)`.

### Photo 1

Printed content, paraphrased:

- Maximizing likelihood is equivalent to minimizing `D_KL(p_data || p_theta)`.
- The likelihood itself cannot be computed directly because the normalizing constant `Z_theta` is intractable.
- The gradient of the log-likelihood can still be estimated with MCMC.
- The gradient decomposes into a data term and a model-expectation term:

```math
\nabla_\theta \log p_\theta(x)
  = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z_\theta.
```

- The partition-gradient term can be rewritten as a model expectation:

```math
\nabla_\theta \log Z_\theta
  = \mathbb{E}_{x \sim p_\theta(x)}[-\nabla_\theta E_\theta(x)].
```

Handwritten notes:

- Top notes try to rewrite KL / entropy / cross-entropy relations. The legible intent:

```math
D_\mathrm{KL}(p^* \| q) = H(p^*, q) - H(p^*)
```

and minimizing KL is equivalent to minimizing cross-entropy because the data entropy term is constant with respect to `theta`.

- "Q: contrastive; why did they mention here?" near the KL / likelihood derivation.
- Left margin:

```math
p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}
```

- Near the finite/discrete analog of the partition function:

```math
Z_\theta = \sum_{x \in X} \exp(-E_\theta(x))
```

- A sequence-style note: `x = x_1 x_2 ... x_s + x_A` or similar, likely thinking of proof/code strings and local edits.
- Bottom note near the one-sample Monte Carlo estimate is partially unclear, but the core question is: if the EBM is over LLM/proof candidates, what are we sampling, and are we using MCMC to estimate the model expectation?
- Margin notes compare `p_theta` averages vs `p*` / data averages, asking which distribution the expectation should be under.

### Photo 3

Printed content, paraphrased:

- MCMC sampling from EBMs is itself difficult, so much of the literature focuses on efficient MCMC methods.
- Langevin and Hamiltonian Monte Carlo use the fact that the score is easy to compute:

```math
\nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x).
```

- Langevin MCMC starts from a simple prior sample and iterates a noisy gradient update.
- Running MCMC to convergence can be computationally expensive.
- Contrastive divergence starts the chain at a datapoint and runs a fixed small number of MCMC steps, but truncated MCMC can bias gradients and hurt learning.

Handwritten notes:

- Top: "Q: nabla_x log p_theta(x) -> score."
- Question marker next to "Since drawing random samples is far from being trivial."
- Left margin: "Q: why Langevin and not SGD?"
- Question marker near the Langevin update equation.
- Arrow and emphasis next to highlighted sentence: "Running MCMC till convergence ... can be computationally expensive."
- Right margin:

```math
\text{estimating } \nabla_\theta \log Z_\theta
  = \mathbb{E}_{p_\theta}[\cdots]
\text{ is difficult.}
```

- Right margin, interpreted:
  - If we cannot do this exactly, it may be too hard.
  - Maybe still do SGD / optimization with an approximate or biased gradient.
  - Big research question: what if we do it anyway?
  - For deep learning, biased / approximate gradients often work. When does that work here? When does it not work?

## Answers to the Notes

### If the energy is not normalizable, then what?

Then it is not a valid probability model over that domain. The EBM can still be used as an unnormalized score or heuristic ranker, but `p_theta(x)` is not a proper density if `Z_theta` is infinite or undefined.

For VeriBench this is manageable:

- If `x` is a finite candidate set for a task, `Z_theta = sum_x exp(-E_theta(x))` is finite.
- If `x` is a bounded-length Lean/code string over a finite vocabulary, the space is finite.
- If `x` is unbounded proof/code text, use a length bound, a base-model prior, or an explicit length penalty.

So the first pilot should avoid the hardest continuous/infinite-support issue by using finite candidate pools per task.

### Is density estimation just nonlinear regression?

Only in the weak sense that `E_theta(x)` can be any scalar function approximator, e.g. a neural network. It is not ordinary supervised regression unless we have target energy labels. The training objective is distributional: lower energy on data/correct candidates and higher energy on model/negative candidates.

For our use case, the practical formulation is conditional ranking:

```math
E_\theta(\text{task}, \text{candidate})
```

The model should assign lower energy to verifier-passing or gold candidates than to corrupted or failing candidates.

### Why Langevin and not SGD?

SGD on `x` would optimize toward a low-energy mode. Langevin is SGD plus noise calibrated so the chain samples from the whole distribution proportional to `exp(-E_theta(x))`, not just the current best mode.

For proof/code text, vanilla Langevin over `x` is not directly available because `x` is discrete. Options:

- sample in a continuous latent embedding space and decode;
- use Metropolis-Hastings over text edits;
- use an LLM proposal distribution and accept/reweight with the EBM;
- avoid sampling at first and do finite-pool ranking / contrastive training.

### Do we really need full MCMC?

Not for the first experiment. The full paper is explaining maximum-likelihood EBM training, where exact model expectations are hard. But the research bet here can be:

> Short-run, biased, or finite-pool negative sampling may still be enough to learn a useful energy function for Lean/proof candidate ranking.

That is exactly the kind of "do it anyway" question worth testing.

## Concrete Research Question

Can a transformer energy function trained with cheap negative sampling, short-run MCMC, or contrastive divergence rank VeriBench candidate formalizations/proofs better than a base LLM score, even when the MCMC sampler is not run to convergence?

Subquestions:

- How biased can the negative sampler be before ranking quality collapses?
- Does the EBM learn semantic verification signals or mostly superficial formatting signals?
- Does a verifier-informed negative pool reduce the need for expensive model samples?
- When does optimizing/searching for low energy suffice, even if we cannot treat the result as a calibrated density?

## First VeriBench Three-Example Pilot

Use `veribench_three_example_manifest.json`.

The seed tasks are:

- `easy_set/1_MyAdd`: arithmetic sanity check.
- `easy_set/21_is_palindrome`: helper-heavy digit/string reasoning.
- `cs_set/binary_search`: classic algorithmic specification with sorted-input precondition and index soundness/completeness.

Initial split policy:

- For the tiny pilot, use all three to debug the pipeline and deliberately overfit.
- For an actual benchmark experiment, preserve the spirit of VeriBench by keeping most tasks as held-out test tasks. Use only a small train/dev slice for fitting the energy model and hyperparameters.

## Minimal Pipeline

1. Build candidate pools per task:
   - gold Lean file;
   - existing agent-generated attempts;
   - corrupted theorem statements;
   - wrong implementation/spec pairings;
   - small local text edits.

2. Define an energy scorer:

```math
E_\theta(t, y) \in \mathbb{R}
```

where `t` is the VeriBench task context and `y` is a candidate Lean artifact.

3. Start with a transformer scalar scorer:
   - use an open-source code/Lean-capable transformer as encoder or feature extractor;
   - add a scalar energy head;
   - do not start with autoregressive likelihood as the main objective.

4. Train with finite-pool contrastive/ranking loss first:

```math
E_\theta(t, y^+) + m < E_\theta(t, y^-)
```

or binary logistic contrastive loss over positive vs negative candidates.

5. Add "MCMC anyway" only after ranking works:
   - proposal: local text edit, theorem-name swap, proof deletion, statement corruption, or LLM-sampled variant;
   - acceptance: energy-based Metropolis step;
   - measure whether short-run chains improve hard negatives.

## What Would Count as Progress?

- The gold/verifier-passing candidate has the lowest energy in each three-example pool.
- Generated failing candidates get higher energy than passing candidates.
- Short-run negative sampling produces harder negatives than random corruptions.
- The learned energy improves ranking beyond a base LLM log-probability or simple syntax heuristics.

## Bottom TL;DR

The right first move is not to solve EBM likelihood training in full generality. Use VeriBench as a finite candidate-ranking problem, train a transformer energy scorer with cheap contrastive negatives, and then test whether short-run MCMC gives useful hard negatives despite being biased and non-convergent.
