# Research Brief - Partition Functions, Langevin CD, and Dynamic Weighting

## 1. What the Partition Function Does

An EBM over a space `X` is usually written as:

```text
p_theta(x) = exp(-E_theta(x)) p0(x) / Z(theta)
Z(theta) = integral exp(-E_theta(x)) p0(x) dx
```

The partition function is not optional if the model is meant to define normalized probabilities. It turns an arbitrary energy landscape into a probability distribution. The manifold hypothesis does not remove this need. It says data may concentrate near a lower-dimensional structure, but it does not say how much probability mass each region receives, how off-manifold mass is controlled, or whether the density integrates to one.

What is optional is explicitly evaluating `Z(theta)` for every update. For maximum likelihood,

```text
grad_theta[-E_data log p_theta(x)]
  = E_data[grad_theta E_theta(x)]
    - E_p_theta[grad_theta E_theta(x)]
```

The derivative of `log Z(theta)` becomes the negative-phase expectation under `p_theta`. This is why training can avoid computing `Z(theta)` numerically while still needing approximate samples from the normalized model.

`Z(theta)` also drops out of the score with respect to `x`:

```text
grad_x log p_theta(x)
  = -grad_x E_theta(x) + grad_x log p0(x)
```

because `Z(theta)` has no dependence on `x`. This is why score matching and Langevin sampling can work without explicit partition-function evaluation.

## 2. Is MCMC Required?

MCMC is not a mathematical absolute. Alternatives include score matching, denoising score matching, noise-contrastive estimation, minimum probability flow, diffusion/recovery-likelihood variants, amortized samplers, variational approximations, normalizing-flow proposals, and deterministic particle methods such as SVGD.

But for a generic undirected model with only an energy function and no tractable sampler, exact independent sampling is usually unavailable. MCMC is the standard fallback because it only needs energy gradients or energy differences. Langevin dynamics is still an MCMC method: it is a Markov chain in continuous state space.

So the practical question is not "MCMC or no MCMC" but:

```text
How much biased dynamics can we tolerate before the learned EBM stops being useful?
```

Short-run CD, persistent CD, diffusion-assisted CD, and score-based methods all answer this differently.

## 3. Continuous Energy Landscapes Are Legitimate

Textbook PMF/PDF restrictions are not fundamental. EBMs can live on finite sets, countable sets, Euclidean spaces, manifolds, graphs, or mixed spaces, as long as a base measure is specified and `Z(theta)` is finite. What is invalid is treating an energy landscape alone as a probability model without specifying the reference measure and normalization.

For continuous Langevin EBMs, differentiability also matters. If `E_theta(x)` is not sufficiently smooth, the score and Langevin drift are not well-defined in the usual way.

## 4. The Base Distribution `p0(x)`

If `p0(x)` does not have trainable parameters, it is constant with respect to `theta` during parameter updates. It does not contribute to `grad_theta E_theta` unless it is parameterized.

It can still matter for sampling because the Langevin drift uses:

```text
grad_x log p_theta(x) = -grad_x E_theta(x) + grad_x log p0(x)
```

Examples:

- Uniform base on a bounded region: `grad_x log p0(x) = 0` inside the region.
- Standard Gaussian base: `grad_x log p0(x) = -x`.
- Empirical/data initialization for CD: this is an initialization distribution, not the target distribution.

The model is "sampling itself" in the sense that the current `E_theta` defines the Langevin drift. The initial noise or data-start distribution is not the target unless the chain is already at stationarity.

## 5. Why CD-1 Fails for Generation

CD-k uses a chain distribution:

```text
q_k = T_theta^k q_0
```

instead of the true model distribution `p_theta`. With small `k`, `q_k` stays close to the data-start distribution. This makes the gradient biased. It is not merely noisy. The gradient shapes energy locally near data, which can be enough for discrimination or representation learning, but it does not force the global energy surface to put correct mass across modes.

Noise is still a problem, especially for one-step Langevin in high dimensions, but the primary generative failure mode is systematic negative-phase bias.

## 6. Dynamic Weighted CD Hypothesis

A time-weighted CD objective uses:

```text
q_alpha = sum_{t=1}^T alpha_t q_t
sum_t alpha_t = 1
alpha_t >= 0
```

and approximates the negative phase as:

```text
E_{q_alpha}[grad_theta E_theta(x)]
```

If `alpha_T` carries most mass and `T` is large enough for mixing, this approaches the likelihood negative phase. If the weights place substantial mass on early trajectory states, the objective is biased, but it may be useful as a regularizer or hard-negative curriculum.

The proposed idea is viable as a heuristic, but the sign needs care:

- Data samples are positive-phase points.
- All generated trajectory samples are negative-phase points under the likelihood-gradient view.
- Early states can be low-weight hard negatives.
- Later states can be higher-weight negative samples because they are closer to the current model.
- Calling later chain states "positive examples" would require a different objective and could collapse the model toward its own early mistakes.

Zipf or alignment weighting could be tested, but it should be treated as an adaptive importance or curriculum rule. If `alpha_t` depends on `theta`, sample quality, or learned diagnostics, then ignoring `grad_theta alpha_t` makes the update a heuristic rather than the gradient of a clean scalar likelihood.

## 7. Parallelization Angle

The promising scaling route is not to make one Markov chain non-sequential. The dependence of `x_{t+1}` on `x_t` is real. The scalable route is to run many short chains in parallel and use all intermediate states:

```text
P chains x_{p,1:T}
negative phase = sum_t alpha_t mean_p grad_theta E_theta(x_{p,t})
```

This maps well to GPUs and molecular dynamics kernels. The tradeoff is that it exchanges asymptotic sampling correctness for throughput and biased but diverse negative-phase coverage.

## 8. Literature Grounding

- Hinton, 2002, "Training Products of Experts by Minimizing Contrastive Divergence": introduces CD as short-chain contrastive training for products of experts. Source: https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf
- Tieleman, 2008, "Training Restricted Boltzmann Machines using Approximations to the Likelihood Gradient": introduces persistent CD/fantasy particles to reduce CD bias. Source: https://www.cs.toronto.edu/~tijmen/pcd/pcd.pdf
- Hyvarinen, 2005, "Estimation of Non-Normalized Statistical Models by Score Matching": avoids partition-function evaluation through score matching. Source: https://www.jmlr.org/papers/v6/hyvarinen05a.html
- Gutmann and Hyvarinen, 2010, "Noise-Contrastive Estimation": learns unnormalized models through classification against noise. Source: https://proceedings.mlr.press/v9/gutmann10a.html
- Welling and Teh, 2011, "Bayesian Learning via Stochastic Gradient Langevin Dynamics": connects stochastic gradients and Langevin sampling. Source: https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf
- Sohl-Dickstein et al., 2011/2012, "Minimum Probability Flow Learning": trains unnormalized models by moving probability mass away from data for infinitesimal time. Source: https://arxiv.org/abs/1206.1106
- Song and Ermon, 2019, "Generative Modeling by Estimating Gradients of the Data Distribution": score-based generative modeling with annealed Langevin dynamics. Source: https://papers.nips.cc/paper/2019/hash/3001ef257407d5a371a96dcd947c7d93-Abstract.html
- Song et al., 2021, "Score-Based Generative Modeling through Stochastic Differential Equations": continuous-time score-based view. Source: https://openreview.net/forum?id=PxTIG12RRHS
- Du and Mordatch, 2019, "Implicit Generation and Modeling with Energy Based Models": modern neural EBM generation with Langevin dynamics. Source: https://arxiv.org/abs/1903.08689
- Nijkamp et al., 2019, "Learning Non-Convergent Non-Persistent Short-Run MCMC toward Energy-Based Model": treats short-run MCMC as a generator-like process. Source: https://arxiv.org/abs/1904.09770
- Romero Merino et al., 2018/2019, "Weighted Contrastive Divergence": weights the negative phase in RBM CD. Source: https://arxiv.org/abs/1801.02567
- Luo et al., 2023, "Training Energy-Based Models with Diffusion Contrastive Divergences": interprets CD through diffusion processes and addresses short-run MCMC bias. Source: https://arxiv.org/abs/2307.01668
- Liu and Wang, 2016, "Stein Variational Gradient Descent": deterministic interacting particles as a parallelizable approximate inference method. Source: https://arxiv.org/abs/1608.04471

## Bottom Line

The partition function can be bypassed computationally but not conceptually. MCMC can be replaced or amortized, but Langevin is still MCMC. The dynamic weighting idea is worth testing as a biased, massively parallel negative-phase estimator. The cleanest hypothesis is not that bad samples become positives; it is that a weighted mixture of early hard negatives and later model-like negatives may reduce CD-1 bias without paying for full mixing.
