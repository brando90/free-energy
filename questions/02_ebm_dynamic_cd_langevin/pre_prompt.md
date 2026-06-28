# Pre-Prompt - Dynamic Weighted CD for Continuous EBMs

You are investigating whether continuous EBMs can be trained with short, highly parallel Langevin trajectories whose intermediate states are dynamically weighted. The target is not to prove that the partition function disappears; it is to separate three questions that are often conflated:

1. Do we need to evaluate `Z(theta)` explicitly?
2. Do we need samples from the normalized model distribution?
3. Can biased short-run samples still provide a useful scalable training signal?

The proposed hypothesis is:

```text
Use a trajectory x_1, ..., x_T from Langevin dynamics.
Give early states lower negative-phase weights because they are data-local and biased.
Give later states higher negative-phase weights because they are closer to model samples.
Train with a weighted negative phase instead of using only x_1 as in CD-1.
```

Important sign correction:

- In likelihood-gradient EBMs, data are positive-phase samples.
- Model or chain samples are negative-phase samples.
- Later Langevin samples should not become "positive examples" unless the training objective is changed. The safer prototype treats them as higher-confidence negative-phase samples.

Minimal executable test:

- Dataset: 2D eight-Gaussians.
- Models: identical MLP energy networks.
- Baselines:
  - CD-1 with one Langevin step from data.
  - Dynamic weighted CD with `T > 1` Langevin steps and increasing weights over the collected trajectory.
- Outputs:
  - energy surface plots,
  - generated samples from long-run Langevin,
  - mode coverage metrics,
  - short written verdict.
