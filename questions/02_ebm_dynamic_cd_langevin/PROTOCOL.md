# Protocol - CD-1 vs Dynamic Weighted CD on 2D EBMs

## Objective

Test whether a weighted mixture of short Langevin trajectory states improves a toy continuous EBM relative to CD-1.

## Dataset

Use an eight-Gaussians distribution in 2D:

- 8 equally spaced modes on a circle.
- Standard deviation `0.08`.
- Radius `2.0`.

## Models

Train two identical MLP energy networks:

- input dimension: 2
- hidden width: 128
- layers: 3 hidden layers
- activation: SiLU
- scalar energy output

## Methods

### CD-1

1. Sample minibatch `x_data`.
2. Initialize `x_0 = x_data + small_noise`.
3. Run one Langevin step.
4. Minimize:

```text
mean(E(x_data)) - mean(E(x_1)) + energy_l2
```

### Dynamic Weighted CD

1. Sample minibatch `x_data`.
2. Initialize `x_0 = x_data + small_noise`.
3. Run `T` Langevin steps and keep all `x_1, ..., x_T`; the default prototype uses `T = 8`.
4. Use increasing normalized weights:

```text
alpha_t proportional_to exp(ramp * (t - 1) / max(T - 1, 1))
```

5. Minimize:

```text
mean(E(x_data)) - sum_t alpha_t mean(E(x_t)) + energy_l2
```

## Metrics

Report:

- data energy,
- random-noise energy,
- generated-sample mode coverage,
- mode-count entropy,
- average distance to nearest true mode,
- runtime,
- generated energy-surface and sample plots.

## Verdict Criteria

Dynamic weighted CD is promising if, at equal training steps and similar compute, it shows:

- better mode coverage,
- lower average distance to true modes,
- less pathological energy surface,
- no obvious collapse to one or two modes.

CD-1 is favored if dynamic weighting only adds compute without improving those metrics.

## Non-Goals

- This protocol does not prove convergence.
- It does not establish molecular-scale usefulness.
- It does not compare against score matching, NCE, PCD, diffusion CD, or amortized samplers.
- It does not treat later chain states as positive examples; they remain negative-phase samples.
