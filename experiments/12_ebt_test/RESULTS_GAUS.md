# Gaussian Energy / Langevin Mini Experiment Results

## Current status

The previous high-dimensional entries for `1024D` and `4096D` were removed because they were invalid for the intended convergence test: particles started inside the success threshold due to tiny `target_radius` and `init_scale` values.

Going forward, successful results should require random starts that are independent of targets and not trivially inside `success_threshold` at initialization.

## Valid 38D baseline result with exactly 20 Langevin timestamps

Constraint: `--langevin-steps 20`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run test_gaussian_langevin_35d.py \
  --dim 38 \
  --num-targets 3 \
  --target-radius 0.25 \
  --attenuation-dim 4 \
  --train-steps 1500 \
  --log-every 500 \
  --device cpu \
  --starts-per-target 4 \
  --langevin-steps 20 \
  --langevin-step-size 0.2 \
  --langevin-noise 0.0 \
  --langevin-final-step-size 0.02 \
  --langevin-final-noise 0.0 \
  --samples-per-target 256 \
  --beta 50 \
  --gaussian-alpha 1 \
  --init-scale 0.05 \
  --seed 17
```

Final output at `step=1500`:

```text
dist=0.3934->0.3058
nearest_dist=0.2880
energy=-3.2207->-3.2422
grad_align=0.3603
grad_agree=0.583
max_dist=0.3371
assigned_success=1.000
nearest_success=1.000
```

Caveat: this 38D result also uses a relatively small initialization scale. It is retained as historical context, not as evidence for the stricter high-dimensional far-start requirement.

## Invalidated results

Removed from this file:

- `1024D` result with `target_radius=0.05`, `init_scale=0.0005`
- `4096D` result with `target_radius=0.01`, `init_scale=0.0001`

Reason: particles were already within `success_threshold=0.5` at step 1, so those runs did not demonstrate convergence.
