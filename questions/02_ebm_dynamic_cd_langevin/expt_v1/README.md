# Experiment v1 - Dynamic Weighted CD

This experiment compares CD-1 with a dynamic weighted CD variant on a 2D eight-Gaussians target.

## Run

Smoke test:

```bash
uv run python questions/02_ebm_dynamic_cd_langevin/expt_v1/src/dynamic_weighted_cd.py \
  --steps 80 \
  --batch-size 128 \
  --grid-size 80 \
  --eval-samples 512 \
  --out-dir questions/02_ebm_dynamic_cd_langevin/expt_v1/results/smoke
```

Longer local run:

```bash
uv run python questions/02_ebm_dynamic_cd_langevin/expt_v1/src/dynamic_weighted_cd.py \
  --steps 800 \
  --batch-size 256 \
  --langevin-steps 8 \
  --out-dir questions/02_ebm_dynamic_cd_langevin/expt_v1/results
```

SNAP 5-seed run:

```bash
cd questions/02_ebm_dynamic_cd_langevin/expt_v1
./run_on_snap.sh snap_5seed
```

Useful overrides:

```bash
HOST=skampere2.stanford.edu GPU=0 SEEDS="0 1 2 3 4" STEPS=800 ./run_on_snap.sh snap_5seed
```

Current SNAP aggregate:

| Method | Coverage | Entropy | TV to Uniform | Nearest Dist | Radial Error | Train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CD-1 | 8.00 +/- 0.00 | 0.996 +/- 0.002 | 0.053 +/- 0.017 | 0.902 +/- 0.010 | 0.756 +/- 0.013 | 2.31 +/- 0.46 |
| Dynamic weighted CD | 8.00 +/- 0.00 | 0.996 +/- 0.001 | 0.051 +/- 0.017 | 0.783 +/- 0.022 | 0.630 +/- 0.025 | 5.21 +/- 0.28 |

## Outputs

- `report.json` - config and metrics.
- `energy_surfaces.png` - learned energy contours for each method.
- `samples.png` - target samples and long-run generated samples.
- `verdict.md` - one-paragraph result summary.
- `snap_*/aggregate.json` and `snap_*/aggregate.md` - multi-seed summaries
  from SNAP runs.

## Interpretation

Dynamic weighted CD is not an unbiased likelihood-gradient estimator unless the trajectory distribution has mixed to the model distribution or the weighting has a valid importance-correction interpretation. In this prototype it is intentionally treated as a scalable negative-phase heuristic.
