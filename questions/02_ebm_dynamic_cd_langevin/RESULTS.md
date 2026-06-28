# Results - Dynamic Weighted CD for Continuous EBMs

## Answer So Far

Claude's response is useful as a roadmap, but not as a drop-in spec. The most
important correction is that later Langevin/CD states should be treated as
higher-confidence negative-phase samples, not positive examples, unless the
objective is explicitly changed.

The dynamic weighting idea is worth continuing. On the initial 5-seed SNAP run,
dynamic weighted CD preserved full mode coverage and improved sample proximity
to the eight-Gaussians modes, at higher per-step compute.

## SNAP 5-Seed Run

Command:

```bash
cd questions/02_ebm_dynamic_cd_langevin/expt_v1
./run_on_snap.sh snap_5seed
```

Host and device:

- `skampere2.stanford.edu`
- `NVIDIA H200`, GPU 0
- seeds `0 1 2 3 4`
- `800` training steps per seed

Aggregate:

| Method | Coverage | Entropy | TV to Uniform | Nearest Dist | Radial Error | Train s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CD-1 | 8.00 +/- 0.00 | 0.996 +/- 0.002 | 0.053 +/- 0.017 | 0.902 +/- 0.010 | 0.756 +/- 0.013 | 2.31 +/- 0.46 |
| Dynamic weighted CD | 8.00 +/- 0.00 | 0.996 +/- 0.001 | 0.051 +/- 0.017 | 0.783 +/- 0.022 | 0.630 +/- 0.025 | 5.21 +/- 0.28 |

Interpretation:

- Both methods covered all eight modes.
- Dynamic weighted CD improved mean nearest-mode distance by `0.120`.
- Dynamic weighted CD improved radial error by `0.126`.
- Mode balance was roughly tied.
- Runtime was about `2.25x` CD-1 for this implementation.

Artifacts:

- `expt_v1/results/snap_5seed/aggregate.md`
- `expt_v1/results/snap_5seed/aggregate.json`
- `expt_v1/results/snap_5seed/seed_*/report.json`
- `expt_v1/results/snap_5seed/seed_*/energy_surfaces.png`
- `expt_v1/results/snap_5seed/seed_*/samples.png`

## Local Runs

- Smoke test: `expt_v1/results/smoke/`
- Local CPU default: `expt_v1/results/`

The local CPU default was one seed and should be treated only as a sanity check.
The SNAP 5-seed run is the first result to cite.

## Next Experiment

Run a schedule sweep:

```text
T in {1, 2, 4, 8, 16}
ramp in {0, 1, 2, 4}
seeds in {0, 1, 2, 3, 4}
```

The decision criterion should be matched energy evaluations, not just matched
optimizer steps, because dynamic weighted CD uses more Langevin work per update.
