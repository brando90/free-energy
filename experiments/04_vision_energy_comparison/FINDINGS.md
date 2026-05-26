# Findings

## 2026-05-26 -- SNAP H200 smoke + real run

Command:

```bash
cd /Users/brandomiranda/free-energy
./experiments/04_vision_energy_comparison/run_on_snap.sh both
```

Remote environment:

- host: `skampere2.stanford.edu`
- GPU: `NVIDIA H200`
- Python: `/lfs/skampere2/0/brando9/miniconda/bin/python3`
- Torch: `2.5.1+cu124`

Artifacts:

- `results/snap_smoke/metrics.json`
- `results/snap_smoke/report.md`
- `results/snap_smoke/accuracy.png`
- `results/snap_real/metrics.json`
- `results/snap_real/report.md`
- `results/snap_real/accuracy.png`
- `logs/run_both_20260526T200127Z.log`

Real run (`digits`, 6 classifier epochs, 4 diffusion epochs):

| model | objective | accuracy | loss / mse | energy margin |
|---|---|---:|---:|---:|
| CNN | cross entropy | 0.6400 | 0.9679 | |
| tiny ViT | cross entropy | 0.4689 | 1.5935 | |
| conditional EBM | energy CE | 0.8756 | 0.4441 | 2.0752 |
| contrastive novel EBM prototype | energy CE + contrastive clean/noisy margin | 0.9022 | 0.4354 | 2.1261 |
| diffusion denoiser | DDPM noise prediction | | 0.7390 | |

Interpretation:

- The scaffold works on SNAP and writes reproducible artifacts.
- The EBM-style classifiers are not yet a fair architecture win claim: the CNN
  and tiny ViT baselines are deliberately tiny and undertrained in this smoke run.
- The useful result is that the full comparison harness runs end-to-end,
  including a diffusion/iterative baseline.
- Next step: make the vision ladder more credible with FashionMNIST and CIFAR-10
  or a Kesh/Zane-recommended dataset, then compute-matched baselines.
