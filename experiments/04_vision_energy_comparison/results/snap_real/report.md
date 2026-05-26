# Vision Energy Comparison Run: `snap_real`

Dataset: `digits`
Seed: `0`
Device: `cuda`

## Results

| model | objective | accuracy | loss / mse | energy margin |
|---|---|---:|---:|---:|
| cnn | cross_entropy | 0.6400 | 0.9679 |  |
| tiny_vit | cross_entropy | 0.4689 | 1.5935 |  |
| ebm | energy_ce | 0.8756 | 0.4441 | 2.0752 |
| novel_ebm | energy_ce | 0.9022 | 0.4354 | 2.1261 |
| diffusion | ddpm_noise_prediction |  | 0.7390 |  |

Plot: `results/snap_real/accuracy.png`

## Interpretation

This is a derisking scaffold. Digits/MNIST results are not a publishable vision claim by themselves.
They are a check that CNN, tiny ViT, EBM-style, novel-EBM-style, and diffusion objectives all run under one protocol.
