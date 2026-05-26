# Vision Energy Comparison Run: `snap_smoke`

Dataset: `digits`
Seed: `0`
Device: `cuda`

## Results

| model | objective | accuracy | loss / mse | energy margin |
|---|---|---:|---:|---:|
| cnn | cross_entropy | 0.1250 | 2.2989 |  |
| tiny_vit | cross_entropy | 0.1094 | 2.3432 |  |
| ebm | energy_ce | 0.1875 | 2.2944 | -0.0825 |
| novel_ebm | energy_ce | 0.1016 | 2.2949 | -0.0771 |
| diffusion | ddpm_noise_prediction |  | 0.9956 |  |

Plot: `results/snap_smoke/accuracy.png`

## Interpretation

This is a derisking scaffold. Digits/MNIST results are not a publishable vision claim by themselves.
They are a check that CNN, tiny ViT, EBM-style, novel-EBM-style, and diffusion objectives all run under one protocol.
