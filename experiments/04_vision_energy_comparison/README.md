# `04_vision_energy_comparison` -- vision derisking for AR/EBM/diffusion ideas

This experiment is the fast vision rung for the Free Energy project.

The goal is not to claim MNIST is the final benchmark. The goal is to build a
cheap, reproducible scaffold where we can test whether prototype energy/free
energy architectures behave sensibly on images before spending real GPU budget.

## Best current recommendation

Use a benchmark ladder rather than one dataset:

1. **Digits / MNIST smoke.** Fastest iteration, catches broken training loops.
2. **FashionMNIST.** Same shape as MNIST, less saturated, still cheap.
3. **CIFAR-10 or SVHN.** More realistic color/natural-image stress test.
4. **Later dataset to decide with Kesh/Zane.** Use their vision judgment before
   making a paper claim.

For baselines, compare:

| Family | Baseline here | Why it matters |
|---|---|---|
| Standard discriminative vision | small CNN | sanity floor; should work immediately |
| Transformer vision baseline | tiny ViT/patch transformer | closest vision analogue to transformer-style sequence processing |
| Normal EBM | conditional energy classifier | tests whether energy scoring is at least sane on labels |
| Novel EBM | contrastive/refinement energy classifier | prototype hook for this project's proposed model |
| Diffusion | tiny DDPM-style denoiser | required generative/iterative baseline in the pipeline |

This makes the EBM comparison believable because the paper can say what each
method optimizes and where the compute goes, instead of comparing one custom
model to a weak straw baseline.

## Run locally

```bash
cd /Users/brandomiranda/free-energy
python experiments/04_vision_energy_comparison/run_vision_benchmark.py --smoke
python experiments/04_vision_energy_comparison/run_vision_benchmark.py --tag local_real --epochs 5
```

The default dataset is `digits`, the built-in 8x8 sklearn handwritten-digit
dataset. Use it as a MNIST proxy for smoke tests. Use `--dataset mnist` only when
`torchvision` and the MNIST files are available.

## Run on SNAP

```bash
cd /Users/brandomiranda/free-energy
./experiments/04_vision_energy_comparison/run_on_snap.sh both
```

Defaults:

- host: `skampere2.stanford.edu`
- user: `brando9`
- device: `cuda`
- remote base: `/dfs/scratch0/brando9/free-energy/experiments/04_vision_energy_comparison`

Results are pulled back into:

```text
experiments/04_vision_energy_comparison/results/
```

## What counts as success

For now:

- CNN and tiny ViT train above chance on the digits/MNIST smoke.
- Conditional EBM trains above chance and has a positive energy margin.
- Novel contrastive EBM does not collapse.
- Diffusion denoiser test MSE decreases below the initial/noisy baseline.
- SNAP smoke and real runs produce JSON and markdown reports.

For the paper:

- MNIST/FashionMNIST is only a method check.
- CIFAR-10/SVHN or a Kesh-recommended dataset is needed before claiming a real
  vision result.

## Reviewer asks

- `@eobbad`: does this toy/vision setup actually expose the AR/EBM pros and cons
  we care about, or is it too classification-heavy?
- `@Srivatsava`: check whether the split/metric discipline mirrors the VeriBench
  protocol closely enough for the data-centric paper.
- `@brando90`: decide whether to promote this from smoke scaffold to paper result.
