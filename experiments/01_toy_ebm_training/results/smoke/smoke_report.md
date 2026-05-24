# Toy EBM Training Report

Status: `pass`

## Config

```json
{
  "batch_size": 4,
  "device": "cpu",
  "device_resolved": "cpu",
  "epochs": 20,
  "grad_clip": 5.0,
  "hidden_dim": 32,
  "log_every": 10,
  "lr": 0.01,
  "models": [
    "linear",
    "mlp"
  ],
  "num_heads": 4,
  "num_layers": 2,
  "num_test_tasks": 4,
  "num_train_tasks": 12,
  "output_dir": "experiments/01_toy_ebm_training/results/smoke",
  "pair_batch_size": 512,
  "require_improvement": true,
  "seed": 17,
  "seq_len": 6,
  "support_size": 64,
  "tag": "smoke",
  "target_temperature": 1.0,
  "weight_decay": 0.0001
}
```

## Uniform Baseline

- test KL(p_star || uniform): `0.2912`
- test TV distance: `0.3070`
- test NLL: `4.1589`

## Held-Out Results

| model | params | test KL | test TV | test NLL | target mode rank | mode match | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 25 | 0.1200 | 0.2040 | 3.9878 | 1.00 | 1.00 | 0.7 |
| mlp | 1889 | 0.1084 | 0.1965 | 3.9761 | 1.00 | 1.00 | 0.6 |

Lower KL/TV/NLL and lower target-mode rank are better. Higher mode match is better.
