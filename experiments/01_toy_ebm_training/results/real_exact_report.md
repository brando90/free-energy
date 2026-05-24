# Toy EBM Training Report

Status: `pass`

## Config

```json
{
  "batch_size": 8,
  "device": "cpu",
  "device_resolved": "cpu",
  "epochs": 80,
  "grad_clip": 5.0,
  "hidden_dim": 64,
  "log_every": 10,
  "lr": 0.003,
  "models": [
    "linear",
    "mlp",
    "cnn",
    "resnet",
    "transformer"
  ],
  "num_heads": 4,
  "num_layers": 2,
  "num_test_tasks": 16,
  "num_train_tasks": 48,
  "output_dir": "/Users/brandomiranda/free-energy/experiments/01_toy_ebm_training/results",
  "pair_batch_size": 8192,
  "require_improvement": true,
  "seed": 17,
  "seq_len": 9,
  "support_size": 512,
  "tag": "real_exact",
  "target_temperature": 1.0,
  "weight_decay": 0.0001
}
```

## Uniform Baseline

- test KL(p_star || uniform): `0.2149`
- test TV distance: `0.2597`
- test NLL: `6.2383`

## Held-Out Results

| model | params | test KL | test TV | test NLL | target mode rank | mode match | seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 37 | 0.1077 | 0.1987 | 6.1311 | 13.06 | 0.44 | 1.1 |
| mlp | 6593 | 0.0742 | 0.1900 | 6.0976 | 1.69 | 0.56 | 1.4 |
| cnn | 25025 | 0.0810 | 0.1900 | 6.1044 | 2.06 | 0.50 | 228.0 |
| resnet | 62593 | 0.0741 | 0.1900 | 6.0975 | 1.81 | 0.56 | 560.1 |
| transformer | 67841 | 0.0894 | 0.1902 | 6.1128 | 4.50 | 0.44 | 46.5 |

Lower KL/TV/NLL and lower target-mode rank are better. Higher mode match is better.
