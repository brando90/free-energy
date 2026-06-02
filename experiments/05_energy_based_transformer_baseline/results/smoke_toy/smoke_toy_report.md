# Toy EBT Baseline Report

Status: `pass`

## Config

```json
{
  "alpha": 1.0,
  "batch_size": 32,
  "detach_between_steps": false,
  "device": "cpu",
  "device_resolved": "cpu",
  "direct_epochs": 20,
  "direct_lr": 0.003,
  "ebt_epochs": 60,
  "ebt_lr": 0.001,
  "ebt_steps": 2,
  "eval_batch_size": 64,
  "eval_samples": 1,
  "eval_steps": [
    1,
    2,
    4
  ],
  "hidden_dim": 48,
  "init_scale": 0.0,
  "log_every": 4,
  "max_grad_norm": 5.0,
  "num_heads": 4,
  "num_layers": 2,
  "num_test": 128,
  "num_train": 256,
  "output_dir": "experiments/05_energy_based_transformer_baseline/results/smoke_toy",
  "prediction_grad_clip": 1.0,
  "require_ebt_signal": true,
  "seed": 7,
  "seq_len": 4,
  "tag": "smoke",
  "weight_decay": 0.0001
}
```

## Held-Out Test Metrics

| model | loss | token acc | sequence acc | energy drop |
| --- | ---: | ---: | ---: | ---: |
| direct_transformer | 0.0005 | 1.000 | 1.000 | n/a |
| energy_based_transformer | 0.2912 | 0.846 | 0.523 | 5.5112 |

## EBT Test-Time Compute Sweep

| refinement steps | loss | token acc | sequence acc | energy drop |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.5874 | 0.656 | 0.164 | 3.9917 |
| 2 | 0.2912 | 0.846 | 0.523 | 5.5112 |
| 4 | 0.3371 | 0.867 | 0.469 | 10.3279 |

The direct transformer is the feed-forward baseline. The EBT result uses the same
task but predicts by descending the learned energy landscape over candidate logits.
