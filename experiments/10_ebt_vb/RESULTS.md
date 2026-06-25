# Results

## Goedel-Prover-V2-8B evaluation (parallel compile, 32 workers)

- Run: `evaluate_veribench_outputs_from_safetensors.py`
- Run date: `2026-06-24`
- Workers: `32`
- Input manifest: `experiments/10_ebt_vb/data/context_gold/manifest.jsonl`
- Source outputs: `experiments/09_vb_testing_ipynb/results/goedel_prover_v2_8b_sglang_896_hidden_states_gpus0_5_full_884_bs16`
- Output directory: `experiments/10_ebt_vb/results/veribench_output_eval_workers32`

The input manifest currently contains **689** rows (train/val/test split counts are 550/65/74).

| Split | Rows | IC1 | IC2 | TE1 | D1 | D2 | S_tilde |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 550 | 0.120000 | 0.005455 | 0.000000 | 0.970909 | 0.926571 | 0.000000 |
| val | 65 | 0.076923 | 0.000000 | 0.000000 | 0.984615 | 0.936190 | 0.000000 |
| test | 74 | 0.094595 | 0.000000 | 0.000000 | 0.986486 | 0.921686 | 0.000000 |
| all | 689 | 0.113208 | 0.004354 | 0.000000 | 0.973875 | 0.926954 | 0.000000 |

## Output files

- `experiments/10_ebt_vb/results/veribench_output_eval_workers32/veribench_output_scores.csv`
- `experiments/10_ebt_vb/results/veribench_output_eval_workers32/veribench_output_aggregate.csv`

## EBT validation-overfit trainer

- Run date: `2026-06-24`
- Trainer: `experiments/10_ebt_vb/train_ebt.py`
- Config: `experiments/10_ebt_vb/configs/train_config.yaml`
- Run directory: `experiments/10_ebt_vb/runs/val_overfit/20260624_154937`
- Dataset split: `val`
- Validation rows: `65`
- Vocab: compact VeriBench dataset vocab, `6724` tokens
- Target cap: `2048` tokens per sample
- Steps: `1000`
- GPU: one A100 via `CUDA_VISIBLE_DEVICES=0`

The initial full-Goedel-vocab setup used dense EBT states shaped like `[batch, target_tokens, 151669]`.
That OOMed on one 80GB A100, including with batch size 1, because validation contains long gold Lean targets
up to `8100` tokens. The trainer now defaults to the compact VeriBench token vocabulary and trains on capped
gold-prefix targets so every validation task is still represented.

| Metric | Value |
|---|---:|
| Peak CUDA memory | 45.55 GB |
| Sampled GPU utilization | 89% |
| Best logged loss | 9.1331 |
| Best-loss step | 840 |
| Best logged final-step loss | 9.0984 |
| Final logged loss | 9.3183 |
| Final logged final-step loss | 9.3241 |
| Final logged perplexity | 11205.24 |
| Final logged throughput | 19501.16 target tokens/sec |

Artifacts:

- `experiments/10_ebt_vb/runs/val_overfit/20260624_154937/metrics.jsonl`
- `experiments/10_ebt_vb/runs/val_overfit/20260624_154937/summary.json`
- `experiments/10_ebt_vb/runs/val_overfit/20260624_154937/checkpoint_final.pt`

Conclusion: the trainer is functional and stable on one A100, but this default run did not strongly overfit
the validation split. Loss improved modestly around the best checkpoint and then regressed by the final logged
batch.

## How the EBT overfit experiments are run

The recent overfit checks use the cloned Alexi EBT code under:

- `experiments/10_ebt_vb/alexiglad_EBT/train_veribench_overfit.py`
- `experiments/10_ebt_vb/alexiglad_EBT/sample_veribench_overfit.py`

Common setup:

- Dataset: first `5` examples from the VeriBench `val` split.
- Compact target vocabulary: `6724` local VeriBench/Goedel tokens from `data/context_gold/vocab.json`.
- Target format: decoder input is `BOS + gold Lean prefix`; labels are the corresponding next gold Lean tokens.
- Sampling: autoregressive greedy decoding with `temperature=0`, using local token ids and decoding back through the Goedel tokenizer mapping.
- Goedel context ablation: `--no-context` zeros the projected Goedel hidden-state conditioning, so the model sees only the decoder sequence.
- Architecture unless otherwise stated: `6` EBT transformer layers, `6` heads, `hidden_dim=384`.
- Optimizer settings unless otherwise stated: `AdamW`, `lr=0.0012`, alpha lr multiplier `1.5`, learnable MCMC step size, `mcmc_steps=2`.
- Stochasticity: training uses random model initialization and random dense-vocab MCMC initial conditions; greedy sampling is deterministic apart from CUDA nondeterminism.

## Alexi EBT 5-example overfit and sampling results

These runs were intended as sanity checks: if the EBT setup is healthy, it should memorize five validation examples, especially when the gold proof is truncated to five tokens.

### Full 256-token targets with Goedel context

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps ... --max-items 5 --max-target-tokens 256 --batch-size 5 \
  --hidden-dim 384 --num-layers 6 --num-heads 6
```

| Run | Variant | Best logged loss | Sampled checkpoint | Exact recovered | Token matches |
|---|---|---:|---:|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_124131` | normal context-conditioned | 9.2092 | 600 | 0/5 | 1/1280 |
| `runs/alexiglad_ebt_veribench5/20260625_124614` | no MCMC detach, larger MCMC step | 9.2355 | 300 | 0/5 | 0/1280 |

Conclusion: the context-conditioned EBT did not recover the 5 examples. Generations were random-looking tokenizer fragments, not Lean code.

### Larger hidden dimension

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps 1000 --max-items 5 --max-target-tokens 256 --batch-size 5 \
  --hidden-dim 1024 --num-layers 6 --num-heads 8
```

`1024` is not divisible by `6`, so this run used `8` heads.

| Run | Best logged loss | Sampled checkpoints | Exact recovered | Token matches |
|---|---:|---|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_132907` | 9.2288 | 200, 600 | 0/5, 0/5 | 0/1280, 0/1280 |

Conclusion: increasing hidden size to `1024` did not improve recovery.

### Full 256-token targets without Goedel context

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps 1000 --max-items 5 --max-target-tokens 256 --batch-size 5 \
  --hidden-dim 384 --num-layers 6 --num-heads 6 --no-context
```

| Run | Best logged loss | Sampled checkpoints | Exact recovered | Token matches |
|---|---:|---|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_134438` | 9.2006 | 100, 400 | 0/5, 0/5 | 0/1280, 0/1280 |

Conclusion: removing Goedel activations did not fix the failure to memorize.

### Five-token target truncation without Goedel context

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python alexiglad_EBT/train_veribench_overfit.py \
  --max-items 5 --max-target-tokens 5 --batch-size 5 \
  --hidden-dim 384 --num-layers 6 --num-heads 6 --no-context
```

| Run | Steps | Best logged loss | Sampled checkpoints | Exact recovered | Token matches |
|---|---:|---:|---|---:|---:|
| `runs/alexiglad_ebt_veribench5/20260625_135002` | 368 interrupted | 8.7603 | 200, 300 | 0/5, 0/5 | 0/25, 0/25 |
| `runs/alexiglad_ebt_veribench5/20260625_135231` | 10000 | 4.4294 at step 9965 | 7000, 10000 | 0/5, 0/5 | 4/25, 6/25 |

The 10k-step run used exactly:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run python alexiglad_EBT/train_veribench_overfit.py \
  --steps 10000 --max-items 5 --max-target-tokens 5 --batch-size 5 \
  --log-every 100 --save-every 1000 \
  --hidden-dim 384 --num-layers 6 --num-heads 6 \
  --warmup-steps 50 --lr 0.0012 --no-context
```

10k-step details:

| Metric | Value |
|---|---:|
| Final loss | 7.7497 |
| Final final-step loss | 7.7151 |
| Best observed loss | 4.4294 |
| Best observed final-step loss | 4.1443 |
| Best observed step | 9965 |
| Best saved checkpoint by loss | 7000 |
| Peak CUDA memory | 0.38 GB |

Saved sampling outputs:

- `runs/alexiglad_ebt_veribench5/20260625_135231/sample_checkpoint_step_007000.jsonl`
- `runs/alexiglad_ebt_veribench5/20260625_135231/sample_checkpoint_step_010000.jsonl`

Conclusion: longer training improved token-level recovery from `0/25` to `6/25`, but still did not recover any complete 5-token example. Outputs mostly repeated frequent Lean-ish tokens such as `import`, `Std`, and `_option`.
