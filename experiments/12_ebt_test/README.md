# 12_ebt_test — Synthetic Modus Ponens Chains

This environment is a local, synthetic replacement for the Lean-dependent
`11_ebt_leandojo` workflow.

## What it contains

- `generate_dataset.py` creates a chain dataset with:
  - facts: `fact: A`
  - rules: `rule: A -> B`
  - target sequence: `A -> B -> C`
  - no query token in the input prompt
- vocabulary is local and explicit (~35 tokens): `A-Z`, `->`, `|`, `fact:`, `rule:`, `query:` plus `<pad>/<eos>/<bos>/<unk>`
- trainable models:
  - `train_ebt.py` uses an EBT-style optimizer
- `train_transformer.py` is a standard autoregressive baseline
- WandB tracking is supported; set `WANDB_API_KEY` and pass `wandb.enabled=true`.

## Quick start

```bash
cd experiments/12_ebt_test
python generate_dataset.py --num-samples 10000
python dataloader.py
python train_ebt.py
python train_transformer.py
```

## Fast 10-minute control

The default model shape in both configs is a 4-layer, 128-dimensional transformer.

```bash
bash scripts/train_ebt_fast.sh
bash scripts/train_transformer_fast.sh
```

For explicit long-run CLI usage:

```bash
WANDB_API_KEY=... python train_ebt.py wandb.enabled=true
WANDB_API_KEY=... python train_transformer.py wandb.enabled=true
```

Both scripts now log:
- `teacher_forcing_token_accuracy` / `teacher_forcing_exact_accuracy`
- `autoregressive_token_accuracy` / `autoregressive_exact_accuracy`

## Notes

- `train_ebt.py` and `train_transformer.py` log to `runs/.../metrics.jsonl`
  and `runs/.../validation_metrics.jsonl`.
- Set `allow_cpu=true` for quick CPU sanity checks.
