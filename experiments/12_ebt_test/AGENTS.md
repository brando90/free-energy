# AGENTS.md — 12_ebt_test

## Goal
Create a synthetic Modus Ponens/chain-composition workload:

- Input fact and implication rules in a compact token language.
 - Output the implied chain as a token sequence (e.g., `A -> B -> C`).
 - Input includes only `fact:` + rules (no `query:` token in the prompt).

This folder keeps the useful experiment structure from `11_ebt_leandojo`
(Hydra configs, train scripts, metric logging), but drops Lean/context
dependencies and keeps everything local.

## Top-level layout
- `generate_dataset.py` – generate a programmatic 10,000-example chain dataset.
- `tokenizer.py` – local 35-token vocabulary (`A`–`Z`, `->`, `|`, `fact:`, `rule:`, `query:` plus standard specials).
- `dataloader.py` – dataset + dataloader utilities.
- `ebt.py` – compact Energy-Based Transformer over local vocab.
- `train_ebt.py` – trainer for EBT.
- `train_transformer.py` – autoregressive baseline trainer with the same data.
- `configs/ebt_config.yaml` – EBT defaults.
- `configs/transformer_config.yaml` – AR baseline defaults.
- `scripts/train_ebt_fast.sh`, `scripts/train_transformer_fast.sh` – launch files for quick 10-min-style runs.

## Environment
- Python: >=3.11,<3.13.
- Current deps are in `pyproject.toml`.
- GPU recommended.
- `allow_cpu=true` is supported for sanity checks and tiny runs.

## Run recipes
- Generate data:
  - `python generate_dataset.py --num-samples 10000`
- Dataloader smoke test:
  - `python dataloader.py`
- EBT:
  - `python train_ebt.py`
- Baseline AR Transformer:
  - `python train_transformer.py`

WandB tracking:
- Add `WANDB_API_KEY=...` and run with `wandb.enabled=true`.
- Both trainers now log:
  - `teacher_forcing_token_accuracy`, `teacher_forcing_exact_accuracy`
  - `autoregressive_token_accuracy`, `autoregressive_exact_accuracy`
- Fast override for debugging:
  - `python train_ebt.py train.max_steps=250 loader.batch_size=16`
  - `python train_transformer.py train.max_steps=250 loader.batch_size=64`
