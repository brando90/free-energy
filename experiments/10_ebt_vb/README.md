# experiments/10_ebt_vb

Utilities for running Goedel Prover / VeriBench preparation experiments.

Included in this folder:

- `veribench_dataloader.py`: Hugging Face-backed prompt tokenizer and dataset
  loader for VeriBench tasks.
- `convert_hidden_states_to_safetensors.py`: converts SGLang hidden-state JSON
  into compact per-task `.safetensors` tensors.
- `prepare_context_gold_dataset.py`: builds a context-only activation dataset
  with gold VeriBench Lean target tokens and a compact local vocabulary.
- `veribench_context_gold_dataloader.py`: pads context activations and gold
  autoregressive target labels for training.
- `train_context_gold_ebt.py`: small EBT-style overfit trainer conditioned on
  context activations only.
- `veribench_task.py`: one-task abstraction that binds prompt token count, Goedel
  8B hidden-state activations, and gold target Lean local tokens; includes
  helpers to create a single EBT sample.
- `veribench_embedding_dataloader.py`: wraps `VeriBenchTask` samples and returns
  Goedel context activations plus cleaned gold target token ids.
- `ebt.py`: scaffold EBT module that internally optimizes dense full-vocab token
  logits, maps them through a learned `vocab_to_embed` projection, and feeds an
  `nn.TransformerDecoder`.

Typical context/gold preparation:

```bash
uv run python prepare_context_gold_dataset.py --splits train val test --exclude-zero-context
uv run python veribench_context_gold_dataloader.py --split val --batch-size 2
```

Validation overfit smoke:

```bash
uv run python train_context_gold_ebt.py --train-split val --eval-split val --max-steps 10 --eval-every 5
```

Single task example:

```bash
uv run python - <<'PY'
from pathlib import Path
from veribench_task import VeriBenchTask

task = next(VeriBenchTask.iter_tasks(split="val", data_dir=Path("data/context_gold")))
sample = task.as_ebt_sample()
print(task.task_name, sample["context_activations"].shape, sample["labels"].shape)
PY
```

EBT dataloader smoke:

```bash
uv run python veribench_embedding_dataloader.py --split val --batch-size 2 --max-items 2
```
