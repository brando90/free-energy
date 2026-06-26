#!/usr/bin/env python3
"""VeriBench context-activation dataloader for EBT training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from veribench_task import DEFAULT_DATA_DIR, VeriBenchTask


def _load_vocab(data_dir: Path) -> dict[str, Any]:
    return json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))


class GoedelIdMapper:
    """Maps compact local dataset ids to original Goedel tokenizer ids."""

    local_to_original: torch.Tensor
    model_name: str
    vocab_size: int
    pad_original_id: int
    eos_original_id: int

    def __init__(
        self,
        *,
        model_name: str,
        vocab: dict[str, Any],
        revision: str | None = None,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        eos_original_id = int(tokenizer.eos_token_id)
        pad_original_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_original_id)
        bos_original_id = int(tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_original_id)
        unk_original_id = int(tokenizer.unk_token_id if tokenizer.unk_token_id is not None else eos_original_id)

        id_to_original = vocab["id_to_original"]
        local_to_original = torch.full((len(id_to_original),), unk_original_id, dtype=torch.long)
        for local_id, original_id in enumerate(id_to_original):
            if original_id is not None:
                local_to_original[local_id] = int(original_id)
        local_to_original[int(vocab["local_pad_id"])] = pad_original_id
        local_to_original[int(vocab["local_bos_id"])] = bos_original_id
        local_to_original[int(vocab["local_eos_id"])] = eos_original_id
        local_to_original[int(vocab["local_unk_id"])] = unk_original_id

        self.local_to_original = local_to_original
        self.model_name = model_name
        self.vocab_size = int(len(tokenizer))
        self.pad_original_id = pad_original_id
        self.eos_original_id = eos_original_id

    def original_ids(self, local_ids: torch.Tensor) -> torch.Tensor:
        local_ids = local_ids.to(dtype=torch.long, device=self.local_to_original.device)
        return self.local_to_original.index_select(0, local_ids.reshape(-1)).reshape(local_ids.shape)


class VeriBenchEmbeddingDataset(Dataset[dict[str, Any]]):
    """Dataset over VeriBenchTask samples with Goedel-token-id targets."""

    def __init__(
        self,
        *,
        data_dir: Path = DEFAULT_DATA_DIR,
        split: str | list[str] | tuple[str, ...] | None = None,
        families: list[str] | tuple[str, ...] | str | None = None,
        max_items: int | None = None,
        max_target_tokens: int | None = None,
        activation_dtype: str = "bf16",
        model_name: str | None = None,
        model_revision: str | None = None,
        validate_context: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.vocab = _load_vocab(self.data_dir)
        self.model_name = model_name or str(self.vocab["tokenizer"])
        self.max_target_tokens = max_target_tokens
        self.tasks = self._load_tasks(
            split=split,
            families=families,
            max_items=max_items,
            activation_dtype=activation_dtype,
            validate_context=validate_context,
        )
        self.id_mapper = GoedelIdMapper(
            model_name=self.model_name,
            revision=model_revision,
            vocab=self.vocab,
        )

    def _load_tasks(
        self,
        *,
        split: str | list[str] | tuple[str, ...] | None,
        families: list[str] | tuple[str, ...] | str | None,
        max_items: int | None,
        activation_dtype: str,
        validate_context: bool,
    ) -> list[VeriBenchTask]:
        tasks: list[VeriBenchTask] = []
        split_filter: set[str] | None = None
        if split is not None:
            if isinstance(split, str):
                split_filter = {split}
            else:
                split_filter = {str(item) for item in split}
        family_filter: set[str] | None = None
        if families:
            if isinstance(families, str):
                family_filter = {families}
            else:
                family_filter = {str(family) for family in families}
        for task in VeriBenchTask.iter_tasks(
            split=None,
            data_dir=self.data_dir,
            activation_dtype=activation_dtype,
        ):
            if split_filter is not None and str(task.split) not in split_filter:
                continue
            if family_filter is not None and str(task.family or "") not in family_filter:
                continue
            if not task.context_activations_path.exists():
                continue
            if validate_context:
                try:
                    with safe_open(task.context_activations_path, framework="pt", device="cpu") as handle:
                        if "hidden_states" not in handle.keys():
                            continue
                except Exception:
                    continue
            tasks.append(task)
            if max_items is not None and len(tasks) >= max_items:
                break
        return tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def __getitem__(self, index: int) -> dict[str, Any]:
        task = self.tasks[index]
        sample = task.as_ebt_sample()
        labels = sample["labels"]
        decoder_input_ids = sample["decoder_input_ids"]
        if self.max_target_tokens is not None:
            max_target_tokens = int(self.max_target_tokens)
            labels = labels[:max_target_tokens]
            decoder_input_ids = decoder_input_ids[:max_target_tokens]

        return {
            "task_name": task.task_name,
            "task_index": torch.tensor(index, dtype=torch.long),
            "split": task.split,
            "family": task.family or "",
            "context_activations": sample["context_activations"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_input_original_ids": self.id_mapper.original_ids(decoder_input_ids),
            "labels": labels,
            "label_original_ids": self.id_mapper.original_ids(labels),
            "context_token_count": sample["context_token_count"],
            "target_token_count": torch.tensor(labels.shape[0], dtype=torch.long),
        }


def collate_veribench_embedding_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    batch = len(samples)
    hidden_dim = int(samples[0]["context_activations"].shape[1])
    max_context = max(int(sample["context_activations"].shape[0]) for sample in samples)
    max_target = max(int(sample["labels"].shape[0]) for sample in samples)
    max_decoder = max(int(sample["decoder_input_ids"].shape[0]) for sample in samples)

    context_dtype = samples[0]["context_activations"].dtype
    context = torch.zeros(batch, max_context, hidden_dim, dtype=context_dtype)
    context_mask = torch.zeros(batch, max_context, dtype=torch.bool)
    label_mask = torch.zeros(batch, max_target, dtype=torch.bool)
    decoder_mask = torch.zeros(batch, max_decoder, dtype=torch.bool)
    labels = torch.full((batch, max_target), -100, dtype=torch.long)
    label_original_ids = torch.full((batch, max_target), -100, dtype=torch.long)
    decoder_input_ids = torch.zeros(batch, max_decoder, dtype=torch.long)
    decoder_input_original_ids = torch.zeros(batch, max_decoder, dtype=torch.long)

    for i, sample in enumerate(samples):
        c_len = int(sample["context_activations"].shape[0])
        t_len = int(sample["labels"].shape[0])
        d_len = int(sample["decoder_input_ids"].shape[0])
        context[i, :c_len] = sample["context_activations"]
        context_mask[i, :c_len] = True
        label_mask[i, :t_len] = True
        decoder_mask[i, :d_len] = True
        labels[i, :t_len] = sample["labels"]
        label_original_ids[i, :t_len] = sample["label_original_ids"]
        decoder_input_ids[i, :d_len] = sample["decoder_input_ids"]
        decoder_input_original_ids[i, :d_len] = sample["decoder_input_original_ids"]

    return {
        "task_name": [sample["task_name"] for sample in samples],
        "task_index": torch.stack([sample["task_index"] for sample in samples]),
        "split": [sample["split"] for sample in samples],
        "family": [sample["family"] for sample in samples],
        "context_activations": context,
        "context_attention_mask": context_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_input_original_ids": decoder_input_original_ids,
        "decoder_attention_mask": decoder_mask,
        "labels": labels,
        "label_original_ids": label_original_ids,
        "label_attention_mask": label_mask,
        "context_token_count": torch.stack([sample["context_token_count"] for sample in samples]),
        "target_token_count": torch.stack([sample["target_token_count"] for sample in samples]),
    }


def make_veribench_embedding_dataloader(
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    **dataset_kwargs: Any,
) -> DataLoader[dict[str, Any]]:
    dataset = VeriBenchEmbeddingDataset(**dataset_kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_veribench_embedding_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=2)
    args = parser.parse_args()

    loader = make_veribench_embedding_dataloader(
        data_dir=args.data_dir,
        split=args.split,
        families=args.families,
        max_items=args.max_items,
        batch_size=args.batch_size,
    )
    batch = next(iter(loader))
    print(f"dataset_size={len(loader.dataset)}")
    print(f"task_name={batch['task_name']}")
    print(f"context_activations={tuple(batch['context_activations'].shape)} {batch['context_activations'].dtype}")
    print(f"labels={tuple(batch['labels'].shape)} {batch['labels'].dtype}")
    print(f"label_original_ids={tuple(batch['label_original_ids'].shape)} {batch['label_original_ids'].dtype}")
    print(f"decoder_input_original_ids={tuple(batch['decoder_input_original_ids'].shape)} {batch['decoder_input_original_ids'].dtype}")


if __name__ == "__main__":
    main()
