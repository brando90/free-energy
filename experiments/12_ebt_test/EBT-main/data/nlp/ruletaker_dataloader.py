"""RuleTaker dataset adapter for causal-label NLP experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer


DEFAULT_RULETAKER_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "ruletaker"


class RuleTakerDataset(Dataset):
    """HF RuleTaker prompt-to-label dataset.

    Each sample is formatted as:
    ``prompt tokens, BOS, label tokens, EOS``.
    Labels are masked over prompt/BOS context, so training only predicts the
    answer label and EOS. Metrics use only label tokens, excluding EOS.
    """

    def __init__(self, hparams, split: str = "train") -> None:
        self.hparams = hparams
        self.split = split
        cache_dir = getattr(hparams, "dataset_dir", "") or str(DEFAULT_RULETAKER_CACHE_DIR)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer,
            clean_up_tokenization_spaces=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = int(self.tokenizer.pad_token_id)
        self.eos_token_id = int(self.tokenizer.eos_token_id)
        self.bos_token_id = int(
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self.max_length = int(hparams.context_length) + 1

        hf_split = "validation" if split == "val" else split
        if hf_split == "validation":
            hf_split = "dev"
        self.dataset = load_dataset(
            "tasksource/ruletaker",
            split=hf_split,
            cache_dir=str(self.cache_dir),
        )
        max_items = int(getattr(hparams, "ruletaker_max_items", 0) or 0)
        if max_items > 0:
            self.dataset = self.dataset.select(range(min(max_items, len(self.dataset))))

    def __len__(self) -> int:
        return len(self.dataset)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @staticmethod
    def _prompt(context: str, question: str) -> str:
        return f"Context: {context}\nQuestion: {question}\nAnswer:"

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.dataset[idx]
        prompt_text = self._prompt(str(row["context"]), str(row["question"]))
        target_text = str(row["label"])

        target_ids_list = self._encode(" " + target_text)
        target_ids = torch.tensor(target_ids_list, dtype=torch.long)
        reserved = 1 + len(target_ids_list) + 1
        max_prompt_len = max(1, self.max_length - reserved)
        prompt_ids_list = self._encode(prompt_text)[-max_prompt_len:]
        prompt_ids = torch.tensor(prompt_ids_list, dtype=torch.long)

        bos = torch.tensor([self.bos_token_id], dtype=torch.long)
        eos = torch.tensor([self.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, bos, target_ids, eos])
        if input_ids.numel() > self.max_length:
            raise ValueError(
                f"RuleTaker sample length {input_ids.numel()} exceeds max_length={self.max_length}"
            )

        labels = torch.full_like(input_ids, -100)
        target_start = int(prompt_ids.numel())
        labels[target_start : target_start + int(target_ids.numel()) + 1] = torch.cat([target_ids, eos])

        target_metric_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        target_metric_mask[target_start : target_start + int(target_ids.numel())] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "target_metric_mask": target_metric_mask,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.bool),
            "prompt_token_ids": prompt_ids,
            "target_token_ids": target_ids,
            "target_attention_mask": torch.ones_like(target_ids, dtype=torch.bool),
            "prompt_len": torch.tensor(prompt_ids.numel(), dtype=torch.long),
            "target_len": torch.tensor(target_ids.numel(), dtype=torch.long),
            "target_start": torch.tensor(target_start, dtype=torch.long),
            "row_id": torch.tensor(idx, dtype=torch.long),
            "config": str(row.get("config", "")),
            "label_text": target_text,
            "bos_token_id": torch.tensor(self.bos_token_id, dtype=torch.long),
            "pad_token_id": torch.tensor(self.pad_token_id, dtype=torch.long),
        }


def collate_ruletaker(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty RuleTaker batch")

    batch_size = len(samples)
    pad_id = int(samples[0]["pad_token_id"].item())
    max_len = max(int(sample["input_ids"].shape[0]) for sample in samples)
    max_prompt = max(int(sample["prompt_token_ids"].shape[0]) for sample in samples)
    max_target = max(int(sample["target_token_ids"].shape[0]) for sample in samples)

    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    target_metric_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    prompt_ids = torch.full((batch_size, max_prompt), pad_id, dtype=torch.long)
    target_ids = torch.full((batch_size, max_target), pad_id, dtype=torch.long)
    target_attention_mask = torch.zeros((batch_size, max_target), dtype=torch.bool)

    for i, sample in enumerate(samples):
        seq_len = int(sample["input_ids"].shape[0])
        prompt_len = int(sample["prompt_token_ids"].shape[0])
        target_len = int(sample["target_token_ids"].shape[0])
        input_ids[i, :seq_len] = sample["input_ids"]
        labels[i, :seq_len] = sample["labels"]
        target_metric_mask[i, :seq_len] = sample["target_metric_mask"]
        attention_mask[i, :seq_len] = True
        prompt_ids[i, :prompt_len] = sample["prompt_token_ids"]
        target_ids[i, :target_len] = sample["target_token_ids"]
        target_attention_mask[i, :target_len] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "target_metric_mask": target_metric_mask,
        "attention_mask": attention_mask,
        "prompt_token_ids": prompt_ids,
        "target_token_ids": target_ids,
        "target_attention_mask": target_attention_mask,
        "prompt_len": torch.stack([sample["prompt_len"] for sample in samples]),
        "target_len": torch.stack([sample["target_len"] for sample in samples]),
        "target_start": torch.stack([sample["target_start"] for sample in samples]),
        "row_id": torch.stack([sample["row_id"] for sample in samples]),
        "config": [sample["config"] for sample in samples],
        "label_text": [sample["label_text"] for sample in samples],
        "bos_token_id": torch.stack([sample["bos_token_id"] for sample in samples]),
        "pad_token_id": torch.stack([sample["pad_token_id"] for sample in samples]),
    }
