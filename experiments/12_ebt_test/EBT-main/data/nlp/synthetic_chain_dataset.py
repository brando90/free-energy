#!/usr/bin/env python3
"""Synthetic Modus Ponens chain dataset for NLP EBT experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def fact_symbols(count: int = 128) -> list[str]:
    symbols: list[str] = []
    idx = 0
    while len(symbols) < count:
        n = idx
        chars: list[str] = []
        while True:
            chars.append(chr(ord("A") + (n % 26)))
            n = n // 26 - 1
            if n < 0:
                break
        symbols.append("".join(reversed(chars)))
        idx += 1
    return symbols


@dataclass(frozen=True)
class SyntheticChainTokenizer:
    tokens: tuple[str, ...]

    @classmethod
    def default(cls) -> "SyntheticChainTokenizer":
        return cls(
            tokens=(
                "<pad>",
                "<eos>",
                "<bos>",
                "<unk>",
                *tuple(fact_symbols(128)),
                "->",
                "and",
                "or",
                "Start",
                "Rules",
                "Goal",
                "Prove",
                "<SEP>",
                "[",
                "]",
                ",",
                "|",
                "fact:",
                "rule:",
                "query:",
            )
        )

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id.get(token, self.unk_token_id) for token in tokens]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.tokens[int(token_id)] for token_id in token_ids]


DEFAULT_SYNTHETIC_CHAIN_PATH = (
    Path(__file__).resolve().parent / "synthetic_chain_dataset" / "synthetic_chain_dataset.jsonl"
)


class SyntheticChainDataset(Dataset):
    """JSONL-backed prompt-to-chain dataset.

    Samples are formatted as a language-model sequence:
    prompt tokens, ``<bos>``, target chain tokens, ``<eos>``.
    Loss labels are masked over the prompt, so training predicts only the
    chain and EOS. Accuracy metrics use only the chain tokens.
    """

    def __init__(self, hparams, split: str = "train") -> None:
        self.hparams = hparams
        self.split = split
        self.tokenizer = SyntheticChainTokenizer.default()
        data_path = getattr(hparams, "synthetic_chain_data_path", "")
        self.data_path = Path(data_path).expanduser().resolve() if data_path else DEFAULT_SYNTHETIC_CHAIN_PATH
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Missing synthetic chain dataset at {self.data_path}. "
                "Run ../generate_dataset.py or pass --synthetic_chain_data_path."
            )
        self.records = self._load_records()
        if not self.records:
            raise ValueError(f"No synthetic chain records for split={split!r} in {self.data_path}")

    def _load_records(self) -> list[dict[str, Any]]:
        max_items = int(getattr(self.hparams, "synthetic_chain_max_items", 0) or 0)
        records: list[dict[str, Any]] = []
        with self.data_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("split", "train")) != self.split:
                    continue
                records.append(row)
                if max_items > 0 and len(records) >= max_items:
                    break
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.records[idx]
        prompt_ids = torch.tensor(self.tokenizer.encode([str(t) for t in row["prompt_tokens"]]), dtype=torch.long)
        target_ids = torch.tensor(self.tokenizer.encode([str(t) for t in row["target_tokens"]]), dtype=torch.long)

        bos = torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long)
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        input_ids = torch.cat([prompt_ids, bos, target_ids, eos])

        labels = torch.full_like(input_ids, -100)
        target_start = int(prompt_ids.numel())
        labels[target_start : target_start + int(target_ids.numel()) + 1] = torch.cat([target_ids, eos])

        metric_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        metric_mask[target_start : target_start + int(target_ids.numel())] = True

        return {
            "input_ids": input_ids,
            "labels": labels,
            "target_metric_mask": metric_mask,
            "prompt_token_ids": prompt_ids,
            "target_token_ids": target_ids,
            "prompt_len": torch.tensor(prompt_ids.numel(), dtype=torch.long),
            "target_len": torch.tensor(target_ids.numel(), dtype=torch.long),
            "row_id": torch.tensor(int(row["id"]), dtype=torch.long),
        }


def collate_synthetic_chain(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty synthetic-chain batch")
    tokenizer = SyntheticChainTokenizer.default()
    batch_size = len(samples)
    max_len = max(int(sample["input_ids"].shape[0]) for sample in samples)
    max_prompt = max(int(sample["prompt_token_ids"].shape[0]) for sample in samples)
    max_target = max(int(sample["target_token_ids"].shape[0]) for sample in samples)

    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    metric_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    prompt_ids = torch.full((batch_size, max_prompt), tokenizer.pad_token_id, dtype=torch.long)
    target_ids = torch.full((batch_size, max_target), tokenizer.pad_token_id, dtype=torch.long)
    target_attention_mask = torch.zeros((batch_size, max_target), dtype=torch.bool)

    for i, sample in enumerate(samples):
        seq_len = int(sample["input_ids"].shape[0])
        prompt_len = int(sample["prompt_token_ids"].shape[0])
        target_len = int(sample["target_token_ids"].shape[0])
        input_ids[i, :seq_len] = sample["input_ids"]
        labels[i, :seq_len] = sample["labels"]
        metric_mask[i, :seq_len] = sample["target_metric_mask"]
        attention_mask[i, :seq_len] = True
        prompt_ids[i, :prompt_len] = sample["prompt_token_ids"]
        target_ids[i, :target_len] = sample["target_token_ids"]
        target_attention_mask[i, :target_len] = True

    return {
        "input_ids": input_ids,
        "labels": labels,
        "target_metric_mask": metric_mask,
        "attention_mask": attention_mask,
        "prompt_token_ids": prompt_ids,
        "target_token_ids": target_ids,
        "target_attention_mask": target_attention_mask,
        "prompt_len": torch.stack([sample["prompt_len"] for sample in samples]),
        "target_len": torch.stack([sample["target_len"] for sample in samples]),
        "row_id": torch.stack([sample["row_id"] for sample in samples]),
    }
