#!/usr/bin/env python3
"""Lean Workbook Plus dataloader for prompt-conditioned EBT training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from leanworkbook_plus_benchmark import (
    DEFAULT_DATASET,
    DEFAULT_DATA_DIR,
    download_dataset,
    make_prompt,
    write_validation_indices,
)


DEFAULT_MODEL = "Goedel-LM/Goedel-Prover-V2-8B"


@dataclass(frozen=True)
class LeanWorkbookSample:
    row_index: int
    task_id: str
    status: str
    prompt_text: str
    formal_statement: str
    proof_text: str
    prompt_ids: list[int]
    label_ids: list[int]


def _read_index_payload(indices_file: Path | None) -> set[int] | None:
    if indices_file is None or not indices_file.exists():
        return None
    payload = json.loads(indices_file.read_text(encoding="utf-8"))
    return {int(idx) for idx in payload["indices"]}


class LeanWorkbookDataset(Dataset[LeanWorkbookSample]):
    def __init__(
        self,
        *,
        data_file: Path,
        tokenizer_name: str = DEFAULT_MODEL,
        split: str = "train",
        indices_file: Path | None = None,
        max_prompt_tokens: int = 512,
        max_label_tokens: int = 256,
        status_filter: Sequence[str] | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        index_set = _read_index_payload(indices_file)
        wanted_status = set(status_filter) if status_filter else None

        rows = [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.samples: list[LeanWorkbookSample] = []
        for row_idx, row in enumerate(rows):
            if index_set is not None:
                in_val = row_idx in index_set
                if split == "train" and in_val:
                    continue
                if split in {"val", "validation"} and not in_val:
                    continue
            if wanted_status is not None and str(row["status"]) not in wanted_status:
                continue
            proof_text = str(row["tactic"]).strip()
            if not proof_text:
                continue
            prompt_text = make_prompt(str(row["id"]), str(row["natural_language_statement"]), str(row["formal_statement"]))
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_prompt_tokens,
            )["input_ids"]
            label_ids = self.tokenizer(
                proof_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_label_tokens,
            )["input_ids"]
            if not label_ids:
                continue
            self.samples.append(
                LeanWorkbookSample(
                    row_index=row_idx,
                    task_id=str(row["id"]),
                    status=str(row["status"]),
                    prompt_text=prompt_text,
                    formal_statement=str(row["formal_statement"]),
                    proof_text=proof_text,
                    prompt_ids=list(prompt_ids),
                    label_ids=list(label_ids),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LeanWorkbookSample:
        return self.samples[index]


class LeanWorkbookCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, batch: Sequence[LeanWorkbookSample]) -> dict[str, Any]:
        prompt_len = max(len(item.prompt_ids) for item in batch)
        label_len = max(len(item.label_ids) for item in batch)
        prompt_ids = torch.full((len(batch), prompt_len), self.pad_token_id, dtype=torch.long)
        label_ids = torch.full((len(batch), label_len), self.pad_token_id, dtype=torch.long)
        prompt_mask = torch.zeros((len(batch), prompt_len), dtype=torch.bool)
        label_mask = torch.zeros((len(batch), label_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            prompt_ids[i, : len(item.prompt_ids)] = torch.tensor(item.prompt_ids, dtype=torch.long)
            label_ids[i, : len(item.label_ids)] = torch.tensor(item.label_ids, dtype=torch.long)
            prompt_mask[i, : len(item.prompt_ids)] = True
            label_mask[i, : len(item.label_ids)] = True
        return {
            "row_index": torch.tensor([item.row_index for item in batch], dtype=torch.long),
            "task_id": [item.task_id for item in batch],
            "status": [item.status for item in batch],
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "label_token_ids": label_ids,
            "label_attention_mask": label_mask,
            "formal_statement": [item.formal_statement for item in batch],
            "proof_text": [item.proof_text for item in batch],
        }


def make_leanworkbook_dataloader(
    *,
    data_file: Path | None = None,
    tokenizer_name: str = DEFAULT_MODEL,
    split: str = "train",
    indices_file: Path | None = None,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    max_prompt_tokens: int = 512,
    max_label_tokens: int = 256,
    status_filter: Sequence[str] | None = None,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader[dict[str, Any]]:
    if data_file is None:
        data_file = download_dataset(DEFAULT_DATASET, out_dir=DEFAULT_DATA_DIR)
    if indices_file is None:
        indices_file = DEFAULT_DATA_DIR / "leanworkbook_plus_val500_indices.json"
        if not indices_file.exists():
            write_validation_indices(data_file, out_file=indices_file)
    dataset = LeanWorkbookDataset(
        data_file=data_file,
        tokenizer_name=tokenizer_name,
        split=split,
        indices_file=indices_file,
        max_prompt_tokens=max_prompt_tokens,
        max_label_tokens=max_label_tokens,
        status_filter=status_filter,
    )
    collator = LeanWorkbookCollator(dataset.tokenizer.pad_token_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collator,
    )
