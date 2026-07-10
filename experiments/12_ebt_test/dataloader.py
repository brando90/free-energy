#!/usr/bin/env python3
"""Data loading utilities for synthetic Modus Ponens chain examples."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from tokenizer import SyntheticTokenizer


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parent
    / "EBT-main"
    / "data"
    / "nlp"
    / "synthetic_chain_dataset"
    / "synthetic_chain_dataset.jsonl"
)


@dataclass
class ChainRecord:
    row_id: int
    split: str
    prompt_tokens: list[str]
    target_tokens: list[str]
    path_tokens: list[str]
    starts: list[str]
    goal: str
    depth: int
    rules: list[str]


class SyntheticChainDataset(Dataset[dict[str, Any]]):
    """Toy synthetic dataset with prompt + target chain tokens."""

    def __init__(
        self,
        *,
        data_path: Path = DEFAULT_DATA_PATH,
        split: str = "train",
        max_items: int | None = None,
        max_prompt_tokens: int | None = None,
        max_target_tokens: int | None = None,
    ) -> None:
        self.path = Path(data_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Missing dataset file {self.path}. Run `python generate_dataset.py` first."
            )
        self.split = split
        self.tokenizer = SyntheticTokenizer.default()
        self.max_prompt_tokens = max_prompt_tokens
        self.max_target_tokens = max_target_tokens
        self.records = self._load_records(max_items=max_items)

    def _load_records(self, max_items: int | None = None) -> list[ChainRecord]:
        records: list[ChainRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if self.split != "all" and str(row.get("split", "train")) != self.split:
                    continue
                prompt = [str(t) for t in row["prompt_tokens"]]
                target = [str(t) for t in row["target_tokens"]]
                records.append(
                    ChainRecord(
                        row_id=int(row["id"]),
                        split=str(row.get("split", "train")),
                        prompt_tokens=prompt,
                        target_tokens=target,
                        path_tokens=[str(x) for x in row.get("path_tokens", [])],
                        starts=[str(x) for x in row.get("starts", [row.get("start", "")]) if str(x)],
                        goal=str(row.get("goal", row.get("query", ""))),
                        depth=int(row.get("depth", 0)),
                        rules=[
                            str(rule.get("text", ""))
                            for rule in row.get("rules", [])
                            if isinstance(rule, dict)
                        ],
                    )
                )
                if max_items is not None and len(records) >= max_items:
                    break
        if not records:
            raise ValueError(f"No rows for split={self.split!r} in {self.path}")
        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        prompt_tokens = rec.prompt_tokens[: self.max_prompt_tokens] if self.max_prompt_tokens is not None else rec.prompt_tokens
        target_tokens = rec.target_tokens[: self.max_target_tokens] if self.max_target_tokens is not None else rec.target_tokens

        prompt_ids = torch.tensor(self.tokenizer.encode(prompt_tokens), dtype=torch.long)
        target_ids = torch.tensor(self.tokenizer.encode(target_tokens), dtype=torch.long)
        if target_ids.numel() == 0:
            target_ids = torch.empty(0, dtype=torch.long)

        # Autoregressive sample input with a BOS token and a final EOS token.
        full_input = torch.cat(
            [
                prompt_ids,
                torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long),
                target_ids,
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long),
            ]
        )
        labels = torch.full_like(full_input, fill_value=-100)
        # Align labels for next-token prediction:
        #  - bos token is conditioned on target[0]
        #  - each target token is conditioned on the previous target token
        #  - eos is conditioned on the last target token
        target_start = int(prompt_ids.numel())
        target_full = torch.cat([target_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)])
        labels[target_start : target_start + target_full.shape[0]] = target_full

        return {
            "row_id": torch.tensor(rec.row_id, dtype=torch.long),
            "split": rec.split,
            "prompt_token_ids": prompt_ids,
            "target_token_ids": target_ids,
            "prompt_len": torch.tensor(len(prompt_ids), dtype=torch.long),
            "target_len": torch.tensor(len(target_ids), dtype=torch.long),
            "path_tokens": rec.path_tokens,
            "starts": rec.starts,
            "goal": rec.goal,
            "depth": torch.tensor(rec.depth, dtype=torch.long),
            "rules": rec.rules,
            "full_input_ids": full_input,
            "labels": labels,
            "target_start": torch.tensor(target_start, dtype=torch.long),
        }


def collate_chain_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate empty batch")

    max_prompt = max(int(sample["prompt_token_ids"].shape[0]) for sample in samples)
    max_target = max(int(sample["target_token_ids"].shape[0]) for sample in samples)
    max_input = max(int(sample["full_input_ids"].shape[0]) for sample in samples)

    batch_size = len(samples)
    tokenizer = SyntheticTokenizer.default()
    pad = tokenizer.pad_token_id
    prompt = torch.full((batch_size, max_prompt), pad, dtype=torch.long)
    target = torch.full((batch_size, max_target), pad, dtype=torch.long)
    full_input = torch.full((batch_size, max_input), pad, dtype=torch.long)
    labels = torch.full((batch_size, max_input), -100, dtype=torch.long)
    prompt_mask = torch.zeros((batch_size, max_prompt), dtype=torch.bool)
    target_mask = torch.zeros((batch_size, max_target), dtype=torch.bool)
    input_mask = torch.zeros((batch_size, max_input), dtype=torch.bool)
    target_attention = torch.zeros((batch_size, max_target), dtype=torch.bool)

    for i, sample in enumerate(samples):
        p = sample["prompt_token_ids"]
        t = sample["target_token_ids"]
        x = sample["full_input_ids"]
        y = sample["labels"]
        plen = int(p.shape[0])
        tlen = int(t.shape[0])
        ilen = int(x.shape[0])

        prompt[i, :plen] = p
        target[i, :tlen] = t
        full_input[i, :ilen] = x
        labels[i, :ilen] = y
        prompt_mask[i, :plen] = True
        target_mask[i, :tlen] = True
        input_mask[i, :ilen] = True
        target_attention[i, :tlen] = True

    return {
        "row_id": torch.stack([sample["row_id"] for sample in samples]),
        "split": [sample["split"] for sample in samples],
        "prompt_token_ids": prompt,
        "target_token_ids": target,
        "full_input_ids": full_input,
        "labels": labels,
        "prompt_attention_mask": prompt_mask,
        "target_attention_mask": target_attention,
        "input_attention_mask": input_mask,
        "prompt_len": torch.stack([sample["prompt_len"] for sample in samples]),
        "target_len": torch.stack([sample["target_len"] for sample in samples]),
        "target_start": torch.stack([sample["target_start"] for sample in samples]),
        "path_tokens": [sample["path_tokens"] for sample in samples],
        "starts": [sample["starts"] for sample in samples],
        "goal": [sample["goal"] for sample in samples],
        "depth": torch.stack([sample["depth"] for sample in samples]),
        "rules": [sample["rules"] for sample in samples],
        "dataset_size": torch.tensor(len(samples), dtype=torch.long),
        "target_token_count": target_attention.sum(dim=1),
    }


def make_dataloader(
    *,
    data_path: Path,
    split: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    max_items: int | None,
    max_prompt_tokens: int | None = None,
    max_target_tokens: int | None = None,
) -> DataLoader[dict[str, Any]]:
    dataset = SyntheticChainDataset(
        data_path=data_path,
        split=split,
        max_items=max_items,
        max_prompt_tokens=max_prompt_tokens,
        max_target_tokens=max_target_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate_chain_samples,
        drop_last=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=4)
    args = parser.parse_args()

    loader = make_dataloader(
        data_path=args.data_path,
        split=args.split,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        max_items=args.max_items,
    )
    batch = next(iter(loader))
    print(f"dataset_size={len(loader.dataset)}")
    print(f"prompt_token_ids={tuple(batch['prompt_token_ids'].shape)}")
    print(f"target_token_ids={tuple(batch['target_token_ids'].shape)}")
    print(f"full_input_ids={tuple(batch['full_input_ids'].shape)}")
    print(f"depth={batch['depth'].tolist()}")
    print(f"goal={batch['goal']}")


if __name__ == "__main__":
    main()
