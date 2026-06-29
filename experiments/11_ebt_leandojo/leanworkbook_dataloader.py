#!/usr/bin/env python3
"""Lean Workbook context-activation dataloader for chunked EBT training."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data" / "context_gold"
DEFAULT_INDICES_FILE = HERE / "data" / "leanworkbook_plus_val500_indices.json"
DEFAULT_ACTIVATIONS_DIR = HERE / "results" / "leanworkbook_plus_goedel_hidden_states_gpus0_3_contextonly" / "hidden_states_safetensors"
DEFAULT_MODEL = "Goedel-LM/Goedel-Prover-V2-8B"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_validation_indices(path: Path) -> set[int]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        payload = payload["indices"]
    return {int(index) for index in payload}


def _activation_dtype_from_str(value: str) -> torch.dtype:
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported activation dtype: {value!r}")


def _has_hidden_states(path: Path) -> bool:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return "hidden_states" in handle.keys()
    except Exception:
        return False


def chunk_token_ids(token_ids: list[int], *, chunk_size: int, pad_id: int) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    padded = list(token_ids)
    remainder = len(padded) % chunk_size
    if remainder:
        padded.extend([int(pad_id)] * (chunk_size - remainder))
    if not padded:
        padded = [int(pad_id)] * chunk_size
    return torch.tensor(padded, dtype=torch.long).reshape(-1, chunk_size)


def flatten_until_eos(chunks: torch.Tensor, *, eos_id: int, pad_id: int | None = None) -> list[int]:
    flat = chunks.detach().cpu().reshape(-1).tolist()
    out: list[int] = []
    for token_id in flat:
        token_id = int(token_id)
        if token_id == int(eos_id):
            break
        if pad_id is not None and token_id == int(pad_id):
            continue
        out.append(token_id)
    return out


class GoedelIdMapper:
    """Maps compact local ids to original Goedel tokenizer ids and text."""

    local_to_original: torch.Tensor

    def __init__(
        self,
        *,
        model_name: str,
        vocab: dict[str, Any],
        revision: str | None = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        eos_original_id = int(self.tokenizer.eos_token_id)
        pad_original_id = int(
            self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_original_id
        )
        bos_original_id = int(
            self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else eos_original_id
        )
        unk_original_id = int(
            self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else eos_original_id
        )

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

    def original_ids(self, local_ids: torch.Tensor) -> torch.Tensor:
        local_ids = local_ids.to(dtype=torch.long, device=self.local_to_original.device)
        return self.local_to_original.index_select(0, local_ids.reshape(-1)).reshape(local_ids.shape)

    def decode_local_ids(self, local_ids: list[int] | torch.Tensor) -> str:
        if not torch.is_tensor(local_ids):
            local_ids = torch.tensor(local_ids, dtype=torch.long)
        original = self.original_ids(local_ids).detach().cpu().tolist()
        return self.tokenizer.decode(original, skip_special_tokens=True)


@dataclass(frozen=True)
class LeanWorkbookRecord:
    row_index: int
    task_id: str
    status: str
    formal_statement: str
    natural_language_statement: str
    target_text: str
    target_local_ids: list[int]
    target_original_ids: list[int]
    safetensors_path: Path


class LeanWorkbookEmbeddingDataset(Dataset[dict[str, Any]]):
    """Dataset over cleaned Lean Workbook targets and Goedel context activations."""

    def __init__(
        self,
        *,
        data_dir: Path = DEFAULT_DATA_DIR,
        activations_dir: Path = DEFAULT_ACTIVATIONS_DIR,
        indices_file: Path = DEFAULT_INDICES_FILE,
        split: str = "train",
        max_items: int | None = None,
        random_sample_items: int | None = None,
        random_sample_seed: int = 0,
        max_target_tokens: int | None = None,
        chunk_size: int = 8,
        activation_dtype: str = "bf16",
        model_name: str | None = None,
        model_revision: str | None = None,
        validate_context: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.activations_dir = Path(activations_dir)
        self.indices_file = Path(indices_file)
        self.split = str(split)
        self.max_target_tokens = max_target_tokens
        self.chunk_size = int(chunk_size)
        self.activation_dtype = _activation_dtype_from_str(activation_dtype)
        self.vocab = _load_json(self.data_dir / "vocab.json")
        self.vocab_size = int(self.vocab["vocab_size"])
        self.pad_id = int(self.vocab["local_pad_id"])
        self.bos_id = int(self.vocab["local_bos_id"])
        self.eos_id = int(self.vocab["local_eos_id"])
        self.model_name = model_name or str(self.vocab.get("tokenizer") or DEFAULT_MODEL)
        self.id_mapper = GoedelIdMapper(model_name=self.model_name, revision=model_revision, vocab=self.vocab)
        self.records = self._load_records(
            max_items=max_items,
            random_sample_items=random_sample_items,
            random_sample_seed=random_sample_seed,
            validate_context=validate_context,
        )

    def _wanted_split(self, row_index: int, validation_indices: set[int]) -> bool:
        split = self.split.lower()
        is_val = row_index in validation_indices
        if split in {"val", "validation"}:
            return is_val
        if split == "train":
            return not is_val
        if split in {"all", "*"}:
            return True
        raise ValueError(f"Unsupported split: {self.split!r}")

    def _load_records(
        self,
        *,
        max_items: int | None,
        random_sample_items: int | None,
        random_sample_seed: int,
        validate_context: bool,
    ) -> list[LeanWorkbookRecord]:
        rows = _read_jsonl(self.data_dir / "manifest.jsonl")
        validation_indices = _read_validation_indices(self.indices_file)
        records: list[LeanWorkbookRecord] = []
        for manifest_pos, row in enumerate(rows):
            row_index = int(row.get("index", manifest_pos))
            if not self._wanted_split(row_index, validation_indices):
                continue
            path = self.activations_dir / f"{row_index:06d}.safetensors"
            if not path.exists():
                continue
            if validate_context and not _has_hidden_states(path):
                continue
            target_local_ids = [int(x) for x in row["target_local_ids"]]
            target_original_ids = [int(x) for x in row["target_original_ids"]]
            records.append(
                LeanWorkbookRecord(
                    row_index=row_index,
                    task_id=str(row["task_id"]),
                    status=str(row["status"]),
                    formal_statement=str(row["formal_statement"]),
                    natural_language_statement=str(row["natural_language_statement"]),
                    target_text=str(row["target_text"]),
                    target_local_ids=target_local_ids,
                    target_original_ids=target_original_ids,
                    safetensors_path=path,
                )
            )
            if random_sample_items is None and max_items is not None and len(records) >= max_items:
                break
        if random_sample_items is not None:
            sample_size = min(int(random_sample_items), len(records))
            records = random.Random(int(random_sample_seed)).sample(records, sample_size)
        if max_items is not None:
            records = records[: int(max_items)]
        return records

    def __len__(self) -> int:
        return len(self.records)

    def _load_context(self, path: Path) -> torch.Tensor:
        tensors = load_file(path, device="cpu")
        if "hidden_states" not in tensors:
            raise KeyError(f"hidden_states not found in {path}")
        return tensors["hidden_states"].to(self.activation_dtype).contiguous()

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target = list(record.target_local_ids)
        target_original = list(record.target_original_ids)
        if self.max_target_tokens is not None:
            max_target_tokens = int(self.max_target_tokens)
            target = target[:max_target_tokens]
            target_original = target_original[:max_target_tokens]

        labels = chunk_token_ids(target + [self.eos_id], chunk_size=self.chunk_size, pad_id=self.eos_id)
        decoder_input_ids = chunk_token_ids([self.bos_id] + target, chunk_size=self.chunk_size, pad_id=self.eos_id)
        label_original_ids = self.id_mapper.original_ids(labels)
        decoder_input_original_ids = self.id_mapper.original_ids(decoder_input_ids)

        return {
            "row_index": torch.tensor(record.row_index, dtype=torch.long),
            "task_id": record.task_id,
            "status": record.status,
            "formal_statement": record.formal_statement,
            "natural_language_statement": record.natural_language_statement,
            "target_text": record.target_text,
            "target_local_ids": torch.tensor(target, dtype=torch.long),
            "target_original_ids": torch.tensor(target_original, dtype=torch.long),
            "context_activations": self._load_context(record.safetensors_path),
            "decoder_input_ids": decoder_input_ids,
            "decoder_input_original_ids": decoder_input_original_ids,
            "labels": labels,
            "label_original_ids": label_original_ids,
            "context_token_count": torch.tensor(0, dtype=torch.long),
            "target_token_count": torch.tensor(labels.numel(), dtype=torch.long),
        }


class TargetChunkBatchSampler(Sampler[list[int]]):
    """Groups examples by chunk count under a chunk budget."""

    def __init__(
        self,
        dataset: LeanWorkbookEmbeddingDataset,
        *,
        max_items_per_batch: int,
        max_target_chunks_per_batch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.lengths = [
            chunk_token_ids(
                (record.target_local_ids[: dataset.max_target_tokens] if dataset.max_target_tokens else record.target_local_ids)
                + [dataset.eos_id],
                chunk_size=dataset.chunk_size,
                pad_id=dataset.eos_id,
            ).shape[0]
            for record in dataset.records
        ]
        self.max_items_per_batch = int(max_items_per_batch)
        self.max_target_chunks_per_batch = int(max_target_chunks_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        batches = self._batches()
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._batches())

    def _batches(self) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in sorted(range(len(self.lengths)), key=lambda i: self.lengths[i]):
            length = self.lengths[idx]
            next_size = len(current) + 1
            next_max = max(current_max, length)
            if current and (
                next_size > self.max_items_per_batch
                or next_size * next_max > self.max_target_chunks_per_batch
            ):
                batches.append(current)
                current = []
                current_max = 0
            current.append(idx)
            current_max = max(current_max, length)
        if current:
            batches.append(current)
        return batches


def collate_leanworkbook_embedding_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    batch = len(samples)
    hidden_dim = int(samples[0]["context_activations"].shape[1])
    chunk_size = int(samples[0]["labels"].shape[1])
    max_context = max(int(sample["context_activations"].shape[0]) for sample in samples)
    max_chunks = max(int(sample["labels"].shape[0]) for sample in samples)
    max_raw_target = max(int(sample["target_local_ids"].shape[0]) for sample in samples)
    context_dtype = samples[0]["context_activations"].dtype

    context = torch.zeros(batch, max_context, hidden_dim, dtype=context_dtype)
    context_mask = torch.zeros(batch, max_context, dtype=torch.bool)
    labels = torch.full((batch, max_chunks, chunk_size), -100, dtype=torch.long)
    label_original_ids = torch.full((batch, max_chunks, chunk_size), -100, dtype=torch.long)
    decoder_input_ids = torch.zeros(batch, max_chunks, chunk_size, dtype=torch.long)
    decoder_input_original_ids = torch.zeros(batch, max_chunks, chunk_size, dtype=torch.long)
    label_attention_mask = torch.zeros(batch, max_chunks, chunk_size, dtype=torch.bool)
    decoder_attention_mask = torch.zeros(batch, max_chunks, dtype=torch.bool)
    target_local_ids = torch.full((batch, max_raw_target), -100, dtype=torch.long)
    target_original_ids = torch.full((batch, max_raw_target), -100, dtype=torch.long)

    for i, sample in enumerate(samples):
        c_len = int(sample["context_activations"].shape[0])
        context[i, :c_len] = sample["context_activations"]
        context_mask[i, :c_len] = True
        chunks = int(sample["labels"].shape[0])
        labels[i, :chunks] = sample["labels"]
        label_original_ids[i, :chunks] = sample["label_original_ids"]
        decoder_input_ids[i, :chunks] = sample["decoder_input_ids"]
        decoder_input_original_ids[i, :chunks] = sample["decoder_input_original_ids"]
        label_attention_mask[i, :chunks] = True
        decoder_attention_mask[i, :chunks] = True
        raw_len = int(sample["target_local_ids"].shape[0])
        target_local_ids[i, :raw_len] = sample["target_local_ids"]
        target_original_ids[i, :raw_len] = sample["target_original_ids"]

    return {
        "row_index": torch.stack([sample["row_index"] for sample in samples]),
        "task_id": [sample["task_id"] for sample in samples],
        "status": [sample["status"] for sample in samples],
        "formal_statement": [sample["formal_statement"] for sample in samples],
        "natural_language_statement": [sample["natural_language_statement"] for sample in samples],
        "target_text": [sample["target_text"] for sample in samples],
        "target_local_ids": target_local_ids,
        "target_original_ids": target_original_ids,
        "context_activations": context,
        "context_attention_mask": context_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_input_original_ids": decoder_input_original_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
        "label_original_ids": label_original_ids,
        "label_attention_mask": label_attention_mask,
        "context_token_count": context_mask.sum(dim=1).to(torch.long),
        "target_token_count": label_attention_mask.sum(dim=(1, 2)).to(torch.long),
    }


def make_leanworkbook_embedding_dataloader(
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    max_target_chunks_per_batch: int | None = None,
    seed: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    drop_last: bool = False,
    **dataset_kwargs: Any,
) -> DataLoader[dict[str, Any]]:
    dataset = LeanWorkbookEmbeddingDataset(**dataset_kwargs)
    persistent_workers = bool(persistent_workers) and int(num_workers) > 0
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": persistent_workers,
        "collate_fn": collate_leanworkbook_embedding_samples,
    }
    if max_target_chunks_per_batch is None:
        loader_kwargs.update(batch_size=int(batch_size), shuffle=bool(shuffle), drop_last=bool(drop_last))
    else:
        loader_kwargs["batch_sampler"] = TargetChunkBatchSampler(
            dataset,
            max_items_per_batch=int(batch_size),
            max_target_chunks_per_batch=int(max_target_chunks_per_batch),
            shuffle=bool(shuffle),
            seed=int(seed),
        )
    if int(num_workers) > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**loader_kwargs)


def make_leanworkbook_dataloader(**kwargs: Any) -> DataLoader[dict[str, Any]]:
    return make_leanworkbook_embedding_dataloader(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--activations-dir", type=Path, default=DEFAULT_ACTIVATIONS_DIR)
    parser.add_argument("--indices-file", type=Path, default=DEFAULT_INDICES_FILE)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-items", type=int, default=2)
    args = parser.parse_args()
    loader = make_leanworkbook_embedding_dataloader(
        data_dir=args.data_dir,
        activations_dir=args.activations_dir,
        indices_file=args.indices_file,
        split=args.split,
        chunk_size=args.chunk_size,
        max_items=args.max_items,
        batch_size=args.batch_size,
    )
    batch = next(iter(loader))
    print(f"dataset_size={len(loader.dataset)}")
    print(f"row_index={batch['row_index'].tolist()}")
    print(f"context_activations={tuple(batch['context_activations'].shape)} {batch['context_activations'].dtype}")
    print(f"labels={tuple(batch['labels'].shape)}")
    print(f"label_attention_mask={tuple(batch['label_attention_mask'].shape)}")


if __name__ == "__main__":
    main()
