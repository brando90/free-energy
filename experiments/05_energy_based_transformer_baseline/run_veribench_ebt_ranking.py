#!/usr/bin/env python3
"""VeriBench Energy-Based Transformer reranking baseline.

This is the Lean/VeriBench half of the EBT baseline. It treats the EBT as a
learned verifier: E(task, candidate Lean artifact) should be low for compatible
or verifier-passing candidates and high for incompatible candidates.

Unlike `run_ebt_toy.py`, this script does not refine a discrete Lean sequence by
gradient descent. That would require a serious tokenizer/generator and verifier
loop. This is the cheap baseline we can run now: candidate generation comes from
gold files, generated agents, corruptions, and wrong-task gold files; the EBT
scores candidates and supports best-of-N selection by minimum energy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
START_OFF_DIR = REPO_ROOT / "experiments" / "00_start_off"
sys.path.insert(0, str(START_OFF_DIR))

import pilot_ebm_ranking as pilot  # noqa: E402


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "veribench"
DEFAULT_VERIBENCH_ROOT = Path.home() / "veribench"
DEFAULT_SPLIT_DIR = REPO_ROOT / "experiments" / "02_ar_pros_cons" / "data" / "splits"
RANK_EPS = 1e-8

PAD = "<pad>"
UNK = "<unk>"


@dataclass(frozen=True)
class RankingMetrics:
    mean_reciprocal_rank: float
    all_gold_top: bool
    mean_gold_rank: float
    mean_energy_margin: float


class CharVocab:
    def __init__(self, texts: list[str], max_vocab_size: int) -> None:
        counts: dict[str, int] = {}
        for text in texts:
            for ch in text:
                counts[ch] = counts.get(ch, 0) + 1
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        chars = [ch for ch, _count in ordered[: max(0, max_vocab_size - 2)]]
        self.itos = [PAD, UNK, *chars]
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        ids = [self.stoi.get(ch, self.stoi[UNK]) for ch in text]
        if len(ids) > max_length:
            head = max_length // 2
            tail = max_length - head
            ids = ids[:head] + ids[-tail:]
        mask = [1] * len(ids)
        if len(ids) < max_length:
            pad = max_length - len(ids)
            ids.extend([self.stoi[PAD]] * pad)
            mask.extend([0] * pad)
        return ids, mask

    def to_json(self) -> dict[str, Any]:
        return {"size": len(self.itos), "tokens": self.itos}


class CharEnergyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.position = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.energy_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids) + self.position[:, : input_ids.shape[1]]
        key_padding_mask = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=key_padding_mask)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.energy_head(pooled).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def scrub_corruption_markers(text: str) -> str:
    text = re.sub(r"/- corrupted:[\s\S]*?-/\n?", "", text)
    text = text.replace("-- corrupted expected failure", "")
    text = text.replace("corrupted_", "")
    text = text.replace("corrupted", "")
    return text


def candidate_input_text(candidate: pilot.Candidate, task_meta: dict[str, Any], scrub: bool) -> str:
    candidate_text = scrub_corruption_markers(candidate.text) if scrub else candidate.text
    return "\n".join(
        [
            f"TASK_ID: {candidate.task_id}",
            f"TASK_ROLE: {task_meta.get('role', '')}",
            f"TASK_REASON: {task_meta.get('why', '')}",
            "LEAN_CANDIDATE:",
            candidate_text,
        ]
    )


def add_wrong_task_gold_negatives(
    pools: dict[str, list[pilot.Candidate]],
    max_wrong_per_task: int | None,
) -> dict[str, list[pilot.Candidate]]:
    gold_by_task = {
        task_id: next(candidate for candidate in candidates if candidate.candidate_id == "gold")
        for task_id, candidates in pools.items()
    }
    augmented: dict[str, list[pilot.Candidate]] = {}
    for task_id, candidates in pools.items():
        new_candidates = list(candidates)
        added = 0
        for other_task_id, other_gold in gold_by_task.items():
            if other_task_id == task_id:
                continue
            if max_wrong_per_task is not None and added >= max_wrong_per_task:
                break
            new_candidates.append(
                pilot.Candidate(
                    task_id=task_id,
                    candidate_id=f"wrong_task_gold__{safe_name(other_task_id)}",
                    kind="wrong_task_gold",
                    label=0,
                    source_path=other_gold.source_path,
                    text=other_gold.text,
                )
            )
            added += 1
        augmented[task_id] = new_candidates
    return augmented


def load_candidate_pools(args: argparse.Namespace) -> tuple[dict[str, list[pilot.Candidate]], dict[str, Any], Path]:
    manifest = pilot.load_manifest(args.manifest)
    veribench_root = pilot.resolve_veribench_root(manifest, str(args.veribench_root))
    pools = pilot.build_candidate_pools(manifest, veribench_root, args.max_generated_agents)
    if args.add_wrong_task_gold_negatives:
        pools = add_wrong_task_gold_negatives(pools, args.max_wrong_task_gold_negatives)
    return pools, manifest, veribench_root


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing VeriBench split file: {path}. Generate splits with "
            "`cd experiments/02_ar_pros_cons && python -m data.setup --include-generated-agents --smoke`."
        )
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_task_id(row: dict[str, Any]) -> str:
    family = str(row.get("family", "")).strip()
    task_id = str(row["task_id"]).strip()
    return f"{family}/{task_id}" if family and not task_id.startswith(f"{family}/") else task_id


def candidate_from_split_row(row: dict[str, Any], task_id: str) -> pilot.Candidate:
    source_kind = str(row["source_kind"])
    lean_path = Path(row["lean_path"]).expanduser()
    text = lean_path.read_text(encoding="utf-8", errors="replace")
    candidate_id = "gold" if source_kind == "gold" else str(row.get("variant_id") or lean_path.stem)
    return pilot.Candidate(
        task_id=task_id,
        candidate_id=candidate_id,
        kind=source_kind,
        label=1 if source_kind == "gold" else 0,
        source_path=str(lean_path),
        text=text,
    )


def cap_pools(pools: dict[str, list[pilot.Candidate]], max_tasks: int) -> dict[str, list[pilot.Candidate]]:
    if max_tasks <= 0 or len(pools) <= max_tasks:
        return pools
    return {task_id: pools[task_id] for task_id in sorted(pools)[:max_tasks]}


def manifest_for_pools(pools_by_split: dict[str, dict[str, list[pilot.Candidate]]]) -> dict[str, Any]:
    examples = []
    seen: set[str] = set()
    for split, pools in pools_by_split.items():
        for task_id in pools:
            if task_id in seen:
                continue
            seen.add(task_id)
            examples.append(
                {
                    "id": task_id,
                    "role": split,
                    "why": "VeriBench train/val/test split task",
                }
            )
    return {"name": "veribench_split_ebt_reranking", "examples": examples}


def load_split_pools(args: argparse.Namespace) -> tuple[dict[str, dict[str, list[pilot.Candidate]]], dict[str, Any], Path]:
    split_dir = args.split_dir.expanduser().resolve()
    rows_by_split = {split: read_jsonl(split_dir / f"{split}.jsonl") for split in ("train", "val", "test")}

    pools_by_split: dict[str, dict[str, list[pilot.Candidate]]] = {}
    for split, rows in rows_by_split.items():
        rows_by_task: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            rows_by_task.setdefault(split_task_id(row), []).append(row)

        pools: dict[str, list[pilot.Candidate]] = {}
        for task_id, task_rows in rows_by_task.items():
            gold_rows = [row for row in task_rows if row["source_kind"] == "gold"]
            if not gold_rows:
                continue
            gold = candidate_from_split_row(gold_rows[0], task_id)
            candidates = [gold]
            for row in task_rows:
                if row["source_kind"] == "gold":
                    continue
                candidates.append(candidate_from_split_row(row, task_id))
            if args.add_corruption_negatives:
                candidates.extend(pilot.make_corruptions(task_id, gold.text))
            pools[task_id] = candidates

        max_tasks = {
            "train": args.max_split_train_tasks,
            "val": args.max_split_val_tasks,
            "test": args.max_split_test_tasks,
        }[split]
        pools = cap_pools(pools, max_tasks)
        if args.add_wrong_task_gold_negatives:
            pools = add_wrong_task_gold_negatives(pools, args.max_wrong_task_gold_negatives)
        pools_by_split[split] = pools

    manifest = manifest_for_pools(pools_by_split)
    summary_path = split_dir / "summary.json"
    veribench_root = Path(args.veribench_root).expanduser().resolve()
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        veribench_root = Path(summary.get("veribench_root", veribench_root)).expanduser().resolve()
    return pools_by_split, manifest, veribench_root


def make_pairs(
    pools: dict[str, list[pilot.Candidate]],
    manifest: dict[str, Any],
    holdout_task_id: str | None,
) -> list[tuple[pilot.Candidate, pilot.Candidate, dict[str, Any]]]:
    meta_by_id = {example["id"]: example for example in manifest["examples"]}
    pairs = []
    for task_id, candidates in pools.items():
        if holdout_task_id and task_id == holdout_task_id:
            continue
        positives = [candidate for candidate in candidates if candidate.label == 1]
        negatives = [candidate for candidate in candidates if candidate.label == 0]
        if not positives:
            raise ValueError(f"No positive candidate for task {task_id}")
        for positive in positives:
            for negative in negatives:
                pairs.append((positive, negative, meta_by_id[task_id]))
    return pairs


def encode_texts(
    vocab: CharVocab,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [vocab.encode(text, max_length) for text in texts]
    ids = torch.tensor([item[0] for item in encoded], dtype=torch.long, device=device)
    mask = torch.tensor([item[1] for item in encoded], dtype=torch.long, device=device)
    return ids, mask


def score_texts(
    model: CharEnergyTransformer,
    vocab: CharVocab,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[float]:
    energies: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            ids, mask = encode_texts(vocab, texts[start : start + batch_size], max_length, device)
            batch_energies = model(ids, mask)
            energies.extend(float(value) for value in batch_energies.detach().cpu())
    return energies


def evaluate(
    model: CharEnergyTransformer,
    vocab: CharVocab,
    pools: dict[str, list[pilot.Candidate]],
    manifest: dict[str, Any],
    max_length: int,
    batch_size: int,
    device: torch.device,
    scrub: bool,
) -> dict[str, Any]:
    meta_by_id = {example["id"]: example for example in manifest["examples"]}
    task_results = []
    reciprocal_ranks = []
    gold_ranks = []
    margins = []

    for task_id, candidates in pools.items():
        texts = [candidate_input_text(candidate, meta_by_id[task_id], scrub) for candidate in candidates]
        energies = score_texts(model, vocab, texts, max_length, batch_size, device)
        ranked_pairs = sorted(zip(energies, candidates), key=lambda pair: pair[0])
        ranked = []
        for rank, (energy, candidate) in enumerate(ranked_pairs, start=1):
            ranked.append(
                {
                    "rank": rank,
                    "energy": energy,
                    "task_id": candidate.task_id,
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind,
                    "label": candidate.label,
                    "source_path": candidate.source_path,
                    "num_chars": len(candidate.text),
                    "num_lines": candidate.text.count("\n") + 1,
                    "num_sorry": candidate.text.count("sorry"),
                }
            )
        gold = next(row for row in ranked if row["label"] == 1)
        best_negative_energy = min(row["energy"] for row in ranked if row["label"] == 0)
        gold_rank = 1 + sum(
            1
            for row in ranked
            if row["label"] == 0 and row["energy"] <= float(gold["energy"]) + RANK_EPS
        )
        reciprocal_ranks.append(1.0 / gold_rank)
        gold_ranks.append(gold_rank)
        margins.append(best_negative_energy - float(gold["energy"]))
        task_results.append(
            {
                "task_id": task_id,
                "num_candidates": len(candidates),
                "gold_rank": gold_rank,
                "gold_energy": gold["energy"],
                "best_negative_minus_gold_energy": margins[-1],
                "top_candidate": ranked[0],
                "ranked_candidates": ranked,
            }
        )

    metrics = RankingMetrics(
        mean_reciprocal_rank=sum(reciprocal_ranks) / len(reciprocal_ranks),
        all_gold_top=all(rank == 1 for rank in gold_ranks),
        mean_gold_rank=sum(gold_ranks) / len(gold_ranks),
        mean_energy_margin=sum(margins) / len(margins),
    )
    return {"metrics": asdict(metrics), "task_results": task_results}


def train_split_experiment(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = resolve_device(args.device)
    pools_by_split, manifest, veribench_root = load_split_pools(args)
    train_pools = pools_by_split["train"]
    pairs = make_pairs(train_pools, manifest, None)
    if not pairs:
        raise ValueError("No training pairs were built from the train split")

    meta_by_id = {example["id"]: example for example in manifest["examples"]}
    train_texts = []
    for task_id, candidates in train_pools.items():
        train_texts.extend(
            candidate_input_text(candidate, meta_by_id[task_id], args.scrub_corruption_markers)
            for candidate in candidates
        )
    vocab = CharVocab(train_texts, args.max_vocab_size)
    model = CharEnergyTransformer(
        vocab_size=len(vocab.itos),
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = random.Random(args.seed + 505)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before_by_split = {
        split: evaluate(
            model,
            vocab,
            pools,
            manifest,
            args.max_length,
            args.eval_batch_size,
            device,
            args.scrub_corruption_markers,
        )
        for split, pools in pools_by_split.items()
    }

    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator.shuffle(pairs)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            pos_texts = [
                candidate_input_text(pos, meta, args.scrub_corruption_markers)
                for pos, _neg, meta in batch_pairs
            ]
            neg_texts = [
                candidate_input_text(neg, meta, args.scrub_corruption_markers)
                for _pos, neg, meta in batch_pairs
            ]
            pos_ids, pos_mask = encode_texts(vocab, pos_texts, args.max_length, device)
            neg_ids, neg_mask = encode_texts(vocab, neg_texts, args.max_length, device)
            e_pos = model(pos_ids, pos_mask)
            e_neg = model(neg_ids, neg_mask)
            loss = F.softplus(e_pos - e_neg).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * len(batch_pairs)
            total_count += len(batch_pairs)

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            val_eval = evaluate(
                model,
                vocab,
                pools_by_split["val"],
                manifest,
                args.max_length,
                args.eval_batch_size,
                device,
                args.scrub_corruption_markers,
            )
            history.append(
                {
                    "epoch": epoch,
                    "train_pairwise_loss": total_loss / max(1, total_count),
                    "val_mrr": val_eval["metrics"]["mean_reciprocal_rank"],
                    "val_mean_gold_rank": val_eval["metrics"]["mean_gold_rank"],
                }
            )

    after_by_split = {
        split: evaluate(
            model,
            vocab,
            pools,
            manifest,
            args.max_length,
            args.eval_batch_size,
            device,
            args.scrub_corruption_markers,
        )
        for split, pools in pools_by_split.items()
    }

    before_test_mrr = before_by_split["test"]["metrics"]["mean_reciprocal_rank"]
    after_test = after_by_split["test"]["metrics"]
    after_test_mrr = after_test["mean_reciprocal_rank"]
    if after_test["all_gold_top"]:
        status = "pass"
    elif math.isfinite(after_test_mrr) and after_test_mrr > before_test_mrr:
        status = "warn"
    else:
        status = "fail"
    if args.require_gold_top and not after_test["all_gold_top"]:
        status = "fail"

    report = {
        "status": status,
        "mode": "train_val_test_splits",
        "config": {
            **jsonable_args(args),
            "device_resolved": str(device),
            "vocab_size": len(vocab.itos),
            "num_pairs": len(pairs),
            "num_tasks_by_split": {split: len(pools) for split, pools in pools_by_split.items()},
        },
        "veribench_root": str(veribench_root),
        "model": {
            "kind": "char_transformer_scalar_energy",
            "num_parameters": sum(param.numel() for param in model.parameters()),
        },
        "before_training_by_split": before_by_split,
        "after_training_by_split": after_by_split,
        "history": history,
        "train_seconds": time.time() - started,
        "vocab": vocab.to_json(),
    }

    json_path = args.output_dir / f"{args.tag}_veribench_splits_report.json"
    md_path = args.output_dir / f"{args.tag}_veribench_splits_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report, md_path)
    torch.save(model.state_dict(), args.output_dir / f"{args.tag}_char_energy_state.pt")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    if args.require_gold_top and status == "fail":
        raise RuntimeError("VeriBench split EBT run failed the requested gold-top criterion")
    return report


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.use_splits:
        return train_split_experiment(args)

    set_seed(args.seed)
    device = resolve_device(args.device)
    pools, manifest, veribench_root = load_candidate_pools(args)
    pairs = make_pairs(pools, manifest, args.holdout_task_id)
    if not pairs:
        raise ValueError("No training pairs were built")

    all_texts = []
    meta_by_id = {example["id"]: example for example in manifest["examples"]}
    for task_id, candidates in pools.items():
        all_texts.extend(candidate_input_text(candidate, meta_by_id[task_id], args.scrub_corruption_markers) for candidate in candidates)
    vocab = CharVocab(all_texts, args.max_vocab_size)
    model = CharEnergyTransformer(
        vocab_size=len(vocab.itos),
        max_length=args.max_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = random.Random(args.seed + 505)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    before = evaluate(
        model,
        vocab,
        pools,
        manifest,
        args.max_length,
        args.eval_batch_size,
        device,
        args.scrub_corruption_markers,
    )

    history = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        generator.shuffle(pairs)
        total_loss = 0.0
        total_count = 0
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            pos_texts = [candidate_input_text(pos, meta, args.scrub_corruption_markers) for pos, _neg, meta in batch_pairs]
            neg_texts = [candidate_input_text(neg, meta, args.scrub_corruption_markers) for _pos, neg, meta in batch_pairs]
            pos_ids, pos_mask = encode_texts(vocab, pos_texts, args.max_length, device)
            neg_ids, neg_mask = encode_texts(vocab, neg_texts, args.max_length, device)
            e_pos = model(pos_ids, pos_mask)
            e_neg = model(neg_ids, neg_mask)
            loss = F.softplus(e_pos - e_neg).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += float(loss.detach().cpu()) * len(batch_pairs)
            total_count += len(batch_pairs)

        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            history.append({"epoch": epoch, "train_pairwise_loss": total_loss / max(1, total_count)})

    after = evaluate(
        model,
        vocab,
        pools,
        manifest,
        args.max_length,
        args.eval_batch_size,
        device,
        args.scrub_corruption_markers,
    )

    before_mrr = before["metrics"]["mean_reciprocal_rank"]
    after_mrr = after["metrics"]["mean_reciprocal_rank"]
    if after["metrics"]["all_gold_top"]:
        status = "pass"
    elif math.isfinite(after_mrr) and after_mrr > before_mrr:
        status = "warn"
    else:
        status = "fail"
    if args.require_gold_top and not after["metrics"]["all_gold_top"]:
        status = "fail"

    report = {
        "status": status,
        "config": {
            **jsonable_args(args),
            "device_resolved": str(device),
            "vocab_size": len(vocab.itos),
            "num_pairs": len(pairs),
            "num_tasks": len(pools),
        },
        "veribench_root": str(veribench_root),
        "model": {
            "kind": "char_transformer_scalar_energy",
            "num_parameters": sum(param.numel() for param in model.parameters()),
        },
        "before_training": before,
        "after_training": after,
        "history": history,
        "train_seconds": time.time() - started,
        "vocab": vocab.to_json(),
    }

    json_path = args.output_dir / f"{args.tag}_veribench_report.json"
    md_path = args.output_dir / f"{args.tag}_veribench_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown_report(report, md_path)
    torch.save(model.state_dict(), args.output_dir / f"{args.tag}_char_energy_state.pt")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {md_path}", flush=True)
    if args.require_gold_top and status == "fail":
        raise RuntimeError("VeriBench EBT ranking run failed the requested gold-top criterion")
    return report


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    if "after_training_by_split" in report:
        summary_rows = []
        for split in ("train", "val", "test"):
            before = report["before_training_by_split"][split]["metrics"]
            after = report["after_training_by_split"][split]["metrics"]
            summary_rows.append(
                "| {split} | before | {mrr:.3f} | {top} | {rank:.2f} | {margin:.4f} |".format(
                    split=split,
                    mrr=before["mean_reciprocal_rank"],
                    top=before["all_gold_top"],
                    rank=before["mean_gold_rank"],
                    margin=before["mean_energy_margin"],
                )
            )
            summary_rows.append(
                "| {split} | after | {mrr:.3f} | {top} | {rank:.2f} | {margin:.4f} |".format(
                    split=split,
                    mrr=after["mean_reciprocal_rank"],
                    top=after["all_gold_top"],
                    rank=after["mean_gold_rank"],
                    margin=after["mean_energy_margin"],
                )
            )

        test_rows = []
        for task in report["after_training_by_split"]["test"]["task_results"]:
            test_rows.append(
                "| {task} | {n} | {rank} | {top} | {margin:.4f} |".format(
                    task=task["task_id"],
                    n=task["num_candidates"],
                    rank=task["gold_rank"],
                    top=task["top_candidate"]["candidate_id"],
                    margin=task["best_negative_minus_gold_energy"],
                )
            )

        lines = [
            "# VeriBench EBT Split Report",
            "",
            f"Status: `{report['status']}`",
            "",
            "## Train / Val / Test Summary",
            "",
            "| split | phase | MRR | all gold top | mean gold rank | mean margin |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *summary_rows,
            "",
            "## Test Tasks After Training",
            "",
            "| task | candidates | gold rank | top candidate | best negative - gold energy |",
            "| --- | ---: | ---: | --- | ---: |",
            *test_rows,
            "",
            "Lower energy is better. Ties with negatives count against the gold rank.",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    before = report["before_training"]["metrics"]
    after = report["after_training"]["metrics"]
    rows = []
    for task in report["after_training"]["task_results"]:
        rows.append(
            "| {task} | {n} | {rank} | {top} | {margin:.4f} |".format(
                task=task["task_id"],
                n=task["num_candidates"],
                rank=task["gold_rank"],
                top=task["top_candidate"]["candidate_id"],
                margin=task["best_negative_minus_gold_energy"],
            )
        )
    lines = [
        "# VeriBench EBT Ranking Report",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
        "| phase | MRR | all gold top | mean gold rank | mean margin |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| before | {mrr:.3f} | {top} | {rank:.2f} | {margin:.4f} |".format(
            mrr=before["mean_reciprocal_rank"],
            top=before["all_gold_top"],
            rank=before["mean_gold_rank"],
            margin=before["mean_energy_margin"],
        ),
        "| after | {mrr:.3f} | {top} | {rank:.2f} | {margin:.4f} |".format(
            mrr=after["mean_reciprocal_rank"],
            top=after["all_gold_top"],
            rank=after["mean_gold_rank"],
            margin=after["mean_energy_margin"],
        ),
        "",
        "## Per-Task After Training",
        "",
        "| task | candidates | gold rank | top candidate | best negative - gold energy |",
        "| --- | ---: | ---: | --- | ---: |",
        *rows,
        "",
        "Lower energy is better. Positive margin means the gold candidate has lower",
        "energy than the best negative candidate for that task.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="local")
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--veribench-root", type=Path, default=DEFAULT_VERIBENCH_ROOT)
    parser.add_argument("--use-splits", action="store_true", help="Train on train.jsonl and evaluate val/test JSONL splits.")
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-generated-agents", type=int, default=3)
    parser.add_argument("--add-wrong-task-gold-negatives", action="store_true", default=True)
    parser.add_argument("--no-wrong-task-gold-negatives", dest="add_wrong_task_gold_negatives", action="store_false")
    parser.add_argument("--max-wrong-task-gold-negatives", type=int, default=2)
    parser.add_argument("--add-corruption-negatives", action="store_true", default=True)
    parser.add_argument("--no-corruption-negatives", dest="add_corruption_negatives", action="store_false")
    parser.add_argument("--scrub-corruption-markers", action="store_true", default=True)
    parser.add_argument("--keep-corruption-markers", dest="scrub_corruption_markers", action="store_false")
    parser.add_argument("--holdout-task-id", default=None)
    parser.add_argument("--max-split-train-tasks", type=int, default=0, help="0 means all train tasks.")
    parser.add_argument("--max-split-val-tasks", type=int, default=0, help="0 means all val tasks.")
    parser.add_argument("--max-split-test-tasks", type=int, default=0, help="0 means all test tasks.")
    parser.add_argument("--max-vocab-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--require-gold-top", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
