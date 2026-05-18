#!/usr/bin/env python3
"""Train a tiny transformer scalar energy model on the three-example pool.

This is the SNAP-cluster starter, not the final research system. It uses the
same candidate pools as `pilot_ebm_ranking.py` and trains a cross-encoder
energy function E(task, candidate), where lower energy should mean better Lean
artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pilot_ebm_ranking as pilot


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "results" / "transformer_energy"


def candidate_input_text(candidate: pilot.Candidate, task_meta: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"TASK_ID: {candidate.task_id}",
            f"TASK_ROLE: {task_meta.get('role', '')}",
            f"TASK_REASON: {task_meta.get('why', '')}",
            "LEAN_CANDIDATE:",
            candidate.text,
        ]
    )


def make_pairs(
    pools: dict[str, list[pilot.Candidate]],
    manifest: dict[str, Any],
) -> list[tuple[pilot.Candidate, pilot.Candidate, dict[str, Any]]]:
    meta_by_id = {example["id"]: example for example in manifest["examples"]}
    pairs: list[tuple[pilot.Candidate, pilot.Candidate, dict[str, Any]]] = []
    for task_id, candidates in pools.items():
        positives = [c for c in candidates if c.label == 1]
        negatives = [c for c in candidates if c.label == 0]
        if not positives:
            raise ValueError(f"No positive candidate for task {task_id}")
        for positive in positives:
            for negative in negatives:
                pairs.append((positive, negative, meta_by_id[task_id]))
    return pairs


def import_training_deps():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    return torch, nn, F, AutoModel, AutoTokenizer


def build_energy_model(model_name: str):
    torch, nn, _F, AutoModel, _AutoTokenizer = import_training_deps()

    class EnergyModel(nn.Module):
        def __init__(self, base_name: str):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(base_name)
            hidden = self.encoder.config.hidden_size
            self.energy_head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 1),
            )

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            return self.energy_head(pooled).squeeze(-1)

    return EnergyModel(model_name)


def tokenize_batch(tokenizer, texts: list[str], device: str, max_length: int):
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in batch.items()}


def score_candidates(model, tokenizer, candidates, meta_by_id, device: str, max_length: int):
    torch, _nn, _F, _AutoModel, _AutoTokenizer = import_training_deps()
    rows = []
    model.eval()
    with torch.no_grad():
        for candidate in candidates:
            text = candidate_input_text(candidate, meta_by_id[candidate.task_id])
            batch = tokenize_batch(tokenizer, [text], device, max_length)
            energy = float(model(**batch).item())
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "kind": candidate.kind,
                    "label": candidate.label,
                    "energy": energy,
                    "source_path": candidate.source_path,
                }
            )
    return sorted(rows, key=lambda row: row["energy"])


def evaluate(model, tokenizer, pools, manifest, device: str, max_length: int):
    meta_by_id = {example["id"]: example for example in manifest["examples"]}
    task_results = []
    reciprocal_ranks = []
    for task_id, candidates in pools.items():
        ranked = score_candidates(model, tokenizer, candidates, meta_by_id, device, max_length)
        for idx, row in enumerate(ranked, start=1):
            row["rank"] = idx
        gold_rank = next(row["rank"] for row in ranked if row["label"] == 1)
        reciprocal_ranks.append(1.0 / gold_rank)
        task_results.append(
            {
                "task_id": task_id,
                "gold_rank": gold_rank,
                "top_candidate": ranked[0],
                "ranked_candidates": ranked,
            }
        )
    return {
        "mean_reciprocal_rank": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "all_gold_top": all(task["gold_rank"] == 1 for task in task_results),
        "task_results": task_results,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    torch, _nn, F, _AutoModel, AutoTokenizer = import_training_deps()

    manifest = pilot.load_manifest(args.manifest)
    veribench_root = pilot.resolve_veribench_root(manifest, args.veribench_root)
    pools = pilot.build_candidate_pools(manifest, veribench_root, args.max_generated_agents)
    pairs = make_pairs(pools, manifest)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = build_energy_model(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for start in range(0, len(pairs), args.batch_size):
            batch_pairs = pairs[start : start + args.batch_size]
            pos_texts = [candidate_input_text(pos, meta) for pos, _neg, meta in batch_pairs]
            neg_texts = [candidate_input_text(neg, meta) for _pos, neg, meta in batch_pairs]
            pos_batch = tokenize_batch(tokenizer, pos_texts, device, args.max_length)
            neg_batch = tokenize_batch(tokenizer, neg_texts, device, args.max_length)

            e_pos = model(**pos_batch)
            e_neg = model(**neg_batch)
            loss = -F.logsigmoid(e_neg - e_pos).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_pairs)

        eval_report = evaluate(model, tokenizer, pools, manifest, device, args.max_length)
        epoch_record = {
            "epoch": epoch,
            "loss": total_loss / max(1, len(pairs)),
            **eval_report,
        }
        history.append(epoch_record)
        print(
            f"epoch={epoch} loss={epoch_record['loss']:.4f} "
            f"mrr={epoch_record['mean_reciprocal_rank']:.4f} "
            f"all_gold_top={epoch_record['all_gold_top']}"
        )

    report = {
        "model_name": args.model_name,
        "veribench_root": str(veribench_root),
        "num_pairs": len(pairs),
        "num_tasks": len(pools),
        "epochs": args.epochs,
        "device": device,
        "history": history,
        "final": history[-1] if history else None,
    }

    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    torch.save(model.state_dict(), output_dir / "energy_model_state.pt")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=pilot.DEFAULT_MANIFEST)
    parser.add_argument("--veribench-root", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-name", default="microsoft/codebert-base")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-generated-agents", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    report = train(args)
    final = report["final"]
    if final is None:
        return 1
    print(f"wrote={args.output_dir / 'training_report.json'}")
    return 0 if math.isfinite(final["loss"]) else 1


if __name__ == "__main__":
    sys.exit(main())
