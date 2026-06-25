#!/usr/bin/env python3
"""Autoregressively sample the 5-example VeriBench EBT overfit checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from model.nlp.veribench_context_ebt import VeriBenchContextEBT, VeriBenchContextEBTConfig
from veribench_embedding_dataloader import VeriBenchEmbeddingDataset, collate_veribench_embedding_samples

try:
    from inference.nlp.generate_text import sample_top_p
except ModuleNotFoundError:
    def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        return torch.gather(probs_idx, -1, next_token)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def load_run_args(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def build_model(run_args: dict[str, Any], vocab_size: int, device: torch.device) -> VeriBenchContextEBT:
    max_target_tokens = int(run_args.get("max_target_tokens", 256))
    cfg = VeriBenchContextEBTConfig(
        vocab_size=vocab_size,
        context_dim=4096,
        hidden_dim=int(run_args.get("hidden_dim", 384)),
        num_layers=int(run_args.get("num_layers", 6)),
        num_heads=int(run_args.get("num_heads", 6)),
        ffn_dim_multiplier=4.0,
        max_seq_len=max_target_tokens * 2 + 8,
        max_mcmc_steps=int(run_args.get("mcmc_steps", 2)),
        mcmc_num_steps=int(run_args.get("mcmc_steps", 2)),
        mcmc_step_size=float(run_args.get("mcmc_step_size", 0.5)),
        mcmc_step_size_learnable=True,
        no_mcmc_detach=bool(run_args.get("no_mcmc_detach", False)),
        truncate_mcmc=bool(run_args.get("truncate_mcmc", False)),
        use_context=not bool(run_args.get("no_context", False)),
    )
    return VeriBenchContextEBT(cfg).to(device)


def decode_local_ids(dataset: VeriBenchEmbeddingDataset, tokenizer: AutoTokenizer, local_ids: list[int]) -> str:
    if not local_ids:
        return ""
    ids = torch.tensor(local_ids, dtype=torch.long)
    original_ids = dataset.id_mapper.original_ids(ids).tolist()
    return tokenizer.decode(original_ids, skip_special_tokens=True)


def trim_at_eos(ids: list[int], eos_id: int) -> list[int]:
    return ids[: ids.index(eos_id)] if eos_id in ids else ids


def sample_batch(
    model: VeriBenchContextEBT,
    batch: dict[str, Any],
    *,
    bos_id: int,
    eos_id: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    device = next(model.parameters()).device
    context = batch["context_activations"]
    context_mask = batch["context_attention_mask"]
    batch_size = context.shape[0]
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        target_mask = torch.ones_like(generated, dtype=torch.bool)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            with torch.enable_grad():
                out = model(
                    generated,
                    context,
                    context_mask=context_mask,
                    target_mask=target_mask,
                    learning=False,
                    capture_intermediates=False,
                )
        logits = out["predicted_distributions"][-1][:, -1].float()
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p).reshape(-1)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token[:, None]], dim=1)
        finished |= next_token.eq(eos_id)
        if bool(finished.all()):
            break
    return generated[:, 1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_args = load_run_args(args.run_dir)
    data_dir = args.data_dir or Path(run_args["data_dir"])
    max_target_tokens = int(run_args.get("max_target_tokens", 256))
    max_new_tokens = args.max_new_tokens or max_target_tokens
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else args.run_dir / args.checkpoint

    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = VeriBenchEmbeddingDataset(
        data_dir=data_dir,
        split=str(run_args.get("split", "val")),
        max_items=int(run_args.get("max_items", 5)),
        max_target_tokens=max_target_tokens,
        activation_dtype="bf16",
        validate_context=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=len(dataset),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_veribench_embedding_samples,
    )
    batch = move_batch(next(iter(loader)), device)
    model = build_model(run_args, int(dataset.vocab["vocab_size"]), device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(dataset.model_name, trust_remote_code=True)
    bos_id = int(dataset.vocab["local_bos_id"])
    eos_id = int(dataset.vocab["local_eos_id"])
    generations = sample_batch(
        model,
        batch,
        bos_id=bos_id,
        eos_id=eos_id,
        max_new_tokens=max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    ).cpu()

    output_path = args.output or (args.run_dir / f"sample_{checkpoint_path.stem}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    exact = 0
    token_matches = 0
    token_total = 0
    for i, task_name in enumerate(batch["task_name"]):
        gold = batch["labels"][i].detach().cpu()
        mask = batch["label_attention_mask"][i].detach().cpu().bool()
        gold_ids = gold[mask].tolist()
        pred_ids = trim_at_eos(generations[i].tolist(), eos_id)
        compare_len = min(len(gold_ids), len(pred_ids))
        matches = sum(int(a == b) for a, b in zip(gold_ids[:compare_len], pred_ids[:compare_len]))
        token_matches += matches
        token_total += len(gold_ids)
        is_exact = pred_ids == gold_ids
        exact += int(is_exact)
        rows.append(
            {
                "task_name": task_name,
                "split": batch["split"][i],
                "family": batch["family"][i],
                "exact": is_exact,
                "token_matches": matches,
                "gold_tokens": len(gold_ids),
                "pred_tokens": len(pred_ids),
                "token_accuracy_vs_gold_len": matches / max(1, len(gold_ids)),
                "generation": decode_local_ids(dataset, tokenizer, pred_ids),
                "gold": decode_local_ids(dataset, tokenizer, gold_ids),
                "generated_local_ids": pred_ids,
                "gold_local_ids": gold_ids,
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "examples": len(rows),
        "exact": exact,
        "exact_rate": exact / max(1, len(rows)),
        "token_matches": token_matches,
        "gold_tokens": token_total,
        "token_accuracy_vs_gold_len": token_matches / max(1, token_total),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    for row in rows:
        print(
            f"{row['task_name']}: exact={row['exact']} "
            f"token_acc={row['token_accuracy_vs_gold_len']:.4f} "
            f"pred_tokens={row['pred_tokens']} gold_tokens={row['gold_tokens']}"
        )
        print("GENERATION_HEAD:", row["generation"][:240].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
