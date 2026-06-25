#!/usr/bin/env python3
"""Overfit alexiglad/EBT-style model on 5 VeriBench examples with Goedel activations."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIR = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from model.nlp.veribench_context_ebt import VeriBenchContextEBT, VeriBenchContextEBTConfig
from veribench_embedding_dataloader import VeriBenchEmbeddingDataset, collate_veribench_embedding_samples


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=EXPERIMENT_DIR / "data" / "context_gold")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-items", type=int, default=5)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--alpha-lr-mult", type=float, default=1.5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--mcmc-step-size", type=float, default=0.5)
    parser.add_argument("--no-mcmc-detach", action="store_true")
    parser.add_argument("--truncate-mcmc", action="store_true")
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = args.run_dir or (EXPERIMENT_DIR / "runs" / "alexiglad_ebt_veribench5" / time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    activation_dir = run_dir / "intermediate_activations"
    activation_dir.mkdir(parents=True, exist_ok=True)

    dataset = VeriBenchEmbeddingDataset(
        data_dir=args.data_dir,
        split=args.split,
        max_items=args.max_items,
        max_target_tokens=args.max_target_tokens,
        activation_dtype="bf16",
        validate_context=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_veribench_embedding_samples,
    )
    if len(dataset) == 0:
        raise ValueError("No VeriBench samples loaded")

    cfg = VeriBenchContextEBTConfig(
        vocab_size=int(dataset.vocab["vocab_size"]),
        context_dim=4096,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim_multiplier=4.0,
        max_seq_len=args.max_target_tokens * 2 + 8,
        max_mcmc_steps=args.mcmc_steps,
        mcmc_num_steps=args.mcmc_steps,
        mcmc_step_size=args.mcmc_step_size,
        mcmc_step_size_learnable=True,
        no_mcmc_detach=args.no_mcmc_detach,
        truncate_mcmc=args.truncate_mcmc,
        use_context=not args.no_context,
    )
    model = VeriBenchContextEBT(cfg).to(device)
    alpha_params = [model.alpha]
    alpha_ids = {id(p) for p in alpha_params}
    other_params = [p for p in model.parameters() if id(p) not in alpha_ids and p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": alpha_params, "lr": args.lr * args.alpha_lr_mult, "weight_decay": 0.0},
            {"params": other_params, "lr": args.lr, "weight_decay": 0.01},
        ],
        betas=(0.9, 0.999),
        fused=torch.cuda.is_available(),
    )

    def lr_lambda(step: int) -> float:
        return min(1.0, max(1, step) / max(1, args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    metrics_path = run_dir / "metrics.jsonl"
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(serializable_args, indent=2, sort_keys=True), encoding="utf-8")
    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"dataset_size={len(dataset)} tasks={[task.task_name for task in dataset.tasks]}")

    step = 0
    latest: dict[str, Any] = {}
    start = time.perf_counter()
    while step < args.steps:
        for batch in loader:
            batch = move_batch(batch, device)
            step += 1
            capture = step == 1 or step % args.save_every == 0 or step == args.steps
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model.loss(
                    batch["decoder_input_ids"],
                    batch["labels"],
                    batch["context_activations"],
                    context_mask=batch["context_attention_mask"],
                    target_mask=batch["label_attention_mask"],
                    capture_intermediates=capture,
                )
            loss = out["loss"]
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            token_count = int(batch["label_attention_mask"].sum().detach().cpu())
            elapsed = time.perf_counter() - start
            latest = {
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "initial_loss": float(out["initial_loss"].detach().float().cpu()),
                "final_step_loss": float(out["final_step_loss"].detach().float().cpu()),
                "perplexity": float(out["perplexity"].detach().float().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().float().cpu()),
                "alpha": float(model.alpha.detach().float().cpu()),
                "lr_alpha": float(optimizer.param_groups[0]["lr"]),
                "lr_model": float(optimizer.param_groups[1]["lr"]),
                "tokens": token_count,
                "elapsed_sec": elapsed,
                "cuda_max_mem_gb": torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(latest, sort_keys=True) + "\n")
            if step == 1 or step % args.log_every == 0:
                print(
                    "step={step} loss={loss:.4f} final={final_step_loss:.4f} "
                    "ppl={perplexity:.1f} alpha={alpha:.4f} mem={cuda_max_mem_gb:.2f}GB".format(**latest),
                    flush=True,
                )
            if capture:
                torch.save(
                    {
                        "step": step,
                        "task_name": batch["task_name"],
                        "decoder_input_ids": batch["decoder_input_ids"].detach().cpu(),
                        "labels": batch["labels"].detach().cpu(),
                        "context_attention_mask": batch["context_attention_mask"].detach().cpu(),
                        "label_attention_mask": batch["label_attention_mask"].detach().cpu(),
                        "metrics": latest,
                        "intermediates": out["intermediates"],
                    },
                    activation_dir / f"step_{step:06d}.pt",
                )
            if step % args.save_every == 0 or step == args.steps:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "metrics": latest,
                    },
                    run_dir / f"checkpoint_step_{step:06d}.pt",
                )
            if step >= args.steps:
                break

    summary = {"run_dir": str(run_dir), "steps": step, "latest_metrics": latest}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
