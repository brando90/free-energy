#!/usr/bin/env python3
"""Limited language-modeling reproduction for Alexi EBT NLP models.

This intentionally avoids full FineWeb/RedPajama pretraining. It exercises the
same EBT_NLP and Baseline_Transformer_NLP forward/loss code on a tiny local text
subset so the experiment can run quickly in this repo environment.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.nlp.baseline_transformer import Baseline_Transformer_NLP
from model.nlp.ebt import EBT_NLP


TEXTS = [
    "Lean programs describe definitions, theorems, and proofs with precise dependent types.",
    "A language model predicts the next token from a prefix and improves by minimizing cross entropy.",
    "Energy based transformers refine a dense token prediction through gradient based inference steps.",
    "The small reproduction uses repeated examples so the model can overfit a limited language subset.",
    "Formal verification benefits from clear specifications, executable functions, and correctness lemmas.",
    "The baseline transformer predicts logits in one forward pass while the EBT optimizes token states.",
    "This corpus is intentionally tiny and deterministic; it is not a replacement for FineWeb pretraining.",
    "Evaluation reports train and validation loss on held out repeated sentences from the same distribution.",
]


class TinyLMDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        *,
        context_length: int,
        size: int,
        split: str,
    ) -> None:
        texts = TEXTS[:6] if split == "train" else TEXTS[6:]
        repeated = "\n".join(texts * max(1, (size // len(texts)) + 2))
        ids = tokenizer(repeated, add_special_tokens=False)["input_ids"]
        if len(ids) < context_length + 2:
            ids = ids * (((context_length + 2) // max(1, len(ids))) + 1)
        self.samples: list[torch.Tensor] = []
        stride = max(1, context_length // 2)
        for i in range(size):
            start = (i * stride) % (len(ids) - context_length - 1)
            self.samples.append(torch.tensor(ids[start : start + context_length + 1], dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.samples[index]}


def collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {"input_ids": torch.stack([sample["input_ids"] for sample in samples])}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hparams_for(args: argparse.Namespace, model_name: str) -> SimpleNamespace:
    values: dict[str, Any] = {
        "model_name": model_name,
        "tokenizer": args.tokenizer,
        "context_length": args.context_length,
        "batch_size_per_device": args.batch_size,
        "embedding_dim": args.embedding_dim,
        "num_transformer_blocks": args.layers,
        "multiheaded_attention_heads": args.heads,
        "ffn_dim_multiplier": args.ffn_dim_multiplier,
        "weight_initialization_method": "xavier",
        "weight_initialization_gain": 1.0,
        "execution_mode": "pretrain",
        "debug_unused_parameters": False,
        "mcmc_step_size": args.mcmc_step_size,
        "mcmc_step_size_lr_multiplier": args.alpha_lr_multiplier,
        "mcmc_num_steps": args.mcmc_steps,
        "ebt_type": "time_embed",
        "normalize_initial_condition": True,
        "denoising_initial_condition": "random_noise",
        "mcmc_step_size_learnable": True,
        "no_mcmc_detach": False,
        "ebt_norm": "rms",
        "ebt_act_func": "silu",
        "dyt_alpha_init": 0.5,
        "mcmc_replay_buffer": False,
        "gaussian_random_noise_scaling": 1.0,
        "normalize_initial_condition_only_first_step": False,
        "randomize_mcmc_step_size_scale": 1.0,
        "randomize_mcmc_num_steps": 0,
        "randomize_mcmc_num_steps_min": 0,
        "randomize_mcmc_num_steps_final_landscape": False,
        "langevin_dynamics_noise": 0.0,
        "langevin_dynamics_noise_learnable": False,
        "vocab_to_embed_uses_prob_dist": False,
        "num_modality_processing_mlp_layers": 1,
        "truncate_mcmc": False,
        "clamp_futures_grad": False,
        "clamp_futures_grad_max_change": 9.0,
        "absolute_clamp": 0.0,
        "clamp_max_after_warm_up": 0.0,
        "sharpen_predicted_distribution": 0.0,
        "reconstruction_coeff": 1.0,
        "contrastive_loss": False,
        "contrastive_loss_coeff": 0.0005,
        "soften_target_prob_dist": 0.0,
        "discrete_contrastive_loss_true_logit_val": 0.0,
        "norm_pred": False,
        "norm_pred_not_final_step": False,
        "scale_alpha_with_energy": False,
        "scale_alpha_with_energy_temp": 9.0,
    }
    return SimpleNamespace(**values)


def make_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    if hasattr(model, "alpha"):
        alpha_params = [model.alpha]
        alpha_ids = {id(p) for p in alpha_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in alpha_ids]
        return torch.optim.AdamW(
            [
                {"params": alpha_params, "lr": args.lr * args.alpha_lr_multiplier, "weight_decay": 0.0},
                {"params": other_params, "lr": args.lr, "weight_decay": args.weight_decay},
            ],
            betas=(0.9, 0.999),
            fused=torch.cuda.is_available(),
        )
    return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=torch.cuda.is_available())


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    finals: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = move_batch(batch, device)
        with torch.enable_grad():
            metrics = model.forward_loss_wrapper(batch, "valid")
        losses.append(float(metrics["loss"].detach().float().cpu()))
        finals.append(float(metrics.get("final_step_loss", metrics["loss"]).detach().float().cpu()))
    model.train()
    loss = float(np.mean(losses)) if losses else float("nan")
    final = float(np.mean(finals)) if finals else loss
    return {"loss": loss, "final_step_loss": final, "perplexity": float(np.exp(final))}


def train_one(model_name: str, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, clean_up_tokenization_spaces=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    train_ds = TinyLMDataset(tokenizer, context_length=args.context_length, size=args.train_samples, split="train")
    val_ds = TinyLMDataset(tokenizer, context_length=args.context_length, size=args.val_samples, split="val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate, drop_last=False)
    model_cls = {"ebt": EBT_NLP, "baseline_transformer": Baseline_Transformer_NLP}[model_name]
    model = model_cls(hparams_for(args, model_name)).to(device)
    optimizer = make_optimizer(model, args)

    metrics_path = run_dir / f"{model_name}_metrics.jsonl"
    best = {"step": 0, "loss": float("inf"), "final_step_loss": float("inf")}
    step = 0
    start = time.perf_counter()
    while step < args.steps:
        for batch in train_loader:
            step += 1
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                metrics = model.forward_loss_wrapper(batch, "train")
            loss = metrics["loss"]
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"{model_name} non-finite loss at step {step}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            row = {
                "model": model_name,
                "step": step,
                "loss": float(loss.detach().float().cpu()),
                "final_step_loss": float(metrics.get("final_step_loss", loss).detach().float().cpu()),
                "perplexity": float(metrics.get("perplexity", torch.exp(loss.detach())).detach().float().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().float().cpu()),
                "elapsed_sec": time.perf_counter() - start,
                "alpha": float(model.alpha.detach().float().cpu()) if hasattr(model, "alpha") else None,
            }
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                val = evaluate(model, val_loader, device, args.eval_batches)
                row.update({f"val_{key}": value for key, value in val.items()})
                if val["final_step_loss"] < best["final_step_loss"]:
                    best = {"step": step, **val}
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                print(json.dumps(row, sort_keys=True), flush=True)
            if step >= args.steps:
                break

    torch.save({"model": model.state_dict(), "best": best, "args": vars(args)}, run_dir / f"{model_name}_final.pt")
    return {"model": model_name, "best": best, "final_step": step, "metrics": str(metrics_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--val-samples", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--ffn-dim-multiplier", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.0012)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--alpha-lr-multiplier", type=float, default=1.5)
    parser.add_argument("--mcmc-step-size", type=float, default=0.5)
    parser.add_argument("--mcmc-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--models", nargs="+", default=["baseline_transformer", "ebt"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--run-dir", type=Path, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_float32_matmul_precision("medium")
    run_dir = args.run_dir or ROOT.parent / "runs" / "alexiglad_lm_subset" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str), encoding="utf-8")
    summaries = [train_one(model_name, args, run_dir) for model_name in args.models]
    (run_dir / "summary.json").write_text(json.dumps({"run_dir": str(run_dir), "summaries": summaries}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "summaries": summaries}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
