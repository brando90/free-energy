#!/usr/bin/env python3
"""Hydra trainer for VeriBench EBT runs."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler

from ebt import GoedelVocabEBT
from veribench_embedding_dataloader import (
    VeriBenchEmbeddingDataset,
    collate_veribench_embedding_samples,
)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _metric_value(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu())
    return float(value)


def _optional_container(value: Any) -> Any:
    if value is None:
        return None
    return OmegaConf.to_container(value, resolve=True)


class TargetTokenBatchSampler(Sampler[list[int]]):
    """Groups examples by target length under a token budget."""

    def __init__(
        self,
        dataset: VeriBenchEmbeddingDataset,
        *,
        max_items_per_batch: int,
        max_target_tokens_per_batch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        cap = dataset.max_target_tokens
        self.lengths = [
            min(int(task.target_tokens) + 1, int(cap)) if cap is not None else int(task.target_tokens) + 1
            for task in dataset.tasks
        ]
        self.max_items_per_batch = int(max_items_per_batch)
        self.max_target_tokens_per_batch = int(max_target_tokens_per_batch)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches: list[list[int]] = []
        current: list[int] = []
        current_max = 0
        for idx in order:
            length = self.lengths[idx]
            proposed_max = max(current_max, length)
            proposed_size = len(current) + 1
            proposed_tokens = proposed_max * proposed_size
            over_items = proposed_size > self.max_items_per_batch
            over_tokens = proposed_tokens > self.max_target_tokens_per_batch
            if current and (over_items or over_tokens):
                batches.append(current)
                current = []
                current_max = 0
            current.append(idx)
            current_max = max(current_max, length)
        if current:
            batches.append(current)

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        count = 0
        current_size = 0
        current_max = 0
        for length in sorted(self.lengths):
            proposed_size = current_size + 1
            proposed_max = max(current_max, length)
            if current_size and (
                proposed_size > self.max_items_per_batch
                or proposed_size * proposed_max > self.max_target_tokens_per_batch
            ):
                count += 1
                current_size = 0
                current_max = 0
            current_size += 1
            current_max = max(current_max, length)
        return count + int(current_size > 0)


def _make_loader(
    cfg: DictConfig,
    *,
    split: str | list[str] | None = None,
    shuffle: bool | None = None,
    max_items: int | None = None,
) -> DataLoader[dict[str, Any]]:
    families = _optional_container(cfg.data.families)
    configured_splits = _optional_container(cfg.data.get("splits")) if "splits" in cfg.data else None
    selected_split = split if split is not None else (configured_splits if configured_splits is not None else cfg.data.split)
    dataset = VeriBenchEmbeddingDataset(
        data_dir=_as_path(cfg.data.data_dir),
        split=selected_split,
        families=families,
        max_items=cfg.data.max_items if max_items is None else max_items,
        max_target_tokens=cfg.data.max_target_tokens,
        activation_dtype=cfg.data.activation_dtype,
        model_name=cfg.data.model_name or cfg.model.model_name,
        model_revision=cfg.data.model_revision or cfg.model.revision,
        validate_context=cfg.data.validate_context,
    )
    if len(dataset) == 0:
        raise ValueError("No VeriBench samples matched the configured split/family filters")

    persistent_workers = bool(cfg.loader.persistent_workers) and int(cfg.loader.num_workers) > 0
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "num_workers": int(cfg.loader.num_workers),
        "pin_memory": bool(cfg.loader.pin_memory),
        "persistent_workers": persistent_workers,
        "collate_fn": collate_veribench_embedding_samples,
    }
    if cfg.loader.max_target_tokens_per_batch is not None:
        loader_kwargs["batch_sampler"] = TargetTokenBatchSampler(
            dataset,
            max_items_per_batch=int(cfg.loader.batch_size),
            max_target_tokens_per_batch=int(cfg.loader.max_target_tokens_per_batch),
            shuffle=bool(cfg.loader.shuffle) if shuffle is None else bool(shuffle),
            seed=int(cfg.seed),
        )
    else:
        loader_kwargs["batch_size"] = int(cfg.loader.batch_size)
        loader_kwargs["shuffle"] = bool(cfg.loader.shuffle) if shuffle is None else bool(shuffle)
        loader_kwargs["drop_last"] = bool(cfg.loader.drop_last)
    if int(cfg.loader.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.loader.prefetch_factor)
    return DataLoader(**loader_kwargs)


def _init_wandb(cfg: DictConfig, run_dir: Path, dataset_size: int, validation_size: int) -> Any:
    if not bool(cfg.wandb.enabled):
        return None
    try:
        import wandb
    except Exception as exc:
        print(f"wandb_unavailable={exc!r}", flush=True)
        return None
    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=(None if cfg.wandb.entity is None else str(cfg.wandb.entity)),
        name=(None if cfg.wandb.name is None else str(cfg.wandb.name)),
        dir=str(run_dir),
        config={
            **OmegaConf.to_container(cfg, resolve=True),
            "dataset_size": dataset_size,
            "validation_size": validation_size,
        },
        tags=list(_optional_container(cfg.wandb.tags) or []),
    )
    return run


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    cfg: DictConfig,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int | None = None,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {
        "loss": 0.0,
        "initial_loss": 0.0,
        "final_step_loss": 0.0,
        "perplexity": 0.0,
        "final_energy": 0.0,
        "final_token_accuracy": 0.0,
    }
    exact_total = 0.0
    total_tokens = 0
    total_rows = 0
    batches = 0
    start = time.perf_counter()
    for batch in loader:
        batch = _move_batch(batch, device)
        tokens = int(batch["label_attention_mask"].sum().detach().cpu())
        rows = int(batch["label_attention_mask"].shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            loss_dict = model.loss(
                batch["context_activations"],
                batch[str(cfg.data.label_field)],
                context_attention_mask=batch["context_attention_mask"],
                label_attention_mask=batch["label_attention_mask"],
                task_indices=batch["task_index"],
                learning=False,
            )
        weight = max(tokens, 1)
        for key in totals:
            totals[key] += _metric_value(loss_dict[key]) * weight
        exact_total += _metric_value(loss_dict["final_exact_accuracy"]) * rows
        total_tokens += tokens
        total_rows += rows
        batches += 1
        if max_batches is not None and batches >= max_batches:
            break
    denom = max(total_tokens, 1)
    metrics = {key: value / denom for key, value in totals.items()}
    metrics["final_exact_accuracy"] = exact_total / max(total_rows, 1)
    metrics.update(
        {
            "tokens": float(total_tokens),
            "rows": float(total_rows),
            "batches": float(batches),
            "elapsed_sec": time.perf_counter() - start,
        }
    )
    if was_training:
        model.train()
    return metrics


def _resolve_dataset_int(value: Any, dataset: VeriBenchEmbeddingDataset, key: str) -> int | None:
    if value is None:
        return None
    if str(value) == "dataset":
        return int(dataset.vocab[key])
    return int(value)


def _make_model(cfg: DictConfig, dataset: VeriBenchEmbeddingDataset) -> GoedelVocabEBT:
    return GoedelVocabEBT(
        model_name=cfg.model.model_name,
        revision=cfg.model.revision,
        vocab_size=_resolve_dataset_int(cfg.model.vocab_size, dataset, "vocab_size"),
        pad_token_id=_resolve_dataset_int(cfg.model.pad_token_id, dataset, "local_pad_id"),
        context_dim=cfg.model.context_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dim_feedforward=int(cfg.model.dim_feedforward),
        dropout=float(cfg.model.dropout),
        mcmc_num_steps=int(cfg.model.mcmc_num_steps),
        mcmc_step_size=float(cfg.model.mcmc_step_size),
        mcmc_step_size_learnable=bool(cfg.model.mcmc_step_size_learnable),
        gaussian_random_noise_scaling=float(cfg.model.gaussian_random_noise_scaling),
        denoising_initial_condition=cfg.model.denoising_initial_condition,
        normalize_initial_condition=bool(cfg.model.normalize_initial_condition),
        normalize_initial_condition_only_first_step=bool(cfg.model.normalize_initial_condition_only_first_step),
        langevin_dynamics_noise=float(cfg.model.langevin_dynamics_noise),
        truncate_mcmc=bool(cfg.model.truncate_mcmc),
        no_mcmc_detach=bool(cfg.model.no_mcmc_detach),
        clamp_futures_grad=bool(cfg.model.clamp_futures_grad),
        clamp_futures_grad_max_change=float(cfg.model.clamp_futures_grad_max_change),
        absolute_clamp=float(cfg.model.absolute_clamp),
        sharpen_predicted_distribution=float(cfg.model.sharpen_predicted_distribution),
        norm_pred=bool(cfg.model.norm_pred),
        norm_pred_not_final_step=bool(cfg.model.norm_pred_not_final_step),
        reconstruction_coeff=float(cfg.model.reconstruction_coeff),
        soften_target_prob_dist=float(cfg.model.soften_target_prob_dist),
        loss_on_final_step_only=bool(cfg.model.loss_on_final_step_only),
        use_context_activations=bool(cfg.model.use_context_activations),
        max_task_embeddings=int(cfg.model.max_task_embeddings),
        max_target_positions=int(cfg.model.max_target_positions),
    )


def _make_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    named_params = list(model.named_parameters())
    alpha_params = [param for name, param in named_params if name.endswith("alpha") and param.requires_grad]
    alpha_param_ids = {id(param) for param in alpha_params}
    other_params = [param for _, param in named_params if id(param) not in alpha_param_ids and param.requires_grad]
    if alpha_params:
        params: Any = [
            {
                "params": alpha_params,
                "weight_decay": 0.0,
                "lr": float(cfg.optim.lr) * float(cfg.optim.mcmc_step_size_lr_multiplier),
            },
            {
                "params": other_params,
                "weight_decay": float(cfg.optim.weight_decay),
                "lr": float(cfg.optim.lr),
            },
        ]
    else:
        params = other_params

    kwargs = {
        "lr": float(cfg.optim.lr),
        "weight_decay": float(cfg.optim.weight_decay),
        "betas": tuple(float(x) for x in cfg.optim.betas),
        "eps": float(cfg.optim.eps),
    }
    if torch.cuda.is_available() and bool(cfg.optim.fused):
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(params, **kwargs)
    except TypeError:
        kwargs.pop("fused", None)
        return torch.optim.AdamW(params, **kwargs)


def _make_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler.LRScheduler | None:
    if not bool(cfg.optim.use_scheduler):
        return None

    warmup_steps = int(cfg.optim.warm_up_steps)
    max_steps = int(cfg.optim.max_scheduling_steps)
    min_lr_scale = float(cfg.optim.min_lr_scale)
    if max_steps <= warmup_steps:
        raise ValueError("optim.max_scheduling_steps must be greater than optim.warm_up_steps")

    base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    min_lrs = [lr / min_lr_scale for lr in base_lrs]
    warmup_divider = float(cfg.optim.warm_up_base_lr_divider)

    def lr_lambda_for_group(group_idx: int):
        base_lr = base_lrs[group_idx]
        min_lr = min_lrs[group_idx]

        def lr_lambda(step: int) -> float:
            if warmup_steps > 0 and step <= warmup_steps:
                if warmup_divider == -1:
                    return step / warmup_steps
                return (1.0 / warmup_divider) + (1.0 - (1.0 / warmup_divider)) * (step / warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr / base_lr) + (1.0 - (min_lr / base_lr)) * cosine

        return lr_lambda

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lr_lambda_for_group(i) for i in range(len(optimizer.param_groups))],
    )


def _save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict[str, float],
    cfg: DictConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(
        {
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        path,
    )


def _prune_checkpoints(run_dir: Path, keep: int) -> None:
    if keep <= 0:
        return
    checkpoints = sorted(run_dir.glob("checkpoint_step_*.pt"), key=lambda p: p.stat().st_mtime)
    for checkpoint in checkpoints[:-keep]:
        checkpoint.unlink(missing_ok=True)


@hydra.main(version_base=None, config_path="configs", config_name="train_config")
def main(cfg: DictConfig) -> None:
    configured_splits = _optional_container(cfg.data.get("splits")) if "splits" in cfg.data else None
    if configured_splits is None and cfg.data.split != "val" and not bool(cfg.allow_non_val_split):
        raise ValueError("This trainer defaults to validation overfit only; set allow_non_val_split=true")
    if configured_splits is not None and any(split != "val" for split in configured_splits) and not bool(cfg.allow_non_val_split):
        raise ValueError("Non-val training splits require allow_non_val_split=true")

    if not torch.cuda.is_available() and not bool(cfg.allow_cpu):
        raise RuntimeError("CUDA is required by default. Set allow_cpu=true only for debugging.")

    _seed_everything(int(cfg.seed))
    if cfg.matmul_precision:
        torch.set_float32_matmul_precision(str(cfg.matmul_precision))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")

    loader = _make_loader(cfg)
    dataset = loader.dataset
    validation_loader = _make_loader(
        cfg,
        split=str(cfg.validation.split),
        shuffle=False,
        max_items=(None if cfg.validation.max_items is None else int(cfg.validation.max_items)),
    )
    validation_dataset = validation_loader.dataset
    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"dataset_size={len(dataset)} split={configured_splits or cfg.data.split} families={cfg.data.families}")
    print(f"validation_size={len(validation_dataset)} split={cfg.validation.split}")

    model = _make_model(cfg, dataset).to(device)
    if bool(cfg.compile_model):
        model = torch.compile(model)
    optimizer = _make_optimizer(model, cfg)
    scheduler = _make_scheduler(optimizer, cfg)

    metrics_path = run_dir / "metrics.jsonl"
    max_steps = int(cfg.train.max_steps)
    grad_accum_steps = max(1, int(cfg.train.grad_accum_steps))
    log_every = max(1, int(cfg.train.log_every))
    save_every = max(1, int(cfg.train.save_every))
    val_every = max(1, int(cfg.validation.every_steps))
    amp_enabled = bool(cfg.train.amp_bf16) and device.type == "cuda"
    wandb_run = _init_wandb(cfg, run_dir, len(dataset), len(validation_dataset))

    step = 0
    accum = 0
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()
    last_log = train_start
    running_loss = 0.0
    running_tokens = 0
    running_batches = 0
    latest_metrics: dict[str, float] = {}
    latest_validation_metrics: dict[str, float] = {}

    while step < max_steps:
        for batch in loader:
            batch_start = time.perf_counter()
            batch = _move_batch(batch, device)
            token_count = int(batch["label_attention_mask"].sum().detach().cpu())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                loss_dict = model.loss(
                    batch["context_activations"],
                    batch[str(cfg.data.label_field)],
                    context_attention_mask=batch["context_attention_mask"],
                    label_attention_mask=batch["label_attention_mask"],
                    task_indices=batch["task_index"],
                )
                loss = loss_dict["loss"] / grad_accum_steps

            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {float(loss.detach().cpu())}")

            loss.backward()
            accum += 1
            running_loss += _metric_value(loss_dict["loss"])
            running_tokens += token_count
            running_batches += 1

            if accum >= grad_accum_steps:
                if float(cfg.train.grad_clip_norm) > 0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
                    )
                else:
                    grad_norm = math.nan
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accum = 0

                now = time.perf_counter()
                latest_metrics = {
                    "step": float(step),
                    "loss": running_loss / max(1, running_batches),
                    "initial_loss": _metric_value(loss_dict["initial_loss"]),
                    "final_step_loss": _metric_value(loss_dict["final_step_loss"]),
                    "perplexity": _metric_value(loss_dict["perplexity"]),
                    "final_energy": _metric_value(loss_dict["final_energy"]),
                    "final_token_accuracy": _metric_value(loss_dict["final_token_accuracy"]),
                    "final_exact_accuracy": _metric_value(loss_dict["final_exact_accuracy"]),
                    "grad_norm": grad_norm,
                    "tokens": float(running_tokens),
                    "tokens_per_sec": running_tokens / max(now - last_log, 1e-9),
                    "batch_sec": now - batch_start,
                    "elapsed_sec": now - train_start,
                    "cuda_max_mem_gb": (
                        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                    ),
                }
                for i, group in enumerate(optimizer.param_groups):
                    latest_metrics[f"lr_group_{i}"] = float(group["lr"])

                if step == 1 or step % log_every == 0:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(latest_metrics, sort_keys=True) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(_prefixed(latest_metrics, "train"), step=step)
                    print(
                        "step={step} loss={loss:.4f} final={final_step_loss:.4f} "
                        "acc={final_token_accuracy:.3f} exact={final_exact_accuracy:.3f} "
                        "ppl={perplexity:.3f} tok/s={tokens_per_sec:.1f} "
                        "mem={cuda_max_mem_gb:.2f}GB".format(**latest_metrics),
                        flush=True,
                    )
                    running_loss = 0.0
                    running_tokens = 0
                    running_batches = 0
                    last_log = now

                if step == 1 or step % val_every == 0:
                    latest_validation_metrics = _evaluate(
                        model=model,
                        loader=validation_loader,
                        cfg=cfg,
                        device=device,
                        amp_enabled=amp_enabled,
                        max_batches=(None if cfg.validation.max_batches is None else int(cfg.validation.max_batches)),
                    )
                    latest_validation_metrics["step"] = float(step)
                    with (run_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(latest_validation_metrics, sort_keys=True) + "\n")
                    if wandb_run is not None:
                        wandb_run.log(_prefixed(latest_validation_metrics, "validation"), step=step)
                    print(
                        "validation step={step:.0f} loss={loss:.4f} "
                        "acc={final_token_accuracy:.3f} exact={final_exact_accuracy:.3f} "
                        "rows={rows:.0f}".format(**latest_validation_metrics),
                        flush=True,
                    )

                if step % save_every == 0:
                    _save_checkpoint(
                        path=run_dir / f"checkpoint_step_{step}.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        metrics=latest_metrics,
                        cfg=cfg,
                    )
                    _prune_checkpoints(run_dir, int(cfg.train.keep_last_checkpoints))

                if step >= max_steps:
                    break

    _save_checkpoint(
        path=run_dir / "checkpoint_final.pt",
        model=model,
        optimizer=optimizer,
        step=step,
        metrics=latest_metrics,
        cfg=cfg,
    )
    summary = {
        "step": step,
        "dataset_size": len(dataset),
        "validation_dataset_size": len(validation_dataset),
        "split": configured_splits or cfg.data.split,
        "validation_split": cfg.validation.split,
        "families": _optional_container(cfg.data.families),
        "latest_metrics": latest_metrics,
        "latest_validation_metrics": latest_validation_metrics,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
