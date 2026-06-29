#!/usr/bin/env python3
"""Hydra trainer for chunked Lean Workbook EBT runs."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ebt import GoedelVocabEBT
from lean_compile import compile_theorem
from leanworkbook_dataloader import (
    LeanWorkbookEmbeddingDataset,
    make_leanworkbook_embedding_dataloader,
)

TOKEN_WEIGHTED_METRICS = (
    "loss",
    "initial_loss",
    "final_step_loss",
    "perplexity",
    "final_energy",
    "final_token_accuracy",
)


class HybridMuonAdamW(torch.optim.Optimizer):
    """Use Muon for 2-D tensors and AdamW for tensors Muon cannot update."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("HybridMuonAdamW requires at least one optimizer")
        params = [param for optimizer in optimizers for group in optimizer.param_groups for param in group["params"]]
        super().__init__(params, {})
        self.optimizers = optimizers
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]
        self.state = defaultdict(dict)

    def step(self, closure: Any = None) -> Any:
        loss = closure() if closure is not None else None
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for optimizer, child_state in zip(self.optimizers, state_dict["optimizers"], strict=True):
            optimizer.load_state_dict(child_state)


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


def _make_loader(
    cfg: DictConfig,
    *,
    split: str | list[str] | None = None,
    shuffle: bool | None = None,
    max_items: int | None = None,
    random_sample_items: int | None = None,
    random_sample_seed: int | None = None,
) -> DataLoader[dict[str, Any]]:
    selected_split = split if split is not None else cfg.data.split
    loader = make_leanworkbook_embedding_dataloader(
        data_dir=_as_path(cfg.data.data_dir),
        activations_dir=_as_path(cfg.data.activations_dir),
        indices_file=_as_path(cfg.data.indices_file),
        split=selected_split,
        max_items=cfg.data.max_items if max_items is None else max_items,
        random_sample_items=random_sample_items,
        random_sample_seed=int(cfg.seed) if random_sample_seed is None else int(random_sample_seed),
        max_target_tokens=cfg.data.max_target_tokens,
        chunk_size=int(cfg.data.chunk_size),
        activation_dtype=cfg.data.activation_dtype,
        model_name=cfg.data.model_name or cfg.model.model_name,
        model_revision=cfg.data.model_revision or cfg.model.revision,
        validate_context=cfg.data.validate_context,
        batch_size=int(cfg.loader.batch_size),
        shuffle=bool(cfg.loader.shuffle) if shuffle is None else bool(shuffle),
        num_workers=int(cfg.loader.num_workers),
        max_target_chunks_per_batch=(
            None
            if cfg.loader.max_target_chunks_per_batch is None
            else int(cfg.loader.max_target_chunks_per_batch)
        ),
        seed=int(cfg.seed),
        pin_memory=bool(cfg.loader.pin_memory),
        persistent_workers=bool(cfg.loader.persistent_workers),
        prefetch_factor=(None if int(cfg.loader.num_workers) == 0 else int(cfg.loader.prefetch_factor)),
        drop_last=bool(cfg.loader.drop_last),
    )
    if len(loader.dataset) == 0:
        raise ValueError(f"No Lean Workbook samples matched split={selected_split!r}")
    return loader


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


def _epoch_metrics(
    *,
    examples_seen: int,
    batches_seen: int,
    dataset_size: int,
    batches_per_epoch: int,
) -> dict[str, float]:
    examples = max(0, int(examples_seen))
    batches = max(0, int(batches_seen))
    dataset_denom = max(1, int(dataset_size))
    batch_denom = max(1, int(batches_per_epoch))
    return {
        "epoch": examples / dataset_denom,
        "epoch_count": math.floor(examples / dataset_denom),
        "epoch_progress": (examples % dataset_denom) / dataset_denom,
        "examples_seen": float(examples),
        "batches_seen": float(batches),
        "batches_per_epoch": float(batch_denom),
    }


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def _append_jsonl(path: Path, row: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _raw_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _mcmc_step_size(model: torch.nn.Module) -> float:
    alpha = getattr(_raw_model(model), "alpha", None)
    if alpha is None:
        return math.nan
    return float(torch.clamp(alpha.detach().float().cpu(), min=0.0001))


def _batch_loss(
    *,
    model: torch.nn.Module,
    batch: dict[str, Any],
    cfg: DictConfig,
    learning: bool,
) -> dict[str, torch.Tensor]:
    return model.loss(
        batch["context_activations"],
        batch[str(cfg.data.label_field)],
        decoder_input_ids=batch["decoder_input_ids"],
        context_attention_mask=batch["context_attention_mask"],
        label_attention_mask=batch["label_attention_mask"],
        decoder_attention_mask=batch["decoder_attention_mask"],
        learning=learning,
    )


def _evaluate_loss(
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
    totals = dict.fromkeys(TOKEN_WEIGHTED_METRICS, 0.0)
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
            loss_dict = _batch_loss(model=model, batch=batch, cfg=cfg, learning=False)
        weight = max(tokens, 1)
        for key in TOKEN_WEIGHTED_METRICS:
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
            "mcmc_step_size": _mcmc_step_size(model),
        }
    )
    if was_training:
        model.train()
    return metrics


def _compile_validate(
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
    raw_model = _raw_model(model)
    dataset: LeanWorkbookEmbeddingDataset = loader.dataset
    total_rows = 0
    pass_count = 0
    batches = 0
    start = time.perf_counter()
    sample_steps = None if cfg.validation.sample_steps is None else int(cfg.validation.sample_steps)
    max_target_chunks = int(cfg.validation.max_target_chunks)
    compile_timeout = int(cfg.validation.compile_timeout)
    compile_expect_timeout = int(cfg.validation.compile_expect_timeout)
    compile_workers = max(1, int(cfg.validation.get("compile_workers", 1)))
    progress_every = max(0, int(cfg.validation.get("progress_every", 0)))
    repl_path = _as_path(cfg.validation.repl_path)
    lean_env_path = _as_path(cfg.validation.lean_env_path)
    sampling_elapsed = 0.0
    compile_elapsed = 0.0

    def compile_one(theorem_code: str) -> bool:
        passed, _ = compile_theorem(
            theorem_code,
            repl_path=repl_path,
            lean_env_path=lean_env_path,
            timeout=compile_timeout,
            expect_timeout=compile_expect_timeout,
        )
        return bool(passed)

    for batch in loader:
        batch = _move_batch(batch, device)
        target_len = int(batch["labels"].shape[1]) if cfg.validation.target_chunks_from_labels else max_target_chunks
        target_len = min(target_len, max_target_chunks)
        sample_start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            sampled = raw_model.sample(
                batch["context_activations"],
                target_len=target_len,
                context_attention_mask=batch["context_attention_mask"],
                steps=sample_steps,
                eos_token_id=dataset.eos_id,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_elapsed += time.perf_counter() - sample_start

        theorem_codes = [dataset.id_mapper.decode_local_ids(token_ids) for token_ids in sampled["token_ids"]]
        compile_start = time.perf_counter()
        if compile_workers == 1 or len(theorem_codes) <= 1:
            results = [compile_one(code) for code in theorem_codes]
        else:
            with ThreadPoolExecutor(max_workers=min(compile_workers, len(theorem_codes))) as executor:
                results = list(executor.map(compile_one, theorem_codes))
        compile_elapsed += time.perf_counter() - compile_start
        pass_count += sum(int(passed) for passed in results)
        total_rows += len(results)
        batches += 1
        if progress_every and batches % progress_every == 0:
            print(
                f"validation_progress batches={batches} rows={total_rows} "
                f"compile_pass={pass_count} sample_sec={sampling_elapsed:.1f} compile_sec={compile_elapsed:.1f}",
                flush=True,
            )
        if max_batches is not None and batches >= max_batches:
            break

    pass_rate = pass_count / max(total_rows, 1)
    metrics = {
        "compile_pass_rate": pass_rate,
        "lean_proof_correct_pct": 100.0 * pass_rate,
        "compile_pass_count": float(pass_count),
        "rows": float(total_rows),
        "batches": float(batches),
        "sampling_elapsed_sec": sampling_elapsed,
        "compile_elapsed_sec": compile_elapsed,
        "elapsed_sec": time.perf_counter() - start,
        "mcmc_step_size": _mcmc_step_size(model),
    }
    if was_training:
        model.train()
    return metrics


def _run_validation(
    *,
    model: torch.nn.Module,
    validation_loader: DataLoader[dict[str, Any]],
    cfg: DictConfig,
    device: torch.device,
    amp_enabled: bool,
    step: int,
    examples_seen: int,
    batches_seen: int,
    dataset_size: int,
    batches_per_epoch: int,
) -> dict[str, float]:
    metrics = _evaluate_loss(
        model=model,
        loader=validation_loader,
        cfg=cfg,
        device=device,
        amp_enabled=amp_enabled,
        max_batches=(None if cfg.validation.max_batches is None else int(cfg.validation.max_batches)),
    )
    if bool(cfg.validation.compile_enabled):
        compile_metrics = _compile_validate(
            model=model,
            loader=validation_loader,
            cfg=cfg,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=(None if cfg.validation.max_batches is None else int(cfg.validation.max_batches)),
        )
        metrics.update(compile_metrics)
    metrics.update(
        _epoch_metrics(
            examples_seen=examples_seen,
            batches_seen=batches_seen,
            dataset_size=dataset_size,
            batches_per_epoch=batches_per_epoch,
        )
    )
    metrics["step"] = float(step)
    return metrics


def _resolve_dataset_int(value: Any, dataset: LeanWorkbookEmbeddingDataset, key: str) -> int | None:
    if value is None:
        return None
    if str(value) == "dataset":
        return int(dataset.vocab[key])
    return int(value)


def _make_model(cfg: DictConfig, dataset: LeanWorkbookEmbeddingDataset) -> GoedelVocabEBT:
    model_cfg = cfg.model
    kwargs: dict[str, Any] = {
        "model_name": model_cfg.model_name,
        "revision": model_cfg.revision,
        "vocab_size": _resolve_dataset_int(model_cfg.vocab_size, dataset, "vocab_size"),
        "pad_token_id": _resolve_dataset_int(model_cfg.pad_token_id, dataset, "local_pad_id"),
        "eos_token_id": _resolve_dataset_int(model_cfg.eos_token_id, dataset, "local_eos_id"),
        "bos_token_id": _resolve_dataset_int(model_cfg.bos_token_id, dataset, "local_bos_id"),
        "context_dim": model_cfg.context_dim,
        "hidden_dim": model_cfg.hidden_dim,
        "denoising_initial_condition": model_cfg.denoising_initial_condition,
        "chunk_size": int(cfg.data.chunk_size),
    }
    int_keys = ("num_layers", "num_heads", "dim_feedforward", "mcmc_num_steps", "max_target_positions")
    float_keys = (
        "dropout", "mcmc_step_size", "gaussian_random_noise_scaling", "langevin_dynamics_noise",
        "clamp_futures_grad_max_change", "absolute_clamp", "sharpen_predicted_distribution",
        "reconstruction_coeff", "soften_target_prob_dist",
    )
    bool_keys = (
        "mcmc_step_size_learnable", "normalize_initial_condition", "normalize_initial_condition_only_first_step",
        "truncate_mcmc", "no_mcmc_detach", "clamp_futures_grad", "norm_pred", "norm_pred_not_final_step",
        "loss_on_final_step_only",
    )
    for key in int_keys:
        kwargs[key] = int(model_cfg[key])
    for key in float_keys:
        kwargs[key] = float(model_cfg[key])
    for key in bool_keys:
        kwargs[key] = bool(model_cfg[key])
    return GoedelVocabEBT(**kwargs)


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

    optimizer_name = str(cfg.optim.get("name", "muon")).lower()
    if optimizer_name == "muon":
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("optim.name=muon requires a PyTorch build with torch.optim.Muon")

        muon_groups: list[dict[str, Any]] = []
        adamw_groups: list[dict[str, Any]] = []
        source_groups = params if isinstance(params, list) and params and isinstance(params[0], dict) else [
            {
                "params": params,
                "weight_decay": float(cfg.optim.weight_decay),
                "lr": float(cfg.optim.lr),
            }
        ]
        for group in source_groups:
            group_params = list(group["params"])
            muon_params = [param for param in group_params if param.ndim == 2]
            adamw_params = [param for param in group_params if param.ndim != 2]
            base_group = {key: value for key, value in group.items() if key != "params"}
            if muon_params:
                muon_groups.append({**base_group, "params": muon_params})
            if adamw_params:
                adamw_groups.append({**base_group, "params": adamw_params})

        optimizers: list[torch.optim.Optimizer] = []
        if muon_groups:
            optimizers.append(
                torch.optim.Muon(
                    muon_groups,
                    lr=float(cfg.optim.lr),
                    weight_decay=float(cfg.optim.weight_decay),
                    momentum=float(cfg.optim.momentum),
                    nesterov=bool(cfg.optim.nesterov),
                    ns_coefficients=tuple(float(x) for x in cfg.optim.ns_coefficients),
                    eps=float(cfg.optim.eps),
                    ns_steps=int(cfg.optim.ns_steps),
                    adjust_lr_fn=_optional_container(cfg.optim.adjust_lr_fn),
                )
            )
        if adamw_groups:
            adamw_kwargs = {
                "lr": float(cfg.optim.lr),
                "weight_decay": float(cfg.optim.weight_decay),
                "betas": tuple(float(x) for x in cfg.optim.betas),
                "eps": float(cfg.optim.eps),
            }
            if torch.cuda.is_available() and bool(cfg.optim.fused):
                adamw_kwargs["fused"] = True
            try:
                optimizers.append(torch.optim.AdamW(adamw_groups, **adamw_kwargs))
            except TypeError:
                adamw_kwargs.pop("fused", None)
                optimizers.append(torch.optim.AdamW(adamw_groups, **adamw_kwargs))
        return optimizers[0] if len(optimizers) == 1 else HybridMuonAdamW(optimizers)

    if optimizer_name == "adamw":
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

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


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
        random_sample_items=(
            None if cfg.validation.random_sample_size is None else int(cfg.validation.random_sample_size)
        ),
        random_sample_seed=int(cfg.validation.random_sample_seed),
    )
    validation_dataset = validation_loader.dataset
    print(f"run_dir={run_dir}")
    print(f"device={device}")
    print(f"dataset_size={len(dataset)} split={cfg.data.split}")
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
    save_every = max(1, int(cfg.train.save_every_steps))
    val_every = max(1, int(cfg.validation.every_steps))
    dataset_size = len(dataset)
    batches_per_epoch = len(loader)
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
    examples_seen = 0
    batches_seen = 0
    latest_metrics: dict[str, float] = {}
    latest_validation_metrics: dict[str, float] = {}

    latest_validation_metrics = _run_validation(
        model=model,
        validation_loader=validation_loader,
        cfg=cfg,
        device=device,
        amp_enabled=amp_enabled,
        step=0,
        examples_seen=examples_seen,
        batches_seen=batches_seen,
        dataset_size=dataset_size,
        batches_per_epoch=batches_per_epoch,
    )
    _append_jsonl(run_dir / "validation_metrics.jsonl", latest_validation_metrics)
    if wandb_run is not None:
        wandb_run.log(_prefixed(latest_validation_metrics, "validation"), step=0)
    if "compile_pass_rate" in latest_validation_metrics:
        print(
            "validation step={step:.0f} loss={loss:.4f} acc={final_token_accuracy:.3f} "
            "ppl={perplexity:.3f} compile_pass={compile_pass_count:.0f}/{rows:.0f} "
            "rate={compile_pass_rate:.4f} epoch={epoch:.4f}".format(**latest_validation_metrics),
            flush=True,
        )
    else:
        print(
            "validation step={step:.0f} loss={loss:.4f} "
            "acc={final_token_accuracy:.3f} exact={final_exact_accuracy:.3f} "
            "ppl={perplexity:.3f} rows={rows:.0f} epoch={epoch:.4f}".format(**latest_validation_metrics),
            flush=True,
        )

    while step < max_steps:
        for batch in loader:
            batch_start = time.perf_counter()
            batch = _move_batch(batch, device)
            batch_rows = int(batch["label_attention_mask"].shape[0])
            token_count = int(batch["label_attention_mask"].sum().detach().cpu())
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                loss_dict = _batch_loss(model=model, batch=batch, cfg=cfg, learning=True)
                loss = loss_dict["loss"] / grad_accum_steps

            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"Non-finite loss at step {step + 1}: {float(loss.detach().cpu())}")

            loss.backward()
            accum += 1
            running_loss += _metric_value(loss_dict["loss"])
            running_tokens += token_count
            running_batches += 1
            examples_seen += batch_rows
            batches_seen += 1

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
                    "mcmc_step_size": _mcmc_step_size(model),
                    "cuda_max_mem_gb": (
                        torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
                    ),
                    **_epoch_metrics(
                        examples_seen=examples_seen,
                        batches_seen=batches_seen,
                        dataset_size=dataset_size,
                        batches_per_epoch=batches_per_epoch,
                    ),
                }
                for i, group in enumerate(optimizer.param_groups):
                    latest_metrics[f"lr_group_{i}"] = float(group["lr"])

                if step == 1 or step % log_every == 0:
                    _append_jsonl(metrics_path, latest_metrics)
                    if wandb_run is not None:
                        wandb_run.log(_prefixed(latest_metrics, "train"), step=step)
                    print(
                        "step={step} loss={loss:.4f} final={final_step_loss:.4f} "
                        "acc={final_token_accuracy:.3f} exact={final_exact_accuracy:.3f} "
                        "ppl={perplexity:.3f} epoch={epoch:.4f} tok/s={tokens_per_sec:.1f} "
                        "mem={cuda_max_mem_gb:.2f}GB".format(**latest_metrics),
                        flush=True,
                    )
                    running_loss = 0.0
                    running_tokens = 0
                    running_batches = 0
                    last_log = now

                if step % val_every == 0:
                    latest_validation_metrics = _run_validation(
                        model=model,
                        validation_loader=validation_loader,
                        cfg=cfg,
                        device=device,
                        amp_enabled=amp_enabled,
                        step=step,
                        examples_seen=examples_seen,
                        batches_seen=batches_seen,
                        dataset_size=dataset_size,
                        batches_per_epoch=batches_per_epoch,
                    )
                    _append_jsonl(run_dir / "validation_metrics.jsonl", latest_validation_metrics)
                    if wandb_run is not None:
                        wandb_run.log(_prefixed(latest_validation_metrics, "validation"), step=step)
                    if "compile_pass_rate" in latest_validation_metrics:
                        print(
                            "validation step={step:.0f} loss={loss:.4f} acc={final_token_accuracy:.3f} "
                            "ppl={perplexity:.3f} compile_pass={compile_pass_count:.0f}/{rows:.0f} "
                            "rate={compile_pass_rate:.4f} epoch={epoch:.4f}".format(**latest_validation_metrics),
                            flush=True,
                        )
                    else:
                        print(
                            "validation step={step:.0f} loss={loss:.4f} "
                            "acc={final_token_accuracy:.3f} exact={final_exact_accuracy:.3f} "
                            "ppl={perplexity:.3f} rows={rows:.0f} epoch={epoch:.4f}".format(**latest_validation_metrics),
                            flush=True,
                        )

                if step % save_every == 0:
                    _save_checkpoint(
                        path=run_dir / f"checkpoint_step_{step}.pt",
                        model=model,
                        optimizer=optimizer,
                        step=step,
                        metrics={**latest_metrics, **_prefixed(latest_validation_metrics, "validation")},
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
        metrics={**latest_metrics, **_prefixed(latest_validation_metrics, "validation")},
        cfg=cfg,
    )
    summary = {
        "step": step,
        "dataset_size": len(dataset),
        "validation_dataset_size": len(validation_dataset),
        "split": cfg.data.split,
        "validation_split": cfg.validation.split,
        "latest_metrics": latest_metrics,
        "latest_validation_metrics": latest_validation_metrics,
        "examples_seen": examples_seen,
        "batches_seen": batches_seen,
        "batches_per_epoch": batches_per_epoch,
        "run_dir": str(run_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
