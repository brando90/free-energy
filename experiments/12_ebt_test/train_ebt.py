#!/usr/bin/env python3
"""Hydra trainer for the synthetic EBT chain task."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataloader import make_dataloader
from ebt import SyntheticEBT
from tokenizer import SyntheticTokenizer

TOKEN_METRICS = (
    "loss",
    "reconstruction_loss",
    "perplexity",
    "final_token_accuracy",
    "final_exact_accuracy",
    "teacher_forcing_token_accuracy",
    "teacher_forcing_exact_accuracy",
    "autoregressive_token_accuracy",
    "autoregressive_exact_accuracy",
    "contrastive_rollout_loss",
    "contrastive_rollout_accuracy",
)


def _is_exact_metric(name: str) -> bool:
    return name.endswith("exact_accuracy")


def _log_train_step(
    run: Any | None,
    metrics: dict[str, float],
    step: int,
    epoch: int | float,
    batch_rows: int | None = None,
    batch_tokens: float | None = None,
) -> None:
    if run is None:
        return
    train_metrics: dict[str, float] = {}
    exact_metrics: dict[str, float] = {}

    for key, value in metrics.items():
        if _is_exact_metric(key):
            exact_metrics[key] = value
        else:
            train_metrics[key] = value

    record = {
        **{f"train/{k}": v for k, v in train_metrics.items()},
        **{f"exact_accuracy/train/{k}": v for k, v in exact_metrics.items()},
        "progress/epoch": float(epoch),
    }
    if batch_rows is not None:
        record["progress/train/rows"] = float(batch_rows)
    if batch_tokens is not None:
        record["progress/train/tokens"] = float(batch_tokens)
    record["progress/train/step"] = float(step)
    run.log(record, step=step)


# Teacher forcing comes from model loss and is named `final_*` in model output for
# compatibility with the original training code.


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, non_blocking=True)
        else:
            out[key] = value
    return out


def _append_target_eos(
    batch: dict[str, Any],
    tokenizer: SyntheticTokenizer,
) -> tuple[torch.Tensor, torch.Tensor]:
    target = batch["target_token_ids"]
    mask = batch["target_attention_mask"]
    eos = torch.full((target.shape[0], 1), tokenizer.eos_token_id, dtype=torch.long, device=target.device)
    target_with_eos = torch.cat([target, eos], dim=1)
    eos_mask = torch.ones((target.shape[0], 1), dtype=mask.dtype, device=mask.device)
    target_mask_with_eos = torch.cat([mask, eos_mask], dim=1)
    return target_with_eos, target_mask_with_eos


def _append_jsonl(path: Path, row: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _optional_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "undefined"}:
            return None
    return value


def _build_train_record(
    *,
    loss: float,
    perplexity: float,
    teacher_token_accuracy: float,
    teacher_exact_accuracy: float,
    autoregressive_token_accuracy: float,
    autoregressive_exact_accuracy: float,
    contrastive_rollout_loss: float = 0.0,
    contrastive_rollout_accuracy: float = 0.0,
) -> dict[str, float]:
    return {
        "loss": float(loss),
        "perplexity": float(perplexity),
        "teacher_forcing_token_accuracy": float(teacher_token_accuracy),
        "teacher_forcing_exact_accuracy": float(teacher_exact_accuracy),
        "autoregressive_token_accuracy": float(autoregressive_token_accuracy),
        "autoregressive_exact_accuracy": float(autoregressive_exact_accuracy),
        "contrastive_rollout_loss": float(contrastive_rollout_loss),
        "contrastive_rollout_accuracy": float(contrastive_rollout_accuracy),
}


class _HybridMuonAdamW:
    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        adamw_params: list[torch.nn.Parameter],
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float],
        eps: float,
        momentum: float,
        nesterov: bool,
        ns_coefficients: tuple[float, float, float],
        ns_steps: int,
        adjust_lr_fn: str | None = None,
        fused_adamw: bool = False,
    ) -> None:
        self._optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            self._optimizers.append(
                torch.optim.Muon(
                    muon_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    nesterov=nesterov,
                    ns_coefficients=ns_coefficients,
                    eps=eps,
                    ns_steps=ns_steps,
                    adjust_lr_fn=adjust_lr_fn,
                )
            )
        if adamw_params:
            adamw_kwargs = {
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
            }
            if torch.cuda.is_available() and fused_adamw:
                adamw_kwargs["fused"] = True
            try:
                self._optimizers.append(torch.optim.AdamW(adamw_params, **adamw_kwargs))
            except TypeError:
                adamw_kwargs.pop("fused", None)
                self._optimizers.append(torch.optim.AdamW(adamw_params, **adamw_kwargs))

        if not self._optimizers:
            raise ValueError("No parameters supplied to optimizer")

        self.param_groups: list[dict[str, Any]] = []
        for optimizer in self._optimizers:
            self.param_groups.extend(optimizer.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self._optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None) -> None:
        for optimizer in self._optimizers:
            optimizer.step(closure=closure)

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizers": [
                optimizer.state_dict()
                for optimizer in self._optimizers
            ]
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for optimizer, sub_state in zip(self._optimizers, state.get("optimizers", [])):
            optimizer.load_state_dict(sub_state)


def _log_val_metrics(
    run: Any | None,
    step: int,
    epoch: int | float,
    metrics: dict[str, float],
    prefix: str = "val/",
) -> None:
    if run is None:
        return
    val_metrics: dict[str, float] = {}
    exact_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("final_"):
            continue
        if _is_exact_metric(key):
            exact_metrics[key] = value
        else:
            val_metrics[key] = value
    record = {
        **{f"{prefix}{k}": v for k, v in val_metrics.items()},
        **{f"exact_accuracy/val/{k}": v for k, v in exact_metrics.items()},
        "progress/step": float(step),
        "progress/epoch": float(epoch),
    }
    run.log(record, step=step)


def _init_wandb(cfg: DictConfig, run_dir: Path):
    if not bool(cfg.wandb.enabled):
        return None

    import wandb

    run = wandb.init(
        project=str(cfg.wandb.project),
        entity=_optional_value(cfg.wandb.get("entity")),
        name=_optional_value(cfg.wandb.get("name")),
        group=_optional_value(cfg.wandb.get("group")),
        notes=_optional_value(cfg.wandb.get("notes")),
        tags=[str(t) for t in (cfg.wandb.get("tags", []) or [])],
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=str(cfg.wandb.get("mode", "online")),
        dir=str(run_dir),
    )
    return run


def _autoregressive_eval(
    model: torch.nn.Module,
    batch: dict[str, Any],
    tokenizer: SyntheticTokenizer,
    best_of_n: int = 1,
) -> tuple[float, float, float]:
    prompt_ids = batch["prompt_token_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    target_ids = batch["target_token_ids"]
    target_mask = batch["target_attention_mask"].to(dtype=torch.bool)

    target_len = int(target_ids.shape[1])
    if target_len <= 0:
        return 0.0, 0.0, 0.0

    with torch.no_grad():
        was_training = model.training
        model.eval()
        sample = model.sample(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_len=target_len,
            best_of_n=best_of_n,
        )
        predicted = sample["token_ids"].to(dtype=torch.long)
        if was_training:
            model.train()

    pad = torch.tensor(tokenizer.pad_token_id, dtype=torch.long, device=prompt_ids.device)
    predicted_padded = torch.where(target_mask, predicted[:, :target_len], pad)
    target_padded = torch.where(target_mask, target_ids[:, :target_len], pad)

    token_masked = target_mask[:, :target_len]
    token_matches = ((predicted_padded == target_padded) & token_masked).sum(dtype=torch.float32)
    token_count = float(token_masked.sum().item())
    token_accuracy = (token_matches / max(token_count, 1.0)).item()
    row_matches = (((predicted_padded == target_padded) | ~token_masked).all(dim=1).to(dtype=torch.float32)).sum().item()
    row_count = float(target_ids.shape[0])
    exact_accuracy = row_matches / max(row_count, 1.0)

    return float(token_accuracy), float(exact_accuracy), token_count


def _evaluate(
    *,
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    tokenizer: SyntheticTokenizer,
    amp: bool,
    contrastive_rollout_enabled: bool,
    contrastive_rollout_coeff: float,
    best_of_n: int,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {k: 0.0 for k in TOKEN_METRICS}
    total_teacher_tokens = 0.0
    total_teacher_rows = 0.0
    total_auto_tokens = 0.0
    total_auto_token_correct = 0.0
    total_auto_rows = 0.0
    with torch.enable_grad():
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            target_ids, target_mask = _append_target_eos(batch, tokenizer)

            with torch.autocast(device_type=str(device), dtype=torch.bfloat16, enabled=amp):
                loss_dict = model.loss(
                    prompt_ids=batch["prompt_token_ids"],
                    prompt_attention_mask=batch["prompt_attention_mask"],
                    target_ids=target_ids,
                    target_mask=target_mask,
                    contrastive_rollout_enabled=contrastive_rollout_enabled,
                    contrastive_rollout_coeff=contrastive_rollout_coeff,
                    learning=False,
                )

            batch_tokens = float(target_mask.sum().item())
            batch_rows = int(target_ids.shape[0])
            totals["loss"] += float(loss_dict["loss"].detach().item()) * batch_tokens
            totals["reconstruction_loss"] += float(loss_dict["reconstruction_loss"].detach().item()) * batch_tokens
            totals["perplexity"] += float(loss_dict["perplexity"].detach().item()) * batch_tokens
            totals["final_token_accuracy"] += float(loss_dict["final_token_accuracy"].detach().item()) * batch_tokens
            totals["final_exact_accuracy"] += float(loss_dict["final_exact_accuracy"].detach().item()) * batch_rows

            teacher_token_acc = float(loss_dict["final_token_accuracy"].detach().item())
            teacher_exact_acc = float(loss_dict["final_exact_accuracy"].detach().item())
            totals["teacher_forcing_token_accuracy"] += teacher_token_acc * batch_tokens
            totals["teacher_forcing_exact_accuracy"] += teacher_exact_acc * batch_rows
            if "contrastive_rollout_loss" in loss_dict:
                totals["contrastive_rollout_loss"] += float(loss_dict["contrastive_rollout_loss"].detach().item()) * batch_rows
            if "contrastive_rollout_accuracy" in loss_dict:
                totals["contrastive_rollout_accuracy"] += float(loss_dict["contrastive_rollout_accuracy"].detach().item()) * batch_rows

            auto_token_acc, auto_exact_acc, auto_tokens = _autoregressive_eval(model, batch, tokenizer, best_of_n=best_of_n)
            totals["autoregressive_token_accuracy"] += auto_token_acc * auto_tokens
            totals["autoregressive_exact_accuracy"] += auto_exact_acc * batch_rows
            total_auto_token_correct += auto_token_acc * auto_tokens
            total_auto_rows += batch_rows
            total_auto_tokens += auto_tokens

            total_teacher_tokens += batch_tokens
            total_teacher_rows += batch_rows

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    model.train()

    out: dict[str, float] = {}
    if total_teacher_tokens > 0:
        out["loss"] = totals["loss"] / total_teacher_tokens
        out["reconstruction_loss"] = totals["reconstruction_loss"] / total_teacher_tokens
        out["perplexity"] = totals["perplexity"] / total_teacher_tokens
        out["final_token_accuracy"] = totals["final_token_accuracy"] / total_teacher_tokens
        out["teacher_forcing_token_accuracy"] = totals["teacher_forcing_token_accuracy"] / total_teacher_tokens
    else:
        out["loss"] = 0.0
        out["reconstruction_loss"] = 0.0
        out["perplexity"] = 0.0
        out["final_token_accuracy"] = 0.0
        out["teacher_forcing_token_accuracy"] = 0.0

    out["final_exact_accuracy"] = (
        totals["final_exact_accuracy"] / max(total_teacher_rows, 1.0)
    )
    out["teacher_forcing_exact_accuracy"] = (
        totals["teacher_forcing_exact_accuracy"] / max(total_teacher_rows, 1.0)
    )
    out["contrastive_rollout_loss"] = (
        totals["contrastive_rollout_loss"] / max(total_teacher_rows, 1.0)
    ) if total_teacher_rows > 0 else 0.0
    out["contrastive_rollout_accuracy"] = (
        totals["contrastive_rollout_accuracy"] / max(total_teacher_rows, 1.0)
    ) if total_teacher_rows > 0 else 0.0
    out["autoregressive_token_accuracy"] = total_auto_token_correct / max(total_auto_tokens, 1.0)
    out["autoregressive_exact_accuracy"] = totals["autoregressive_exact_accuracy"] / max(total_auto_rows, 1.0)
    return out


def _build_model(cfg: DictConfig, vocab_size: int) -> SyntheticEBT:
    return SyntheticEBT(
        vocab_size=vocab_size,
        context_dim=int(cfg.model.context_dim),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dim_feedforward=int(cfg.model.dim_feedforward),
        mcmc_num_steps=int(cfg.model.mcmc_num_steps),
        mcmc_step_size=float(cfg.model.mcmc_step_size),
        mcmc_step_size_learnable=bool(cfg.model.mcmc_step_size_learnable),
        max_target_positions=int(cfg.model.max_target_positions),
        mcmc_noise=float(cfg.model.mcmc_noise),
        truncate_mcmc=bool(cfg.model.truncate_mcmc),
        dropout=float(cfg.model.dropout),
        alpha_max_norm=float(cfg.model.alpha_max_norm),
        contrastive_rollout_enabled=bool(cfg.model.contrastive_rollout_enabled),
        contrastive_rollout_coeff=float(cfg.model.contrastive_rollout_coeff),
        train_mcmc_num_steps_min=int(cfg.model.train_mcmc_num_steps_min),
        train_mcmc_num_steps_max=int(cfg.model.train_mcmc_num_steps_max),
        randomize_mcmc_step_size_scale=float(cfg.model.randomize_mcmc_step_size_scale),
        replay_buffer_enabled=bool(cfg.model.replay_buffer_enabled),
        replay_buffer_size=int(cfg.model.replay_buffer_size),
        replay_buffer_sample_fraction=float(cfg.model.replay_buffer_sample_fraction),
        gaussian_energy_loss_enabled=bool(cfg.model.gaussian_energy_loss_enabled),
        gaussian_energy_alpha=float(cfg.model.gaussian_energy_alpha),
    )


def _build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    optimizer_name = str(getattr(cfg.optim, "name", "muon")).lower()

    muon_cfg = {
        "momentum": 0.95,
        "nesterov": True,
        "ns_coefficients": (3.4445, -4.775, 2.0315),
        "ns_steps": 5,
        "adjust_lr_fn": None,
        "fused_adamw": bool(torch.cuda.is_available()),
    }
    if getattr(cfg.optim, "momentum", None) is not None:
        muon_cfg["momentum"] = float(cfg.optim.momentum)
    if getattr(cfg.optim, "nesterov", None) is not None:
        muon_cfg["nesterov"] = bool(cfg.optim.nesterov)
    if getattr(cfg.optim, "ns_coefficients", None) is not None:
        muon_cfg["ns_coefficients"] = tuple(float(x) for x in cfg.optim.ns_coefficients)
    if getattr(cfg.optim, "ns_steps", None) is not None:
        muon_cfg["ns_steps"] = int(cfg.optim.ns_steps)
    if getattr(cfg.optim, "adjust_lr_fn", None) is not None:
        muon_cfg["adjust_lr_fn"] = cfg.optim.adjust_lr_fn

    if getattr(torch.optim, "Muon", None) is None:
        if optimizer_name == "muon":
            raise RuntimeError("optim.name=muon requires a PyTorch build with torch.optim.Muon")
        muon_cfg["fused_adamw"] = False

    if optimizer_name == "muon":
        param_list = list(model.parameters())
        muon_params = [param for param in param_list if param.requires_grad and param.ndim >= 2]
        adamw_params = [param for param in param_list if param.requires_grad and param.ndim < 2]

        if not muon_params and not adamw_params:
            raise ValueError("No trainable parameters found for optimization")
        if not adamw_params:
            return torch.optim.Muon(
                muon_params,
                lr=float(cfg.optim.lr),
                weight_decay=float(cfg.optim.weight_decay),
                momentum=muon_cfg["momentum"],
                nesterov=muon_cfg["nesterov"],
                ns_coefficients=muon_cfg["ns_coefficients"],
                eps=float(cfg.optim.eps),
                ns_steps=muon_cfg["ns_steps"],
                adjust_lr_fn=muon_cfg["adjust_lr_fn"],
            )
        return _HybridMuonAdamW(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            betas=tuple(float(x) for x in cfg.optim.betas),
            eps=float(cfg.optim.eps),
            momentum=muon_cfg["momentum"],
            nesterov=muon_cfg["nesterov"],
            ns_coefficients=muon_cfg["ns_coefficients"],
            ns_steps=muon_cfg["ns_steps"],
            adjust_lr_fn=muon_cfg["adjust_lr_fn"],
            fused_adamw=bool(muon_cfg["fused_adamw"]),
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            betas=tuple(float(x) for x in cfg.optim.betas),
            eps=float(cfg.optim.eps),
        )
    if str(cfg.optim.name).lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            betas=tuple(float(x) for x in cfg.optim.betas),
            eps=float(cfg.optim.eps),
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optim.name}")


@hydra.main(version_base=None, config_path="configs", config_name="ebt_config")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available() and not bool(cfg.allow_cpu):
        raise RuntimeError("CUDA unavailable; set allow_cpu=true for CPU-only debug")

    _seed(int(cfg.seed))
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    tokenizer = SyntheticTokenizer.default()
    vocab_size = int(tokenizer.vocab_size if str(cfg.model.vocab_size) == "local" else int(cfg.model.vocab_size))
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(cfg.allow_cpu) else "cpu")

    train_loader = make_dataloader(
        data_path=_as_path(cfg.data.data_path),
        split=str(cfg.data.train_split),
        batch_size=int(cfg.loader.batch_size),
        shuffle=bool(cfg.loader.shuffle),
        num_workers=int(cfg.loader.num_workers),
        max_items=None if cfg.data.max_items is None else int(cfg.data.max_items),
        max_prompt_tokens=cfg.data.get("max_prompt_tokens"),
        max_target_tokens=cfg.data.get("max_target_tokens"),
    )
    val_loader = make_dataloader(
        data_path=_as_path(cfg.data.data_path),
        split=str(cfg.data.val_split),
        batch_size=int(cfg.loader.batch_size),
        shuffle=False,
        num_workers=int(cfg.loader.num_workers),
        max_items=cfg.validation.get("max_items"),
        max_prompt_tokens=cfg.data.get("max_prompt_tokens"),
        max_target_tokens=cfg.data.get("max_target_tokens"),
    )

    model = _build_model(cfg, vocab_size).to(device=device)
    optimizer = _build_optimizer(model, cfg)
    amp_enabled = bool(cfg.train.amp_bf16) and device.type == "cuda"
    contrastive_rollout_enabled = bool(cfg.model.contrastive_rollout_enabled)
    contrastive_rollout_coeff = float(cfg.model.contrastive_rollout_coeff)
    best_of_n = max(1, int(cfg.model.best_of_n))

    grad_accum = max(1, int(cfg.train.grad_accum_steps))
    max_steps = int(cfg.train.max_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every_steps)
    validate_every = int(cfg.validation.every_steps)
    val_max_batches = None if cfg.validation.max_batches is None else int(cfg.validation.max_batches)

    metrics_log = run_dir / "metrics.jsonl"
    val_log = run_dir / "validation_metrics.jsonl"

    wandb_run = _init_wandb(cfg, run_dir)
    epoch = 0

    initial_train_batch = next(iter(train_loader))
    initial_train_batch = _move_batch(initial_train_batch, device)
    initial_target_ids, initial_target_mask = _append_target_eos(initial_train_batch, tokenizer)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        initial_loss_dict = model.loss(
            prompt_ids=initial_train_batch["prompt_token_ids"],
            prompt_attention_mask=initial_train_batch["prompt_attention_mask"],
            target_ids=initial_target_ids,
            target_mask=initial_target_mask,
            contrastive_rollout_enabled=contrastive_rollout_enabled,
            contrastive_rollout_coeff=contrastive_rollout_coeff,
            learning=False,
        )
    initial_auto_token_accuracy, initial_auto_exact_accuracy, _ = _autoregressive_eval(
        model=model,
        batch=initial_train_batch,
        tokenizer=tokenizer,
        best_of_n=best_of_n,
    )

    initial_train_record = _build_train_record(
        loss=float(initial_loss_dict["loss"].detach().item()),
        perplexity=float(initial_loss_dict["perplexity"].detach().item()),
        teacher_token_accuracy=float(initial_loss_dict["final_token_accuracy"].detach().item()),
        teacher_exact_accuracy=float(initial_loss_dict["final_exact_accuracy"].detach().item()),
        autoregressive_token_accuracy=float(initial_auto_token_accuracy),
        autoregressive_exact_accuracy=float(initial_auto_exact_accuracy),
        contrastive_rollout_loss=float(initial_loss_dict.get("contrastive_rollout_loss", torch.tensor(0.0)).detach().item()),
        contrastive_rollout_accuracy=float(initial_loss_dict.get("contrastive_rollout_accuracy", torch.tensor(0.0)).detach().item()),
    )
    _append_jsonl(metrics_log, initial_train_record)
    _log_train_step(
        run=wandb_run,
        metrics=initial_train_record,
        step=0,
        epoch=0,
        batch_rows=int(initial_target_ids.shape[0]),
        batch_tokens=float(initial_target_mask.sum().item()),
    )
    print(
        f"step=0 loss={initial_train_record['loss']:.4f} ppl={initial_train_record['perplexity']:.4f} "
        f"tf_tok={initial_train_record['teacher_forcing_token_accuracy']:.4f} "
        f"ar_tok={initial_train_record['autoregressive_token_accuracy']:.4f}"
    )

    initial_val = _evaluate(
        model=model,
        loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        amp=amp_enabled,
        contrastive_rollout_enabled=contrastive_rollout_enabled,
        contrastive_rollout_coeff=contrastive_rollout_coeff,
        best_of_n=best_of_n,
        max_batches=val_max_batches,
    )
    _append_jsonl(val_log, initial_val)
    _log_val_metrics(wandb_run, step=0, epoch=0, metrics=initial_val)
    metric_preview = list(initial_val.items())[:8]
    print(
        "val "
        + " ".join(
            f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}"
            for k, v in metric_preview
        )
    )

    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch_idx in range(10_000):
        epoch = epoch_idx
        for batch in train_loader:
            step += 1
            batch = _move_batch(batch, device)
            target_ids, target_mask = _append_target_eos(batch, tokenizer)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss_dict = model.loss(
                    prompt_ids=batch["prompt_token_ids"],
                    prompt_attention_mask=batch["prompt_attention_mask"],
                    target_ids=target_ids,
                    target_mask=target_mask,
                    contrastive_rollout_enabled=contrastive_rollout_enabled,
                    contrastive_rollout_coeff=contrastive_rollout_coeff,
                    learning=True,
                )
            scaled = loss_dict["loss"] / grad_accum
            scaled.backward()

            if step % grad_accum == 0:
                if float(cfg.train.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                auto_token_accuracy, auto_exact_accuracy, _ = _autoregressive_eval(model, batch, tokenizer, best_of_n=best_of_n)
                record = _build_train_record(
                    loss=float(loss_dict["loss"].detach().item()),
                    perplexity=float(loss_dict["perplexity"].detach().item()),
                    teacher_token_accuracy=float(loss_dict["final_token_accuracy"].detach().item()),
                    teacher_exact_accuracy=float(loss_dict["final_exact_accuracy"].detach().item()),
                    autoregressive_token_accuracy=float(auto_token_accuracy),
                    autoregressive_exact_accuracy=float(auto_exact_accuracy),
                    contrastive_rollout_loss=float(loss_dict.get("contrastive_rollout_loss", torch.tensor(0.0)).detach().item()),
                    contrastive_rollout_accuracy=float(loss_dict.get("contrastive_rollout_accuracy", torch.tensor(0.0)).detach().item()),
                )
                _append_jsonl(metrics_log, record)
                print(
                    f"step={step} loss={record['loss']:.4f} ppl={record['perplexity']:.4f} "
                    f"tf_tok={record['teacher_forcing_token_accuracy']:.4f} ar_tok={record['autoregressive_token_accuracy']:.4f}"
                )
                if wandb_run is not None:
                    _log_train_step(
                        run=wandb_run,
                        metrics=record,
                        step=step,
                        epoch=epoch,
                        batch_rows=int(target_ids.shape[0]),
                        batch_tokens=float(target_mask.sum().item()),
                    )

            if step % validate_every == 0:
                metrics = _evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    amp=amp_enabled,
                    contrastive_rollout_enabled=contrastive_rollout_enabled,
                    contrastive_rollout_coeff=contrastive_rollout_coeff,
                    best_of_n=best_of_n,
                    max_batches=val_max_batches,
                )
                _append_jsonl(val_log, metrics)
                metric_preview = list(metrics.items())[:8]
                print(
                    "val "
                    + " ".join(
                        f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}"
                        for k, v in metric_preview
                    )
                )
                _log_val_metrics(wandb_run, step=step, epoch=epoch, metrics=metrics)

            if step >= max_steps:
                break

        if step >= max_steps:
            break

        if len(train_loader) == 0:
            break

    torch.save(model.state_dict(), run_dir / "checkpoint_final.pt")

    final_val = _evaluate(
        model=model,
        loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        amp=amp_enabled,
        contrastive_rollout_enabled=contrastive_rollout_enabled,
        contrastive_rollout_coeff=contrastive_rollout_coeff,
        best_of_n=best_of_n,
        max_batches=val_max_batches,
    )
    _append_jsonl(val_log, {"phase": "final", **final_val})
    print("final", " ".join(f"{k}={v:.4f}" for k, v in final_val.items() if isinstance(v, float)))
    final_val_record = dict(final_val)
    _log_val_metrics(wandb_run, step=step, epoch=epoch, metrics=final_val_record)

    if wandb_run is not None:
        wandb_run.finish()

    if step % save_every == 0:
        torch.save(model.state_dict(), run_dir / f"checkpoint_step_{step}.pt")


if __name__ == "__main__":
    main()
