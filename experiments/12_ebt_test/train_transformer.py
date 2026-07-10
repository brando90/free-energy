#!/usr/bin/env python3
"""Autoregressive Transformer baseline for the synthetic chain task."""

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
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataloader import make_dataloader
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
)


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


class BaselineTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        max_positions: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.max_positions = int(max_positions)
        self.pos_embed = nn.Embedding(max_positions, self.hidden_dim)
        self.embed = nn.Embedding(self.vocab_size, self.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.to_vocab = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = hidden + self.pos_embed(positions).unsqueeze(0).to(hidden.dtype)
        mask = torch.triu(
            torch.ones(input_ids.shape[1], input_ids.shape[1], device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding = ~attention_mask.bool()
        hidden = self.encoder(hidden, mask=mask, src_key_padding_mask=key_padding)
        return self.to_vocab(hidden)

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        flat_logits = logits.reshape(-1, self.vocab_size)
        flat_labels = labels.reshape(-1)
        ignore_mask = flat_labels == -100
        flat_labels = flat_labels.masked_fill(ignore_mask, -100)
        loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
        pred = flat_logits.argmax(dim=-1)
        correct = (pred == flat_labels) & ~ignore_mask
        token_total = torch.sum(~ignore_mask).clamp_min(1).to(dtype=torch.float32)
        token_accuracy = correct.sum(dtype=torch.float32) / token_total

        labels_matrix = labels.view(input_ids.shape[0], input_ids.shape[1])
        pred_matrix = pred.view(input_ids.shape[0], input_ids.shape[1])
        exact_by_row = []
        for row in range(labels_matrix.shape[0]):
            row_mask = labels_matrix[row] != -100
            if int(row_mask.sum()) == 0:
                exact_by_row.append(torch.tensor(0.0, device=labels.device))
                continue
            row_pred = pred_matrix[row][row_mask]
            row_label = labels_matrix[row][row_mask]
            exact_by_row.append((row_pred == row_label).all().to(torch.float32))

        final_exact_accuracy = torch.stack(exact_by_row).mean()

        return {
            "loss": loss,
            "reconstruction_loss": loss.detach(),
            "perplexity": torch.exp(loss).detach(),
            "final_token_accuracy": token_accuracy.detach(),
            "final_exact_accuracy": final_exact_accuracy.detach(),
        }


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _append_jsonl(path: Path, row: dict[str, float]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _build_train_record(
    *,
    loss: float,
    perplexity: float,
    teacher_token_accuracy: float,
    teacher_exact_accuracy: float,
    autoregressive_token_accuracy: float,
    autoregressive_exact_accuracy: float,
) -> dict[str, float]:
    return {
        "loss": float(loss),
        "perplexity": float(perplexity),
        "teacher_forcing_token_accuracy": float(teacher_token_accuracy),
        "teacher_forcing_exact_accuracy": float(teacher_exact_accuracy),
        "autoregressive_token_accuracy": float(autoregressive_token_accuracy),
        "autoregressive_exact_accuracy": float(autoregressive_exact_accuracy),
    }


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
    run.log(
        {
            **{f"{prefix}{k}": v for k, v in val_metrics.items()},
            **{f"exact_accuracy/val/{k}": v for k, v in exact_metrics.items()},
            "progress/step": float(step),
            "progress/epoch": float(epoch),
        },
        step=step,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _optional_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "undefined"}:
            return None
    return value


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


def _decode_autoregressive(
    model: BaselineTransformer,
    prompt_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    max_len: int,
    target_attention_mask: torch.Tensor | None,
    amp: bool,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> torch.Tensor:
    if max_len <= 0:
        return torch.empty((prompt_ids.shape[0], 0), dtype=torch.long, device=prompt_ids.device)

    prompt_lengths = prompt_attention_mask.sum(dim=1).to(dtype=torch.long)
    if int(prompt_lengths.numel()) == 0:
        return torch.empty((prompt_ids.shape[0], 0), dtype=torch.long, device=prompt_ids.device)

    # Some batches append a synthetic EOS token to the prompt separator.
    # Trim it so decoding starts from the exact training prefix: `prompt + BOS`.
    safe_idx = (prompt_lengths.clamp(min=1) - 1).clamp(min=0)
    last_tok = prompt_ids.gather(dim=1, index=safe_idx.view(-1, 1)).view(-1)
    has_sep_eos = (prompt_attention_mask.gather(dim=1, index=safe_idx.view(-1, 1)).view(-1) > 0) & (last_tok == eos_id)
    prompt_effective_lengths = (prompt_lengths - has_sep_eos.to(dtype=torch.long)).clamp(min=0)
    batch_size = int(prompt_ids.shape[0])
    generated = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=prompt_ids.device)
    model_max_positions = int(getattr(model, "max_positions", model.pos_embed.num_embeddings))
    was_training = model.training
    model.eval()

    if target_attention_mask is None:
        target_lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=prompt_ids.device)
    else:
        target_lengths = target_attention_mask.sum(dim=1).to(dtype=torch.long).clamp(max=max_len)

    groups: list[list[int]] = []
    group_bounds: list[tuple[int, int]] = []
    order = sorted(
        range(batch_size),
        key=lambda i: (int(prompt_effective_lengths[i].item()), int(target_lengths[i].item())),
        reverse=True,
    )
    for row in order:
        row_prompt = int(prompt_effective_lengths[row].item())
        row_target = int(target_lengths[row].item())
        if row_target <= 0:
            continue
        placed = False
        for group_idx, (group_prompt, group_target) in enumerate(group_bounds):
            next_prompt = max(group_prompt, row_prompt)
            next_target = max(group_target, row_target)
            if next_prompt + 1 + next_target <= model_max_positions:
                groups[group_idx].append(row)
                group_bounds[group_idx] = (next_prompt, next_target)
                placed = True
                break
        if not placed:
            groups.append([row])
            group_bounds.append((row_prompt, row_target))

    for rows, (group_prompt_len, group_target_len) in zip(groups, group_bounds):
        if group_target_len <= 0:
            continue
        idx = torch.tensor(rows, dtype=torch.long, device=prompt_ids.device)
        prompt_context = prompt_ids.index_select(dim=0, index=idx)[:, :group_prompt_len]
        context_mask = torch.zeros((len(rows), group_prompt_len), dtype=torch.bool, device=prompt_ids.device)
        for local_row, source_row in enumerate(rows):
            context_mask[local_row, : int(prompt_effective_lengths[source_row].item())] = True

        bos = torch.full((len(rows), 1), bos_id, dtype=torch.long, device=prompt_ids.device)
        running_input = torch.cat([prompt_context, bos], dim=1)
        running_attention = torch.cat(
            [context_mask, torch.ones((len(rows), 1), dtype=torch.bool, device=prompt_ids.device)],
            dim=1,
        )

        for step in range(group_target_len):
            with torch.autocast(device_type=prompt_ids.device.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(input_ids=running_input, attention_mask=running_attention)
            next_token = logits[:, -1, :].argmax(dim=-1)
            active = target_lengths.index_select(dim=0, index=idx) > step
            generated[idx[active], step] = next_token[active]
            running_input = torch.cat([running_input, next_token[:, None]], dim=1)
            running_attention = torch.cat(
                [running_attention, torch.ones((len(rows), 1), dtype=torch.bool, device=prompt_ids.device)],
                dim=1,
            )

    if was_training:
        model.train()

    return generated


def _append_autoregressive_metrics(
    totals: dict[str, float],
    *,
    auto_token_correct: int,
    auto_token_total: int,
    auto_exact_correct: float,
    auto_rows: int,
) -> None:
    totals["autoregressive_token_correct"] += float(auto_token_correct)
    totals["autoregressive_tokens"] += float(auto_token_total)
    totals["autoregressive_exact_correct"] += float(auto_exact_correct)
    totals["autoregressive_rows"] += float(auto_rows)


def _evaluate(
    *,
    model: nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    tokenizer: SyntheticTokenizer,
    amp: bool,
    max_batches: int | None = None,
    ar_max_batches: int | None = 8,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {metric: 0.0 for metric in TOKEN_METRICS}
    totals["autoregressive_token_correct"] = 0.0
    totals["autoregressive_tokens"] = 0.0
    totals["autoregressive_exact_correct"] = 0.0
    totals["autoregressive_rows"] = 0.0
    rows_for_exact = 0
    tokens = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            prompt_ids = batch["prompt_token_ids"]
            prompt_attention_mask = batch["prompt_attention_mask"]
            target_ids = batch["target_token_ids"]
            target_attention = batch["target_attention_mask"]
            token_count = int((batch["labels"] != -100).sum().item())
            row_count = int(batch["labels"].shape[0])

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss_dict = model.loss(
                    input_ids=batch["full_input_ids"],
                    attention_mask=batch["input_attention_mask"],
                    labels=batch["labels"],
                )

            totals["loss"] += float(loss_dict["loss"].detach().item()) * token_count
            totals["reconstruction_loss"] += float(loss_dict["reconstruction_loss"].detach().item()) * token_count
            totals["perplexity"] += float(loss_dict["perplexity"].detach().item()) * token_count
            totals["final_token_accuracy"] += float(loss_dict["final_token_accuracy"].detach().item()) * token_count
            totals["final_exact_accuracy"] += float(loss_dict["final_exact_accuracy"].detach().item()) * row_count

            if ar_max_batches is None or batch_idx < ar_max_batches:
                max_target = int(target_ids.shape[1])
                generated = _decode_autoregressive(
                    model=model,
                    prompt_ids=prompt_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    max_len=max_target,
                    target_attention_mask=target_attention,
                    amp=amp,
                    bos_id=tokenizer.bos_token_id,
                    eos_id=tokenizer.eos_token_id,
                    pad_id=tokenizer.pad_token_id,
                )
                target_mask = target_attention.to(dtype=torch.bool)
                if max_target > 0:
                    generated_masked = torch.where(target_mask, generated, torch.tensor(tokenizer.pad_token_id, device=device))
                    target_masked = torch.where(target_mask, target_ids, torch.tensor(tokenizer.pad_token_id, device=device))
                    token_total = int(target_mask.sum().item())
                    token_correct = int(((generated_masked == target_masked) & target_mask).sum().item())
                    exact = ((generated_masked == target_masked) | ~target_mask).all(dim=1).to(dtype=torch.float32)
                    exact_correct = float(exact.sum().item())
                else:
                    token_total = 0
                    token_correct = 0
                    exact_correct = 0.0

                _append_autoregressive_metrics(
                    totals,
                    auto_token_correct=token_correct,
                    auto_token_total=token_total,
                    auto_exact_correct=exact_correct,
                    auto_rows=row_count,
                )

            rows_for_exact += row_count
            tokens += token_count

            if max_batches is not None and batch_idx + 1 >= max_batches:
                break

    model.train()
    denom_tokens = max(tokens, 1)
    denom_rows = max(rows_for_exact, 1)
    final_token_accuracy = totals["final_token_accuracy"] / denom_tokens
    final_exact_accuracy = totals["final_exact_accuracy"] / denom_rows
    out = {
        "loss": totals["loss"] / denom_tokens,
        "reconstruction_loss": totals["reconstruction_loss"] / denom_tokens,
        "perplexity": totals["perplexity"] / denom_tokens,
        "final_token_accuracy": final_token_accuracy,
        "final_exact_accuracy": final_exact_accuracy,
        "teacher_forcing_token_accuracy": final_token_accuracy,
        "teacher_forcing_exact_accuracy": final_exact_accuracy,
        "autoregressive_token_accuracy": totals["autoregressive_token_correct"] / max(totals["autoregressive_tokens"], 1),
        "autoregressive_exact_accuracy": totals["autoregressive_exact_correct"] / max(totals["autoregressive_rows"], 1),
    }
    return out


def _build_model(cfg: DictConfig, tokenizer: SyntheticTokenizer) -> BaselineTransformer:
    return BaselineTransformer(
        vocab_size=int(tokenizer.vocab_size if str(cfg.model.vocab_size) == "local" else int(cfg.model.vocab_size)),
        hidden_dim=int(cfg.model.hidden_dim),
        num_layers=int(cfg.model.num_layers),
        num_heads=int(cfg.model.num_heads),
        dim_feedforward=int(cfg.model.dim_feedforward),
        max_positions=int(cfg.model.max_positions),
        dropout=float(cfg.model.dropout),
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
        if not muon_params:
            adamw_kwargs = {
                "lr": float(cfg.optim.lr),
                "weight_decay": float(cfg.optim.weight_decay),
                "betas": tuple(float(x) for x in cfg.optim.betas),
                "eps": float(cfg.optim.eps),
            }
            if bool(muon_cfg["fused_adamw"]):
                adamw_kwargs["fused"] = True
            try:
                return torch.optim.AdamW(adamw_params, **adamw_kwargs)
            except TypeError:
                adamw_kwargs.pop("fused", None)
                return torch.optim.AdamW(adamw_params, **adamw_kwargs)

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
    if optimizer_name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=float(cfg.optim.lr),
            weight_decay=float(cfg.optim.weight_decay),
            betas=tuple(float(x) for x in cfg.optim.betas),
            eps=float(cfg.optim.eps),
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optim.name}")


@hydra.main(version_base=None, config_path="configs", config_name="transformer_config")
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available() and not bool(cfg.allow_cpu):
        raise RuntimeError("CUDA unavailable; set allow_cpu=true for CPU-only debug.")

    _seed(int(cfg.seed))
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "config_resolved.yaml")

    tokenizer = SyntheticTokenizer.default()
    device = torch.device("cuda" if torch.cuda.is_available() and not bool(cfg.allow_cpu) else "cpu")
    amp_enabled = bool(cfg.train.amp_bf16) and device.type == "cuda"

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

    model = _build_model(cfg, tokenizer).to(device=device)
    optimizer = _build_optimizer(model, cfg)

    max_steps = int(cfg.train.max_steps)
    log_every = int(cfg.train.log_every)
    save_every = int(cfg.train.save_every_steps)
    validate_every = int(cfg.validation.every_steps)
    grad_accum = max(1, int(cfg.train.grad_accum_steps))
    metrics_path = run_dir / "metrics.jsonl"
    validation_path = run_dir / "validation_metrics.jsonl"
    val_max_batches = None if cfg.validation.max_batches is None else int(cfg.validation.max_batches)
    val_ar_max_batches = int(cfg.validation.get("ar_max_batches", 8))
    if val_ar_max_batches <= 0:
        val_ar_max_batches = 0
    train_ar_rows = int(cfg.train.get("ar_metric_rows", 16))
    if train_ar_rows <= 0:
        train_ar_rows = 0

    wandb_run = _init_wandb(cfg, run_dir)
    epoch = 0

    initial_train_batch = next(iter(train_loader))
    initial_train_batch = _move_batch(initial_train_batch, device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        initial_loss_dict = model.loss(
            input_ids=initial_train_batch["full_input_ids"],
            attention_mask=initial_train_batch["input_attention_mask"],
            labels=initial_train_batch["labels"],
        )
    target_ids = initial_train_batch["target_token_ids"]
    target_attention = initial_train_batch["target_attention_mask"].bool()
    ar_rows = min(int(target_ids.shape[0]), train_ar_rows)
    max_target = int(target_ids[:ar_rows].shape[1]) if ar_rows > 0 else 0
    if ar_rows > 0 and max_target > 0:
        with torch.inference_mode():
            initial_generated = _decode_autoregressive(
                model=model,
                prompt_ids=initial_train_batch["prompt_token_ids"][:ar_rows],
                prompt_attention_mask=initial_train_batch["prompt_attention_mask"][:ar_rows],
                max_len=max_target,
                target_attention_mask=target_attention[:ar_rows],
                amp=amp_enabled,
                bos_id=tokenizer.bos_token_id,
                eos_id=tokenizer.eos_token_id,
                pad_id=tokenizer.pad_token_id,
            )
        pad = torch.tensor(tokenizer.pad_token_id, dtype=torch.long, device=device)
        pred = torch.where(target_attention[:ar_rows], initial_generated, pad)
        tgt = torch.where(target_attention[:ar_rows], target_ids[:ar_rows], pad)
        initial_auto_token_acc = ((pred == tgt) & target_attention[:ar_rows]).sum(dtype=torch.float32) / target_attention[:ar_rows].sum().clamp_min(1).to(dtype=torch.float32)
        initial_auto_exact_acc = (((pred == tgt) | ~target_attention[:ar_rows]).all(dim=1).to(dtype=torch.float32)).mean()
    else:
        initial_auto_token_acc = torch.tensor(0.0, device=device)
        initial_auto_exact_acc = torch.tensor(0.0, device=device)

    initial_train_record = _build_train_record(
        loss=float(initial_loss_dict["loss"].detach().item()),
        perplexity=float(initial_loss_dict["perplexity"].detach().item()),
        teacher_token_accuracy=float(initial_loss_dict["final_token_accuracy"].detach().item()),
        teacher_exact_accuracy=float(initial_loss_dict["final_exact_accuracy"].detach().item()),
        autoregressive_token_accuracy=float(initial_auto_token_acc.detach().item()),
        autoregressive_exact_accuracy=float(initial_auto_exact_acc.detach().item()),
    )
    _append_jsonl(metrics_path, initial_train_record)
    _log_train_step(
        run=wandb_run,
        metrics=initial_train_record,
        step=0,
        epoch=0,
        batch_rows=int(target_ids.shape[0]),
        batch_tokens=float(target_attention.sum().item()),
    )
    print(
        f"step=0 loss={initial_train_record['loss']:.4f} "
        f"ppl={initial_train_record['perplexity']:.4f} "
        f"tf_tok={initial_train_record['teacher_forcing_token_accuracy']:.4f} "
        f"ar_tok={initial_train_record['autoregressive_token_accuracy']:.4f}"
    )

    initial_val = _evaluate(
        model=model,
        loader=val_loader,
        device=device,
        tokenizer=tokenizer,
        amp=amp_enabled,
        max_batches=val_max_batches,
        ar_max_batches=val_ar_max_batches,
    )
    _append_jsonl(validation_path, initial_val)
    _log_val_metrics(wandb_run, step=0, epoch=0, metrics=initial_val)
    print(
        f"val step={0} loss={initial_val['loss']:.4f} "
        f"ppl={initial_val['perplexity']:.4f} "
        f"tf_token={initial_val['teacher_forcing_token_accuracy']:.4f} "
        f"ar_token={initial_val['autoregressive_token_accuracy']:.4f}"
    )

    optimizer.zero_grad(set_to_none=True)
    step = 0
    for epoch_idx in range(10_000):
        epoch = epoch_idx
        for batch in train_loader:
            step += 1
            batch = _move_batch(batch, device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
                loss_dict = model.loss(
                    input_ids=batch["full_input_ids"],
                    attention_mask=batch["input_attention_mask"],
                    labels=batch["labels"],
                )
            (loss_dict["loss"] / grad_accum).backward()

            if step % grad_accum == 0:
                if float(cfg.train.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                target_ids = batch["target_token_ids"]
                target_attention = batch["target_attention_mask"].bool()
                ar_rows = min(int(target_ids.shape[0]), train_ar_rows)
                max_target = int(target_ids[:ar_rows].shape[1]) if ar_rows > 0 else 0
                if ar_rows > 0 and max_target > 0:
                    with torch.inference_mode():
                        generated = _decode_autoregressive(
                            model=model,
                            prompt_ids=batch["prompt_token_ids"][:ar_rows],
                            prompt_attention_mask=batch["prompt_attention_mask"][:ar_rows],
                            max_len=max_target,
                            target_attention_mask=target_attention[:ar_rows],
                            amp=amp_enabled,
                            bos_id=tokenizer.bos_token_id,
                            eos_id=tokenizer.eos_token_id,
                            pad_id=tokenizer.pad_token_id,
                        )
                    pad = torch.tensor(tokenizer.pad_token_id, dtype=torch.long, device=device)
                    pred = torch.where(target_attention[:ar_rows], generated, pad)
                    tgt = torch.where(target_attention[:ar_rows], target_ids[:ar_rows], pad)
                    auto_token_acc = ((pred == tgt) & target_attention[:ar_rows]).sum(dtype=torch.float32) / target_attention[:ar_rows].sum().clamp_min(1).to(dtype=torch.float32)
                    auto_exact_acc = (((pred == tgt) | ~target_attention[:ar_rows]).all(dim=1).to(dtype=torch.float32)).mean()
                else:
                    auto_token_acc = torch.tensor(0.0, device=device)
                    auto_exact_acc = torch.tensor(0.0, device=device)

                stats = _build_train_record(
                    loss=float(loss_dict["loss"].detach().item()),
                    perplexity=float(loss_dict["perplexity"].detach().item()),
                    teacher_token_accuracy=float(loss_dict["final_token_accuracy"].detach().item()),
                    teacher_exact_accuracy=float(loss_dict["final_exact_accuracy"].detach().item()),
                    autoregressive_token_accuracy=float(auto_token_acc.detach().item()),
                    autoregressive_exact_accuracy=float(auto_exact_acc.detach().item()),
                )
                _append_jsonl(metrics_path, stats)
                print(
                    f"step={step} loss={stats['loss']:.4f} "
                    f"ppl={stats['perplexity']:.4f} "
                    f"tf_token={stats['teacher_forcing_token_accuracy']:.4f} "
                    f"ar_token={stats['autoregressive_token_accuracy']:.4f}"
                )
                if wandb_run is not None:
                    _log_train_step(
                        run=wandb_run,
                        metrics=stats,
                        step=step,
                        epoch=epoch,
                        batch_rows=int(target_ids.shape[0]),
                        batch_tokens=float(target_attention.sum().item()),
                    )

            if step % validate_every == 0:
                val_metrics = _evaluate(
                    model=model,
                    loader=val_loader,
                    device=device,
                    tokenizer=tokenizer,
                    amp=amp_enabled,
                    max_batches=val_max_batches,
                    ar_max_batches=val_ar_max_batches,
                )
                _append_jsonl(validation_path, val_metrics)
                print(
                    f"val step={step} loss={val_metrics['loss']:.4f} "
                    f"ppl={val_metrics['perplexity']:.4f} "
                    f"tf_token={val_metrics['teacher_forcing_token_accuracy']:.4f} "
                    f"ar_token={val_metrics['autoregressive_token_accuracy']:.4f}"
                )
                _log_val_metrics(wandb_run, step=step, epoch=epoch, metrics=val_metrics)

            if save_every > 0 and step % save_every == 0:
                torch.save(model.state_dict(), run_dir / f"checkpoint_step_{step}.pt")

            if step >= max_steps:
                break
        if step >= max_steps:
            break

    torch.save(model.state_dict(), run_dir / "checkpoint_final.pt")

    final = _evaluate(model=model, loader=val_loader, device=device, tokenizer=tokenizer, amp=amp_enabled, max_batches=val_max_batches)
    _append_jsonl(validation_path, {"phase": "final", **final})
    print(
        f"final_loss={final['loss']:.4f} "
        f"final_tf_token={final['teacher_forcing_token_accuracy']:.4f} "
        f"final_ar_token={final['autoregressive_token_accuracy']:.4f}"
    )
    if wandb_run is not None:
        _log_val_metrics(wandb_run, step=step, epoch=epoch, metrics=final)
        wandb_run.finish()


if __name__ == "__main__":
    main()
