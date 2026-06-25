"""VeriBench-conditioned EBT model using the cloned alexiglad/EBT transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from model.ar_ebt_time_embed import EBTModelArgs, EBTTimeConcat


@dataclass
class VeriBenchContextEBTConfig:
    vocab_size: int
    context_dim: int = 4096
    hidden_dim: int = 384
    num_layers: int = 6
    num_heads: int = 6
    ffn_dim_multiplier: float = 4.0
    max_seq_len: int = 520
    max_mcmc_steps: int = 2
    mcmc_num_steps: int = 2
    mcmc_step_size: float = 0.5
    mcmc_step_size_learnable: bool = True
    gaussian_random_noise_scaling: float = 1.0
    normalize_initial_condition: bool = True
    normalize_initial_condition_only_first_step: bool = False
    denoising_initial_condition: str = "random_noise"
    langevin_dynamics_noise: float = 0.0
    truncate_mcmc: bool = False
    no_mcmc_detach: bool = False
    reconstruction_coeff: float = 1.0
    weight_initialization: str = "xavier"
    weight_initialization_gain: float = 1.0
    use_context: bool = True


class VeriBenchContextEBT(nn.Module):
    """EBT next-token learner conditioned on Goedel hidden-state context.

    The cloned EBT NLP model normally conditions on previous natural-language
    tokens only. This variant keeps that EBT MCMC/token-energy core but adds a
    projected pooled Goedel context activation to both the real-token and
    predicted-token embeddings.
    """

    def __init__(self, cfg: VeriBenchContextEBTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vocab_size = int(cfg.vocab_size)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.mcmc_step_size)), requires_grad=cfg.mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(torch.tensor(float(cfg.langevin_dynamics_noise)), requires_grad=False)

        self.embeddings = nn.Embedding(self.vocab_size, cfg.hidden_dim)
        self.vocab_to_embed = nn.Linear(self.vocab_size, cfg.hidden_dim, bias=False)
        self.context_proj = nn.Sequential(
            nn.Linear(cfg.context_dim, cfg.hidden_dim, bias=False),
            nn.LayerNorm(cfg.hidden_dim),
        )

        args = EBTModelArgs(
            dim=cfg.hidden_dim,
            n_layers=cfg.num_layers,
            n_heads=cfg.num_heads,
            ffn_dim_multiplier=cfg.ffn_dim_multiplier,
            max_seq_len=cfg.max_seq_len,
            weight_initialization=cfg.weight_initialization,
            weight_initialization_gain=cfg.weight_initialization_gain,
            ebt_norm="rms",
            ebt_act_func="silu",
        )
        self.transformer = EBTTimeConcat(args, max_mcmc_steps=cfg.max_mcmc_steps)

    def _context_embedding(self, context_activations: torch.Tensor, context_mask: torch.Tensor | None) -> torch.Tensor:
        if not self.cfg.use_context:
            batch_size = int(context_activations.shape[0])
            return torch.zeros(
                batch_size,
                1,
                self.cfg.hidden_dim,
                dtype=self.embeddings.weight.dtype,
                device=context_activations.device,
            )
        if context_mask is None:
            pooled = context_activations.mean(dim=1)
        else:
            mask = context_mask.to(device=context_activations.device, dtype=context_activations.dtype).unsqueeze(-1)
            pooled = (context_activations * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.context_proj(pooled).unsqueeze(1)

    def _init_logits(self, batch_size: int, target_len: int, device: torch.device) -> torch.Tensor:
        shape = (batch_size, target_len, self.vocab_size)
        if self.cfg.denoising_initial_condition == "random_noise":
            return torch.randn(*shape, device=device) * self.cfg.gaussian_random_noise_scaling
        if self.cfg.denoising_initial_condition == "zeros":
            return torch.zeros(*shape, device=device)
        raise NotImplementedError(self.cfg.denoising_initial_condition)

    def _logits_to_embeddings(self, logits: torch.Tensor, mcmc_step: int) -> torch.Tensor:
        if self.cfg.normalize_initial_condition:
            if not self.cfg.normalize_initial_condition_only_first_step or mcmc_step == 0:
                logits = torch.softmax(logits, dim=-1)
        return self.vocab_to_embed(logits)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        context_activations: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        learning: bool = True,
        capture_intermediates: bool = False,
    ) -> dict[str, Any]:
        batch_size, target_len = decoder_input_ids.shape
        context_activations = context_activations.to(dtype=self.embeddings.weight.dtype)
        context_embedding = self._context_embedding(context_activations, context_mask)
        real_embeddings = self.embeddings(decoder_input_ids.clamp_min(0)) + context_embedding
        predicted_tokens = self._init_logits(batch_size, target_len, decoder_input_ids.device)

        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        predicted_distributions: list[torch.Tensor] = []
        predicted_energies: list[torch.Tensor] = []
        intermediates: list[dict[str, Any]] = []

        with torch.set_grad_enabled(True):
            for i, mcmc_step in enumerate(range(int(self.cfg.mcmc_num_steps))):
                if self.cfg.no_mcmc_detach:
                    predicted_tokens.requires_grad_(True)
                else:
                    predicted_tokens = predicted_tokens.detach().requires_grad_()

                if float(self.langevin_dynamics_noise_std.detach()) != 0.0:
                    predicted_tokens = predicted_tokens + torch.randn_like(predicted_tokens.detach()) * langevin_std

                predicted_embeddings = self._logits_to_embeddings(predicted_tokens, mcmc_step) + context_embedding
                all_embeddings = torch.cat((real_embeddings, predicted_embeddings), dim=1)
                transformer_out = self.transformer(
                    all_embeddings,
                    start_pos=0,
                    mcmc_step=mcmc_step,
                    return_intermediates=capture_intermediates,
                )
                if capture_intermediates:
                    energy_preds, transformer_intermediates = transformer_out
                else:
                    energy_preds = transformer_out
                    transformer_intermediates = None
                if target_mask is not None:
                    energy_preds = energy_preds.squeeze(-1).masked_fill(~target_mask.bool(), 0.0).unsqueeze(-1)
                predicted_energies.append(energy_preds.reshape(-1, 1))

                create_graph = bool(learning)
                if self.cfg.truncate_mcmc and i != int(self.cfg.mcmc_num_steps) - 1:
                    create_graph = False
                grad = torch.autograd.grad(energy_preds.sum(), predicted_tokens, create_graph=create_graph)[0]
                if torch.isnan(grad).any() or torch.isinf(grad).any():
                    raise ValueError("NaN or Inf gradients detected during EBT MCMC.")
                predicted_tokens = predicted_tokens - alpha * grad
                predicted_distributions.append(predicted_tokens)

                if capture_intermediates:
                    intermediates.append(
                        {
                            "mcmc_step": mcmc_step,
                            "predicted_logits": predicted_tokens.detach().to("cpu", dtype=torch.float16),
                            "predicted_embeddings": predicted_embeddings.detach().to("cpu", dtype=torch.float16),
                            "energies": energy_preds.detach().to("cpu", dtype=torch.float16),
                            "context_embedding": context_embedding.detach().to("cpu", dtype=torch.float16),
                            "transformer": transformer_intermediates,
                        }
                    )

        return {
            "predicted_distributions": predicted_distributions,
            "predicted_energies": predicted_energies,
            "intermediates": intermediates,
        }

    def loss(
        self,
        decoder_input_ids: torch.Tensor,
        labels: torch.Tensor,
        context_activations: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        capture_intermediates: bool = False,
    ) -> dict[str, Any]:
        out = self.forward(
            decoder_input_ids,
            context_activations,
            context_mask=context_mask,
            target_mask=target_mask,
            learning=True,
            capture_intermediates=capture_intermediates,
        )
        targets = labels.reshape(-1)
        if target_mask is not None:
            targets = targets.masked_fill(~target_mask.reshape(-1).bool(), -100)

        total_loss = torch.zeros((), device=labels.device, dtype=torch.float32)
        initial_loss = None
        final_step_loss = None
        for i, logits in enumerate(out["predicted_distributions"]):
            step_loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets, ignore_index=-100)
            if self.cfg.truncate_mcmc:
                if i == len(out["predicted_distributions"]) - 1:
                    total_loss = step_loss
            else:
                total_loss = total_loss + step_loss
            if i == 0:
                initial_loss = step_loss.detach()
            if i == len(out["predicted_distributions"]) - 1:
                final_step_loss = step_loss.detach()
        if not self.cfg.truncate_mcmc:
            total_loss = total_loss / max(1, len(out["predicted_distributions"]))

        return {
            "loss": self.cfg.reconstruction_coeff * total_loss,
            "initial_loss": initial_loss if initial_loss is not None else total_loss.detach(),
            "final_step_loss": final_step_loss if final_step_loss is not None else total_loss.detach(),
            "perplexity": torch.exp(final_step_loss if final_step_loss is not None else total_loss.detach()),
            "intermediates": out["intermediates"],
        }
