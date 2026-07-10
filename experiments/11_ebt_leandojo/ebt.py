#!/usr/bin/env python3
"""Chunk-aware EBT over Lean token logits conditioned on Goedel context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoTokenizer


DEFAULT_MODEL = "Goedel-LM/Goedel-Prover-V2-8B"


@dataclass
class EBTOutput:
    logits: torch.Tensor
    predicted_distributions: list[torch.Tensor]
    energies: list[torch.Tensor]
    decoder_hidden: torch.Tensor


class GoedelVocabEBT(nn.Module):
    """Energy model over differentiable chunked full-vocab token logits.

    This follows the NLP EBT pattern from alexiglad/EBT: token ids enter only as
    conditioning/targets; the optimized state is an internal
    `[B, T_chunk, chunk_size, vocab]` tensor. Each EBT target position represents
    `chunk_size` Lean token slots. A shared token projection maps each
    vocabulary distribution to `token_embed_dim`, and chunk slots are
    concatenated into one decoder hidden state of size
    `token_embed_dim * chunk_size`.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        revision: str | None = None,
        vocab_size: int | None = None,
        pad_token_id: int | None = None,
        context_dim: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int = 1,
        num_heads: int = 32,
        dim_feedforward: int = 8192,
        dropout: float = 0.0,
        mcmc_num_steps: int = 2,
        mcmc_step_size: float = 0.1,
        mcmc_step_size_learnable: bool = False,
        gaussian_random_noise_scaling: float = 1.0,
        denoising_initial_condition: str = "random_noise",
        normalize_initial_condition: bool = True,
        normalize_initial_condition_only_first_step: bool = False,
        langevin_dynamics_noise: float = 0.0,
        truncate_mcmc: bool = False,
        no_mcmc_detach: bool = False,
        clamp_futures_grad: bool = False,
        clamp_futures_grad_max_change: float = 9.0,
        absolute_clamp: float = 0.0,
        sharpen_predicted_distribution: float = 0.0,
        norm_pred: bool = False,
        norm_pred_not_final_step: bool = False,
        reconstruction_coeff: float = 1.0,
        soften_target_prob_dist: float = 0.0,
        loss_on_final_step_only: bool = False,
        contrastive_rollout_enabled: bool = True,
        contrastive_rollout_chunks: int = 8,
        contrastive_rollout_coeff: float = 1.0,
        contrastive_rollout_steps: int | None = None,
        max_target_positions: int = 4096,
        chunk_size: int = 1,
        token_embed_dim: int = 256,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        architecture: str = "decoder",
    ) -> None:
        super().__init__()
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)

        self.vocab_size = int(vocab_size or len(tokenizer))
        self.chunk_size = int(chunk_size)
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.token_embed_dim = int(token_embed_dim)
        if self.token_embed_dim <= 0:
            raise ValueError("token_embed_dim must be positive")
        inferred_hidden_dim = self.token_embed_dim * self.chunk_size
        if hidden_dim is not None and int(hidden_dim) != inferred_hidden_dim:
            raise ValueError(
                f"hidden_dim must equal token_embed_dim * chunk_size "
                f"({self.token_embed_dim} * {self.chunk_size} = {inferred_hidden_dim}), got {hidden_dim}"
            )
        self.hidden_dim = inferred_hidden_dim
        self.context_dim = int(context_dim or self.hidden_dim)
        if pad_token_id is None:
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.pad_token_id = int(pad_token_id)
        self.eos_token_id = int(eos_token_id if eos_token_id is not None else tokenizer.eos_token_id)
        self.bos_token_id = int(
            bos_token_id
            if bos_token_id is not None
            else (tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id)
        )
        self.mcmc_num_steps = int(mcmc_num_steps)
        self.architecture = str(architecture).lower()
        if self.architecture not in {"decoder", "encoder"}:
            raise ValueError(f"Unsupported architecture: {architecture!r}")
        self.gaussian_random_noise_scaling = float(gaussian_random_noise_scaling)
        self.denoising_initial_condition = denoising_initial_condition
        self.normalize_initial_condition = bool(normalize_initial_condition)
        self.normalize_initial_condition_only_first_step = bool(normalize_initial_condition_only_first_step)
        self.truncate_mcmc = bool(truncate_mcmc)
        self.no_mcmc_detach = bool(no_mcmc_detach)
        self.clamp_futures_grad = bool(clamp_futures_grad)
        self.clamp_futures_grad_max_change = float(clamp_futures_grad_max_change)
        self.absolute_clamp = float(absolute_clamp)
        self.sharpen_predicted_distribution = float(sharpen_predicted_distribution)
        self.norm_pred = bool(norm_pred)
        self.norm_pred_not_final_step = bool(norm_pred_not_final_step)
        self.reconstruction_coeff = float(reconstruction_coeff)
        self.soften_target_prob_dist = float(soften_target_prob_dist)
        self.loss_on_final_step_only = bool(loss_on_final_step_only)
        self.contrastive_rollout_enabled = bool(contrastive_rollout_enabled)
        self.contrastive_rollout_chunks = int(contrastive_rollout_chunks)
        if self.contrastive_rollout_chunks <= 0:
            raise ValueError("contrastive_rollout_chunks must be positive")
        self.contrastive_rollout_coeff = float(contrastive_rollout_coeff)
        self.contrastive_rollout_steps = (
            None if contrastive_rollout_steps is None else int(contrastive_rollout_steps)
        )
        self.alpha = nn.Parameter(torch.tensor(float(mcmc_step_size)), requires_grad=mcmc_step_size_learnable)
        self.langevin_dynamics_noise_std = nn.Parameter(
            torch.tensor(float(langevin_dynamics_noise)),
            requires_grad=False,
        )

        self.vocab_to_token_embed = nn.Linear(self.vocab_size, self.token_embed_dim, bias=False)
        self.context_to_hidden = (
            nn.Linear(self.context_dim, self.hidden_dim, bias=False)
            if self.context_dim != self.hidden_dim
            else nn.Identity()
        )
        self.target_position_embeddings = nn.Embedding(int(max_target_positions), self.hidden_dim)
        if self.norm_pred:
            self.pred_norm = nn.RMSNorm(self.vocab_size)
        if self.architecture == "decoder":
            layer = nn.TransformerDecoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.downstream_model = nn.TransformerDecoder(layer, num_layers=num_layers)
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.downstream_model = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.energy_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))
        self.prediction_norm = nn.LayerNorm(self.hidden_dim)
        self.token_embed_to_vocab = nn.Linear(self.token_embed_dim, self.vocab_size, bias=False)

    def init_logits(
        self,
        *,
        batch_size: int,
        target_len: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        shape = (batch_size, target_len, self.chunk_size, self.vocab_size)
        if self.denoising_initial_condition == "random_noise":
            return torch.randn(*shape, device=device) * self.gaussian_random_noise_scaling
        if self.denoising_initial_condition == "zeros":
            return torch.zeros(*shape, device=device)
        raise NotImplementedError(
            f"{self.denoising_initial_condition!r} denoising_initial_condition is not supported"
        )

    def vocab_state_to_embeddings(
        self,
        x: torch.Tensor,
        *,
        mcmc_step: int,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        if x.shape[-2:] != (self.chunk_size, self.vocab_size):
            raise ValueError(
                f"Expected chunk/vocab dimensions {(self.chunk_size, self.vocab_size)}, got {tuple(x.shape[-2:])}"
            )
        
        # we initialize the normal initial condition  
        # we ALWAYS softmax before turning into the EBT model
        if self.normalize_initial_condition and not already_normalized:
            if not self.normalize_initial_condition_only_first_step or mcmc_step == 0:
                x = torch.softmax(x, dim=-1)
        
        token_embeddings = self.vocab_to_token_embed(x)
        return token_embeddings.reshape(*x.shape[:-2], self.chunk_size * self.token_embed_dim)

    def ids_to_logits(self, ids: torch.Tensor, *, true_logit: float = 10.0, false_logit: float = 0.0) -> torch.Tensor:
        if ids.dim() != 3:
            raise ValueError(f"Expected chunked ids [B, T, C], got shape {tuple(ids.shape)}")
        if ids.shape[-1] != self.chunk_size:
            raise ValueError(f"Expected chunk_size {self.chunk_size}, got {ids.shape[-1]}")
        clamped = ids.clamp(min=0, max=self.vocab_size - 1).to(dtype=torch.long)
        logits = torch.full(
            (*clamped.shape, self.vocab_size),
            float(false_logit),
            device=ids.device,
            dtype=torch.float32,
        )
        return logits.scatter_(-1, clamped.unsqueeze(-1), float(true_logit))

    def ids_to_embeddings(
        self,
        ids: torch.Tensor,
        *,
        true_logit: float = 10.0,
        false_logit: float = 0.0,
    ) -> torch.Tensor:
        if ids.dim() != 3:
            raise ValueError(f"Expected chunked ids [B, T, C], got shape {tuple(ids.shape)}")
        if ids.shape[-1] != self.chunk_size:
            raise ValueError(f"Expected chunk_size {self.chunk_size}, got {ids.shape[-1]}")

        clamped = ids.clamp(min=0, max=self.vocab_size - 1).to(dtype=torch.long)
        weight = self.vocab_to_token_embed.weight
        exp_true = torch.exp(torch.tensor(float(true_logit), device=weight.device, dtype=weight.dtype))
        exp_false = torch.exp(torch.tensor(float(false_logit), device=weight.device, dtype=weight.dtype))
        denom = exp_true + (self.vocab_size - 1) * exp_false
        p_true = exp_true / denom
        p_false = exp_false / denom
        all_vocab = weight.sum(dim=1)
        token_embeddings = (
            p_false * all_vocab.view(*([1] * clamped.dim()), self.token_embed_dim)
            + (p_true - p_false) * F.embedding(clamped, weight.t())
        )
        return token_embeddings.reshape(*clamped.shape[:-1], self.chunk_size * self.token_embed_dim)

    def predict_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.prediction_norm(hidden)
        token_hidden = normalized.reshape(hidden.shape[0], hidden.shape[1], self.chunk_size, self.token_embed_dim)
        return self.token_embed_to_vocab(token_hidden)

    @staticmethod
    def target_causal_mask(target_len: int, *, device: torch.device | str) -> torch.Tensor:
        """Causal self-attention mask for generated EBT tokens only.

        This mask is passed as `tgt_mask`, so it affects target-token
        self-attention. Cross-attention to `memory` remains uncausal and can
        attend to every unpadded Goedel context activation.
        """
        return torch.triu(torch.ones(target_len, target_len, device=device, dtype=torch.bool), diagonal=1)

    def _memory(
        self,
        context_activations: torch.Tensor,
        target_embeddings: torch.Tensor,
        context_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        memory = context_activations.to(device=target_embeddings.device, dtype=target_embeddings.dtype)
        if memory.shape[-1] != self.context_dim:
            raise ValueError(f"Expected context dimension {self.context_dim}, got {memory.shape[-1]}")
        memory_mask = ~context_attention_mask.bool() if context_attention_mask is not None else None
        return self.context_to_hidden(memory), memory_mask

    def energy_from_target_embeddings(
        self,
        target_embeddings: torch.Tensor,
        context_activations: torch.Tensor,
        *,
        context_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # do positional embedding
        positions = torch.arange(target_embeddings.shape[1], device=target_embeddings.device)
        target_embeddings = target_embeddings + self.target_position_embeddings(positions).to(
            dtype=target_embeddings.dtype
        ).unsqueeze(0)

        tgt_len = int(target_embeddings.shape[1])
        target_self_attention_mask = self.target_causal_mask(tgt_len, device=target_embeddings.device)
        target_key_padding_mask = ~decoder_attention_mask.bool() if decoder_attention_mask is not None else None

        # Let PyTorch select the fastest available SDPA backend.
        if self.architecture == "decoder":
            # Goedel activations are used only in decoder mode as cross-attention memory.
            memory, memory_key_padding_mask = self._memory(
                context_activations,
                target_embeddings,
                context_attention_mask,
            )
            hidden = self.downstream_model(
                tgt=target_embeddings,
                memory=memory,
                tgt_mask=target_self_attention_mask,
                tgt_key_padding_mask=target_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        else:
            hidden = self.downstream_model(
                src=target_embeddings,
                mask=target_self_attention_mask,
                src_key_padding_mask=target_key_padding_mask,
            )
        
        # run through the energy head. Note that at the gradient, we take the SUM across ALL tokens!
        token_energy = self.energy_head(hidden).squeeze(-1)
        if decoder_attention_mask is not None:
            token_energy = token_energy.masked_fill(~decoder_attention_mask.bool(), 0.0)
        return token_energy.reshape(-1, 1), hidden

    def energy(
        self,
        x: torch.Tensor,
        context_activations: torch.Tensor,
        *,
        mcmc_step: int,
        context_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        state_already_normalized: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # get embeddings from vocab state
        target_embeddings = self.vocab_state_to_embeddings(
            x,
            mcmc_step=mcmc_step,
            already_normalized=state_already_normalized,
        )
        return self.energy_from_target_embeddings(
            target_embeddings,
            context_activations,
            context_attention_mask=context_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )

    def forward(
        self,
        context_activations: torch.Tensor,
        *,
        target_len: int | None = None,
        decoder_input_logits: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        learning: bool = True,
        return_raw_logits: bool = True,
        return_all_steps: bool = False,
    ) -> EBTOutput:
        if decoder_input_logits is None: # dhether we want to pre-initialize with something
            if target_len is None:
                raise ValueError("target_len is required when decoder_input_logits is None")
            x = self.init_logits( # usually, set the random noise
                batch_size=int(context_activations.shape[0]),
                target_len=target_len,
                device=context_activations.device,
            ) 
            # size: (batch_size, target_len, self.vocab_size)
            # (2, target_len, 3565) --> (2, target_len, 384) when embedding
        else:
            x = decoder_input_logits

        if x.dim() != 4:
            raise ValueError(f"Expected decoder state [B, T, C, V], got shape {tuple(x.shape)}")
        batch_size = int(x.shape[0])
        target_len = int(x.shape[1])
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        total_steps = max(1, int(self.mcmc_num_steps))

        predicted_distributions: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        hidden = None
        prediction_logits = None
        with torch.set_grad_enabled(True):
            # langevin steps
            for step in range(total_steps):
                if self.no_mcmc_detach:
                    x.requires_grad_(True)
                else:
                    x = x.detach().requires_grad_() # detach gradients (by default)

                if float(self.langevin_dynamics_noise_std.detach()) != 0.0:
                    x = x + torch.randn_like(x.detach()) * langevin_std

                mcmc_step = step
                state_already_normalized = False
                if self.normalize_initial_condition:
                    if not self.normalize_initial_condition_only_first_step or mcmc_step == 0:
                        x = torch.softmax(x, dim=-1) # we turn this to a softmax at the very FIRST step (aka from the random noise)
                        state_already_normalized = True

                # go through forward pass
                energy, hidden = self.energy(
                    x, # initial "guess"
                    context_activations, # conditioned on Goedel Activations
                    mcmc_step=mcmc_step,
                    context_attention_mask=context_attention_mask, # masked context and decoder 
                    decoder_attention_mask=decoder_attention_mask,
                    state_already_normalized=state_already_normalized,
                )
                energies.append(energy)
                prediction_logits = self.predict_from_hidden(hidden)

                # calculate gradient
                create_graph = bool(learning)
                if self.truncate_mcmc and step != total_steps - 1:
                    create_graph = False
                grad_x = torch.autograd.grad(energy.sum(), x, create_graph=create_graph)[0]
                
                # clamp grad
                if self.clamp_futures_grad:
                    min_and_max = self.clamp_futures_grad_max_change / torch.clamp(self.alpha.detach(), min=0.0001)
                    grad_x = torch.clamp(grad_x, min=-min_and_max, max=min_and_max)
                if torch.isnan(grad_x).any() or torch.isinf(grad_x).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")

                # update x
                x = x - alpha * grad_x
                if self.absolute_clamp != 0.0: # none of these actually happen in practice
                    x = torch.clamp(x, min=-self.absolute_clamp, max=self.absolute_clamp)
                if self.sharpen_predicted_distribution != 0.0:
                    x = x / self.sharpen_predicted_distribution
                if self.norm_pred and not (self.norm_pred_not_final_step and step == total_steps - 1):
                    x = self.pred_norm(x)

                predicted_distribution = (
                    prediction_logits
                    if return_raw_logits
                    else F.log_softmax(prediction_logits, dim=-1).reshape(
                        batch_size * target_len * self.chunk_size,
                        self.vocab_size,
                    )
                )
                predicted_distributions.append(predicted_distribution)

        selected_energies = energies if return_all_steps else energies[-1:]
        selected_distributions = predicted_distributions if return_all_steps else predicted_distributions[-1:]
        return EBTOutput(
            logits=prediction_logits if prediction_logits is not None else x,
            predicted_distributions=selected_distributions,
            energies=selected_energies,
            decoder_hidden=hidden,
        )

    def _target_embeddings_with_next_chunk(
        self,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        next_chunk_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prefix_input_ids.dim() != 3:
            raise ValueError(f"Expected prefix ids [B, P, C], got shape {tuple(prefix_input_ids.shape)}")
        if next_chunk_embeddings.dim() != 3 or next_chunk_embeddings.shape[1] != 1:
            raise ValueError(
                f"Expected next chunk embeddings [B, 1, H], got shape {tuple(next_chunk_embeddings.shape)}"
            )
        batch_size = int(prefix_input_ids.shape[0])
        prefix_len = int(prefix_input_ids.shape[1])
        total_len = prefix_len + 1
        device = next_chunk_embeddings.device
        dtype = next_chunk_embeddings.dtype
        target_embeddings = torch.zeros(
            batch_size,
            total_len,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )
        if prefix_len:
            prefix_embeddings = self.ids_to_embeddings(prefix_input_ids.to(device=device)).to(dtype=dtype)
            target_embeddings[:, :prefix_len] = prefix_embeddings
        prefix_mask = prefix_attention_mask.to(device=device).bool()
        current_indices = prefix_mask.sum(dim=1).to(dtype=torch.long)
        row_indices = torch.arange(batch_size, device=device)
        target_embeddings[row_indices, current_indices] = next_chunk_embeddings[:, 0]
        decoder_attention_mask = torch.zeros(batch_size, total_len, device=device, dtype=torch.bool)
        if prefix_len:
            decoder_attention_mask[:, :prefix_len] = prefix_mask
        decoder_attention_mask[row_indices, current_indices] = True
        return target_embeddings, decoder_attention_mask, current_indices

    def _denoise_next_chunk(
        self,
        context_activations: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        *,
        context_attention_mask: torch.Tensor | None,
        learning: bool,
        num_steps: int | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size = int(context_activations.shape[0])
        x = self.init_logits(batch_size=batch_size, target_len=1, device=context_activations.device)
        alpha = torch.clamp(self.alpha, min=0.0001)
        langevin_std = torch.clamp(self.langevin_dynamics_noise_std, min=0.000001)
        total_steps = max(1, int(self.mcmc_num_steps if num_steps is None else num_steps))
        predicted_distributions: list[torch.Tensor] = []
        selected_energies: list[torch.Tensor] = []

        with torch.set_grad_enabled(True):
            for step in range(total_steps):
                if self.no_mcmc_detach:
                    x.requires_grad_(True)
                else:
                    x = x.detach().requires_grad_()

                if float(self.langevin_dynamics_noise_std.detach()) != 0.0:
                    x = x + torch.randn_like(x.detach()) * langevin_std

                state_already_normalized = False
                x_for_embedding = x
                if self.normalize_initial_condition:
                    if not self.normalize_initial_condition_only_first_step or step == 0:
                        x_for_embedding = torch.softmax(x, dim=-1)
                        state_already_normalized = True

                next_embeddings = self.vocab_state_to_embeddings(
                    x_for_embedding,
                    mcmc_step=step,
                    already_normalized=state_already_normalized,
                )
                target_embeddings, decoder_attention_mask, current_indices = self._target_embeddings_with_next_chunk(
                    prefix_input_ids,
                    prefix_attention_mask,
                    next_embeddings,
                )
                energy, hidden = self.energy_from_target_embeddings(
                    target_embeddings,
                    context_activations,
                    context_attention_mask=context_attention_mask,
                    decoder_attention_mask=decoder_attention_mask,
                )
                energy_by_chunk = energy.reshape(batch_size, int(target_embeddings.shape[1]))
                selected_energy = energy_by_chunk.gather(1, current_indices.view(-1, 1))
                row_indices = torch.arange(batch_size, device=context_activations.device)
                selected_hidden = hidden[row_indices, current_indices].unsqueeze(1)
                predicted_distributions.append(self.predict_from_hidden(selected_hidden))
                selected_energies.append(selected_energy)

                create_graph = bool(learning)
                if self.truncate_mcmc and step != total_steps - 1:
                    create_graph = False
                grad_x = torch.autograd.grad(selected_energy.sum(), x, create_graph=create_graph)[0]
                if self.clamp_futures_grad:
                    min_and_max = self.clamp_futures_grad_max_change / torch.clamp(self.alpha.detach(), min=0.0001)
                    grad_x = torch.clamp(grad_x, min=-min_and_max, max=min_and_max)
                if torch.isnan(grad_x).any() or torch.isinf(grad_x).any():
                    raise ValueError("NaN or Inf gradients detected during next-chunk MCMC.")

                x = x - alpha * grad_x
                if self.absolute_clamp != 0.0:
                    x = torch.clamp(x, min=-self.absolute_clamp, max=self.absolute_clamp)
                if self.sharpen_predicted_distribution != 0.0:
                    x = x / self.sharpen_predicted_distribution
                if self.norm_pred and not (self.norm_pred_not_final_step and step == total_steps - 1):
                    x = self.pred_norm(x)

        return predicted_distributions, selected_energies

    def _energy_next_chunk_for_ids(
        self,
        context_activations: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        next_chunk_ids: torch.Tensor,
        *,
        context_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        next_embeddings = self.ids_to_embeddings(next_chunk_ids.to(device=context_activations.device))
        target_embeddings, decoder_attention_mask, current_indices = self._target_embeddings_with_next_chunk(
            prefix_input_ids,
            prefix_attention_mask,
            next_embeddings,
        )
        energy, _ = self.energy_from_target_embeddings(
            target_embeddings,
            context_activations,
            context_attention_mask=context_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        energy_by_chunk = energy.reshape(int(context_activations.shape[0]), int(target_embeddings.shape[1]))
        return energy_by_chunk.gather(1, current_indices.view(-1, 1)).squeeze(1)

    def _contrastive_rollout_loss(
        self,
        context_activations: torch.Tensor,
        prefix_input_ids: torch.Tensor,
        prefix_attention_mask: torch.Tensor,
        contrastive_target_ids: torch.Tensor,
        contrastive_target_mask: torch.Tensor,
        *,
        context_attention_mask: torch.Tensor | None,
        learning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.contrastive_rollout_enabled or self.contrastive_rollout_coeff == 0.0:
            zero = torch.zeros((), device=context_activations.device, dtype=torch.float32)
            return zero, zero

        valid_future_chunks = contrastive_target_mask.to(device=context_activations.device).bool().any(dim=-1)
        max_offsets = min(self.contrastive_rollout_chunks, int(contrastive_target_ids.shape[1]))
        if max_offsets <= 0 or not bool(valid_future_chunks.any()):
            zero = torch.zeros((), device=context_activations.device, dtype=torch.float32)
            return zero, zero

        positive_prefix_ids = prefix_input_ids.to(device=context_activations.device)
        positive_prefix_mask = prefix_attention_mask.to(device=context_activations.device).bool()
        generated_prefix_ids = positive_prefix_ids.clone()
        generated_prefix_mask = positive_prefix_mask.clone()
        loss_sum = torch.zeros((), device=context_activations.device, dtype=torch.float32)
        positives_lower = torch.zeros((), device=context_activations.device, dtype=torch.float32)
        comparisons = torch.zeros((), device=context_activations.device, dtype=torch.float32)
        rollout_steps = self.contrastive_rollout_steps

        for offset in range(max_offsets):
            active_mask = valid_future_chunks[:, offset]
            negative_logits, _ = self._denoise_next_chunk(
                context_activations,
                generated_prefix_ids,
                generated_prefix_mask,
                context_attention_mask=context_attention_mask,
                learning=learning,
                num_steps=rollout_steps,
            )
            next_generated = negative_logits[-1].argmax(dim=-1).detach()
            gold_next = contrastive_target_ids[:, offset : offset + 1].to(device=context_activations.device)
            paired_prefix_ids = torch.cat(
                [positive_prefix_ids, generated_prefix_ids],
                dim=0,
            )
            paired_prefix_mask = torch.cat(
                [positive_prefix_mask, generated_prefix_mask],
                dim=0,
            )
            paired_next_ids = torch.cat([gold_next, next_generated], dim=0)
            paired_context = torch.cat([context_activations, context_activations], dim=0)
            paired_context_mask = (
                None if context_attention_mask is None else torch.cat([context_attention_mask, context_attention_mask], dim=0)
            )
            paired_energy = self._energy_next_chunk_for_ids(
                paired_context,
                paired_prefix_ids,
                paired_prefix_mask,
                paired_next_ids,
                context_attention_mask=paired_context_mask,
            )
            positive_energy, negative_energy = paired_energy.chunk(2, dim=0)
            active_float = active_mask.to(dtype=torch.float32, device=context_activations.device)
            loss_sum = loss_sum + (F.softplus(positive_energy - negative_energy) * active_float).sum()
            positives_lower = positives_lower + (
                (positive_energy.detach() < negative_energy.detach()).float() * active_float
            ).sum()
            comparisons = comparisons + active_float.sum()

            next_mask_col = torch.zeros(
                contrastive_target_ids.shape[0],
                1,
                device=context_activations.device,
                dtype=torch.bool,
            )
            next_positive_col = gold_next
            next_negative_col = next_generated
            next_mask_col[:, 0] = active_mask
            positive_prefix_ids = torch.cat([positive_prefix_ids, next_positive_col], dim=1)
            positive_prefix_mask = torch.cat([positive_prefix_mask, next_mask_col], dim=1)
            generated_prefix_ids = torch.cat([generated_prefix_ids, next_negative_col], dim=1)
            generated_prefix_mask = torch.cat([generated_prefix_mask, next_mask_col], dim=1)

        loss = loss_sum / comparisons.clamp_min(1.0)
        accuracy = positives_lower / comparisons.clamp_min(1.0)
        return loss, accuracy

    def loss(
        self,
        context_activations: torch.Tensor,
        label_ids: torch.Tensor,
        *,
        prefix_input_ids: torch.Tensor | None = None,
        prefix_attention_mask: torch.Tensor | None = None,
        teacher_target_ids: torch.Tensor | None = None,
        teacher_target_mask: torch.Tensor | None = None,
        contrastive_target_ids: torch.Tensor | None = None,
        contrastive_target_mask: torch.Tensor | None = None,
        decoder_input_ids: torch.Tensor | None = None,
        context_attention_mask: torch.Tensor | None = None,
        label_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        learning: bool = True,
    ) -> dict[str, torch.Tensor]:
        if prefix_input_ids is None or prefix_attention_mask is None:
            raise ValueError("prefix_input_ids and prefix_attention_mask are required for sampled-prefix loss")
        if teacher_target_ids is None or teacher_target_mask is None:
            raise ValueError("teacher_target_ids and teacher_target_mask are required for sampled-prefix loss")
        if contrastive_target_ids is None or contrastive_target_mask is None:
            raise ValueError("contrastive_target_ids and contrastive_target_mask are required for contrastive loss")
        if teacher_target_ids.dim() != 3 or teacher_target_ids.shape[1] != 1:
            raise ValueError(f"Expected teacher target [B, 1, C], got shape {tuple(teacher_target_ids.shape)}")

        predicted_distributions, selected_energies = self._denoise_next_chunk(
            context_activations,
            prefix_input_ids.to(device=context_activations.device),
            prefix_attention_mask.to(device=context_activations.device),
            context_attention_mask=context_attention_mask,
            learning=learning,
        )
        targets = teacher_target_ids.reshape(-1).to(device=context_activations.device)
        ignore_index = -100
        mask = teacher_target_mask.reshape(-1).to(device=targets.device).bool()
        targets = targets.masked_fill(~mask, ignore_index)

        reconstruction_loss = torch.zeros((), device=context_activations.device, dtype=torch.float32)
        total_steps = len(predicted_distributions)
        initial_loss = None
        final_step_loss = None
        initial_energy = None
        final_energy = None
        perplexity = None
        final_token_accuracy = None
        final_exact_accuracy = None
        final_token_correct_by_row = None
        final_token_count_by_row = None
        final_exact_correct_by_row = None
        
        for step, (predicted_distribution, predicted_energy) in enumerate(zip(predicted_distributions, selected_energies)):
            logits = predicted_distribution.reshape(-1, self.vocab_size)
            
            if self.soften_target_prob_dist != 0.0:
                if total_steps <= 1:
                    label_smoothing = 0.0
                else:
                    label_smoothing = ((total_steps - 1) - step) / (total_steps - 1) * self.soften_target_prob_dist
                step_loss = F.cross_entropy(
                    logits,
                    targets,
                    ignore_index=ignore_index,
                    label_smoothing=label_smoothing,
                )
            
            else:
                step_loss = F.nll_loss(
                    F.log_softmax(logits, dim=-1),
                    targets,
                    ignore_index=ignore_index,
                )

            # update reconstruction step
            if self.truncate_mcmc or self.loss_on_final_step_only:
                if step == total_steps - 1:
                    reconstruction_loss = step_loss
            else:
                reconstruction_loss = reconstruction_loss + step_loss

            # some logging
            if step == 0:
                initial_loss = step_loss.detach()
                initial_energy = predicted_energy.squeeze().mean().detach()
            if step == total_steps - 1:
                final_step_loss = step_loss.detach()
                final_energy = predicted_energy.squeeze().mean().detach()
                perplexity = torch.exp(step_loss).detach()
                with torch.no_grad():
                    predictions = predicted_distribution.argmax(dim=-1)
                    target_2d = teacher_target_ids.to(device=predictions.device)
                    valid_mask = teacher_target_mask.to(device=predictions.device).bool()
                    correct = (predictions == target_2d) & valid_mask
                    final_token_accuracy = correct.sum().float() / valid_mask.sum().clamp_min(1).float()
                    final_token_correct_by_row = correct.flatten(1).sum(dim=1).float()
                    final_token_count_by_row = valid_mask.flatten(1).sum(dim=1).float()
                    per_row_correct = (correct | ~valid_mask).flatten(1).all(dim=1)
                    has_any = valid_mask.flatten(1).any(dim=1)
                    final_exact_correct_by_row = (per_row_correct & has_any).float()
                    final_exact_accuracy = final_exact_correct_by_row.mean()

        if not self.truncate_mcmc: # take the average of the reconstruction loss across ALL steps
            reconstruction_loss = reconstruction_loss / max(1, total_steps)
        contrastive_rollout_loss, contrastive_rollout_accuracy = self._contrastive_rollout_loss(
            context_activations,
            prefix_input_ids.to(device=context_activations.device),
            prefix_attention_mask.to(device=context_activations.device),
            contrastive_target_ids.to(device=context_activations.device),
            contrastive_target_mask.to(device=context_activations.device),
            context_attention_mask=context_attention_mask,
            learning=learning,
        )
        loss = (
            self.reconstruction_coeff * reconstruction_loss
            + self.contrastive_rollout_coeff * contrastive_rollout_loss
        )

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "contrastive_rollout_loss": contrastive_rollout_loss.detach(),
            "contrastive_rollout_accuracy": contrastive_rollout_accuracy.detach(),
            "initial_loss": initial_loss if initial_loss is not None else loss.detach(),
            "final_step_loss": final_step_loss if final_step_loss is not None else loss.detach(),
            "initial_final_pred_energies_gap": (
                (initial_energy - final_energy)
                if initial_energy is not None and final_energy is not None
                else torch.zeros((), device=loss.device)
            ),
            "perplexity": perplexity if perplexity is not None else torch.exp(loss.detach()),
            "final_energy": final_energy if final_energy is not None else torch.zeros((), device=loss.device),
            "final_token_accuracy": (
                final_token_accuracy if final_token_accuracy is not None else torch.zeros((), device=loss.device)
            ),
            "final_exact_accuracy": (
                final_exact_accuracy if final_exact_accuracy is not None else torch.zeros((), device=loss.device)
            ),
            "final_token_correct_by_row": (
                final_token_correct_by_row
                if final_token_correct_by_row is not None
                else torch.zeros(teacher_target_ids.shape[0], device=loss.device)
            ),
            "final_token_count_by_row": (
                final_token_count_by_row
                if final_token_count_by_row is not None
                else torch.zeros(teacher_target_ids.shape[0], device=loss.device)
            ),
            "final_exact_correct_by_row": (
                final_exact_correct_by_row
                if final_exact_correct_by_row is not None
                else torch.zeros(teacher_target_ids.shape[0], device=loss.device)
            ),
        }

    def sample(
        self,
        context_activations: torch.Tensor,
        *,
        target_len: int,
        context_attention_mask: torch.Tensor | None = None,
        steps: int | None = None,
        eos_token_id: int | None = None,
    ) -> dict[str, Any]:
        old_steps = self.mcmc_num_steps
        if steps is not None:
            self.mcmc_num_steps = int(steps)
        try:
            out = self.forward(
                context_activations,
                target_len=target_len,
                context_attention_mask=context_attention_mask,
                learning=False,
                return_all_steps=True,
            )
        finally:
            self.mcmc_num_steps = old_steps
        chunk_ids = out.logits.argmax(dim=-1).detach()
        eos_id = int(self.eos_token_id if eos_token_id is None else eos_token_id)
        token_ids: list[torch.Tensor] = []
        for row in chunk_ids:
            flat = row.reshape(-1)
            eos_hits = (flat == eos_id).nonzero(as_tuple=False)
            if eos_hits.numel():
                flat = flat[: int(eos_hits[0].item())]
            token_ids.append(flat.detach())
        return {
            "logits": out.logits.detach(),
            "chunk_token_ids": chunk_ids,
            "token_ids": token_ids,
            "energies": [energy.detach() for energy in out.energies],
        }
