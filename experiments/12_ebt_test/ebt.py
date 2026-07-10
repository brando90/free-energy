#!/usr/bin/env python3
"""Compact Energy-Based Transformer for the synthetic chain task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class EBTOutput:
    logits: torch.Tensor
    predicted_distributions: list[torch.Tensor]
    energies: list[torch.Tensor]
    decoder_hidden: torch.Tensor
    final_state: torch.Tensor


class SyntheticEBT(nn.Module):
    """Small token-level Energy-Based Transformer with MCMC updates."""

    def __init__(
        self,
        vocab_size: int,
        context_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        dim_feedforward: int = 512,
        mcmc_num_steps: int = 2,
        mcmc_step_size: float = 0.75,
        mcmc_step_size_learnable: bool = False,
        max_target_positions: int = 256,
        mcmc_noise: float = 0.0,
        truncate_mcmc: bool = False,
        contrastive_rollout_enabled: bool = False,
        contrastive_rollout_coeff: float = 0.0,
        dropout: float = 0.1,
        alpha_max_norm: float = 1.0,
        no_mcmc_detach: bool = False,
        normalize_initial_condition: bool = True,
        normalize_initial_condition_only_first_step: bool = False,
        clamp_futures_grad: bool = False,
        clamp_futures_grad_max_change: float = 9.0,
        absolute_clamp: float = 0.0,
        sharpen_predicted_distribution: float = 0.0,
        norm_pred: bool = False,
        norm_pred_not_final_step: bool = False,
        reconstruction_coeff: float = 1.0,
        soften_target_prob_dist: float = 0.0,
        loss_on_final_step_only: bool = False,
        train_mcmc_num_steps_min: int = 0,
        train_mcmc_num_steps_max: int = 0,
        randomize_mcmc_step_size_scale: float = 1.0,
        replay_buffer_enabled: bool = False,
        replay_buffer_size: int = 0,
        replay_buffer_sample_fraction: float = 0.0,
        gaussian_energy_loss_enabled: bool = False,
        gaussian_energy_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        self.vocab_size = int(vocab_size)
        self.context_dim = int(context_dim)
        self.hidden_dim = int(context_dim)
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_dim <= 0:
            raise ValueError("context_dim must be positive")
        self.mcmc_num_steps = int(mcmc_num_steps)
        if self.mcmc_num_steps < 1:
            raise ValueError("mcmc_num_steps must be >= 1")
        self.mcmc_noise = float(mcmc_noise)
        self.truncate_mcmc = bool(truncate_mcmc)
        self.max_target_positions = int(max_target_positions)
        self.contrastive_rollout_enabled = bool(contrastive_rollout_enabled)
        self.contrastive_rollout_coeff = float(contrastive_rollout_coeff)
        self.alpha_max_norm = float(alpha_max_norm)
        self.no_mcmc_detach = bool(no_mcmc_detach)
        self.normalize_initial_condition = bool(normalize_initial_condition)
        self.normalize_initial_condition_only_first_step = bool(normalize_initial_condition_only_first_step)
        self.clamp_futures_grad = bool(clamp_futures_grad)
        self.clamp_futures_grad_max_change = float(clamp_futures_grad_max_change)
        self.absolute_clamp = float(absolute_clamp)
        self.sharpen_predicted_distribution = float(sharpen_predicted_distribution)
        self.norm_pred = bool(norm_pred)
        self.norm_pred_not_final_step = bool(norm_pred_not_final_step)
        self.reconstruction_coeff = float(reconstruction_coeff)
        self.soften_target_prob_dist = float(soften_target_prob_dist)
        self.loss_on_final_step_only = bool(loss_on_final_step_only)
        self.train_mcmc_num_steps_min = int(train_mcmc_num_steps_min)
        self.train_mcmc_num_steps_max = int(train_mcmc_num_steps_max)
        self.randomize_mcmc_step_size_scale = float(randomize_mcmc_step_size_scale)
        self.replay_buffer_enabled = bool(replay_buffer_enabled)
        self.replay_buffer_size = max(0, int(replay_buffer_size))
        self.replay_buffer_sample_fraction = float(replay_buffer_sample_fraction)
        self._replay_buffer: list[torch.Tensor] = []
        self.gaussian_energy_loss_enabled = bool(gaussian_energy_loss_enabled)
        self.gaussian_energy_alpha = float(gaussian_energy_alpha)

        self.vocab_to_hidden = nn.Linear(self.vocab_size, self.hidden_dim, bias=False)
        self.context_to_hidden = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.position_embed = nn.Embedding(self.max_target_positions, self.hidden_dim)
        if self.norm_pred:
            self.pred_norm = nn.RMSNorm(self.vocab_size)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.energy_head = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, 1))
        self.output_head = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

        self.alpha = nn.Parameter(
            torch.tensor(float(mcmc_step_size), dtype=torch.float32),
            requires_grad=bool(mcmc_step_size_learnable),
        )

    def _sample_train_steps(self, learning: bool) -> int:
        if not learning or self.train_mcmc_num_steps_max <= 0:
            return max(1, int(self.mcmc_num_steps))
        min_steps = max(1, self.train_mcmc_num_steps_min)
        max_steps = max(min_steps, self.train_mcmc_num_steps_max)
        return int(torch.randint(min_steps, max_steps + 1, ()).item())

    def _sample_alpha(self, batch_size: int, target_len: int, device: torch.device | str, learning: bool) -> torch.Tensor:
        alpha = torch.clamp(self.alpha, max=self.alpha_max_norm, min=1e-4)
        if not learning or self.randomize_mcmc_step_size_scale == 1.0:
            return alpha
        scale = max(float(self.randomize_mcmc_step_size_scale), 1.0)
        low = alpha / scale
        high = alpha * scale
        return low + torch.rand((batch_size, target_len, 1), device=device, dtype=alpha.dtype) * (high - low)

    def _apply_replay_init(self, target_init: torch.Tensor, learning: bool) -> torch.Tensor:
        if (
            not learning
            or not self.replay_buffer_enabled
            or self.replay_buffer_sample_fraction <= 0.0
            or not self._replay_buffer
        ):
            return target_init
        batch_size, target_len, vocab_size = target_init.shape
        candidates = [state for state in self._replay_buffer if tuple(state.shape[-2:]) == (target_len, vocab_size)]
        if not candidates:
            return target_init
        replay_rows = min(batch_size, max(1, int(round(batch_size * self.replay_buffer_sample_fraction))))
        sampled = random.choices(candidates, k=replay_rows)
        replay_init = torch.cat(sampled, dim=0).to(device=target_init.device, dtype=target_init.dtype)
        out = target_init.clone()
        out[-replay_rows:] = replay_init[:replay_rows]
        return out

    def _update_replay_buffer(self, final_state: torch.Tensor) -> None:
        if not self.replay_buffer_enabled or self.replay_buffer_size <= 0:
            return
        for row in final_state.detach().cpu().split(1, dim=0):
            self._replay_buffer.append(row)
        overflow = len(self._replay_buffer) - self.replay_buffer_size
        if overflow > 0:
            del self._replay_buffer[:overflow]

    def init_logits(self, *, batch_size: int, target_len: int, device: torch.device | str) -> torch.Tensor:
        return torch.randn((batch_size, target_len, self.vocab_size), device=device) * (
            1.0 / (self.vocab_size**0.5)
        )

    @staticmethod
    def _causal_mask(length: int, device: torch.device | str) -> torch.Tensor:
        return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)

    def ids_to_vocab_state(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.dim() != 2:
            raise ValueError(f"Expected token ids [B, T], got {tuple(ids.shape)}")
        if ids.shape[-1] == 0:
            return torch.empty((*ids.shape, self.vocab_size), device=ids.device, dtype=torch.float32)
        return torch.nn.functional.one_hot(ids.to(dtype=torch.long), num_classes=self.vocab_size).float()

    def vocab_state_to_embeddings(
        self,
        x: torch.Tensor,
        *,
        mcmc_step: int,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected vocab states [B, T, V], got shape {tuple(x.shape)}")
        if x.shape[-1] != self.vocab_size:
            raise ValueError(f"Expected vocab size {self.vocab_size}, got {x.shape[-1]}")

        if self.normalize_initial_condition and not already_normalized:
            if not self.normalize_initial_condition_only_first_step or mcmc_step == 0:
                x = torch.softmax(x, dim=-1)
                already_normalized = True
        return self.vocab_to_hidden(x)

    def predict_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.output_head(hidden)

    @staticmethod
    def _context_summary(
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        vocab_to_hidden: nn.Module,
        context_to_hidden: nn.Module,
    ) -> torch.Tensor:
        prompt_emb = vocab_to_hidden(torch.nn.functional.one_hot(prompt_ids, num_classes=vocab_to_hidden.in_features).float())
        masked = prompt_emb * prompt_attention_mask.unsqueeze(-1).to(prompt_emb.dtype)
        denom = prompt_attention_mask.sum(dim=1).clamp_min(1).to(prompt_emb.dtype).unsqueeze(-1)
        avg = masked.sum(dim=1) / denom
        return context_to_hidden(avg)

    def energy(
        self,
        x: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_attention_mask: torch.Tensor | None,
        *,
        mcmc_step: int,
        state_already_normalized: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_hidden = self.vocab_state_to_embeddings(
            x,
            mcmc_step=mcmc_step,
            already_normalized=state_already_normalized,
        )

        positions = torch.arange(target_hidden.shape[1], device=target_hidden.device)
        target_hidden = target_hidden + self.position_embed(positions).unsqueeze(0).to(target_hidden.dtype)

        prompt_ctx = self._context_summary(
            prompt_ids,
            prompt_attention_mask,
            vocab_to_hidden=self.vocab_to_hidden,
            context_to_hidden=self.context_to_hidden,
        )
        target_hidden = target_hidden + prompt_ctx[:, None, :]

        tgt_len = int(target_hidden.shape[1])
        mask = self._causal_mask(tgt_len, device=target_hidden.device)
        decoder_hidden = self.decoder(tgt=target_hidden, memory=target_hidden, tgt_mask=mask, tgt_key_padding_mask=None)
        token_energy = self.energy_head(decoder_hidden).squeeze(-1)
        if target_attention_mask is not None:
            token_energy = token_energy.masked_fill(~target_attention_mask.to(dtype=torch.bool), 0.0)
        return token_energy.reshape(-1, 1), decoder_hidden

    def forward(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_len: int,
        target_attention_mask: torch.Tensor | None = None,
        *,
        target_init: torch.Tensor | None = None,
        learning: bool = True,
        return_all_steps: bool = False,
    ) -> EBTOutput:
        if target_init is None:
            x = self.init_logits(
                batch_size=int(prompt_ids.shape[0]),
                target_len=int(target_len),
                device=prompt_ids.device,
            )
        else:
            x = target_init

        if x.dim() != 3:
            raise ValueError(f"target_init/initial logits must be [B, T, V], got {tuple(x.shape)}")
        if x.shape[-1] != self.vocab_size:
            raise ValueError(f"Expected vocab size {self.vocab_size}, got {x.shape[-1]}")

        batch_size = int(x.shape[0])
        target_len = int(x.shape[1])
        alpha = self._sample_alpha(batch_size, target_len, prompt_ids.device, learning)
        total_steps = self._sample_train_steps(learning)
        noise = float(self.mcmc_noise)

        predicted_distributions: list[torch.Tensor] = []
        energies: list[torch.Tensor] = []
        decoder_hidden = None

        with torch.set_grad_enabled(True):
            for step in range(total_steps):
                if self.no_mcmc_detach:
                    x.requires_grad_(True)
                else:
                    x = x.detach().requires_grad_()

                if noise != 0.0:
                    x = x + torch.randn_like(x.detach()) * noise

                step_energy, decoder_hidden = self.energy(
                    x,
                    prompt_ids=prompt_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    target_attention_mask=target_attention_mask,
                    mcmc_step=step,
                )
                logits = self.predict_from_hidden(decoder_hidden)
                predicted_distributions.append(logits)
                energies.append(step_energy)

                create_graph = bool(learning)
                if self.truncate_mcmc and step != total_steps - 1:
                    create_graph = False
                grad_x = torch.autograd.grad(step_energy.sum(), x, create_graph=create_graph, retain_graph=create_graph)[0]
                if grad_x is None:
                    raise RuntimeError("EBT energy step produced no gradient")
                if self.clamp_futures_grad:
                    max_change = self.clamp_futures_grad_max_change / torch.clamp(alpha, min=1e-4)
                    grad_x = torch.clamp(grad_x, min=-max_change, max=max_change)
                if torch.isnan(grad_x).any() or torch.isinf(grad_x).any():
                    raise ValueError("NaN or Inf gradients detected during MCMC.")

                x = x - alpha * grad_x
                if self.absolute_clamp != 0.0:
                    x = torch.clamp(x, min=-self.absolute_clamp, max=self.absolute_clamp)
                if self.sharpen_predicted_distribution != 0.0:
                    x = x / self.sharpen_predicted_distribution
                if self.norm_pred and not (self.norm_pred_not_final_step and step == total_steps - 1):
                    x = self.pred_norm(x)

        selected_distributions = predicted_distributions if return_all_steps else predicted_distributions[-1:]
        selected_energies = energies if return_all_steps else energies[-1:]
        if decoder_hidden is None:
            raise RuntimeError("EBT forward produced no decoder hidden states")
        return EBTOutput(
            logits=selected_distributions[-1],
            predicted_distributions=selected_distributions,
            energies=selected_energies,
            decoder_hidden=decoder_hidden,
            final_state=x,
        )

    def _contrastive_rollout_loss(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        learning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.contrastive_rollout_enabled or self.contrastive_rollout_coeff == 0.0:
            zero = torch.zeros((), device=target_ids.device, dtype=torch.float32)
            return zero, zero

        if not bool(target_mask.any()):
            zero = torch.zeros((), device=target_ids.device, dtype=torch.float32)
            return zero, zero

        batch_size = int(target_ids.shape[0])
        target_len = int(target_ids.shape[1])
        if target_len <= 1:
            zero = torch.zeros((), device=target_ids.device, dtype=torch.float32)
            return zero, zero

        perm = torch.randperm(target_len, device=target_ids.device)
        negative_ids = target_ids[:, perm]
        negative_mask = target_mask[:, perm]

        positive_energy, _ = self.energy(
            self.ids_to_vocab_state(target_ids),
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            target_attention_mask=target_mask,
            mcmc_step=0,
            state_already_normalized=True,
        )
        negative_energy, _ = self.energy(
            self.ids_to_vocab_state(negative_ids),
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            target_attention_mask=negative_mask,
            mcmc_step=0,
            state_already_normalized=True,
        )

        positive_sum = positive_energy.reshape(batch_size, target_len).sum(dim=1)
        negative_sum = negative_energy.reshape(batch_size, target_len).sum(dim=1)
        active = target_mask.any(dim=1).to(dtype=torch.float32)
        loss = F.softplus(positive_sum - negative_sum)
        loss = (loss * active).sum() / active.sum().clamp_min(1.0)
        accuracy = ((positive_sum < negative_sum).to(dtype=torch.float32) * active).sum() / active.sum().clamp_min(1.0)
        return loss, accuracy

    def loss(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        contrastive_rollout_enabled: bool | None = None,
        contrastive_rollout_coeff: float = 0.0,
        learning: bool = True,
        target_init: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.gaussian_energy_loss_enabled:
            batch_size = int(target_ids.shape[0])
            target_len = int(target_ids.shape[1])
            target_state = self.ids_to_vocab_state(target_ids)
            eps = torch.randn_like(target_state)
            perturbed_target = target_state + self.gaussian_energy_alpha * eps
            energy, _ = self.energy(
                perturbed_target,
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                target_attention_mask=target_mask,
                mcmc_step=0,
                state_already_normalized=True,
            )
            energy_by_row = energy.reshape(batch_size, target_len).sum(dim=1)
            eps_norm_sq = eps.reshape(batch_size, -1).square().sum(dim=1)
            weights = torch.exp(-eps_norm_sq)
            active = target_mask.flatten(1).any(dim=1).to(dtype=energy_by_row.dtype)
            loss = (weights * energy_by_row * active).sum() / active.sum().clamp_min(1.0)
            zero = torch.zeros((), device=target_ids.device, dtype=torch.float32)
            return {
                "loss": loss,
                "reconstruction_loss": loss.detach(),
                "contrastive_rollout_loss": zero,
                "contrastive_rollout_accuracy": zero,
                "initial_loss": loss.detach(),
                "final_step_loss": loss.detach(),
                "initial_final_pred_energies_gap": zero,
                "perplexity": torch.exp(loss.detach().clamp(max=20.0)),
                "final_energy": energy_by_row.detach().mean(),
                "final_token_accuracy": zero,
                "final_exact_accuracy": zero,
                "final_token_correct_by_row": torch.zeros(batch_size, device=target_ids.device),
                "final_token_count_by_row": target_mask.flatten(1).sum(dim=1).float(),
                "final_exact_correct_by_row": torch.zeros(batch_size, device=target_ids.device),
            }

        if target_init is None:
            target_init = self.init_logits(
                batch_size=int(prompt_ids.shape[0]),
                target_len=int(target_ids.shape[1]),
                device=prompt_ids.device,
            )
        target_init = self._apply_replay_init(target_init, learning=learning)
        out = self.forward(
            prompt_ids=prompt_ids,
            prompt_attention_mask=prompt_attention_mask,
            target_len=int(target_ids.shape[1]),
            target_attention_mask=target_mask,
            target_init=target_init,
            learning=learning,
        )
        if learning:
            self._update_replay_buffer(out.final_state)

        flat_targets = target_ids.reshape(-1)
        flat_mask = target_mask.reshape(-1).to(dtype=torch.bool)
        ignore_index = -100
        flat_targets = torch.where(flat_mask, flat_targets, torch.full_like(flat_targets, ignore_index))
        targets = flat_targets
        total_steps = len(out.predicted_distributions)

        reconstruction_loss = torch.tensor(0.0, device=target_ids.device, dtype=torch.float32)
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

        for step, (predicted_distribution, predicted_energy) in enumerate(
            zip(out.predicted_distributions, out.energies)
        ):
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

            if self.truncate_mcmc or self.loss_on_final_step_only:
                if step == total_steps - 1:
                    reconstruction_loss = step_loss
            else:
                reconstruction_loss = reconstruction_loss + step_loss

            if step == 0:
                initial_loss = step_loss.detach()
                initial_energy = predicted_energy.mean().detach()
            if step == total_steps - 1:
                final_step_loss = step_loss.detach()
                final_energy = predicted_energy.mean().detach()
                perplexity = torch.exp(step_loss.detach())
                with torch.no_grad():
                    predictions = predicted_distribution.argmax(dim=-1)
                    valid_mask = target_mask.bool()
                    correct = (predictions == target_ids) & valid_mask
                    token_count = valid_mask.sum().clamp_min(1).to(dtype=torch.float32)
                    final_token_accuracy = correct.sum().to(dtype=torch.float32) / token_count
                    final_token_correct_by_row = correct.flatten(1).sum(dim=1).float()
                    final_token_count_by_row = valid_mask.flatten(1).sum(dim=1).float()
                    exact = (correct | ~valid_mask).flatten(1).all(dim=1)
                    has_any = valid_mask.flatten(1).any(dim=1)
                    final_exact_correct_by_row = (exact & has_any).float()
                    final_exact_accuracy = final_exact_correct_by_row.mean()

        if not self.truncate_mcmc and not self.loss_on_final_step_only:
            reconstruction_loss = reconstruction_loss / max(1, total_steps)

        if contrastive_rollout_enabled is not None:
            contrastive_rollout_enabled = bool(contrastive_rollout_enabled)
        else:
            contrastive_rollout_enabled = self.contrastive_rollout_enabled
        if contrastive_rollout_coeff == 0.0:
            contrastive_rollout_coeff = self.contrastive_rollout_coeff
        run_contrastive = bool(contrastive_rollout_enabled) and contrastive_rollout_coeff > 0.0

        if run_contrastive:
            contrastive_loss, contrastive_accuracy = self._contrastive_rollout_loss(
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                target_ids=target_ids,
                target_mask=target_mask,
                learning=learning,
            )
            loss = self.reconstruction_coeff * reconstruction_loss + contrastive_rollout_coeff * contrastive_loss
        else:
            contrastive_loss = torch.zeros((), device=target_ids.device, dtype=torch.float32)
            contrastive_accuracy = torch.zeros((), device=target_ids.device, dtype=torch.float32)
            loss = self.reconstruction_coeff * reconstruction_loss

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss.detach(),
            "contrastive_rollout_loss": contrastive_loss.detach(),
            "contrastive_rollout_accuracy": contrastive_accuracy.detach(),
            "initial_loss": initial_loss if initial_loss is not None else loss.detach(),
            "final_step_loss": final_step_loss if final_step_loss is not None else loss.detach(),
            "initial_final_pred_energies_gap": (
                initial_energy - final_energy
                if initial_energy is not None and final_energy is not None
                else torch.zeros((), device=loss.device)
            ),
            "perplexity": perplexity if perplexity is not None else torch.exp(reconstruction_loss.detach()),
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
                else torch.zeros(target_ids.shape[0], device=loss.device)
            ),
            "final_token_count_by_row": (
                final_token_count_by_row
                if final_token_count_by_row is not None
                else torch.zeros(target_ids.shape[0], device=loss.device)
            ),
            "final_exact_correct_by_row": (
                final_exact_correct_by_row
                if final_exact_correct_by_row is not None
                else torch.zeros(target_ids.shape[0], device=loss.device)
            ),
        }

    def sample(
        self,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        max_len: int,
        target_attention_mask: torch.Tensor | None = None,
        *,
        steps: int | None = None,
        best_of_n: int = 1,
    ) -> dict[str, Any]:
        old_steps = self.mcmc_num_steps
        if steps is not None:
            self.mcmc_num_steps = int(steps)
        try:
            if best_of_n > 1:
                batch_size = int(prompt_ids.shape[0])
                prompt_ids = prompt_ids.repeat_interleave(best_of_n, dim=0)
                prompt_attention_mask = prompt_attention_mask.repeat_interleave(best_of_n, dim=0)
                if target_attention_mask is not None:
                    target_attention_mask = target_attention_mask.repeat_interleave(best_of_n, dim=0)
            out = self.forward(
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                target_len=max_len,
                target_attention_mask=target_attention_mask,
                learning=False,
                return_all_steps=True,
            )
        finally:
            self.mcmc_num_steps = old_steps

        token_ids = out.logits.argmax(dim=-1).to(dtype=torch.long)
        if best_of_n > 1:
            final_energy = out.energies[-1].reshape(batch_size, best_of_n, -1).sum(dim=-1)
            best_idx = final_energy.argmin(dim=1)
            row_idx = torch.arange(batch_size, device=token_ids.device)
            token_ids = token_ids.reshape(batch_size, best_of_n, max_len)[row_idx, best_idx]
            logits = out.logits.reshape(batch_size, best_of_n, max_len, self.vocab_size)[row_idx, best_idx]
            energies = [energy.reshape(batch_size, best_of_n, -1)[row_idx, best_idx].detach() for energy in out.energies]
        else:
            logits = out.logits.detach()
            energies = [energy.detach() for energy in out.energies]
        return {
            "token_ids": token_ids,
            "logits": logits,
            "energies": energies,
        }
