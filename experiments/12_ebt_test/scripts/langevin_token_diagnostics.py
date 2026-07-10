#!/usr/bin/env python3
"""Token-level Langevin diagnostics for the old compact synthetic-chain EBT run."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg


REPO_ROOT = Path(__file__).resolve().parents[1]
EBT_ROOT = REPO_ROOT / "EBT-main"
DEFAULT_CKPT = (
    EBT_ROOT
    / "logs"
    / "checkpoints"
    / "synthetic_chain_ebt_mponens_mcmc4_scratch_slashmetrics_v1_2026-07-08_13-29-33_"
    / "epoch=epoch=4-step=step=705-val_autoregressive_exact_accuracy.ckpt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "langevin_token_diagnostics_938"

OLD_TOKENS = (
    "<pad>",
    "<eos>",
    "<bos>",
    "<unk>",
    *tuple(chr(ord("A") + i) for i in range(26)),
    "->",
    "|",
    "fact:",
    "rule:",
    "query:",
)


@dataclass(frozen=True)
class OldSyntheticChainTokenizer:
    tokens: tuple[str, ...] = OLD_TOKENS

    @classmethod
    def default(cls) -> "OldSyntheticChainTokenizer":
        return cls()

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    def encode(self, tokens: list[str] | tuple[str, ...]) -> list[int]:
        return [self.token_to_id.get(str(token), self.unk_token_id) for token in tokens]

    def decode(self, token_ids: list[int] | tuple[int, ...]) -> list[str]:
        return [self.tokens[int(token_id)] for token_id in token_ids]


LETTERS = tuple(chr(ord("A") + i) for i in range(26))


def build_rule_set(rng: random.Random, depth: int) -> tuple[list[str], list[dict[str, str]]]:
    start = rng.choice(LETTERS)
    chain = [start]
    while len(chain) < depth + 1:
        candidate = rng.choice(LETTERS)
        if candidate != chain[-1] and candidate not in chain:
            chain.append(candidate)
    rules = [
        {"lhs": chain[i], "rhs": chain[i + 1], "text": f"{chain[i]} -> {chain[i + 1]}"}
        for i in range(len(chain) - 1)
    ]
    return chain, rules


def make_chain_example(
    rng: random.Random,
    row_id: int,
    split: str,
    min_depth: int,
    max_depth: int,
    include_distractors: bool,
) -> dict[str, Any]:
    depth = rng.randint(min_depth, max_depth)
    chain, rules = build_rule_set(rng, depth)
    all_rules = list(rules)
    if include_distractors:
        for _ in range(rng.randint(0, 2)):
            lhs, rhs = rng.sample(LETTERS, 2)
            if {"lhs": lhs, "rhs": rhs, "text": f"{lhs} -> {rhs}"} not in all_rules:
                all_rules.append({"lhs": lhs, "rhs": rhs, "text": f"{lhs} -> {rhs}"})
    rng.shuffle(all_rules)

    prompt_tokens = ["fact:", chain[0]]
    for rule in all_rules:
        prompt_tokens.extend(["|", "rule:", rule["lhs"], "->", rule["rhs"]])

    target_tokens: list[str] = []
    for i, token in enumerate(chain):
        if i:
            target_tokens.append("->")
        target_tokens.append(token)

    return {
        "id": row_id,
        "split": split,
        "depth": depth,
        "start": chain[0],
        "query": chain[-1],
        "facts_rules": all_rules,
        "prompt_tokens": prompt_tokens,
        "path_tokens": list(chain),
        "target_tokens": target_tokens,
    }


def generate_old_records(
    *,
    num_samples: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    min_depth: int,
    max_depth: int,
    include_distractors: bool,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)
    records = []
    for row_id in range(num_samples):
        if row_id < train_end:
            split = "train"
        elif row_id < val_end:
            split = "val"
        else:
            split = "test"
        records.append(
            make_chain_example(
                rng=rng,
                row_id=row_id,
                split=split,
                min_depth=min_depth,
                max_depth=max_depth,
                include_distractors=include_distractors,
            )
        )
    return records


def patch_old_tokenizer() -> None:
    if str(EBT_ROOT) not in sys.path:
        sys.path.insert(0, str(EBT_ROOT))
    import data.nlp.synthetic_chain_dataset as synthetic_chain_dataset

    synthetic_chain_dataset.SyntheticChainTokenizer = OldSyntheticChainTokenizer


def load_ebt_model(checkpoint_path: Path, device: torch.device):
    patch_old_tokenizer()
    from model.nlp.ebt import EBT_NLP

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = dict(checkpoint["hyper_parameters"])
    hparams["compile_model"] = False
    model = EBT_NLP(hparams)
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, hparams


def set_deterministic_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def autoregressive_generate(
    model,
    tokenizer: OldSyntheticChainTokenizer,
    prompt_ids: list[int],
    target_len: int,
    *,
    seed: int,
    device: torch.device,
) -> list[int]:
    decoded = torch.tensor([prompt_ids + [tokenizer.bos_token_id]], dtype=torch.long, device=device)
    generated: list[int] = []
    for step in range(target_len):
        set_deterministic_seed(seed + step, device)
        with torch.enable_grad():
            pred_list, _ = model(
                decoded,
                return_raw_logits=True,
                learning=False,
                no_randomness=True,
            )
        logits = pred_list[-1]
        gather_pos = len(prompt_ids) + step
        next_token = int(torch.argmax(logits[0, gather_pos], dim=-1).item())
        generated.append(next_token)
        next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        decoded = torch.cat([decoded, next_tensor], dim=1)
    return generated


def effective_alpha(model) -> torch.Tensor:
    return torch.clamp(model.alpha, min=0.0001)


def state_to_embeddings(model, state: torch.Tensor, *, assume_prob_dist: bool) -> torch.Tensor:
    if assume_prob_dist:
        probs = state
    elif model.hparams.normalize_initial_condition:
        if model.hparams.normalize_initial_condition_only_first_step:
            probs = model.softmax(state)
        else:
            probs = model.softmax(state)
    else:
        return model.vocab_to_embed(state)

    if model.hparams.vocab_to_embed_uses_prob_dist:
        return torch.matmul(probs, model.embeddings.weight)
    return model.vocab_to_embed(probs)


def energy_for_state(
    model,
    input_ids: torch.Tensor,
    state: torch.Tensor,
    *,
    mcmc_step: int,
    focus_pos: int,
    assume_prob_dist: bool = False,
) -> torch.Tensor:
    real_embeddings = model.embeddings(input_ids)
    pred_embeddings = state_to_embeddings(model, state, assume_prob_dist=assume_prob_dist)
    all_embeddings = torch.cat((real_embeddings, pred_embeddings), dim=1)
    energies = model.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step)
    return energies.reshape(input_ids.shape[0], input_ids.shape[1])[0, focus_pos]


def trace_langevin_token(
    model,
    tokenizer: OldSyntheticChainTokenizer,
    prompt_ids: list[int],
    generated_prefix: list[int],
    focus_token_id: int,
    *,
    focus_decode_step: int,
    seed: int,
    device: torch.device,
    interp_points: int,
) -> dict[str, Any]:
    input_list = prompt_ids + [tokenizer.bos_token_id] + generated_prefix
    input_ids = torch.tensor([input_list], dtype=torch.long, device=device)
    focus_pos = len(prompt_ids) + focus_decode_step
    real_embeddings = model.embeddings(input_ids)
    set_deterministic_seed(seed + focus_decode_step, device)
    predicted_tokens = model.corrupt_embeddings(real_embeddings)

    alpha = effective_alpha(model)
    grad_vectors: list[torch.Tensor] = []
    grad_norms: list[float] = []
    centered_grad_norms: list[float] = []
    grad_constant_fractions: list[float] = []
    prob_delta_norms: list[float] = []
    target_prob_by_state: list[float] = []
    max_prob_by_state: list[float] = []
    argmax_id_by_state: list[int] = []
    step_energies: list[float] = []
    states: list[torch.Tensor] = [predicted_tokens.detach().cpu()]

    for mcmc_step in range(int(model.hparams.mcmc_num_steps)):
        cur = predicted_tokens.detach().requires_grad_().reshape(1, input_ids.shape[1], model.vocab_size)
        if model.hparams.langevin_dynamics_noise != 0:
            noise_std = torch.clamp(model.langevin_dynamics_noise_std, min=0.000001)
            cur = cur + torch.randn_like(cur.detach()) * noise_std

        if model.hparams.normalize_initial_condition:
            if model.hparams.normalize_initial_condition_only_first_step:
                if mcmc_step == 0:
                    cur = model.softmax(cur)
            else:
                cur = model.softmax(cur)

        pred_embeddings = state_to_embeddings(model, cur, assume_prob_dist=bool(model.hparams.normalize_initial_condition))
        all_embeddings = torch.cat((real_embeddings, pred_embeddings), dim=1)
        energy_preds = model.transformer(all_embeddings, start_pos=0, mcmc_step=mcmc_step).reshape(-1, 1)
        grad = torch.autograd.grad([energy_preds.sum()], [cur], create_graph=False)[0]

        if model.hparams.clamp_futures_grad:
            min_and_max = model.hparams.clamp_futures_grad_max_change / model.alpha
            grad = torch.clamp(grad, min=-min_and_max, max=min_and_max)

        if model.hparams.scale_alpha_with_energy:
            exponentiated = torch.exp(
                energy_preds.detach().reshape(1, input_ids.shape[1], 1)
                / model.hparams.scale_alpha_with_energy_temp
            )
            predicted_tokens = cur - alpha * exponentiated * grad
        else:
            predicted_tokens = cur - alpha * grad

        if model.hparams.absolute_clamp != 0.0:
            predicted_tokens = torch.clamp(
                predicted_tokens,
                min=-model.hparams.absolute_clamp,
                max=model.hparams.absolute_clamp,
            )
        if model.hparams.sharpen_predicted_distribution != 0.0:
            predicted_tokens = predicted_tokens / model.hparams.sharpen_predicted_distribution
        if model.hparams.norm_pred and not (
            model.hparams.norm_pred_not_final_step and mcmc_step == int(model.hparams.mcmc_num_steps) - 1
        ):
            predicted_tokens = model.pred_norm(predicted_tokens)

        focus_grad = grad[0, focus_pos].detach().cpu()
        centered_grad = focus_grad - focus_grad.mean()
        grad_norm = torch.linalg.vector_norm(focus_grad).item()
        centered_norm = torch.linalg.vector_norm(centered_grad).item()
        grad_vectors.append(focus_grad)
        grad_norms.append(float(grad_norm))
        centered_grad_norms.append(float(centered_norm))
        if grad_norm > 0:
            constant_norm = math.sqrt(float(focus_grad.numel())) * abs(float(focus_grad.mean().item()))
            grad_constant_fractions.append(float(constant_norm / grad_norm))
        else:
            grad_constant_fractions.append(0.0)
        step_energies.append(float(energy_preds.reshape(1, input_ids.shape[1])[0, focus_pos].detach().cpu().item()))
        states.append(predicted_tokens.detach().cpu())

    ce_losses = [
        float(F.cross_entropy(state[0, focus_pos].unsqueeze(0), torch.tensor([focus_token_id])).item())
        for state in states
    ]
    token_prob_states = [
        torch.softmax(state[0, focus_pos], dim=-1).detach().cpu().tolist()
        for state in states
    ]
    for probs in token_prob_states:
        probs_tensor = torch.tensor(probs)
        target_prob_by_state.append(float(probs_tensor[focus_token_id].item()))
        max_prob_by_state.append(float(probs_tensor.max().item()))
        argmax_id_by_state.append(int(probs_tensor.argmax().item()))
    for before, after in zip(token_prob_states, token_prob_states[1:]):
        prob_delta_norms.append(float(torch.linalg.vector_norm(torch.tensor(after) - torch.tensor(before)).item()))

    grad_matrix = torch.stack(grad_vectors, dim=0)
    grad_unit = F.normalize(grad_matrix, p=2, dim=-1, eps=1e-12)
    cosine = torch.matmul(grad_unit, grad_unit.T).numpy()

    initial_prob = torch.softmax(states[0][0, focus_pos], dim=-1).to(device)
    final_prob = torch.softmax(states[-1][0, focus_pos], dim=-1).to(device)
    target_prob = torch.zeros_like(final_prob)
    target_prob[focus_token_id] = 1.0
    base_prob = torch.softmax(states[-1], dim=-1).to(device)
    final_landscape = int(model.hparams.mcmc_num_steps) - 1

    init_to_final_energy: list[float] = []
    init_to_target_energy: list[float] = []
    ts = torch.linspace(0.0, 1.0, interp_points, device=device)
    with torch.no_grad():
        for t in ts:
            state = base_prob.clone()
            state[0, focus_pos] = (1.0 - t) * initial_prob + t * final_prob
            init_to_final_energy.append(
                float(
                    energy_for_state(
                        model,
                        input_ids,
                        state,
                        mcmc_step=final_landscape,
                        focus_pos=focus_pos,
                        assume_prob_dist=True,
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            )
            state = base_prob.clone()
            state[0, focus_pos] = (1.0 - t) * initial_prob + t * target_prob
            init_to_target_energy.append(
                float(
                    energy_for_state(
                        model,
                        input_ids,
                        state,
                        mcmc_step=final_landscape,
                        focus_pos=focus_pos,
                        assume_prob_dist=True,
                    )
                    .detach()
                    .cpu()
                    .item()
                )
            )

    final_logits = states[-1][0, focus_pos]
    final_pred_id = int(torch.argmax(final_logits).item())
    return {
        "focus_pos": focus_pos,
        "input_tokens": tokenizer.decode(input_list),
        "focus_target_id": focus_token_id,
        "focus_target_token": tokenizer.tokens[focus_token_id],
        "focus_final_pred_id": final_pred_id,
        "focus_final_pred_token": tokenizer.tokens[final_pred_id],
        "grad_norms": grad_norms,
        "centered_grad_norms": centered_grad_norms,
        "grad_constant_fractions": grad_constant_fractions,
        "prob_delta_norms": prob_delta_norms,
        "target_prob_by_state": target_prob_by_state,
        "max_prob_by_state": max_prob_by_state,
        "argmax_id_by_state": argmax_id_by_state,
        "argmax_token_by_state": [tokenizer.tokens[token_id] for token_id in argmax_id_by_state],
        "grad_cosine": cosine.tolist(),
        "ce_losses": ce_losses,
        "token_prob_states": token_prob_states,
        "step_energies": step_energies,
        "interp_t": ts.detach().cpu().tolist(),
        "init_to_final_energy": init_to_final_energy,
        "init_to_target_energy": init_to_target_energy,
    }


def first_mismatch(generated: list[int], target: list[int]) -> int | None:
    for idx, (pred, gold) in enumerate(zip(generated, target)):
        if pred != gold:
            return idx
    if len(generated) != len(target):
        return min(len(generated), len(target))
    return None


def compact_text(tokens: list[str], max_chars: int = 120) -> str:
    text = " ".join(tokens)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def render_frame(
    example: dict[str, Any],
    trace: dict[str, Any],
    title_prefix: str,
    *,
    state_index: int,
) -> np.ndarray:
    fig = plt.figure(figsize=(15, 9), dpi=128)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    ax_grad = fig.add_subplot(grid[0, 0])
    ax_cos = fig.add_subplot(grid[0, 1])
    ax_ce = fig.add_subplot(grid[0, 2])
    ax_interp = fig.add_subplot(grid[1, 0:2])
    ax_hist = fig.add_subplot(grid[1, 2])

    steps = np.arange(len(trace["grad_norms"]))
    ax_grad.plot(steps, trace["grad_norms"], marker="o", linewidth=2, label="raw")
    if "centered_grad_norms" in trace:
        ax_grad.plot(
            steps,
            trace["centered_grad_norms"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="centered",
        )
    if state_index > 0:
        ax_grad.axvline(state_index - 1, color="black", linestyle="--", alpha=0.45)
    ax_grad.set_title("Gradient Magnitude")
    ax_grad.set_xlabel("Langevin step")
    ax_grad.set_ylabel("L2 norm")
    ax_grad.grid(alpha=0.25)
    ax_grad.legend(fontsize=8)

    cos = np.asarray(trace["grad_cosine"])
    im = ax_cos.imshow(cos, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax_cos.set_title("Gradient Cosine Similarity")
    ax_cos.set_xlabel("Step")
    ax_cos.set_ylabel("Step")
    fig.colorbar(im, ax=ax_cos, fraction=0.046, pad=0.04)

    ce_steps = np.arange(len(trace["ce_losses"]))
    ax_ce.plot(ce_steps, trace["ce_losses"], marker="o", linewidth=2, color="#b15f00")
    ax_ce.axvline(state_index, color="black", linestyle="--", alpha=0.45)
    ax_ce.set_title("Target CE During Langevin")
    ax_ce.set_xlabel("State index")
    ax_ce.set_ylabel("CE loss")
    ax_ce.grid(alpha=0.25)

    interp_t = np.asarray(trace["interp_t"])
    ax_interp.plot(
        interp_t,
        trace["init_to_final_energy"],
        marker=".",
        linewidth=2,
        color="#0b6e4f",
        label="initial -> final",
    )
    ax_interp.plot(
        interp_t,
        trace["init_to_target_energy"],
        marker=".",
        linewidth=2,
        color="#5b4b8a",
        label="initial -> target one-hot",
    )
    ax_interp.set_title("Energy Along Interpolations")
    ax_interp.set_xlabel("Interpolation t")
    ax_interp.set_ylabel("Energy")
    ax_interp.grid(alpha=0.25)
    ax_interp.legend(fontsize=9)

    probs = np.asarray(trace["token_prob_states"][state_index], dtype=np.float32)
    token_labels = OLD_TOKENS
    colors = ["#6f7f8f"] * len(probs)
    target_id = int(example["focus_target_id"])
    pred_id = int(np.argmax(probs))
    colors[target_id] = "#c43c39"
    colors[pred_id] = "#2f6fbb" if pred_id != target_id else "#7b4ab2"
    ax_hist.bar(np.arange(len(probs)), probs, color=colors, width=0.85)
    timestamp_label = "t=0 initial" if state_index == 0 else f"after step {state_index - 1}"
    if state_index == 0 or "prob_delta_norms" not in trace:
        delta_text = ""
    else:
        delta_text = f", prev delta={trace['prob_delta_norms'][state_index - 1]:.2g}"
    ax_hist.set_title(f"Token Probabilities ({timestamp_label}{delta_text})")
    ax_hist.set_xlabel("Token")
    ax_hist.set_ylabel("softmax probability")
    ax_hist.set_xticks(np.arange(len(probs)))
    ax_hist.set_xticklabels(token_labels, rotation=90, fontsize=7)
    ax_hist.set_ylim(0.0, max(0.05, float(probs.max()) * 1.15))
    ax_hist.grid(axis="y", alpha=0.2)

    fig.suptitle(
        (
            f"{title_prefix} example {example['selection_index'] + 1} | "
            f"row_id={example['id']} depth={example['depth']} | "
            f"focus token {example['focus_decode_step']} target={example['focus_target_token']} "
            f"pred={token_labels[pred_id]} | {timestamp_label}"
        ),
        fontsize=12,
    )
    fig.text(
        0.01,
        0.01,
        (
            f"prompt: {compact_text(example['prompt_tokens'], 95)}    "
            f"target: {compact_text(example['target_tokens'], 75)}    "
            f"generated: {compact_text(example['generated_tokens'], 75)}"
        ),
        ha="left",
        va="bottom",
        fontsize=8,
        family="monospace",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    canvas.draw()
    image = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return image


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        for frame in frames:
            writer.append_data(frame)


def select_examples(args: argparse.Namespace, model, tokenizer: OldSyntheticChainTokenizer, device: torch.device):
    records = generate_old_records(
        num_samples=args.num_samples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        include_distractors=not args.no_distractors,
    )
    candidates = [record for record in records if args.split == "all" or record["split"] == args.split]
    if args.max_scan is not None:
        candidates = candidates[: args.max_scan]

    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for scan_index, record in enumerate(candidates):
        prompt_ids = tokenizer.encode(record["prompt_tokens"])
        target_ids = tokenizer.encode(record["target_tokens"])
        gen_seed = args.seed + int(record["id"]) * 1009
        generated_ids = autoregressive_generate(
            model,
            tokenizer,
            prompt_ids,
            len(target_ids),
            seed=gen_seed,
            device=device,
        )
        mismatch = first_mismatch(generated_ids, target_ids)
        exact = mismatch is None
        if exact and len(good) < args.num_good:
            focus_step = min(args.good_focus_index, len(target_ids) - 1)
            selected = "good"
            selection_index = len(good)
        elif (not exact) and len(bad) < args.num_bad:
            focus_step = int(mismatch)
            if focus_step >= len(target_ids):
                focus_step = len(target_ids) - 1
            selected = "bad"
            selection_index = len(bad)
        else:
            if len(good) >= args.num_good and len(bad) >= args.num_bad:
                break
            continue

        out = {
            **record,
            "scan_index": scan_index,
            "selection": selected,
            "selection_index": selection_index,
            "exact_match": exact,
            "first_error_index": mismatch,
            "prompt_ids": prompt_ids,
            "target_ids": target_ids,
            "generated_ids": generated_ids,
            "generated_tokens": tokenizer.decode(generated_ids),
            "focus_decode_step": focus_step,
            "focus_target_id": target_ids[focus_step],
            "focus_target_token": tokenizer.tokens[target_ids[focus_step]],
            "generation_seed": gen_seed,
        }
        if selected == "good":
            good.append(out)
        else:
            bad.append(out)

        print(
            f"selected {selected} {selection_index + 1}: "
            f"row_id={record['id']} exact={exact} focus={focus_step}",
            flush=True,
        )
        if len(good) >= args.num_good and len(bad) >= args.num_bad:
            break

    if len(good) < args.num_good or len(bad) < args.num_bad:
        raise RuntimeError(
            f"Only found {len(good)} good and {len(bad)} bad examples "
            f"from {len(candidates)} scanned candidates."
        )
    return good, bad


def load_existing_examples(output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        for row_idx, row in enumerate(rows):
            row["selection_index"] = row_idx
            row.pop("trace", None)
        return rows

    good_path = output_dir / "selected_good.jsonl"
    bad_path = output_dir / "selected_bad.jsonl"
    if not good_path.exists() or not bad_path.exists():
        raise FileNotFoundError(
            f"Cannot reuse selected examples; missing {good_path} or {bad_path}."
        )
    return load_jsonl(good_path), load_jsonl(bad_path)


def add_traces_and_frames(
    examples: list[dict[str, Any]],
    *,
    title_prefix: str,
    model,
    tokenizer: OldSyntheticChainTokenizer,
    device: torch.device,
    interp_points: int,
    frame_dir: Path,
) -> list[np.ndarray]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("*.png"):
        old_frame.unlink()
    frames: list[np.ndarray] = []
    for example in examples:
        focus_step = int(example["focus_decode_step"])
        generated_prefix = example["generated_ids"][:focus_step]
        trace = trace_langevin_token(
            model,
            tokenizer,
            example["prompt_ids"],
            generated_prefix,
            int(example["focus_target_id"]),
            focus_decode_step=focus_step,
            seed=int(example["generation_seed"]),
            device=device,
            interp_points=interp_points,
        )
        example["trace"] = trace
        for state_index in range(len(trace["token_prob_states"])):
            frame = render_frame(example, trace, title_prefix, state_index=state_index)
            frames.append(frame)
            frame_path = (
                frame_dir
                / f"{example['selection_index']:02d}_row_{example['id']}_state_{state_index:02d}.png"
            )
            imageio.imwrite(frame_path, frame)
        print(
            f"rendered {title_prefix.lower()} example {example['selection_index'] + 1} "
            f"({len(trace['token_prob_states'])} frames)",
            flush=True,
        )
    return frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--num-good", type=int, default=10)
    parser.add_argument("--num-bad", type=int, default=10)
    parser.add_argument("--max-scan", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--min-depth", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--no-distractors", action="store_true")
    parser.add_argument("--good-focus-index", type=int, default=0)
    parser.add_argument("--interp-points", type=int, default=41)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--reuse-selected",
        action="store_true",
        help="reuse selected_good.jsonl and selected_bad.jsonl from output-dir instead of rescanning examples",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = OldSyntheticChainTokenizer.default()
    print(f"loading checkpoint: {args.checkpoint_path}", flush=True)
    print(f"device: {device}", flush=True)
    model, hparams = load_ebt_model(args.checkpoint_path, device)
    if int(model.vocab_size) != len(OLD_TOKENS):
        raise RuntimeError(f"Expected old vocab size {len(OLD_TOKENS)}, got {model.vocab_size}")

    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "seed": args.seed,
        "model_vocab_size": int(model.vocab_size),
        "mcmc_num_steps": int(hparams["mcmc_num_steps"]),
        "mcmc_step_size": float(hparams["mcmc_step_size"]),
        "tokens": list(OLD_TOKENS),
    }
    (args.output_dir / "diagnostic_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.reuse_selected:
        good, bad = load_existing_examples(args.output_dir)
        print(f"reusing {len(good)} good and {len(bad)} bad selected examples", flush=True)
    else:
        good, bad = select_examples(args, model, tokenizer, device)

    good_frames = add_traces_and_frames(
        good,
        title_prefix="Good",
        model=model,
        tokenizer=tokenizer,
        device=device,
        interp_points=args.interp_points,
        frame_dir=args.output_dir / "frames" / "good",
    )
    bad_frames = add_traces_and_frames(
        bad,
        title_prefix="Bad",
        model=model,
        tokenizer=tokenizer,
        device=device,
        interp_points=args.interp_points,
        frame_dir=args.output_dir / "frames" / "bad",
    )

    write_jsonl(args.output_dir / "selected_good.jsonl", good)
    write_jsonl(args.output_dir / "selected_bad.jsonl", bad)
    write_video(args.output_dir / "good_langevin.mp4", good_frames, fps=args.fps)
    write_video(args.output_dir / "bad_langevin.mp4", bad_frames, fps=args.fps)

    print(f"wrote {args.output_dir / 'selected_good.jsonl'}", flush=True)
    print(f"wrote {args.output_dir / 'selected_bad.jsonl'}", flush=True)
    print(f"wrote {args.output_dir / 'good_langevin.mp4'}", flush=True)
    print(f"wrote {args.output_dir / 'bad_langevin.mp4'}", flush=True)


if __name__ == "__main__":
    main()
