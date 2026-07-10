#!/usr/bin/env python3
"""Langevin token diagnostics for the current synthetic-chain dataset."""

from __future__ import annotations

import argparse
import json
import math
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
    / "ebt-2xs-chain128-hops1-5-bs64-100k-valinit-val200-val32-lr0.0005-gpu7-v4_2026-07-09_13-45-11_"
    / "last.ckpt"
)
DEFAULT_DATA_PATH = EBT_ROOT / "data" / "nlp" / "synthetic_chain_dataset" / "synthetic_chain_dataset.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "analysis" / "langevin_diag_current"


def ensure_ebt_path() -> None:
    if str(EBT_ROOT) not in sys.path:
        sys.path.insert(0, str(EBT_ROOT))


def load_tokenizer():
    ensure_ebt_path()
    from data.nlp.synthetic_chain_dataset import SyntheticChainTokenizer

    return SyntheticChainTokenizer.default()


def load_ebt_model(checkpoint_path: Path, device: torch.device):
    ensure_ebt_path()
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
    return model, hparams, checkpoint


def set_deterministic_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def read_records(data_path: Path, split: str, max_scan: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if split != "all" and str(row.get("split", "train")) != split:
                continue
            records.append(row)
            if max_scan is not None and len(records) >= max_scan:
                break
    if not records:
        raise ValueError(f"No records found for split={split!r} in {data_path}")
    return records


def autoregressive_generate(
    model,
    tokenizer,
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
        decoded = torch.cat(
            [decoded, torch.tensor([[next_token]], dtype=torch.long, device=device)],
            dim=1,
        )
    return generated


def first_mismatch(generated: list[int], target: list[int]) -> int | None:
    for idx, (pred, gold) in enumerate(zip(generated, target)):
        if pred != gold:
            return idx
    if len(generated) != len(target):
        return min(len(generated), len(target))
    return None


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
    energies = model.transformer(
        torch.cat((real_embeddings, pred_embeddings), dim=1),
        start_pos=0,
        mcmc_step=mcmc_step,
    )
    return energies.reshape(input_ids.shape[0], input_ids.shape[1])[0, focus_pos]


def trace_langevin_token(
    model,
    tokenizer,
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

        pred_embeddings = state_to_embeddings(
            model,
            cur,
            assume_prob_dist=bool(model.hparams.normalize_initial_condition),
        )
        energy_preds = model.transformer(
            torch.cat((real_embeddings, pred_embeddings), dim=1),
            start_pos=0,
            mcmc_step=mcmc_step,
        ).reshape(-1, 1)
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
        if grad_norm > 0:
            constant_norm = math.sqrt(float(focus_grad.numel())) * abs(float(focus_grad.mean().item()))
            constant_fraction = constant_norm / grad_norm
        else:
            constant_fraction = 0.0

        grad_vectors.append(focus_grad)
        grad_norms.append(float(grad_norm))
        centered_grad_norms.append(float(centered_norm))
        grad_constant_fractions.append(float(constant_fraction))
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
    target_prob_by_state: list[float] = []
    max_prob_by_state: list[float] = []
    argmax_id_by_state: list[int] = []
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


def one_line(tokens: list[str], max_chars: int) -> str:
    text = " ".join(tokens)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def wrapped_lines(label: str, tokens: list[str], max_chars: int = 170) -> str:
    text = f"{label}: {' '.join(tokens)}"
    lines: list[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        lines.append(text[:cut])
        text = "  " + text[cut:].lstrip()
    lines.append(text)
    return "\n".join(lines)


def render_frame(example: dict[str, Any], trace: dict[str, Any], title_prefix: str, *, state_index: int, tokens: tuple[str, ...]) -> np.ndarray:
    fig = plt.figure(figsize=(15, 11), dpi=128)
    canvas = FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.72])
    ax_grad = fig.add_subplot(grid[0, 0])
    ax_cos = fig.add_subplot(grid[0, 1])
    ax_ce = fig.add_subplot(grid[0, 2])
    ax_interp = fig.add_subplot(grid[1, 0:2])
    ax_hist = fig.add_subplot(grid[1, 2])
    ax_text = fig.add_subplot(grid[2, :])

    steps = np.arange(len(trace["grad_norms"]))
    ax_grad.plot(steps, trace["grad_norms"], marker="o", linewidth=2, label="raw")
    ax_grad.plot(steps, trace["centered_grad_norms"], marker="s", linewidth=2, linestyle="--", label="centered")
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
    ax_interp.plot(interp_t, trace["init_to_final_energy"], marker=".", linewidth=2, color="#0b6e4f", label="initial -> final")
    ax_interp.plot(interp_t, trace["init_to_target_energy"], marker=".", linewidth=2, color="#5b4b8a", label="initial -> target one-hot")
    ax_interp.set_title("Energy Along Interpolations")
    ax_interp.set_xlabel("Interpolation t")
    ax_interp.set_ylabel("Energy")
    ax_interp.grid(alpha=0.25)
    ax_interp.legend(fontsize=9)

    probs = np.asarray(trace["token_prob_states"][state_index], dtype=np.float32)
    colors = ["#6f7f8f"] * len(probs)
    target_id = int(example["focus_target_id"])
    pred_id = int(np.argmax(probs))
    colors[target_id] = "#c43c39"
    colors[pred_id] = "#2f6fbb" if pred_id != target_id else "#7b4ab2"
    ax_hist.bar(np.arange(len(probs)), probs, color=colors, width=0.85)
    timestamp_label = "t=0 initial" if state_index == 0 else f"after step {state_index - 1}"
    delta_text = "" if state_index == 0 else f", prev delta={trace['prob_delta_norms'][state_index - 1]:.2g}"
    ax_hist.set_title(f"Token Probabilities ({timestamp_label}{delta_text})")
    ax_hist.set_xlabel("Token id")
    ax_hist.set_ylabel("softmax probability")
    ax_hist.set_ylim(0.0, max(0.05, float(probs.max()) * 1.15))
    tick_positions = np.linspace(0, len(probs) - 1, 12, dtype=int)
    ax_hist.set_xticks(tick_positions)
    ax_hist.set_xticklabels([str(i) for i in tick_positions], fontsize=8)
    ax_hist.grid(axis="y", alpha=0.2)

    ax_text.axis("off")
    text = "\n".join(
        [
            wrapped_lines("Prompt", example["prompt_tokens"], 190),
            wrapped_lines("Generated", example["generated_tokens"], 190),
            wrapped_lines("Target", example["target_tokens"], 190),
        ]
    )
    ax_text.text(0.0, 1.0, text, va="top", ha="left", fontsize=8.5, family="monospace")

    fig.suptitle(
        (
            f"{title_prefix} example {example['selection_index'] + 1} | row_id={example['id']} "
            f"depth={example.get('depth')} | focus token {example['focus_decode_step']} "
            f"target={tokens[target_id]} pred={tokens[pred_id]} | {timestamp_label}"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.955])
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


def select_examples(args: argparse.Namespace, model, tokenizer, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = read_records(args.data_path, args.split, args.max_scan)
    good: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    for scan_index, record in enumerate(records):
        prompt_ids = tokenizer.encode([str(token) for token in record["prompt_tokens"]])
        target_ids = tokenizer.encode([str(token) for token in record["target_tokens"]])
        if len(prompt_ids) + 1 + len(target_ids) > int(model.hparams.context_length):
            continue
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
            "id": int(record["id"]),
            "split": str(record.get("split", args.split)),
            "depth": int(record.get("depth", -1)),
            "num_distractors": int(record.get("num_distractors", -1)),
            "scan_index": scan_index,
            "selection": selected,
            "selection_index": selection_index,
            "exact_match": exact,
            "first_error_index": mismatch,
            "prompt_tokens": [str(token) for token in record["prompt_tokens"]],
            "target_tokens": [str(token) for token in record["target_tokens"]],
            "path_tokens": [str(token) for token in record.get("path_tokens", [])],
            "goal": str(record.get("goal", "")),
            "starts": [str(token) for token in record.get("starts", [])],
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
            f"selected {selected} {selection_index + 1}: row_id={record['id']} "
            f"exact={exact} focus={focus_step}",
            flush=True,
        )
        if len(good) >= args.num_good and len(bad) >= args.num_bad:
            break

    if len(good) < args.num_good or len(bad) < args.num_bad:
        raise RuntimeError(
            f"Only found {len(good)} good and {len(bad)} bad examples "
            f"from {len(records)} scanned {args.split} rows."
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
        raise FileNotFoundError(f"Missing reusable selections in {output_dir}")
    return load_jsonl(good_path), load_jsonl(bad_path)


def add_traces_and_frames(
    examples: list[dict[str, Any]],
    *,
    title_prefix: str,
    model,
    tokenizer,
    device: torch.device,
    interp_points: int,
    frame_dir: Path,
) -> list[np.ndarray]:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("*.png"):
        old_frame.unlink()
    frames: list[np.ndarray] = []
    token_tuple = tuple(tokenizer.tokens)
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
            frame = render_frame(example, trace, title_prefix, state_index=state_index, tokens=token_tuple)
            frames.append(frame)
            frame_path = frame_dir / f"{example['selection_index']:02d}_row_{example['id']}_state_{state_index:02d}.png"
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
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--num-good", type=int, default=10)
    parser.add_argument("--num-bad", type=int, default=10)
    parser.add_argument("--max-scan", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--good-focus-index", type=int, default=0)
    parser.add_argument("--interp-points", type=int, default=41)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--reuse-selected", action="store_true")
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
    tokenizer = load_tokenizer()
    print(f"loading checkpoint: {args.checkpoint_path}", flush=True)
    print(f"data path: {args.data_path}", flush=True)
    print(f"device: {device}", flush=True)
    model, hparams, checkpoint = load_ebt_model(args.checkpoint_path, device)
    if int(model.vocab_size) != int(tokenizer.vocab_size):
        raise RuntimeError(f"Model vocab size {model.vocab_size} != tokenizer vocab size {tokenizer.vocab_size}")

    metadata = {
        "checkpoint_path": str(args.checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "data_path": str(args.data_path),
        "output_dir": str(args.output_dir),
        "split": args.split,
        "seed": args.seed,
        "model_vocab_size": int(model.vocab_size),
        "context_length": int(hparams["context_length"]),
        "mcmc_num_steps": int(hparams["mcmc_num_steps"]),
        "mcmc_step_size": float(hparams["mcmc_step_size"]),
        "run_name": str(hparams.get("run_name", "")),
        "tokens": list(tokenizer.tokens),
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
