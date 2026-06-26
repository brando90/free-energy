#!/usr/bin/env python3
"""Sample from the limited Alexi EBT language-modeling subset checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from model.nlp.baseline_transformer import Baseline_Transformer_NLP
from model.nlp.ebt import EBT_NLP
from train_lm_subset import hparams_for

try:
    from inference.nlp.generate_text import sample_top_p
except ModuleNotFoundError:
    def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        return torch.gather(probs_idx, -1, next_token)


def namespace_from_config(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**config)


def build_model(model_name: str, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    args = namespace_from_config(config)
    model_cls = {"baseline_transformer": Baseline_Transformer_NLP, "ebt": EBT_NLP}[model_name]
    model = model_cls(hparams_for(args, model_name)).to(device)
    checkpoint = torch.load(Path(config["run_dir"]) / f"{model_name}_final.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def next_logits(model: torch.nn.Module, model_name: str, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=input_ids.is_cuda):
        if model_name == "ebt":
            with torch.enable_grad():
                outputs = model(input_ids, start_pos=0, learning=False, return_raw_logits=True, no_randomness=True)
            return outputs[0][-1][:, -1].float()
        return model(input_ids, start_pos=0, learning=False, return_raw_logits=True)[:, -1].float()


def generate(
    model: torch.nn.Module,
    model_name: str,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    context_length: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    encoded = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id)
    generated: list[list[int]] = [ids[:] if ids else [eos_id] for ids in encoded]
    finished = [False for _ in prompts]

    for _ in range(max_new_tokens):
        max_len = max(min(len(ids), context_length) for ids in generated)
        batch = torch.full((len(generated), max_len), pad_id, dtype=torch.long, device=device)
        for i, ids in enumerate(generated):
            window = ids[-context_length:]
            batch[i, -len(window) :] = torch.tensor(window, dtype=torch.long, device=device)
        logits = next_logits(model, model_name, batch)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_ids = sample_top_p(probs, top_p).reshape(-1)
        else:
            next_ids = torch.argmax(logits, dim=-1)
        for i, next_id_t in enumerate(next_ids.detach().cpu().tolist()):
            next_id = int(next_id_t)
            if finished[i]:
                continue
            generated[i].append(next_id)
            finished[i] = next_id == eos_id
        if all(finished):
            break

    rows: list[dict[str, Any]] = []
    for prompt, prompt_ids, ids in zip(prompts, encoded, generated):
        continuation = ids[len(prompt_ids) :]
        rows.append(
            {
                "prompt": prompt,
                "generation": tokenizer.decode(continuation, skip_special_tokens=True),
                "full_text": tokenizer.decode(ids, skip_special_tokens=True),
                "generated_token_ids": continuation,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=ROOT.parent / "runs" / "alexiglad_lm_subset" / "20260625_171232")
    parser.add_argument("--models", nargs="+", default=["baseline_transformer", "ebt"])
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=[
            "Lean programs",
            "Energy based transformers",
            "The baseline transformer",
            "Formal verification",
        ],
    )
    args = parser.parse_args()

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    config["run_dir"] = str(args.run_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], clean_up_tokenization_spaces=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    output_path = args.output or args.run_dir / f"samples_t{args.temperature}_top{args.top_p}.json"
    all_rows: dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "models": {},
    }
    for model_name in args.models:
        model = build_model(model_name, config, device)
        rows = generate(
            model,
            model_name,
            tokenizer,
            args.prompts,
            max_new_tokens=args.max_new_tokens,
            context_length=int(config["context_length"]),
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        all_rows["models"][model_name] = rows
        print(f"\n## {model_name}")
        for row in rows:
            print(f"PROMPT: {row['prompt']}")
            print(f"GEN: {row['generation']}")

    output_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
