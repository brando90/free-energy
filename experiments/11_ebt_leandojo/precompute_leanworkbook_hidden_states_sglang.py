#!/usr/bin/env python3
"""Precompute Goedel hidden states for the full Lean Workbook Plus dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoTokenizer

from leanworkbook_plus_benchmark import DEFAULT_DATASET, download_dataset, make_prompt


HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_MODEL = "Goedel-LM/Goedel-Prover-V2-8B"
DEFAULT_OUT_DIR = HERE / "results" / "leanworkbook_plus_goedel_hidden_states_gpus0_3"


@dataclass
class HiddenStateRecord:
    row_index: int
    task_id: str
    status: str
    prompt_tokens: int
    generated_tokens: int
    hidden_dim: int
    safetensors_path: str
    prompt_path: str
    response_path: str
    elapsed_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--data-file", type=pathlib.Path, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--data-parallel-size", type=int, default=4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--mem-fraction-static", type=float, default=0.85)
    parser.add_argument("--disable-cuda-graph", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--row-indices-file", type=pathlib.Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_hidden_states(response: Any) -> Any:
    if isinstance(response, dict):
        for key in ("hidden_states", "hidden_state", "all_hidden_states", "hidden_states_all", "output_hidden_states"):
            value = response.get(key)
            if value is not None:
                return value
        meta_info = response.get("meta_info")
        if isinstance(meta_info, dict):
            for key in ("hidden_states", "all_hidden_states", "hidden_states_all", "output_hidden_states"):
                value = meta_info.get(key)
                if value is not None:
                    return value
        if "outputs" in response and isinstance(response["outputs"], list):
            for item in response["outputs"]:
                value = extract_hidden_states(item)
                if value is not None:
                    return value
    return None


def extract_text(response: Any) -> str:
    if isinstance(response, dict):
        outputs = response.get("outputs")
        if isinstance(outputs, list) and outputs:
            return str(outputs[0].get("text", ""))
        if "text" in response:
            return str(response["text"])
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    outputs = getattr(response, "outputs", None)
    if isinstance(outputs, list) and outputs:
        return str(getattr(outputs[0], "text", ""))
    return str(response)


def _is_empty_list(value: Any) -> bool:
    return isinstance(value, list) and not value


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and not isinstance(value[0], list)


def _is_matrix(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list)


def flatten_hidden_states(raw: Any, *, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("hidden-state payload must be a non-empty list")

    blocks: list[torch.Tensor] = []
    prompt_tokens = 0
    generated_tokens = 0
    hidden_dim: int | None = None
    saw_prompt_block = False

    for index, item in enumerate(raw):
        if _is_empty_list(item):
            if index == 0:
                saw_prompt_block = True
            continue
        is_prompt_block = False
        if _is_matrix(item):
            tensor = torch.tensor(item, dtype=dtype)
            token_count = int(tensor.shape[0])
            is_prompt_block = not saw_prompt_block and not blocks
        elif _is_number_list(item):
            tensor = torch.tensor(item, dtype=dtype).unsqueeze(0)
            token_count = 1
        else:
            raise ValueError(f"unsupported hidden-state item at index {index}: {type(item).__name__}")

        current_hidden_dim = int(tensor.shape[1])
        if hidden_dim is None:
            hidden_dim = current_hidden_dim
        elif current_hidden_dim != hidden_dim:
            raise ValueError(f"hidden dimension changed: {current_hidden_dim} != {hidden_dim}")

        if is_prompt_block:
            prompt_tokens += token_count
            saw_prompt_block = True
        else:
            generated_tokens += token_count
        blocks.append(tensor)

    if not blocks:
        raise ValueError("hidden-state payload contained no activations")
    flat = torch.cat(blocks, dim=0).contiguous()
    context = flat[:prompt_tokens].contiguous()
    return context, {
        "shape": ",".join(str(x) for x in context.shape),
        "prompt_tokens": str(prompt_tokens),
        "generated_tokens": str(generated_tokens),
        "hidden_dim": str(hidden_dim or 0),
        "dtype": str(dtype).replace("torch.", ""),
    }


def load_rows(data_file: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_row_indices(path: pathlib.Path) -> list[int]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("indices", [])
        if not isinstance(payload, list):
            raise ValueError(f"Expected JSON list or {{\"indices\": [...]}} in {path}")
        return [int(item) for item in payload]
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safetensors_path(out_dir: pathlib.Path, row_index: int) -> pathlib.Path:
    return out_dir / "hidden_states_safetensors" / f"{row_index:06d}.safetensors"


def prompt_path(out_dir: pathlib.Path, row_index: int) -> pathlib.Path:
    return out_dir / "prompts" / f"{row_index:06d}.txt"


def response_path(out_dir: pathlib.Path, row_index: int) -> pathlib.Path:
    return out_dir / "responses" / f"{row_index:06d}.txt"


def read_completed(manifest_path: pathlib.Path) -> set[int]:
    if not manifest_path.exists():
        return set()
    done: set[int] = set()
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done.add(int(row["row_index"]))
    return done


def main() -> int:
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from sglang import Engine

    data_file = args.data_file or download_dataset(args.dataset)
    all_rows = load_rows(data_file)
    if args.row_indices_file is not None:
        selected_indices = load_row_indices(args.row_indices_file)
        selected = [(row_index, all_rows[row_index]) for row_index in selected_indices]
    else:
        rows = all_rows[args.start_index :]
        if args.max_items is not None:
            rows = rows[: args.max_items]
        selected = [(args.start_index + i, row) for i, row in enumerate(rows)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.jsonl"
    errors_path = args.out_dir / "errors.json"
    completed = set() if args.overwrite else read_completed(manifest_path)
    pending = [(row_index, row) for row_index, row in selected if row_index not in completed]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    metadata = {
        "dataset": args.dataset,
        "data_file": str(data_file),
        "model": args.model,
        "gpus": args.gpus,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "context_length": args.context_length,
        "mem_fraction_static": args.mem_fraction_static,
        "row_indices_file": str(args.row_indices_file) if args.row_indices_file else None,
        "pending": len(pending),
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"rows_total={len(selected)} pending={len(pending)} out_dir={args.out_dir}", flush=True)
    if not pending:
        if errors_path.exists():
            errors_path.unlink()
        return 0

    engine = Engine(
        model_path=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        dp_size=args.data_parallel_size,
        tp_size=args.tensor_parallel_size,
        load_balance_method="round_robin",
        enable_return_hidden_states=True,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        disable_cuda_graph=args.disable_cuda_graph,
    )
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    sampling_params = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.top_k >= 0:
        sampling_params["top_k"] = args.top_k

    errors: list[dict[str, Any]] = []
    try:
        with manifest_path.open("a", encoding="utf-8") as manifest:
            progress = tqdm(total=len(pending), desc="Precomputing hidden states", unit="item")
            for offset in range(0, len(pending), args.batch_size):
                batch = pending[offset : offset + args.batch_size]
                prompts = [
                    make_prompt(str(row["id"]), str(row["natural_language_statement"]), str(row["formal_statement"]))
                    for _, row in batch
                ]
                started = time.time()
                responses = engine.generate(
                    prompt=prompts,
                    sampling_params=sampling_params,
                    return_hidden_states=True,
                )
                per_item_elapsed = (time.time() - started) / max(1, len(batch))
                if isinstance(responses, dict):
                    responses = [responses]

                for (row_index, row), prompt, response in zip(batch, prompts, responses):
                    try:
                        hidden_raw = extract_hidden_states(response)
                        if hidden_raw is None:
                            raise ValueError("missing hidden states in response")
                        tensor, tensor_meta = flatten_hidden_states(hidden_raw, dtype=torch.bfloat16)
                        hs_path = safetensors_path(args.out_dir, row_index)
                        pr_path = prompt_path(args.out_dir, row_index)
                        rs_path = response_path(args.out_dir, row_index)
                        hs_path.parent.mkdir(parents=True, exist_ok=True)
                        pr_path.parent.mkdir(parents=True, exist_ok=True)
                        rs_path.parent.mkdir(parents=True, exist_ok=True)
                        pr_path.write_text(prompt, encoding="utf-8")
                        rs_path.write_text(extract_text(response), encoding="utf-8")
                        meta = {
                            **tensor_meta,
                            "row_index": str(row_index),
                            "task_id": str(row["id"]),
                            "status": str(row["status"]),
                            "formal_statement": str(row["formal_statement"]),
                            "prompt_path": str(pr_path),
                            "response_path": str(rs_path),
                        }
                        save_file({"hidden_states": tensor}, str(hs_path), metadata=meta)
                        rec = HiddenStateRecord(
                            row_index=row_index,
                            task_id=str(row["id"]),
                            status=str(row["status"]),
                            prompt_tokens=int(meta["prompt_tokens"]),
                            generated_tokens=int(meta["generated_tokens"]),
                            hidden_dim=int(meta["hidden_dim"]),
                            safetensors_path=str(hs_path),
                            prompt_path=str(pr_path),
                            response_path=str(rs_path),
                            elapsed_sec=per_item_elapsed,
                        )
                        manifest.write(json.dumps(asdict(rec), sort_keys=True) + "\n")
                        manifest.flush()
                    except Exception as exc:
                        errors.append({"row_index": row_index, "task_id": row.get("id"), "error": repr(exc)})
                        errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
                progress.update(len(batch))
            progress.close()
    finally:
        engine.shutdown()

    summary = {
        "rows_selected": len(selected),
        "completed_now": len(pending) - len(errors),
        "errors": len(errors),
        "manifest": str(manifest_path),
    }
    if errors:
        errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
    elif errors_path.exists():
        errors_path.unlink()
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
