#!/usr/bin/env python3
"""Convert SGLang hidden-state JSON files into flat safetensors.

SGLang returns hidden states as a ragged JSON structure for these runs:

    [prefill_matrix, decode_vector_1, decode_vector_2, ...]

This script flattens each item into one tensor with shape
`[prompt_tokens + generated_tokens, hidden_dim]` and stores it as
`hidden_states` in a `.safetensors` file.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import torch
from safetensors.torch import save_file
from tqdm import tqdm


DEFAULT_RUN_DIR = (
    pathlib.Path(__file__).resolve().parents[1]
    / "09_vb_testing_ipynb"
    / "results"
    / "goedel_prover_v2_8b_sglang_896_hidden_states_gpus0_5_full_884_bs16"
)
DEFAULT_OUT_DIR = DEFAULT_RUN_DIR / "hidden_states_safetensors"
HIDDEN_STATE_KEYS = (
    "hidden_states",
    "hidden_state",
    "all_hidden_states",
    "hidden_states_all",
    "output_hidden_states",
)


def _is_empty_list(value: Any) -> bool:
    return isinstance(value, list) and not value


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and not isinstance(value[0], list)


def _is_matrix(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and isinstance(value[0], list)


def _extract_hidden_states_payload(raw: Any) -> Any | None:
    if isinstance(raw, dict):
        for key in HIDDEN_STATE_KEYS:
            value = raw.get(key)
            if value is not None:
                return value

        meta_info = raw.get("meta_info")
        if isinstance(meta_info, dict):
            for key in HIDDEN_STATE_KEYS:
                value = meta_info.get(key)
                if value is not None:
                    return value

        outputs = raw.get("outputs")
        if isinstance(outputs, list):
            for output in outputs:
                nested = _extract_hidden_states_payload(output)
                if nested is not None:
                    return nested
        return None

    if isinstance(raw, list):
        if not raw:
            return raw
        first = raw[0]
        if isinstance(first, (list, int, float)):
            return raw
        for item in raw:
            nested = _extract_hidden_states_payload(item)
            if nested is not None:
                return nested
    return None


def flatten_hidden_states(raw: Any, *, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, str]]:
    """Flatten SGLang's ragged hidden-state payload into `[tokens, hidden]`.

    The first matrix is treated as the prompt/prefill block. Some SGLang rows
    have an empty first prefill block followed by decode vectors; those are
    preserved as `prompt_tokens=0` and are otherwise skipped.
    """

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
            if tensor.ndim != 2:
                raise ValueError(f"expected matrix at item {index}, got shape {tuple(tensor.shape)}")
            token_count = int(tensor.shape[0])
            is_prompt_block = not saw_prompt_block and not blocks
        elif _is_number_list(item):
            tensor = torch.tensor(item, dtype=dtype).unsqueeze(0)
            if tensor.ndim != 2:
                raise ValueError(f"expected vector at item {index}, got shape {tuple(tensor.shape)}")
            token_count = 1
        else:
            raise ValueError(f"unsupported hidden-state item at index {index}: {type(item).__name__}")

        current_hidden_dim = int(tensor.shape[1])
        if hidden_dim is None:
            hidden_dim = current_hidden_dim
        elif current_hidden_dim != hidden_dim:
            raise ValueError(
                f"hidden dimension changed at item {index}: {current_hidden_dim} != {hidden_dim}"
            )

        if is_prompt_block:
            prompt_tokens += token_count
            saw_prompt_block = True
        else:
            generated_tokens += token_count
        blocks.append(tensor)

    if not blocks:
        raise ValueError("hidden-state payload contained no non-empty activation blocks")

    flat = torch.cat(blocks, dim=0).contiguous()
    metadata = {
        "shape": ",".join(str(x) for x in flat.shape),
        "prompt_tokens": str(prompt_tokens),
        "generated_tokens": str(generated_tokens),
        "hidden_dim": str(hidden_dim if hidden_dim is not None else 0),
        "dtype": str(dtype).replace("torch.", ""),
    }
    return flat, metadata


def load_generation_rows(run_dir: pathlib.Path) -> list[dict[str, Any]]:
    jsonl_path = run_dir / "generations.jsonl"
    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_output_path(out_dir: pathlib.Path, task_name: str) -> pathlib.Path:
    return out_dir / f"{task_name}.safetensors"


def resolve_source_path(path_text: str, *, run_dir: pathlib.Path) -> pathlib.Path:
    source = pathlib.Path(path_text)
    if source.is_absolute():
        return source
    candidates = [
        source,
        pathlib.Path.cwd() / source,
        run_dir / source,
        run_dir / source.relative_to(run_dir) if source.is_relative_to(run_dir) else None,
    ]
    if len(run_dir.parents) > 3:
        candidates.append(run_dir.parents[3] / source)
    seen: set[pathlib.Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return candidates[0].resolve()


def parse_dtype(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve()
    dtype = parse_dtype(args.dtype)
    rows = load_generation_rows(run_dir)
    if args.limit is not None:
        rows = rows[: args.limit]

    pending: list[tuple[dict[str, Any], pathlib.Path, pathlib.Path]] = []
    skipped = 0
    errors: list[dict[str, str]] = []
    for row in rows:
        task_name = row["task_name"]
        source = resolve_source_path(row["hidden_states_path"], run_dir=run_dir)
        target = task_output_path(out_dir, task_name)
        if target.exists() and not args.overwrite:
            skipped += 1
            continue
        pending.append((row, source, target))

    if not pending:
        manifest = {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "dtype": str(dtype).replace("torch.", ""),
            "total_rows": len(rows),
            "converted": 0,
            "skipped": skipped,
            "errors": errors,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({k: manifest[k] for k in ("total_rows", "converted", "skipped")}, sort_keys=True))
        return 0

    converted = 0
    for row, source, target in tqdm(pending, desc="Converting hidden states"):
        task_name = row["task_name"]
        try:
            raw = json.loads(source.read_text(encoding="utf-8"))
            hidden_states = _extract_hidden_states_payload(raw)
            if hidden_states is None:
                raise ValueError("could not locate hidden-state payload in response json")
            tensor, metadata = flatten_hidden_states(hidden_states, dtype=dtype)
            metadata.update(
                {
                    "task_name": task_name,
                    "split": str(row.get("split") or ""),
                    "task_id": str(row.get("task_id") or ""),
                    "source_json": str(source),
                }
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file({"hidden_states": tensor}, str(target), metadata=metadata)
            converted += 1
        except Exception as exc:
            errors.append(
                {
                    "task_name": task_name,
                    "split": str(row.get("split") or ""),
                    "task_id": str(row.get("task_id") or ""),
                    "source": str(source),
                    "error": repr(exc),
                }
            )

    if errors:
        (out_dir / "errors.json").write_text(
            json.dumps(errors, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "dtype": str(dtype).replace("torch.", ""),
        "total_rows": len(rows),
        "converted": converted,
        "skipped": skipped,
        "errors": errors,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("total_rows", "converted", "skipped")}, sort_keys=True))
    if errors:
        print(f"Errors: {len(errors)}; see {out_dir / 'manifest.json'}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
