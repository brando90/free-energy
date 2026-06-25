#!/usr/bin/env python3
"""Rebuild context/gold target tokens from comment-stripped Lean files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "context_gold"
DEFAULT_VERIBENCH_ROOT = Path(__file__).resolve().parents[2] / "09_vb_testing_ipynb" / "veribench"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def resolve_gold_path(rel_lean_path: str, veribench_root: Path) -> Path:
    path = Path(rel_lean_path)
    if path.is_absolute():
        return path
    for candidate in (veribench_root / path, veribench_root / "veribench_dataset" / path):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(rel_lean_path)


def strip_lean_comments(source: str) -> str:
    """Remove Lean line/block comments without touching string literals."""
    out: list[str] = []
    i = 0
    n = len(source)
    block_depth = 0
    in_string = False
    in_char = False
    escaped = False

    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""

        if block_depth:
            if ch == "/" and nxt == "-":
                block_depth += 1
                i += 2
            elif ch == "-" and nxt == "/":
                block_depth -= 1
                i += 2
            else:
                if ch == "\n":
                    out.append("\n")
                i += 1
            continue

        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue

        if in_char:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_char = False
            i += 1
            continue

        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif ch == "'":
            in_char = True
            out.append(ch)
            i += 1
        elif ch == "-" and nxt == "-":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
        elif ch == "/" and nxt == "-":
            if out and not out[-1].isspace():
                out.append(" ")
            block_depth = 1
            i += 2
        else:
            out.append(ch)
            i += 1

    return "".join(out)


def remove_empty_lines(source: str) -> str:
    lines = [line.rstrip() for line in source.splitlines() if line.strip()]
    return "\n".join(lines) + ("\n" if lines else "")


def clean_lean(source: str) -> str:
    return remove_empty_lines(strip_lean_comments(source))


def build_vocab(tokenized_rows: list[list[int]], tokenizer_name: str) -> dict[str, Any]:
    original_ids = sorted({token_id for row in tokenized_rows for token_id in row})
    id_to_original: list[int | None] = [None, None, None, None, *original_ids]
    original_to_local = {str(token_id): i + 4 for i, token_id in enumerate(original_ids)}
    return {
        "id_to_original": id_to_original,
        "local_bos_id": 1,
        "local_eos_id": 2,
        "local_pad_id": 0,
        "local_unk_id": 3,
        "original_to_local": original_to_local,
        "source": "comment-stripped gold lean target tokens from usable VeriBench rows",
        "special_tokens": {
            "bos": "<LOCAL_BOS>",
            "eos": "<LOCAL_EOS>",
            "pad": "<LOCAL_PAD>",
            "unk": "<LOCAL_UNK>",
        },
        "tokenizer": tokenizer_name,
        "vocab_size": len(id_to_original),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--veribench-root", type=Path, default=DEFAULT_VERIBENCH_ROOT)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.jsonl"
    vocab_path = data_dir / "vocab.json"
    summary_path = data_dir / "summary.json"

    old_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer_name = old_vocab["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    rows = read_jsonl(manifest_path)
    cleaned_texts: list[str] = []
    tokenized_rows: list[list[int]] = []
    for row in rows:
        gold_path = resolve_gold_path(row["rel_lean_path"], args.veribench_root.resolve())
        cleaned = clean_lean(gold_path.read_text(encoding="utf-8"))
        token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"{row['task_name']} cleaned to zero tokens")
        cleaned_texts.append(cleaned)
        tokenized_rows.append([int(token_id) for token_id in token_ids])

    vocab = build_vocab(tokenized_rows, tokenizer_name=tokenizer_name)
    original_to_local = vocab["original_to_local"]
    for row, original_ids in zip(rows, tokenized_rows, strict=True):
        row["target_original_ids"] = original_ids
        row["target_local_ids"] = [int(original_to_local[str(token_id)]) for token_id in original_ids]
        row["target_token_count"] = len(original_ids)

    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "candidate_rows": len(rows),
            "target_cleaning": {
                "comments_removed": True,
                "empty_lines_removed": True,
                "keeps_newlines_between_nonempty_lines": True,
            },
            "tokenizer": tokenizer_name,
            "vocab_size": vocab["vocab_size"],
        }
    )

    write_jsonl(manifest_path, rows)
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    before_tokens = sum(len(row.get("target_original_ids", [])) for row in rows)
    print(f"rewrote_rows={len(rows)}")
    print(f"vocab_size={vocab['vocab_size']}")
    print(f"target_tokens={before_tokens}")
    print(f"sample_chars={len(cleaned_texts[0])}")


if __name__ == "__main__":
    main()
