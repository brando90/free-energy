#!/usr/bin/env python3
"""Rebuild context/gold target tokens from normalized Lean gold files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "context_gold"
DEFAULT_VERIBENCH_ROOT = Path(__file__).resolve().parents[2] / "09_vb_testing_ipynb" / "veribench"


DECL_KEYWORDS = {
    "def",
    "theorem",
    "lemma",
    "abbrev",
    "inductive",
    "structure",
    "class",
    "instance",
}

LEAN_KEYWORDS = {
    "Prop",
    "Type",
    "Type*",
    "Sort",
    "by",
    "do",
    "let",
    "mut",
    "if",
    "then",
    "else",
    "match",
    "with",
    "where",
    "case",
    "at",
    "of",
    "for",
    "in",
    "return",
    "fun",
    "rec",
    "intro",
    "intros",
    "have",
    "show",
    "from",
    "exact",
    "calc",
    "termination_by",
    "decreasing_by",
    "all_goals",
    "private",
    "protected",
    "partial",
    "noncomputable",
    "namespace",
    "end",
    "open",
    "import",
    "set_option",
    "example",
}

PRESERVED_IDENTIFIERS = LEAN_KEYWORDS | {
    "Nat",
    "Int",
    "Bool",
    "String",
    "Char",
    "List",
    "Array",
    "Option",
    "DecidableEq",
    "True",
    "False",
    "true",
    "false",
    "none",
    "some",
    "inl",
    "inr",
    "zero",
    "succ",
    "mk",
    "refl",
    "step",
    "head",
    "tail",
    "rfl",
    "simp",
    "omega",
    "native_decide",
    "decide",
    "decide_eq_true",
    "decide_eq_false",
    "decide_eq_true_eq",
    "Id",
    "run",
    "min",
    "max",
    "length",
    "foldr",
    "headD",
    "head?",
    "getD",
    "getLastD",
    "nil",
    "cons",
    "left",
    "right",
    "constructor",
    "cases",
    "induction",
    "rw",
    "rwa",
    "simpa",
    "using",
    "unfold",
    "subst",
    "rename_i",
    "by_cases",
    "rcases",
    "constructor",
    "exact",
    "intro",
    "ne_of_gt",
    "Nat",
    "Mathlib",
    "Std",
}

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ω][A-Za-z0-9_']*")
DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*"
    r"(?:def|theorem|lemma|abbrev|inductive|structure|class|instance)\s+([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ω][A-Za-z0-9_']*)",
    re.MULTILINE,
)


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


def remove_given_lines(source: str) -> str:
    kept: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == "set_option linter.unusedVariables false":
            continue
        kept.append(line)
    return "\n".join(kept) + ("\n" if kept else "")


def unwrap_namespaces(source: str) -> str:
    namespace_stack: list[str] = []
    removed_namespaces: list[str] = []
    output: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("namespace "):
            namespace_name = stripped.split(maxsplit=1)[1]
            namespace_stack.append(namespace_name)
            removed_namespaces.append(namespace_name)
            continue
        if namespace_stack and stripped == f"end {namespace_stack[-1]}":
            namespace_stack.pop()
            continue
        output.append(line)
    unwrapped = "\n".join(output) + ("\n" if output else "")
    for namespace_name in sorted(set(removed_namespaces), key=len, reverse=True):
        unwrapped = unwrapped.replace(f"{namespace_name}.", "")
    return unwrapped


def _name_stream() -> Any:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for char in alphabet:
        if char not in PRESERVED_IDENTIFIERS:
            yield char
    width = 2
    while True:
        total = len(alphabet) ** width
        for i in range(total):
            n = i
            chars = []
            for _ in range(width):
                chars.append(alphabet[n % len(alphabet)])
                n //= len(alphabet)
            candidate = "".join(reversed(chars))
            if candidate not in PRESERVED_IDENTIFIERS:
                yield candidate
        width += 1


def _identifier_spans_lexical(source: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    in_string = False
    in_char = False
    escaped = False
    i = 0
    while i < len(source):
        ch = source[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if in_char:
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
            i += 1
            continue
        if ch == "'":
            in_char = True
            i += 1
            continue
        match = IDENT_RE.match(source, i)
        if match:
            start = len(source[: match.start()].encode("utf-8"))
            end = len(source[: match.end()].encode("utf-8"))
            spans.append((start, end, match.group(0)))
            i = match.end()
            continue
        i += 1
    return spans


def _identifier_spans_tree_sitter(source: str) -> list[tuple[int, int, str]] | None:
    try:
        from tree_sitter_languages import get_parser

        parser = get_parser("lean")
    except Exception:
        return None
    tree = parser.parse(source.encode("utf-8"))
    spans: list[tuple[int, int, str]] = []

    def traverse(node: Any) -> None:
        if node.type == "identifier":
            text = source.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8")
            spans.append((node.start_byte, node.end_byte, text))
        for child in node.children:
            traverse(child)

    traverse(tree.root_node)
    return spans


def _collect_binder_names(source: str) -> set[str]:
    names: set[str] = set()

    def add_from_text(text: str) -> None:
        for match in IDENT_RE.finditer(text):
            name = match.group(0)
            if name not in PRESERVED_IDENTIFIERS and not name.startswith("_"):
                names.add(name)

    for content in re.findall(r"[\(\{\[]([^()\{\}\[\]]*?:[^()\{\}\[\]]*?)[\)\}\]]", source):
        before_colon = content.split(":", 1)[0]
        add_from_text(before_colon)
    for content in re.findall(r"∀\s+([^,]+),", source):
        add_from_text(content.split(":", 1)[0])
    for pattern in (
        r"\bintros?\s+([^\n:=]+)",
        r"\brename_i\s+([^\n:=]+)",
        r"\brcases\s+[^\n]+?\s+with\s+([^\n]+)",
        r"\bhave\s+([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ω][A-Za-z0-9_']*)",
        r"\bby_cases\s+([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ω][A-Za-z0-9_']*)",
        r"\blet\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ω][A-Za-z0-9_']*)",
        r"\bfor\s+([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ω][A-Za-z0-9_']*)\s+in\b",
    ):
        for match in re.finditer(pattern, source):
            add_from_text(match.group(1))
    for match in re.finditer(r"^\s*\|\s+(.+?)=>", source, flags=re.MULTILINE):
        add_from_text(match.group(1).replace("::", " ").replace(",", " "))
    return names


def collect_user_identifiers(source: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if (
            name
            and name not in seen
            and name not in PRESERVED_IDENTIFIERS
            and not name.startswith("_")
            and not name[0].isdigit()
        ):
            seen.add(name)
            names.append(name)

    for match in DECL_RE.finditer(source):
        add(match.group(1))
    for name in _collect_binder_names(source):
        add(name)
    return names


def anonymize_user_identifiers(source: str) -> tuple[str, dict[str, str]]:
    user_names = collect_user_identifiers(source)
    if not user_names:
        return source, {}
    name_iter = _name_stream()
    mapping = {name: next(name_iter) for name in user_names}
    spans = _identifier_spans_tree_sitter(source) or _identifier_spans_lexical(source)
    source_bytes = source.encode("utf-8")
    replacements = [
        (start, end, mapping[text])
        for start, end, text in spans
        if text in mapping and (start == 0 or source_bytes[start - 1 : start] != b".")
    ]
    code = bytearray(source_bytes)
    for start, end, replacement in reversed(replacements):
        code[start:end] = replacement.encode("utf-8")
    return code.decode("utf-8"), mapping


def clean_lean_with_mapping(source: str) -> tuple[str, dict[str, str]]:
    cleaned = strip_lean_comments(source)
    cleaned = remove_given_lines(cleaned)
    cleaned = unwrap_namespaces(cleaned)
    cleaned = remove_empty_lines(cleaned)
    cleaned, mapping = anonymize_user_identifiers(cleaned)
    return remove_empty_lines(cleaned), mapping


def clean_lean(source: str) -> str:
    return clean_lean_with_mapping(source)[0]


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
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    manifest_path = data_dir / "manifest.jsonl"
    vocab_path = data_dir / "vocab.json"
    summary_path = data_dir / "summary.json"

    old_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    tokenizer_name = old_vocab["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    rows = read_jsonl(manifest_path)
    old_vocab_size = int(old_vocab.get("vocab_size", 0))
    old_token_count = sum(len(row.get("target_original_ids", [])) for row in rows)
    cleaned_texts: list[str] = []
    tokenized_rows: list[list[int]] = []
    rename_maps: list[dict[str, str]] = []
    for row in rows:
        gold_path = resolve_gold_path(row["rel_lean_path"], args.veribench_root.resolve())
        cleaned, rename_map = clean_lean_with_mapping(gold_path.read_text(encoding="utf-8"))
        token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"{row['task_name']} cleaned to zero tokens")
        cleaned_texts.append(cleaned)
        rename_maps.append(rename_map)
        tokenized_rows.append([int(token_id) for token_id in token_ids])

    if args.preview > 0:
        preview_rows = list(zip(rows, cleaned_texts, rename_maps, tokenized_rows, strict=True))[: args.preview]
        for row, cleaned, rename_map, token_ids in preview_rows:
            print("=" * 88)
            print(f"task_name={row['task_name']}")
            print(f"tokens={len(token_ids)}")
            print(f"rename_map={dict(list(rename_map.items())[:30])}")
            print(cleaned[:2000])
        return

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
                "imports_preserved": True,
                "import_std_removed": False,
                "unused_variable_linter_option_removed": True,
                "namespaces_unwrapped": True,
                "user_identifiers_anonymized": True,
            },
            "tokenizer": tokenizer_name,
            "old_target_tokens": old_token_count,
            "old_vocab_size": old_vocab_size,
            "target_tokens": sum(len(row) for row in tokenized_rows),
            "vocab_size": vocab["vocab_size"],
        }
    )

    write_jsonl(manifest_path, rows)
    vocab_path.write_text(json.dumps(vocab, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"rewrote_rows={len(rows)}")
    print(f"old_vocab_size={old_vocab_size}")
    print(f"vocab_size={vocab['vocab_size']}")
    print(f"old_target_tokens={old_token_count}")
    print(f"target_tokens={sum(len(row) for row in tokenized_rows)}")
    print(f"sample_chars={len(cleaned_texts[0])}")


if __name__ == "__main__":
    main()
