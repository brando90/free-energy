#!/usr/bin/env python3
"""Build cleaned Lean Workbook Plus targets and compact Goedel token ids."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "leanworkbook_plus_train.jsonl"
DEFAULT_OUT_DIR = HERE / "context_gold"
DEFAULT_TOKENIZER = "Goedel-LM/Goedel-Prover-V2-8B"

LEAN_KEYWORDS = {
    "Prop", "Type", "Type*", "Sort", "by", "do", "let", "mut", "if", "then", "else", "match",
    "with", "where", "case", "at", "of", "for", "in", "return", "fun", "rec", "intro", "intros",
    "have", "show", "from", "exact", "calc", "termination_by", "decreasing_by", "all_goals",
    "private", "protected", "partial", "noncomputable", "namespace", "end", "open", "import",
    "set_option", "example", "theorem", "lemma", "def", "abbrev", "inductive", "structure",
    "class", "instance", "axiom", "constant", "opaque", "variable",
}
PRESERVED_IDENTIFIERS = LEAN_KEYWORDS | {
    "Nat", "Int", "Rat", "Real", "Complex", "Bool", "String", "Char", "List", "Array", "Option",
    "DecidableEq", "True", "False", "true", "false", "none", "some", "inl", "inr", "zero", "succ",
    "mk", "refl", "rfl", "simp", "omega", "native_decide", "decide", "Id", "min", "max", "length",
    "foldr", "headD", "head?", "getD", "getLastD", "nil", "cons", "left", "right", "constructor",
    "cases", "induction", "rw", "rwa", "simpa", "using", "unfold", "subst", "rename_i", "by_cases",
    "rcases", "Set", "Finset", "Multiset", "Fin", "Subtype", "Quot", "ULift", "PUnit", "Unit",
    "Prod", "Sum", "PSum", "Sigma", "ℕ", "ℤ", "ℚ", "ℝ", "ℂ", "Icc", "Ioc", "Ico", "Ioo", "uIcc",
    "uIoc", "uIco", "uIoo", "Mathlib", "Std",
}
IDENT_RE = re.compile(r"[A-Za-z_α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_α-ωΑ-Ωℕℤℚℝℂ₀₁₂₃₄₅₆₇₈₉']*")
DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*"
    r"(?:def|theorem|lemma|abbrev|inductive|structure|class|instance|axiom|opaque|constant)\s+"
    r"([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_']*)",
    re.MULTILINE,
)
HEADER_PREFIX_RE = re.compile(
    r"^(\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*theorem\s+)"
    r"([A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_']*)",
    re.DOTALL,
)
QUALIFIED_CONSTANTS = {
    "sin": "Real.sin",
    "cos": "Real.cos",
    "tan": "Real.tan",
    "sinh": "Real.sinh",
    "cosh": "Real.cosh",
    "tanh": "Real.tanh",
    "asin": "Real.arcsin",
    "acos": "Real.arccos",
    "atan": "Real.arctan",
    "arcsin": "Real.arcsin",
    "arccos": "Real.arccos",
    "arctan": "Real.arctan",
    "exp": "Real.exp",
    "log": "Real.log",
    "sqrt": "Real.sqrt",
    "floor": "Int.floor",
    "ceil": "Int.ceil",
    "fract": "Int.fract",
    "choose": "Nat.choose",
    "divisors": "Nat.divisors",
    "factorial": "Nat.factorial",
    "fib": "Nat.fib",
    "Coprime": "Nat.Coprime",
    "φ": "Nat.totient",
    "totient": "Nat.totient",
    "ModEq": "Nat.ModEq",
}
PAREN_FACTORIAL_RE = re.compile(r"(\([^()]+\))!")
NUM_FACTORIAL_RE = re.compile(r"\b(\d+)!")
IDENT_FACTORIAL_RE = re.compile(r"\b([A-Za-z_α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_α-ωΑ-Ωℕℤℚℝℂ₀₁₂₃₄₅₆₇₈₉']*)!")
DOUBLE_FACTORIAL_RE = re.compile(r"(\([^()]+\)|\d+|[A-Za-z_α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_α-ωΑ-Ωℕℤℚℝℂ₀₁₂₃₄₅₆₇₈₉']*)!!")
POLY_C_APPLY_RE = re.compile(r"(?<!\.)\bC(?=\s*\()")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def strip_lean_comments(source: str) -> str:
    out: list[str] = []
    i = 0
    block_depth = 0
    in_string = False
    escaped = False
    while i < len(source):
        ch = source[i]
        nxt = source[i + 1] if i + 1 < len(source) else ""
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
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
        elif ch == "-" and nxt == "-":
            i += 2
            while i < len(source) and source[i] != "\n":
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
    kept = [line for line in source.splitlines() if line.strip() != "set_option linter.unusedVariables false"]
    return "\n".join(kept) + ("\n" if kept else "")


def unwrap_namespaces(source: str) -> str:
    namespace_stack: list[str] = []
    removed: list[str] = []
    output: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("namespace "):
            namespace_stack.append(stripped.split(maxsplit=1)[1])
            removed.append(namespace_stack[-1])
            continue
        if namespace_stack and stripped == f"end {namespace_stack[-1]}":
            namespace_stack.pop()
            continue
        output.append(line)
    text = "\n".join(output) + ("\n" if output else "")
    for name in sorted(set(removed), key=len, reverse=True):
        text = text.replace(f"{name}.", "")
    return text


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
            chars: list[str] = []
            for _ in range(width):
                chars.append(alphabet[n % len(alphabet)])
                n //= len(alphabet)
            candidate = "".join(reversed(chars))
            if candidate not in PRESERVED_IDENTIFIERS and candidate not in LEAN_KEYWORDS:
                yield candidate
        width += 1


def _all_identifiers(source: str) -> set[str]:
    return {match.group(0) for match in IDENT_RE.finditer(source)}


def _looks_like_binder_lhs(text: str) -> bool:
    tokens = text.replace(",", " ").split()
    return bool(tokens) and all(token == "_" or IDENT_RE.fullmatch(token) for token in tokens)


def _skip_ws(source: str, i: int) -> int:
    while i < len(source) and source[i].isspace():
        i += 1
    return i


def _parse_header_binder(source: str, i: int) -> int | None:
    if i >= len(source):
        return None
    opening = source[i]
    if opening == "⦃":
        closing = "⦄"
    elif opening in "([{":
        closing = {"(": ")", "[": "]", "{": "}"}[opening]
    else:
        return None
    start = i
    depth = 0
    while i < len(source):
        ch = source[i]
        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                inner = source[start + 1 : i]
                if opening == "[":
                    return i + 1 if "|" not in inner else None
                if ":" not in inner:
                    return None
                if not _looks_like_binder_lhs(inner.split(":", 1)[0].strip()):
                    return None
                if opening == "{" and "|" in inner:
                    return None
                return i + 1
        i += 1
    return None


def _parse_autoimplicit_binders(source: str, i: int) -> int | None:
    start = i
    names: list[str] = []
    while True:
        match = IDENT_RE.match(source, i)
        if not match:
            break
        names.append(match.group(0))
        i = _skip_ws(source, match.end())
    if names and i < len(source) and source[i] == ":":
        return i
    return None if start == i else None


def repair_missing_leading_binder(source: str) -> str:
    match = re.match(
        r"^(\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*theorem\s+"
        r"(?:[A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_']*))\s*:\s*"
        r"([A-Za-z_][A-Za-z0-9_'.*]*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_'.*]*)\)\s*(\(.+)$",
        source,
        re.DOTALL,
    )
    if not match:
        return source
    prefix, binder_type, rest = match.groups()
    candidates = [
        ident
        for ident in IDENT_RE.findall(rest)
        if ident not in PRESERVED_IDENTIFIERS and ident[:1].islower()
    ]
    if not candidates:
        return source
    return f"{prefix} ({candidates[0]} : {binder_type.strip()}) {rest}"


def repair_missing_typed_varlist(source: str) -> str:
    match = re.match(
        r"^(\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*theorem\s+"
        r"(?:[A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_']*))\s+"
        r"((?:[A-Za-z_][A-Za-z0-9_']*\s+)+):\s*"
        r"([A-Za-z_][A-Za-z0-9_'.*]*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_'.*]*)\)\s*(.+)$",
        source,
        re.DOTALL,
    )
    if not match:
        return source
    prefix, vars_text, binder_type, rest = match.groups()
    vars_list = [name for name in vars_text.split() if name]
    candidates = [
        ident
        for ident in IDENT_RE.findall(rest)
        if ident not in PRESERVED_IDENTIFIERS and ident[:1].islower() and ident not in vars_list
    ]
    if not candidates:
        return source
    all_vars = " ".join([candidates[0], *vars_list])
    return f"{prefix} ({all_vars} : {binder_type}) {rest}"


def repair_missing_single_typed_binder(source: str) -> str:
    match = re.match(
        r"^(\s*(?:@\[[^\]]*\]\s*)?(?:private\s+|protected\s+|noncomputable\s+|partial\s+)*theorem\s+"
        r"(?:[A-Za-z_][A-Za-z0-9_']*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_']*))\s*:\s*"
        r"([A-Za-z_][A-Za-z0-9_'.*]*|[α-ωΑ-Ωℕℤℚℝℂ][A-Za-z0-9_'.*]*)\)\s*:\s*(.+)$",
        source,
        re.DOTALL,
    )
    if not match:
        return source
    prefix, binder_type, proposition = match.groups()
    candidates = [
        ident
        for ident in IDENT_RE.findall(proposition)
        if ident not in PRESERVED_IDENTIFIERS and ident[:1].islower()
    ]
    if not candidates:
        return source
    return f"{prefix} ({candidates[0]} : {binder_type}) : {proposition}"


def canonicalize_theorem_header(source: str) -> str:
    source = repair_missing_leading_binder(source)
    source = repair_missing_typed_varlist(source)
    source = repair_missing_single_typed_binder(source)
    match = HEADER_PREFIX_RE.match(source)
    if not match:
        return source
    pos = match.end()
    pos = _skip_ws(source, pos)
    while True:
        next_pos = _parse_header_binder(source, pos)
        if next_pos is not None:
            pos = _skip_ws(source, next_pos)
            continue
        next_pos = _parse_autoimplicit_binders(source, pos)
        if next_pos is not None:
            pos = next_pos
            break
        break
    if pos < len(source) and source[pos] == ":":
        return source
    return source[:pos] + " :" + source[pos:]


def parenthesize_theorem_proposition(source: str) -> str:
    source = canonicalize_theorem_header(source)
    match = HEADER_PREFIX_RE.match(source)
    if not match:
        return source
    pos = _skip_ws(source, match.end())
    while True:
        next_pos = _parse_header_binder(source, pos)
        if next_pos is not None:
            pos = _skip_ws(source, next_pos)
            continue
        next_pos = _parse_autoimplicit_binders(source, pos)
        if next_pos is not None:
            pos = next_pos
            break
        break
    if pos >= len(source) or source[pos] != ":":
        return source
    prop_start = _skip_ws(source, pos + 1)
    sorry_match = re.search(r"\s*:=\s*by\s*sorry\s*$", source)
    if not sorry_match or prop_start >= sorry_match.start():
        return source
    proposition = source[prop_start : sorry_match.start()].strip()
    if not proposition.startswith(("∀", "∃")):
        return source
    if proposition.startswith("(") and proposition.endswith(")"):
        return source
    return source[:prop_start] + "(" + proposition + ")" + source[sorry_match.start() :]


def _identifier_spans_lexical(source: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    in_string = False
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
        if ch == '"':
            in_string = True
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


def _collect_binder_names(source: str) -> set[str]:
    names: set[str] = set()

    def add_from_text(text: str) -> None:
        for match in IDENT_RE.finditer(text):
            name = match.group(0)
            if name not in PRESERVED_IDENTIFIERS and not name.startswith("_"):
                names.add(name)

    header = canonicalize_theorem_header(source)
    match = HEADER_PREFIX_RE.match(header)
    if match:
        pos = _skip_ws(header, match.end())
        while True:
            next_pos = _parse_header_binder(header, pos)
            if next_pos is not None:
                chunk = header[pos:next_pos]
                inner = chunk[1:-1]
                if not chunk.startswith("["):
                    add_from_text(inner.split(":", 1)[0])
                pos = _skip_ws(header, next_pos)
                continue
            next_pos = _parse_autoimplicit_binders(header, pos)
            if next_pos is not None:
                add_from_text(header[pos:next_pos])
                break
            break
    for content in re.findall(r"∀\s+([^,]+),", header):
        add_from_text(content.split(":", 1)[0])
    return names


def anonymize_user_identifiers(source: str) -> tuple[str, dict[str, str]]:
    theorem_match = next(DECL_RE.finditer(source), None)
    theorem_name = theorem_match.group(1) if theorem_match else None
    binder_names = [name for name in sorted(_collect_binder_names(source)) if name != theorem_name]
    existing = _all_identifiers(source)
    mapping: dict[str, str] = {}
    name_iter = _name_stream()

    def next_name() -> str:
        while True:
            candidate = next(name_iter)
            if candidate not in existing and candidate not in mapping.values():
                return candidate

    if theorem_name and theorem_name not in PRESERVED_IDENTIFIERS:
        mapping[theorem_name] = next_name()
    for name in binder_names:
        if name not in PRESERVED_IDENTIFIERS:
            mapping[name] = next_name()
    if not mapping:
        return source, {}
    spans = _identifier_spans_lexical(source)
    source_bytes = source.encode("utf-8")
    replacements = [
        (start, end, mapping[text])
        for start, end, text in spans
        if text in mapping and (start == 0 or source_bytes[start - 1:start] != b".")
    ]
    code = bytearray(source_bytes)
    for start, end, repl in reversed(replacements):
        code[start:end] = repl.encode("utf-8")
    return code.decode("utf-8"), mapping


def qualify_common_constants(source: str) -> str:
    spans = _identifier_spans_lexical(source)
    source_bytes = source.encode("utf-8")
    replacements: list[tuple[int, int, str]] = []
    for start, end, text in spans:
        replacement = QUALIFIED_CONSTANTS.get(text)
        if replacement is None:
            continue
        prev_byte = source_bytes[start - 1 : start] if start > 0 else b""
        if prev_byte == b".":
            continue
        replacements.append((start, end, replacement))
    code = bytearray(source_bytes)
    for start, end, repl in reversed(replacements):
        code[start:end] = repl.encode("utf-8")
    return POLY_C_APPLY_RE.sub("Polynomial.C", code.decode("utf-8"))


def rewrite_postfix_factorial(source: str) -> str:
    prev = None
    while prev != source:
        prev = source
        source = DOUBLE_FACTORIAL_RE.sub(r"(Nat.doubleFactorial \1)", source)
        source = PAREN_FACTORIAL_RE.sub(r"(Nat.factorial \1)", source)
        source = NUM_FACTORIAL_RE.sub(r"(Nat.factorial \1)", source)
        source = IDENT_FACTORIAL_RE.sub(r"(Nat.factorial \1)", source)
    return source


def clean_lean_with_mapping(source: str) -> tuple[str, dict[str, str]]:
    cleaned = remove_empty_lines(unwrap_namespaces(remove_given_lines(strip_lean_comments(source))))
    cleaned = parenthesize_theorem_proposition(cleaned)
    cleaned = rewrite_postfix_factorial(cleaned)
    cleaned = qualify_common_constants(cleaned)
    cleaned, mapping = anonymize_user_identifiers(cleaned)
    return remove_empty_lines(cleaned), mapping


def apply_known_fixes(task_id: str, source: str) -> str:
    if task_id == "lean_workbook_plus_30235":
        source = source.replace("(2 * k + 1) * π", "(2 * (k : ℝ) + 1) * π")
    if task_id in {"lean_workbook_plus_36418", "lean_workbook_plus_76245"}:
        source = re.sub(r"^theorem\s+\S+\s+x y z\s+\(", lambda m: m.group(0).replace(" x y z (", " (x y z : ℝ) ("), source)
    if task_id == "lean_workbook_plus_43296":
        source = re.sub(r"^theorem\s+(\S+)\s+b c:\s*ℝ\)\s*:", r"theorem \1 (a b c : ℝ) :", source)
    if task_id == "lean_workbook_plus_48585":
        source = re.sub(r"^theorem\s+(\S+)\s+tan_eq_v\s+\(f\s*:\s*ℝ\s*→\s*ℝ\)", r"theorem \1 (tan_eq_v : ℝ) (f : ℝ → ℝ)", source)
    if task_id == "lean_workbook_plus_56652":
        source = source.replace("π = 3.14", "π = (157 / 50 : ℝ)")
    if task_id == "lean_workbook_plus_39807":
        source = source.replace("(a b c : ℝ)", "(a c : ℝ) (b : ℤ)")
        source = source.replace("hb : b = ⌊π * 10^6⌋", "hb : b = Int.floor (π * (10 ^ 6 : ℝ))")
        source = source.replace("(2 * ⌊b⌋ ≤ ⌊2 * b⌋ ∧ ⌊2 * b⌋ ≤ 2 * ⌊b⌋ + 1)", "(2 * b ≤ Int.floor (2 * (b : ℝ)) ∧ Int.floor (2 * (b : ℝ)) ≤ 2 * b + 1)")
    if task_id == "lean_workbook_plus_68471":
        source = re.sub(r"^theorem\s+(\S+)\s+b c d:\s*ℝ\)\s*:", r"theorem \1 (a b c d : ℝ) :", source)
    if task_id == "lean_workbook_plus_70971":
        source = source.replace(
            "(a : ℕ → ℝ) (x : ℕ → ℝ) (n : ℕ) (h₁ : a = fun (n:ℕ) ↦ a₁ - (n-1)*π/8) (h₂ : x = fun (n:ℕ) ↦ tan (a n)) : x n = tan (a₁ - (n-1)*π/8)",
            "(a₁ : ℝ) (a : ℕ → ℝ) (x : ℕ → ℝ) (n : ℕ) "
            "(h₁ : a = fun n : ℕ => a₁ - (((n : ℝ) - 1) * π / 8)) "
            "(h₂ : x = fun n : ℕ => Real.tan (a n)) : x n = Real.tan (a₁ - (((n : ℝ) - 1) * π / 8))",
        )
    if task_id == "lean_workbook_plus_72687":
        source = re.sub(r"^theorem\s+(\S+)\s+⦃a b : ℝ⦄", r"theorem \1 {a b : ℝ}", source)
    if task_id == "lean_workbook_plus_71131":
        source = source.replace(
            "p n k = (n! * 3^(n - 1))⁻¹",
            "p n k = (((n! : ℚ) * (3 : ℚ)^(n - 1))⁻¹)",
        )
    if task_id == "lean_workbook_plus_12763":
        source = source.replace(
            "x ∈ closure {n! / π % 1 | n : ℕ}",
            "x ∈ closure {(((n! : ℝ) / π) % 1) | n : ℕ}",
        )
    if task_id == "lean_workbook_plus_23082":
        source = source.replace(
            "∑' n : ℕ, (1/(n!)^2)",
            "∑' n : ℕ, ((1 : ℚ) / ((n! : ℚ)^2))",
        )
    return source


def build_vocab(tokenized_rows: list[list[int]], tokenizer_name: str) -> dict[str, Any]:
    original_ids = sorted({token_id for row in tokenized_rows for token_id in row})
    id_to_original: list[int | None] = [None, None, None, None, *original_ids]
    return {
        "id_to_original": id_to_original,
        "local_bos_id": 1,
        "local_eos_id": 2,
        "local_pad_id": 0,
        "local_unk_id": 3,
        "original_to_local": {str(token_id): i + 4 for i, token_id in enumerate(original_ids)},
        "source": "comment-stripped cleaned Lean Workbook Plus formal_statement targets",
        "special_tokens": {"bos": "<LOCAL_BOS>", "eos": "<LOCAL_EOS>", "pad": "<LOCAL_PAD>", "unk": "<LOCAL_UNK>"},
        "tokenizer": tokenizer_name,
        "vocab_size": len(id_to_original),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.input.resolve())
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    cleaned_rows: list[dict[str, Any]] = []
    cleaned_texts: list[str] = []
    tokenized_rows: list[list[int]] = []
    for index, row in enumerate(rows):
        fixed_source = apply_known_fixes(str(row["id"]), str(row["formal_statement"]))
        cleaned, rename_map = clean_lean_with_mapping(fixed_source)
        token_ids = [int(token_id) for token_id in tokenizer.encode(cleaned, add_special_tokens=False)]
        if not token_ids:
            raise ValueError(f"row {index} cleaned to zero tokens")
        cleaned_rows.append(
            {
                "index": index,
                "task_id": str(row["id"]),
                "status": str(row["status"]),
                "natural_language_statement": str(row["natural_language_statement"]),
                "formal_statement": str(row["formal_statement"]),
                "target_text": cleaned,
                "rename_map": rename_map,
            }
        )
        cleaned_texts.append(cleaned)
        tokenized_rows.append(token_ids)

    if args.preview > 0:
        for row, text, token_ids in zip(cleaned_rows[: args.preview], cleaned_texts[: args.preview], tokenized_rows[: args.preview], strict=True):
            print("=" * 88)
            print(f"index={row['index']} task_id={row['task_id']} tokens={len(token_ids)}")
            print(f"rename_map={row['rename_map']}")
            print(text)
        return

    vocab = build_vocab(tokenized_rows, tokenizer_name=args.tokenizer)
    original_to_local = vocab["original_to_local"]
    for row, original_ids in zip(cleaned_rows, tokenized_rows, strict=True):
        row["target_original_ids"] = original_ids
        row["target_local_ids"] = [int(original_to_local[str(token_id)]) for token_id in original_ids]

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "manifest.jsonl", cleaned_rows)
    (out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {
        "rows": len(cleaned_rows),
        "input": str(args.input.resolve()),
        "out_dir": str(out_dir),
        "tokenizer": args.tokenizer,
        "target_tokens": sum(len(row["target_original_ids"]) for row in cleaned_rows),
        "vocab_size": vocab["vocab_size"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
